"""Desktop app launcher: `tradingagents desktop` puts a double-clickable
TradingAgents icon on the desktop.

Per platform:

- Windows — a ``TradingAgents.lnk`` shortcut (created via PowerShell's
  WScript.Shell COM object, no extra dependencies) that opens a console
  window running the CLI and keeps it open after the run so the report
  stays readable. Uses the bundled ``.ico``.
- macOS — a minimal ``TradingAgents.app`` bundle whose executable tells
  Terminal to run the CLI; the bundled PNG is converted to ``.icns`` with
  the system ``sips`` tool when available.
- Linux — an XDG ``TradingAgents.desktop`` entry (``Terminal=true``) on the
  desktop and in ``~/.local/share/applications`` so it also appears in the
  app menu.

The launcher targets the installed ``tradingagents`` console script when it
is on PATH, else falls back to ``<current python> -m cli.main``, so it works
from both pip installs and source checkouts.
"""

from __future__ import annotations

import platform
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from rich.console import Console

console = Console()

APP_NAME = "TradingAgents"
_STATIC = Path(__file__).parent / "static"


class DesktopError(RuntimeError):
    """A launcher could not be created; the message is user-facing."""


def _resolve_run_command() -> list[str]:
    """The command the launcher should run, as argv."""
    exe = shutil.which("tradingagents")
    if exe:
        return [exe]
    return [sys.executable, "-m", "cli.main"]


def _app_dir() -> Path:
    """Directory for installed launcher assets (icons), under the same
    ~/.tradingagents home the rest of the app uses."""
    return Path.home() / ".tradingagents" / "app"


def _install_icon(suffix: str) -> Path | None:
    """Copy the bundled icon (.png/.ico) next to the launcher assets."""
    source = _STATIC / f"tradingagents{suffix}"
    if not source.exists():
        return None
    target_dir = _app_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    shutil.copyfile(source, target)
    return target


# ---------------------------------------------------------------------------
# Desktop directory resolution
# ---------------------------------------------------------------------------

def _windows_desktop_dir() -> Path:
    """The user's Desktop on Windows, honoring OneDrive redirection."""
    try:
        out = subprocess.run(
            [
                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                "[Environment]::GetFolderPath('Desktop')",
            ],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip()
        if out:
            return Path(out)
    except Exception:  # noqa: BLE001 — any failure falls back to the default
        pass
    return Path.home() / "Desktop"


def _linux_desktop_dir() -> Path:
    """The XDG desktop directory, from user-dirs.dirs when configured."""
    config = Path.home() / ".config" / "user-dirs.dirs"
    try:
        for line in config.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("XDG_DESKTOP_DIR="):
                raw = line.split("=", 1)[1].strip().strip('"')
                return Path(raw.replace("$HOME", str(Path.home())))
    except OSError:
        pass
    return Path.home() / "Desktop"


# ---------------------------------------------------------------------------
# Launcher content builders (pure — unit-testable)
# ---------------------------------------------------------------------------

def _ps_quote(value: str) -> str:
    """Quote a string as a PowerShell single-quoted literal."""
    return "'" + value.replace("'", "''") + "'"


def _windows_shortcut_script(
    lnk_path: Path, command: list[str], icon: Path | None
) -> str:
    """PowerShell that writes a .lnk opening the CLI in a persistent console.

    ``cmd /k`` keeps the window open after the run finishes so the final
    report and decision stay on screen.
    """
    quoted = subprocess.list2cmdline(command)
    arguments = f'/k "{quoted}"'
    lines = [
        "$ws = New-Object -ComObject WScript.Shell",
        f"$s = $ws.CreateShortcut({_ps_quote(str(lnk_path))})",
        "$s.TargetPath = \"$env:ComSpec\"",
        f"$s.Arguments = {_ps_quote(arguments)}",
        f"$s.WorkingDirectory = {_ps_quote(str(Path.home()))}",
        f"$s.Description = {_ps_quote('TradingAgents: Multi-Agents LLM Financial Trading Framework')}",
    ]
    if icon is not None:
        lines.append(f"$s.IconLocation = {_ps_quote(str(icon) + ',0')}")
    lines.append("$s.Save()")
    return "; ".join(lines)


def _xdg_exec_quote(command: list[str]) -> str:
    """Quote an argv for a .desktop Exec= line (Desktop Entry spec)."""
    parts = []
    for arg in command:
        escaped = arg.replace("%", "%%")
        if any(ch in escaped for ch in ' \t"\'\\'):
            escaped = '"' + escaped.replace("\\", "\\\\").replace('"', '\\"') + '"'
        parts.append(escaped)
    return " ".join(parts)


def _linux_desktop_entry(command: list[str], icon: Path | None) -> str:
    lines = [
        "[Desktop Entry]",
        "Type=Application",
        f"Name={APP_NAME}",
        "Comment=Multi-Agents LLM Financial Trading Framework",
        f"Exec={_xdg_exec_quote(command)}",
        "Terminal=true",
        "Categories=Office;Finance;",
    ]
    if icon is not None:
        lines.append(f"Icon={icon}")
    return "\n".join(lines) + "\n"


def _applescript_quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _macos_launcher_script(command: list[str]) -> str:
    """The .app executable: tell Terminal to run the CLI in a new window."""
    shell_cmd = " ".join(_sh_quote(arg) for arg in command)
    escaped = _applescript_quote(shell_cmd)
    return (
        "#!/bin/bash\n"
        "osascript <<'EOF'\n"
        'tell application "Terminal"\n'
        "  activate\n"
        f'  do script "{escaped}"\n'
        "end tell\n"
        "EOF\n"
    )


def _sh_quote(value: str) -> str:
    if not value or any(ch in value for ch in " \t\"'\\$&|;<>()`"):
        return "'" + value.replace("'", "'\"'\"'") + "'"
    return value


def _macos_info_plist() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n'
        "<dict>\n"
        f"  <key>CFBundleName</key><string>{APP_NAME}</string>\n"
        f"  <key>CFBundleDisplayName</key><string>{APP_NAME}</string>\n"
        "  <key>CFBundleIdentifier</key><string>ai.tauric.tradingagents.launcher</string>\n"
        f"  <key>CFBundleExecutable</key><string>{APP_NAME}</string>\n"
        "  <key>CFBundleIconFile</key><string>tradingagents</string>\n"
        "  <key>CFBundlePackageType</key><string>APPL</string>\n"
        "  <key>CFBundleShortVersionString</key><string>1.0</string>\n"
        "</dict>\n"
        "</plist>\n"
    )


# ---------------------------------------------------------------------------
# Per-platform creation
# ---------------------------------------------------------------------------

def _create_windows(command: list[str]) -> Path:
    desktop = _windows_desktop_dir()
    desktop.mkdir(parents=True, exist_ok=True)
    icon = _install_icon(".ico")
    lnk = desktop / f"{APP_NAME}.lnk"
    script = _windows_shortcut_script(lnk, command, icon)
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=30, check=True,
        )
    except FileNotFoundError as exc:
        raise DesktopError(
            "PowerShell was not found — it is required to create the shortcut."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise DesktopError(
            f"Could not create the shortcut: {exc.stderr.strip() or exc}"
        ) from exc
    return lnk


def _create_macos(command: list[str]) -> Path:
    desktop = Path.home() / "Desktop"
    desktop.mkdir(parents=True, exist_ok=True)
    bundle = desktop / f"{APP_NAME}.app"
    macos_dir = bundle / "Contents" / "MacOS"
    resources = bundle / "Contents" / "Resources"
    macos_dir.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)

    (bundle / "Contents" / "Info.plist").write_text(
        _macos_info_plist(), encoding="utf-8"
    )
    launcher = macos_dir / APP_NAME
    launcher.write_text(_macos_launcher_script(command), encoding="utf-8")
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    png = _install_icon(".png")
    if png is not None and shutil.which("sips"):
        # Best-effort icon: sips converts the bundled PNG to .icns; without it
        # the app simply keeps the generic icon.
        subprocess.run(
            ["sips", "-s", "format", "icns", str(png),
             "--out", str(resources / "tradingagents.icns")],
            capture_output=True, timeout=30, check=False,
        )
    return bundle


def _create_linux(command: list[str]) -> Path:
    desktop = _linux_desktop_dir()
    desktop.mkdir(parents=True, exist_ok=True)
    icon = _install_icon(".png")
    entry = _linux_desktop_entry(command, icon)

    launcher = desktop / f"{APP_NAME}.desktop"
    launcher.write_text(entry, encoding="utf-8")
    launcher.chmod(0o755)
    # Some desktops require the file to be marked trusted before double-click
    # works; best-effort, silently skipped where gio is unavailable.
    if shutil.which("gio"):
        subprocess.run(
            ["gio", "set", str(launcher), "metadata::trusted", "true"],
            capture_output=True, timeout=10, check=False,
        )

    # Also register in the application menu.
    applications = Path.home() / ".local" / "share" / "applications"
    try:
        applications.mkdir(parents=True, exist_ok=True)
        (applications / f"{APP_NAME}.desktop").write_text(entry, encoding="utf-8")
    except OSError:
        pass  # menu registration is a bonus, not a requirement
    return launcher


def create_desktop_launcher() -> Path:
    """Create the platform's desktop launcher; returns its path."""
    command = _resolve_run_command()
    system = platform.system()
    if system == "Windows":
        return _create_windows(command)
    if system == "Darwin":
        return _create_macos(command)
    if system == "Linux":
        return _create_linux(command)
    raise DesktopError(f"Unsupported platform for a desktop launcher: {system}")
