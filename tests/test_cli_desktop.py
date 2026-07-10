"""Desktop launcher creation (`tradingagents desktop`).

The Linux path is exercised end-to-end against a temp HOME; Windows/macOS
run through the same content builders with the OS-specific side effects
(PowerShell, sips) mocked, since neither runs in CI.
"""

from pathlib import Path
from unittest import mock

import pytest

from cli import desktop


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


@pytest.mark.unit
class TestRunCommandResolution:
    def test_prefers_installed_console_script(self, monkeypatch):
        monkeypatch.setattr(desktop.shutil, "which", lambda _: "/usr/bin/tradingagents")
        assert desktop._resolve_run_command() == ["/usr/bin/tradingagents"]

    def test_falls_back_to_module_invocation(self, monkeypatch):
        monkeypatch.setattr(desktop.shutil, "which", lambda _: None)
        command = desktop._resolve_run_command()
        assert command[0] == desktop.sys.executable
        assert command[1:] == ["-m", "cli.main"]


@pytest.mark.unit
class TestContentBuilders:
    def test_linux_desktop_entry_shape(self):
        entry = desktop._linux_desktop_entry(
            ["/opt/py env/bin/tradingagents"], Path("/icons/ta.png")
        )
        assert "[Desktop Entry]" in entry
        assert "Terminal=true" in entry
        assert 'Exec="/opt/py env/bin/tradingagents"' in entry  # space → quoted
        assert "Icon=/icons/ta.png" in entry

    def test_xdg_exec_quoting_escapes_percent_and_quotes(self):
        quoted = desktop._xdg_exec_quote(["/bin/run 100%", 'say "hi"'])
        assert "%%" in quoted
        assert '\\"hi\\"' in quoted

    def test_windows_shortcut_script_uses_cmd_k_and_icon(self):
        script = desktop._windows_shortcut_script(
            Path(r"C:\Users\u\Desktop\TradingAgents.lnk"),
            [r"C:\Program Files\ta\tradingagents.exe"],
            Path(r"C:\Users\u\.tradingagents\app\tradingagents.ico"),
        )
        assert "WScript.Shell" in script
        assert "$env:ComSpec" in script
        assert '/k ""C:\\Program Files\\ta\\tradingagents.exe""' in script
        assert "tradingagents.ico,0" in script
        assert script.rstrip().endswith("$s.Save()")

    def test_powershell_single_quote_escaping(self):
        assert desktop._ps_quote("it's") == "'it''s'"

    def test_macos_launcher_script_quotes_for_shell_and_applescript(self):
        script = desktop._macos_launcher_script(["/opt/py env/bin/python", "-m", "cli.main"])
        assert script.startswith("#!/bin/bash")
        assert 'tell application "Terminal"' in script
        # The space-containing path is shell-quoted, then AppleScript-escaped.
        assert "do script \"'/opt/py env/bin/python' -m cli.main\"" in script

    def test_macos_info_plist_names_the_executable(self):
        plist = desktop._macos_info_plist()
        assert "<key>CFBundleExecutable</key><string>TradingAgents</string>" in plist
        assert "tradingagents" in plist  # icon file basename


@pytest.mark.unit
class TestLinuxLauncher:
    def test_creates_desktop_file_and_menu_entry(self, fake_home, monkeypatch):
        monkeypatch.setattr(desktop.shutil, "which", lambda name: None)
        launcher = desktop._create_linux(["/usr/bin/tradingagents"])

        assert launcher == fake_home / "Desktop" / "TradingAgents.desktop"
        content = launcher.read_text()
        assert "Exec=/usr/bin/tradingagents" in content
        assert launcher.stat().st_mode & 0o111  # executable
        menu = fake_home / ".local/share/applications/TradingAgents.desktop"
        assert menu.read_text() == content
        icon = fake_home / ".tradingagents/app/tradingagents.png"
        assert icon.exists()
        assert f"Icon={icon}" in content

    def test_honors_xdg_desktop_dir(self, fake_home, monkeypatch):
        monkeypatch.setattr(desktop.shutil, "which", lambda name: None)
        config = fake_home / ".config"
        config.mkdir()
        (config / "user-dirs.dirs").write_text(
            'XDG_DESKTOP_DIR="$HOME/Schreibtisch"\n', encoding="utf-8"
        )
        launcher = desktop._create_linux(["/usr/bin/tradingagents"])
        assert launcher.parent == fake_home / "Schreibtisch"


@pytest.mark.unit
class TestWindowsLauncher:
    def test_invokes_powershell_with_shortcut_script(self, fake_home, monkeypatch):
        runs = []

        def fake_run(cmd, **kwargs):
            runs.append(cmd)
            result = mock.Mock()
            result.stdout = str(fake_home / "Desktop")
            return result

        monkeypatch.setattr(desktop.subprocess, "run", fake_run)
        lnk = desktop._create_windows([r"C:\ta\tradingagents.exe"])

        assert lnk == fake_home / "Desktop" / "TradingAgents.lnk"
        # First call resolves the Desktop folder, second writes the shortcut.
        assert "GetFolderPath('Desktop')" in runs[0][-1]
        assert "CreateShortcut" in runs[1][-1]

    def test_missing_powershell_raises_desktop_error(self, fake_home, monkeypatch):
        def fake_run(cmd, **kwargs):
            raise FileNotFoundError("powershell")

        monkeypatch.setattr(desktop.subprocess, "run", fake_run)
        with pytest.raises(desktop.DesktopError):
            desktop._create_windows([r"C:\ta\tradingagents.exe"])


@pytest.mark.unit
class TestMacLauncher:
    def test_builds_app_bundle(self, fake_home, monkeypatch):
        monkeypatch.setattr(desktop.shutil, "which", lambda name: None)  # no sips
        bundle = desktop._create_macos(["/usr/local/bin/tradingagents"])

        assert bundle == fake_home / "Desktop" / "TradingAgents.app"
        launcher = bundle / "Contents" / "MacOS" / "TradingAgents"
        assert launcher.stat().st_mode & 0o111
        assert "Terminal" in launcher.read_text()
        assert (bundle / "Contents" / "Info.plist").exists()


@pytest.mark.unit
class TestDispatch:
    @pytest.mark.parametrize(
        ("system", "target"),
        [("Windows", "_create_windows"), ("Darwin", "_create_macos"), ("Linux", "_create_linux")],
    )
    def test_routes_by_platform(self, monkeypatch, system, target):
        monkeypatch.setattr(desktop.platform, "system", lambda: system)
        with mock.patch.object(desktop, target, return_value=Path("/made")) as made:
            assert desktop.create_desktop_launcher() == Path("/made")
        made.assert_called_once()

    def test_unknown_platform_raises(self, monkeypatch):
        monkeypatch.setattr(desktop.platform, "system", lambda: "Plan9")
        with pytest.raises(desktop.DesktopError):
            desktop.create_desktop_launcher()


@pytest.mark.unit
class TestIconAssets:
    def test_bundled_icons_exist_and_are_valid(self):
        png = desktop._STATIC / "tradingagents.png"
        ico = desktop._STATIC / "tradingagents.ico"
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        # ICO header: reserved=0, type=1 (icon), count=1
        assert ico.read_bytes()[:6] == b"\x00\x00\x01\x00\x01\x00"


@pytest.mark.unit
class TestDesktopCommand:
    def test_command_reports_created_path(self, monkeypatch):
        from typer.testing import CliRunner

        import cli.main as m

        monkeypatch.setattr(
            "cli.desktop.create_desktop_launcher", lambda: Path("/desk/TradingAgents.lnk")
        )
        result = CliRunner().invoke(m.app, ["desktop"])
        assert result.exit_code == 0
        assert "TradingAgents.lnk" in result.output

    def test_command_exits_one_on_desktop_error(self, monkeypatch):
        from typer.testing import CliRunner

        import cli.main as m

        def boom():
            raise desktop.DesktopError("no PowerShell")

        monkeypatch.setattr("cli.desktop.create_desktop_launcher", boom)
        result = CliRunner().invoke(m.app, ["desktop"])
        assert result.exit_code == 1
        assert "no PowerShell" in result.output
