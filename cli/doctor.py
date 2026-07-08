"""Preflight checks: catch a broken setup before a run burns tokens.

`tradingagents doctor [TICKER]` verifies the effective configuration (saved
settings + env overrides), the provider's API key, local endpoint
reachability, data-vendor keys, and — when a ticker is given — that the
symbol resolves to a real instrument. No LLM calls are made, so the check
itself costs nothing.
"""

from __future__ import annotations

import os
import sys

from rich.console import Console

from cli.user_settings import load_saved_settings, settings_path
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.llm_clients.api_key_env import get_api_key_env

console = Console()

_PASS, _WARN, _FAIL = "pass", "warn", "fail"
_ICONS = {_PASS: "[green]✓[/green]", _WARN: "[yellow]![/yellow]", _FAIL: "[red]✗[/red]"}

# Extra env vars some providers need beyond the API key.
_EXTRA_REQUIRED_ENV = {
    "azure": ("AZURE_OPENAI_ENDPOINT", "OPENAI_API_VERSION"),
}


def _report(results: list, status: str, name: str, detail: str) -> None:
    results.append(status)
    console.print(f"{_ICONS[status]} [bold]{name}[/bold] — {detail}")


def _check_settings(results: list) -> dict:
    path = settings_path()
    saved = load_saved_settings()
    if saved:
        _report(results, _PASS, "Saved settings", f"loaded from {path}")
    elif path.exists():
        _report(
            results, _WARN, "Saved settings",
            f"{path} exists but couldn't be read — run `tradingagents config reset`",
        )
    else:
        _report(
            results, _WARN, "Saved settings",
            f"none yet ({path}) — created after your first interactive run",
        )
    return saved or {}


def _check_api_key(results: list, provider: str) -> None:
    if provider == "bedrock":
        has_auth = bool(
            os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
            or os.environ.get("AWS_ACCESS_KEY_ID")
            or os.environ.get("AWS_PROFILE")
        )
        _report(
            results, _PASS if has_auth else _WARN, "Bedrock auth",
            "AWS credentials/bearer token found in env"
            if has_auth
            else "no AWS_BEARER_TOKEN_BEDROCK / AWS_ACCESS_KEY_ID / AWS_PROFILE in "
            "env — ~/.aws/credentials or an IAM role may still cover it",
        )
        return

    env_var = get_api_key_env(provider)
    if env_var is None:
        _report(results, _PASS, "API key", f"{provider} needs no API key")
        return

    from tradingagents.llm_clients.openai_client import OPENAI_COMPATIBLE_PROVIDERS

    spec = OPENAI_COMPATIBLE_PROVIDERS.get(provider)
    optional = spec is not None and spec.key_optional
    if os.environ.get(env_var):
        _report(results, _PASS, "API key", f"{env_var} is set")
    elif optional:
        _report(
            results, _PASS, "API key",
            f"{env_var} not set (fine — optional for {provider})",
        )
    else:
        _report(
            results, _FAIL, "API key",
            f"{env_var} is not set — export it or add it to .env",
        )

    for extra in _EXTRA_REQUIRED_ENV.get(provider, ()):
        if os.environ.get(extra):
            _report(results, _PASS, "Provider env", f"{extra} is set")
        else:
            _report(results, _FAIL, "Provider env", f"{extra} is not set")


def _check_endpoint(results: list, provider: str, backend_url: str | None) -> None:
    """Probe local/self-hosted endpoints; remote SaaS endpoints are not pinged."""
    if provider not in ("ollama", "openai_compatible"):
        return
    if not backend_url:
        _report(
            results, _FAIL, "Endpoint",
            "no backend URL configured — pass --backend-url or set "
            "TRADINGAGENTS_LLM_BACKEND_URL",
        )
        return

    import requests

    url = backend_url.rstrip("/") + "/models"
    try:
        # Any HTTP response proves the server is reachable — a 401/403 from a
        # keyed relay is still "up", so only transport errors fail the check.
        response = requests.get(url, timeout=3)
        if response.ok:
            try:
                served = len(response.json().get("data", []))
                detail = f"{backend_url} is up ({served} model(s) served)"
            except ValueError:
                detail = f"{backend_url} is up"
        else:
            detail = f"{backend_url} is up (HTTP {response.status_code} from /models)"
        _report(results, _PASS, "Endpoint", detail)
    except Exception as exc:
        hint = (
            "is `ollama serve` running?" if provider == "ollama"
            else "is the server running on that host/port?"
        )
        _report(
            results, _FAIL, "Endpoint",
            f"could not reach {url} ({type(exc).__name__}) — {hint}",
        )


def _check_data_vendors(results: list) -> None:
    vendors = DEFAULT_CONFIG.get("data_vendors", {})
    vendor_blob = ",".join(str(v) for v in vendors.values())
    if "alpha_vantage" in vendor_blob:
        if os.environ.get("ALPHA_VANTAGE_API_KEY"):
            _report(results, _PASS, "Alpha Vantage", "ALPHA_VANTAGE_API_KEY is set")
        else:
            _report(
                results, _WARN, "Alpha Vantage",
                "configured as a data vendor but ALPHA_VANTAGE_API_KEY is not set",
            )
    if vendors.get("macro_data") == "fred":
        if os.environ.get("FRED_API_KEY"):
            _report(results, _PASS, "FRED", "FRED_API_KEY is set")
        else:
            _report(
                results, _WARN, "FRED",
                "FRED_API_KEY not set — the news analyst runs without macro "
                "indicators (free key: fred.stlouisfed.org)",
            )


def _check_results_dir(results: list) -> None:
    from pathlib import Path

    results_dir = Path(DEFAULT_CONFIG["results_dir"])
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        probe = results_dir / ".doctor-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        _report(results, _PASS, "Results dir", f"{results_dir} is writable")
    except OSError as exc:
        _report(results, _FAIL, "Results dir", f"{results_dir}: {exc}")


def _check_ticker(results: list, ticker: str) -> None:
    from tradingagents.agents.utils.agent_utils import resolve_instrument_identity
    from tradingagents.dataflows.symbol_utils import normalize_symbol

    symbol = normalize_symbol(ticker)
    note = f" (normalized from {ticker!r})" if symbol != ticker.strip().upper() else ""
    identity = resolve_instrument_identity(symbol)
    name = (identity or {}).get("company_name")
    if name:
        _report(results, _PASS, "Ticker", f"{symbol} resolves to {name}{note}")
    else:
        _report(
            results, _WARN, "Ticker",
            f"could not resolve {symbol}{note} — check the spelling and "
            f"exchange suffix (e.g. 0700.HK, RELIANCE.NS)",
        )


def run_doctor(ticker: str | None = None) -> int:
    """Run all checks; returns the process exit code (0 ok, 1 on failures)."""
    results: list = []

    version = sys.version.split()[0]
    _report(results, _PASS, "Python", version)

    saved = _check_settings(results)

    # Effective provider/model view, resolved exactly like a --yes run would.
    from cli.main import _resolve_run_settings
    from cli.models import AssetType

    settings, src, problem = _resolve_run_settings({}, saved, AssetType.STOCK)
    provider = settings["llm_provider"]
    source = {"env": "env", "saved": "saved settings", "default": "defaults"}.get(
        src.get("provider"), "defaults"
    )
    from cli.utils import known_provider_keys

    if provider not in known_provider_keys():
        # An unknown provider would sail through the key check (no env-var
        # mapping means "no key needed"), false-passing a broken setup.
        _report(
            results, _FAIL, "Provider",
            f"{provider!r} (from {source}) is not a known provider key — "
            f"valid: {', '.join(known_provider_keys())}",
        )
    else:
        _report(
            results, _PASS, "Provider",
            f"{provider} (from {source}) — quick={settings['shallow_thinker']}, "
            f"deep={settings['deep_thinker']}, language={settings['output_language']}",
        )
    if problem:
        _report(results, _WARN, "Run settings", problem)

    _check_api_key(results, provider)
    _check_endpoint(results, provider, settings.get("backend_url"))
    _check_data_vendors(results)
    _check_results_dir(results)
    if ticker:
        _check_ticker(results, ticker)

    failures = results.count(_FAIL)
    warnings = results.count(_WARN)
    console.print()
    if failures:
        console.print(
            f"[red]{failures} check(s) failed[/red] — fix the ✗ items above, "
            f"then re-run `tradingagents doctor`."
        )
        return 1
    if warnings:
        console.print(
            f"[green]Ready to run.[/green] [dim]({warnings} advisory note(s) above)[/dim]"
        )
    else:
        console.print("[green]All checks passed — ready to run.[/green]")
    return 0
