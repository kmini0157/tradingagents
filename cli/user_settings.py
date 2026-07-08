"""Persisted CLI run settings, so a returning user never re-answers the wizard.

The wizard's non-per-run choices (analysts, depth, provider, models, effort,
language) are saved after each interactive run and offered back on the next
one — as a one-confirm fast path, as prompt defaults after "customize", and as
the answer source for ``--yes`` runs.

Precedence stays: CLI flag > TRADINGAGENTS_* env var > saved settings >
built-in default. The library's ``DEFAULT_CONFIG`` never reads this file, so
programmatic behavior and the env-override contract are untouched.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from cli.models import AnalystType

SETTINGS_VERSION = 1

# Keys persisted from the selections dict, with a light type check applied on
# load so a hand-edited file degrades field-by-field instead of all-or-nothing.
_STR_KEYS = (
    "last_ticker",
    "llm_provider",
    "backend_url",
    "quick_think_llm",
    "deep_think_llm",
    "google_thinking_level",
    "openai_reasoning_effort",
    "anthropic_effort",
    "output_language",
)

# The fast path replays a full run, so it needs the complete LLM selection;
# anything less falls back to the stepwise wizard (with whatever partial
# defaults did load).
CORE_KEYS = (
    "llm_provider",
    "quick_think_llm",
    "deep_think_llm",
    "output_language",
    "research_depth",
    "analysts",
)

_VALID_ANALYSTS = {a.value for a in AnalystType}


def settings_path() -> Path:
    """Resolve the settings file path (`TRADINGAGENTS_SETTINGS_PATH` overrides)."""
    override = os.environ.get("TRADINGAGENTS_SETTINGS_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".tradingagents" / "settings.json"


def load_saved_settings() -> dict | None:
    """Load saved settings, tolerantly.

    Returns a dict of only the recognized, well-typed fields, or None when the
    file is absent or unreadable. Never raises: a corrupt settings file must
    not take down the CLI, the wizard just runs from scratch.
    """
    path = settings_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None

    settings: dict = {}
    for key in _STR_KEYS:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            settings[key] = value.strip()

    depth = raw.get("research_depth")
    if isinstance(depth, int) and not isinstance(depth, bool) and depth > 0:
        settings["research_depth"] = depth

    analysts = raw.get("analysts")
    if isinstance(analysts, list):
        valid = [a for a in analysts if isinstance(a, str) and a in _VALID_ANALYSTS]
        if valid:
            settings["analysts"] = valid

    return settings or None


def has_core_settings(settings: dict | None) -> bool:
    """Whether saved settings are complete enough to replay a run."""
    return settings is not None and all(k in settings for k in CORE_KEYS)


def save_settings(selections: dict) -> Path | None:
    """Persist the run selections for the next session. Best-effort: returns
    the path on success, None on any failure (a read-only home directory must
    not fail the analysis run that just finished selecting).
    """
    analysts = [
        a.value if isinstance(a, AnalystType) else str(a)
        for a in selections.get("analysts", [])
    ]
    payload = {
        "version": SETTINGS_VERSION,
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_ticker": selections.get("ticker"),
        "analysts": analysts,
        "research_depth": selections.get("research_depth"),
        "llm_provider": selections.get("llm_provider"),
        # Ollama endpoints are re-resolved from OLLAMA_BASE_URL on every run;
        # freezing the resolved URL here would pin a stale host (the client
        # prefers an explicit base_url over the env var).
        "backend_url": (
            None
            if selections.get("llm_provider") == "ollama"
            else selections.get("backend_url")
        ),
        "quick_think_llm": selections.get("shallow_thinker"),
        "deep_think_llm": selections.get("deep_thinker"),
        "google_thinking_level": selections.get("google_thinking_level"),
        "openai_reasoning_effort": selections.get("openai_reasoning_effort"),
        "anthropic_effort": selections.get("anthropic_effort"),
        "output_language": selections.get("output_language"),
    }
    path = settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        return path
    except OSError:
        return None


def clear_settings() -> bool:
    """Delete the settings file. Returns True when a file was removed."""
    path = settings_path()
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
