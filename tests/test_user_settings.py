"""Saved CLI settings: persistence, tolerant loading, and the replay contract.

The settings file sits BELOW env vars and CLI flags in precedence and must
never take down the CLI: absent, corrupt, or hand-edited files degrade to
"no saved settings" (or drop just the bad fields), not to a crash.
"""

import json

import pytest

from cli import user_settings
from cli.models import AnalystType


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setenv("TRADINGAGENTS_SETTINGS_PATH", str(path))
    return path


SELECTIONS = {
    "ticker": "0700.HK",
    "analysts": [AnalystType.MARKET, AnalystType.NEWS],
    "research_depth": 3,
    "llm_provider": "anthropic",
    "backend_url": "https://api.anthropic.com/",
    "shallow_thinker": "claude-sonnet-5",
    "deep_thinker": "claude-fable-5",
    "google_thinking_level": None,
    "openai_reasoning_effort": None,
    "anthropic_effort": "high",
    "output_language": "Korean",
}


def test_settings_path_honors_env_override(settings_file):
    assert user_settings.settings_path() == settings_file


def test_save_load_round_trip(settings_file):
    assert user_settings.save_settings(SELECTIONS) == settings_file
    loaded = user_settings.load_saved_settings()
    assert loaded["last_ticker"] == "0700.HK"
    assert loaded["analysts"] == ["market", "news"]
    assert loaded["research_depth"] == 3
    assert loaded["llm_provider"] == "anthropic"
    assert loaded["backend_url"] == "https://api.anthropic.com/"
    assert loaded["quick_think_llm"] == "claude-sonnet-5"
    assert loaded["deep_think_llm"] == "claude-fable-5"
    assert loaded["anthropic_effort"] == "high"
    assert loaded["output_language"] == "Korean"
    # None-valued knobs are omitted on load, not returned as None.
    assert "google_thinking_level" not in loaded


def test_round_trip_is_replay_complete(settings_file):
    user_settings.save_settings(SELECTIONS)
    assert user_settings.has_core_settings(user_settings.load_saved_settings())


def test_ollama_backend_url_not_frozen(settings_file):
    """The client prefers an explicit base_url over OLLAMA_BASE_URL, so saving
    the resolved URL would pin a stale host across env changes."""
    selections = dict(SELECTIONS, llm_provider="ollama", backend_url="http://old-host:11434/v1")
    user_settings.save_settings(selections)
    raw = json.loads(settings_file.read_text())
    assert raw["backend_url"] is None


def test_missing_file_loads_none(settings_file):
    assert user_settings.load_saved_settings() is None


def test_corrupt_json_loads_none(settings_file):
    settings_file.write_text("{not json", encoding="utf-8")
    assert user_settings.load_saved_settings() is None


def test_non_dict_json_loads_none(settings_file):
    settings_file.write_text('["a", "b"]', encoding="utf-8")
    assert user_settings.load_saved_settings() is None


def test_invalid_fields_dropped_not_fatal(settings_file):
    settings_file.write_text(json.dumps({
        "llm_provider": "openai",
        "research_depth": True,              # bool is not a valid depth
        "analysts": ["market", "astrology"],  # unknown analyst dropped
        "output_language": 42,                # wrong type dropped
        "future_key": "ignored",
    }), encoding="utf-8")
    loaded = user_settings.load_saved_settings()
    assert loaded == {"llm_provider": "openai", "analysts": ["market"]}


def test_incomplete_settings_fail_core_check(settings_file):
    settings_file.write_text(json.dumps({"llm_provider": "openai"}), encoding="utf-8")
    assert not user_settings.has_core_settings(user_settings.load_saved_settings())
    assert not user_settings.has_core_settings(None)


def test_clear_settings(settings_file):
    user_settings.save_settings(SELECTIONS)
    assert user_settings.clear_settings() is True
    assert not settings_file.exists()
    assert user_settings.clear_settings() is False


def test_save_creates_parent_dirs(tmp_path, monkeypatch):
    path = tmp_path / "deep" / "nested" / "settings.json"
    monkeypatch.setenv("TRADINGAGENTS_SETTINGS_PATH", str(path))
    assert user_settings.save_settings(SELECTIONS) == path
    assert path.exists()


def test_save_failure_returns_none(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("file, not dir", encoding="utf-8")
    monkeypatch.setenv("TRADINGAGENTS_SETTINGS_PATH", str(blocker / "settings.json"))
    assert user_settings.save_settings(SELECTIONS) is None
