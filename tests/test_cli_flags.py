"""CLI flags and non-interactive (--yes) selection resolution.

Precedence contract: CLI flag > TRADINGAGENTS_* env var > saved settings >
built-in default. Flags and --yes must never regress the env-skip behavior
pinned by test_cli_env_skip.py.
"""

import os
import unittest
from unittest import mock

import pytest
import typer

import cli.main as m
from cli.models import AnalystType, AssetType
from cli.utils import parse_analysts_option, parse_depth_option

SAVED = {
    "last_ticker": "0700.HK",
    "analysts": ["market", "news"],
    "research_depth": 3,
    "llm_provider": "anthropic",
    "backend_url": "https://api.anthropic.com/",
    "quick_think_llm": "claude-sonnet-5",
    "deep_think_llm": "claude-fable-5",
    "anthropic_effort": "high",
    "output_language": "Korean",
}


def _wizard_mocks(**over):
    """Patch every prompt/announcement get_user_selections can reach."""
    patches = {
        "fetch_announcements": mock.patch.object(m, "fetch_announcements", return_value=None),
        "display_announcements": mock.patch.object(m, "display_announcements"),
        "load_saved_settings": mock.patch.object(
            m, "load_saved_settings", return_value=over.pop("saved", None)
        ),
        "save_settings": mock.patch.object(m, "save_settings", return_value=None),
        "get_ticker": mock.patch.object(m, "get_ticker", return_value="AAPL"),
        "get_analysis_date": mock.patch.object(m, "get_analysis_date", return_value="2026-05-29"),
        "ask_output_language": mock.patch.object(m, "ask_output_language", return_value="English"),
        "select_analysts": mock.patch.object(
            m, "select_analysts", return_value=[AnalystType.MARKET]
        ),
        "select_research_depth": mock.patch.object(m, "select_research_depth", return_value=1),
        "select_llm_provider": mock.patch.object(
            m, "select_llm_provider", return_value=("openai", "https://api.openai.com/v1")
        ),
        "ensure_api_key": mock.patch.object(m, "ensure_api_key"),
        "select_shallow_thinking_agent": mock.patch.object(
            m, "select_shallow_thinking_agent", return_value="gpt-5.4-mini"
        ),
        "select_deep_thinking_agent": mock.patch.object(
            m, "select_deep_thinking_agent", return_value="gpt-5.5"
        ),
        "ask_openai_reasoning_effort": mock.patch.object(
            m, "ask_openai_reasoning_effort", return_value="medium"
        ),
        "ask_anthropic_effort": mock.patch.object(
            m, "ask_anthropic_effort", return_value="high"
        ),
        "ask_gemini_thinking_config": mock.patch.object(
            m, "ask_gemini_thinking_config", return_value="high"
        ),
    }
    return patches


class _WizardHarness(unittest.TestCase):
    def run_selections(self, overrides=None, non_interactive=False, saved=None, env=None):
        patches = _wizard_mocks(saved=saved)
        started = {}
        with mock.patch.dict(os.environ, env or {}, clear=False):
            # Drop pre-existing TRADINGAGENTS_* vars that would trip skip logic.
            for var in (
                "TRADINGAGENTS_OUTPUT_LANGUAGE", "TRADINGAGENTS_LLM_PROVIDER",
                "TRADINGAGENTS_QUICK_THINK_LLM", "TRADINGAGENTS_DEEP_THINK_LLM",
                "TRADINGAGENTS_MAX_DEBATE_ROUNDS", "TRADINGAGENTS_MAX_RISK_ROUNDS",
                "TRADINGAGENTS_OPENAI_REASONING_EFFORT", "TRADINGAGENTS_LLM_BACKEND_URL",
            ):
                if var not in (env or {}):
                    os.environ.pop(var, None)
            for name, patch in patches.items():
                started[name] = patch.start()
            try:
                selections = m.get_user_selections(overrides, non_interactive)
            finally:
                for patch in patches.values():
                    patch.stop()
        return selections, started


@pytest.mark.unit
class TestFlagsSkipPrompts(_WizardHarness):
    def test_all_flags_skip_all_prompts(self):
        overrides = {
            "ticker": "NVDA",
            "date": "2026-05-29",
            "lang": "Korean",
            "analysts": [AnalystType.MARKET, AnalystType.NEWS],
            "depth": 5,
            "provider": "anthropic",
            "quick_llm": "claude-sonnet-5",
            "deep_llm": "claude-fable-5",
        }
        sel, mocks = self.run_selections(overrides)
        for prompt in (
            "get_ticker", "get_analysis_date", "ask_output_language",
            "select_analysts", "select_research_depth", "select_llm_provider",
            "select_shallow_thinking_agent", "select_deep_thinking_agent",
        ):
            mocks[prompt].assert_not_called()
        self.assertEqual(sel["ticker"], "NVDA")
        self.assertEqual(sel["analysis_date"], "2026-05-29")
        self.assertEqual(sel["output_language"], "Korean")
        self.assertEqual(sel["analysts"], [AnalystType.MARKET, AnalystType.NEWS])
        self.assertEqual(sel["research_depth"], 5)
        self.assertEqual(sel["llm_provider"], "anthropic")
        self.assertEqual(sel["backend_url"], "https://api.anthropic.com/")
        self.assertEqual(sel["shallow_thinker"], "claude-sonnet-5")
        self.assertEqual(sel["deep_thinker"], "claude-fable-5")
        # The provider-specific effort knob keeps its own prompt (it has an env
        # var and a saved value; --yes resolves it without asking).
        self.assertEqual(sel["anthropic_effort"], "high")
        mocks["ensure_api_key"].assert_called_once()

    def test_flag_beats_env(self):
        sel, mocks = self.run_selections(
            overrides={"lang": "Korean"},
            env={"TRADINGAGENTS_OUTPUT_LANGUAGE": "Japanese"},
        )
        mocks["ask_output_language"].assert_not_called()
        self.assertEqual(sel["output_language"], "Korean")

    def test_partial_model_flag_prompts_only_missing_mode(self):
        sel, mocks = self.run_selections(overrides={"quick_llm": "gpt-5.4-mini"})
        mocks["select_shallow_thinking_agent"].assert_not_called()
        mocks["select_deep_thinking_agent"].assert_called_once()
        self.assertEqual(sel["shallow_thinker"], "gpt-5.4-mini")
        self.assertEqual(sel["deep_thinker"], "gpt-5.5")

    def test_model_flag_with_env_uses_env_for_other_mode(self):
        env = {
            "TRADINGAGENTS_QUICK_THINK_LLM": "env-quick",
            "TRADINGAGENTS_DEEP_THINK_LLM": "env-deep",
        }
        fake_cfg = dict(m.DEFAULT_CONFIG, quick_think_llm="env-quick", deep_think_llm="env-deep")
        with mock.patch.object(m, "DEFAULT_CONFIG", fake_cfg):
            sel, mocks = self.run_selections(overrides={"deep_llm": "flag-deep"}, env=env)
        mocks["select_shallow_thinking_agent"].assert_not_called()
        mocks["select_deep_thinking_agent"].assert_not_called()
        self.assertEqual(sel["shallow_thinker"], "env-quick")
        self.assertEqual(sel["deep_thinker"], "flag-deep")

    def test_crypto_drops_fundamentals_from_analyst_flag(self):
        overrides = {
            "ticker": "BTC-USD",
            "analysts": [AnalystType.MARKET, AnalystType.FUNDAMENTALS],
        }
        sel, _ = self.run_selections(overrides)
        self.assertEqual(sel["asset_type"], "crypto")
        self.assertEqual(sel["analysts"], [AnalystType.MARKET])

    def test_crypto_fundamentals_only_flag_falls_back_to_prompt(self):
        """When filtering empties the flag selection, the wizard re-prompts
        instead of running an empty team into a ValueError."""
        sel, mocks = self.run_selections(
            overrides={"ticker": "BTC-USD", "analysts": [AnalystType.FUNDAMENTALS]}
        )
        mocks["select_analysts"].assert_called_once()
        self.assertEqual(sel["analysts"], [AnalystType.MARKET])

    def test_backend_url_flag_wins_over_menu(self):
        sel, _ = self.run_selections(overrides={"backend_url": "http://relay:9000/v1"})
        self.assertEqual(sel["backend_url"], "http://relay:9000/v1")


@pytest.mark.unit
class TestNonInteractiveResolution(_WizardHarness):
    def test_yes_replays_saved_settings_without_prompts(self):
        sel, mocks = self.run_selections(non_interactive=True, saved=dict(SAVED))
        for prompt in (
            "get_ticker", "get_analysis_date", "ask_output_language",
            "select_analysts", "select_research_depth", "select_llm_provider",
            "select_shallow_thinking_agent", "select_deep_thinking_agent",
        ):
            mocks[prompt].assert_not_called()
        self.assertEqual(sel["ticker"], "0700.HK")
        self.assertEqual(sel["llm_provider"], "anthropic")
        self.assertEqual(sel["backend_url"], "https://api.anthropic.com/")
        self.assertEqual(sel["shallow_thinker"], "claude-sonnet-5")
        self.assertEqual(sel["deep_thinker"], "claude-fable-5")
        self.assertEqual(sel["anthropic_effort"], "high")
        self.assertEqual(sel["output_language"], "Korean")
        self.assertEqual(sel["research_depth"], 3)
        self.assertEqual(
            sel["analysts"], [AnalystType.MARKET, AnalystType.NEWS]
        )
        # The key check must fail fast rather than prompt.
        mocks["ensure_api_key"].assert_called_once_with("anthropic", interactive=False)
        # Scripted runs never rewrite the settings file.
        mocks["save_settings"].assert_not_called()

    def test_yes_without_saved_uses_builtin_defaults(self):
        sel, _ = self.run_selections(non_interactive=True)
        self.assertEqual(sel["ticker"], "SPY")
        self.assertEqual(sel["llm_provider"], m.DEFAULT_CONFIG["llm_provider"])
        self.assertEqual(sel["research_depth"], m.DEFAULT_CONFIG["max_debate_rounds"])
        self.assertEqual(sel["output_language"], m.DEFAULT_CONFIG["output_language"])
        self.assertEqual(len(sel["analysts"]), len(list(AnalystType)))

    def test_yes_env_beats_saved(self):
        env = {"TRADINGAGENTS_OUTPUT_LANGUAGE": "Japanese"}
        fake_cfg = dict(m.DEFAULT_CONFIG, output_language="Japanese")
        with mock.patch.object(m, "DEFAULT_CONFIG", fake_cfg):
            sel, _ = self.run_selections(non_interactive=True, saved=dict(SAVED), env=env)
        self.assertEqual(sel["output_language"], "Japanese")

    def test_yes_flag_beats_saved(self):
        sel, _ = self.run_selections(
            overrides={"ticker": "NVDA", "depth": 5},
            non_interactive=True,
            saved=dict(SAVED),
        )
        self.assertEqual(sel["ticker"], "NVDA")
        self.assertEqual(sel["research_depth"], 5)
        # Untouched fields still replay from saved settings.
        self.assertEqual(sel["llm_provider"], "anthropic")

    def test_yes_ignores_saved_models_for_other_provider(self):
        """Saved Claude models must not leak into an env-pinned OpenAI run."""
        env = {"TRADINGAGENTS_LLM_PROVIDER": "openai"}
        with mock.patch.object(m, "DEFAULT_CONFIG", dict(m.DEFAULT_CONFIG, llm_provider="openai")):
            sel, _ = self.run_selections(non_interactive=True, saved=dict(SAVED), env=env)
        self.assertEqual(sel["llm_provider"], "openai")
        self.assertNotEqual(sel["shallow_thinker"], "claude-sonnet-5")
        self.assertNotEqual(sel["deep_thinker"], "claude-fable-5")

    def test_yes_openai_compatible_without_url_exits(self):
        with self.assertRaises(typer.Exit) as ctx:
            self.run_selections(
                overrides={"provider": "openai_compatible"}, non_interactive=True
            )
        self.assertEqual(ctx.exception.exit_code, 2)

    def test_yes_crypto_fundamentals_only_exits_cleanly(self):
        """Crypto filters fundamentals out; an empty team must exit 2, not
        crash later at graph construction with a raw ValueError."""
        with self.assertRaises(typer.Exit) as ctx:
            self.run_selections(
                overrides={
                    "ticker": "BTC-USD",
                    "analysts": [AnalystType.FUNDAMENTALS],
                },
                non_interactive=True,
            )
        self.assertEqual(ctx.exception.exit_code, 2)

    def test_yes_crypto_with_saved_fundamentals_only_exits_cleanly(self):
        saved = dict(SAVED, analysts=["fundamentals"])
        with self.assertRaises(typer.Exit) as ctx:
            self.run_selections(
                overrides={"ticker": "BTC-USD"}, non_interactive=True, saved=saved
            )
        self.assertEqual(ctx.exception.exit_code, 2)

    def test_yes_regional_provider_resolves_endpoint(self):
        sel, _ = self.run_selections(
            overrides={"provider": "qwen-cn"}, non_interactive=True
        )
        self.assertEqual(sel["llm_provider"], "qwen-cn")
        self.assertEqual(
            sel["backend_url"], "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )


@pytest.mark.unit
class TestFastPathReuse(_WizardHarness):
    def test_saved_settings_fast_path_skips_wizard(self):
        patches = _wizard_mocks(saved=dict(SAVED))
        fake_stdin = mock.Mock()
        fake_stdin.isatty.return_value = True
        started = {}
        for var in ("TRADINGAGENTS_OUTPUT_LANGUAGE", "TRADINGAGENTS_LLM_PROVIDER",
                    "TRADINGAGENTS_QUICK_THINK_LLM", "TRADINGAGENTS_DEEP_THINK_LLM",
                    "TRADINGAGENTS_MAX_DEBATE_ROUNDS", "TRADINGAGENTS_MAX_RISK_ROUNDS"):
            os.environ.pop(var, None)
        with mock.patch.object(m.sys, "stdin", fake_stdin), \
             mock.patch.object(m, "_confirm_reuse_settings", return_value=True), \
             mock.patch.object(m, "get_ticker", return_value="0700.HK") as ticker_prompt:
            for name, patch in patches.items():
                if name == "get_ticker":
                    continue
                started[name] = patch.start()
            try:
                sel = m.get_user_selections()
            finally:
                for name, patch in patches.items():
                    if name != "get_ticker":
                        patch.stop()

        ticker_prompt.assert_called_once()          # per-run inputs still asked
        started["get_analysis_date"].assert_called_once()
        started["select_analysts"].assert_not_called()   # the rest replays
        started["select_llm_provider"].assert_not_called()
        started["select_shallow_thinking_agent"].assert_not_called()
        self.assertEqual(sel["llm_provider"], "anthropic")
        self.assertEqual(sel["output_language"], "Korean")
        started["save_settings"].assert_called_once()    # interactive runs persist


@pytest.mark.unit
class TestAutosaveAnalystPreservation(unittest.TestCase):
    """A crypto run can't express a fundamentals preference (the analyst is
    filtered out), so autosave must not erase it from the settings file."""

    def _autosave(self, selections, previously_saved):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRADINGAGENTS_NO_SAVE_SETTINGS", None)
            with mock.patch.object(
                m, "load_saved_settings", return_value=previously_saved
            ), mock.patch.object(m, "save_settings", return_value=None) as save:
                m._autosave_settings(selections)
        return save

    def test_crypto_run_preserves_saved_analyst_team(self):
        save = self._autosave(
            {"asset_type": "crypto", "analysts": [AnalystType.MARKET]},
            {"analysts": ["market", "social", "news", "fundamentals"]},
        )
        saved_analysts = save.call_args.args[0]["analysts"]
        self.assertIn(AnalystType.FUNDAMENTALS, saved_analysts)

    def test_stock_run_saves_its_own_selection(self):
        save = self._autosave(
            {"asset_type": "stock", "analysts": [AnalystType.MARKET]},
            {"analysts": ["market", "social", "news", "fundamentals"]},
        )
        self.assertEqual(save.call_args.args[0]["analysts"], [AnalystType.MARKET])


@pytest.mark.unit
class TestFlagParsers(unittest.TestCase):
    def test_parse_analysts_all(self):
        self.assertEqual(parse_analysts_option("all"), list(AnalystType))

    def test_parse_analysts_names_and_aliases(self):
        self.assertEqual(
            parse_analysts_option("market,sentiment news"),
            [AnalystType.MARKET, AnalystType.SOCIAL, AnalystType.NEWS],
        )

    def test_parse_analysts_dedupes(self):
        self.assertEqual(
            parse_analysts_option("social,sentiment"), [AnalystType.SOCIAL]
        )

    def test_parse_analysts_rejects_unknown(self):
        with self.assertRaises(ValueError):
            parse_analysts_option("market,astrology")
        with self.assertRaises(ValueError):
            parse_analysts_option("  ,  ")

    def test_parse_depth_numbers_and_names(self):
        self.assertEqual(parse_depth_option("1"), 1)
        self.assertEqual(parse_depth_option("shallow"), 1)
        self.assertEqual(parse_depth_option("Medium"), 3)
        self.assertEqual(parse_depth_option("DEEP"), 5)

    def test_parse_depth_rejects_other_values(self):
        for bad in ("2", "7", "extreme", ""):
            with self.assertRaises(ValueError):
                parse_depth_option(bad)


@pytest.mark.unit
class TestAssetTypeDetection(unittest.TestCase):
    def test_detect_asset_type_still_exported(self):
        from cli.utils import detect_asset_type
        self.assertEqual(detect_asset_type("BTC-USD"), AssetType.CRYPTO)
        self.assertEqual(detect_asset_type("AAPL"), AssetType.STOCK)


if __name__ == "__main__":
    unittest.main()
