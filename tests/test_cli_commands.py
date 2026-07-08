"""Command-surface regressions: subcommands exist and validate flags early.

`tradingagents analyze` has been documented since 0.2.4 but the app was a
collapsed single-command Typer app, so the subcommand form actually errored
with "Got unexpected extra argument (analyze)". These tests pin the fixed
surface: analyze/doctor/config exist, the bare root still launches the
wizard path, and bad flag values fail fast with exit code 2 before any
wizard prompt or API call.
"""

import json

import pytest
from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


@pytest.mark.unit
class TestCommandSurface:
    def test_root_help_lists_subcommands(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for command in ("analyze", "doctor", "config"):
            assert command in result.output

    def test_analyze_subcommand_parses(self):
        result = runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "TICKER" in result.output

    def test_root_keeps_legacy_checkpoint_flags(self):
        result = runner.invoke(app, ["--help"])
        assert "--checkpoint" in result.output
        assert "--clear-checkpoints" in result.output


@pytest.mark.unit
class TestAnalyzeFlagValidation:
    """Bad values exit 2 at parse time — no wizard, no tokens."""

    @pytest.mark.parametrize(
        "args",
        [
            ["analyze", "BAD!TICKER"],
            ["analyze", "--date", "01-15-2026"],
            ["analyze", "--date", "2999-01-01"],
            ["analyze", "--analysts", "market,astrology"],
            ["analyze", "--depth", "7"],
            ["analyze", "--provider", "not-a-provider"],
            ["analyze", "--backend-url", "localhost:8000"],
        ],
    )
    def test_invalid_flag_values_fail_fast(self, args):
        result = runner.invoke(app, args)
        assert result.exit_code == 2

    def test_ticker_argument_is_normalized(self, monkeypatch):
        captured = {}

        def fake_run_analysis(**kwargs):
            captured.update(kwargs)

        import cli.main as m
        monkeypatch.setattr(m, "run_analysis", fake_run_analysis)
        result = runner.invoke(app, ["analyze", "btcusd", "--yes"])
        assert result.exit_code == 0
        assert captured["overrides"]["ticker"] == "BTC-USD"
        assert captured["non_interactive"] is True

    def test_report_path_implies_save(self, monkeypatch):
        captured = {}

        def fake_run_analysis(**kwargs):
            captured.update(kwargs)

        import cli.main as m
        monkeypatch.setattr(m, "run_analysis", fake_run_analysis)
        result = runner.invoke(
            app, ["analyze", "--report-path", "/tmp/report-dir", "--yes"]
        )
        assert result.exit_code == 0
        assert captured["save_report"] is True

    def test_root_checkpoint_flag_forwards_to_analyze(self, monkeypatch):
        """`tradingagents --checkpoint analyze ...` must not silently drop the
        flag just because it sits before the subcommand."""
        captured = {}

        import cli.main as m
        monkeypatch.setattr(m, "run_analysis", lambda **kw: captured.update(kw))
        result = runner.invoke(app, ["--checkpoint", "analyze", "NVDA", "--yes"])
        assert result.exit_code == 0
        assert captured["checkpoint"] is True

    def test_subcommand_checkpoint_flag_beats_root(self, monkeypatch):
        captured = {}

        import cli.main as m
        monkeypatch.setattr(m, "run_analysis", lambda **kw: captured.update(kw))
        result = runner.invoke(
            app, ["--checkpoint", "analyze", "NVDA", "--no-checkpoint", "--yes"]
        )
        assert result.exit_code == 0
        assert captured["checkpoint"] is False

    def test_root_clear_checkpoints_forwards_to_analyze(self, monkeypatch):
        cleared = []

        import cli.main as m
        monkeypatch.setattr(m, "_clear_all_checkpoints_now", lambda: cleared.append(True))
        monkeypatch.setattr(m, "run_analysis", lambda **kw: None)
        result = runner.invoke(app, ["--clear-checkpoints", "analyze", "NVDA", "-y"])
        assert result.exit_code == 0
        assert cleared


@pytest.mark.unit
class TestConfigCommands:
    def test_config_path_prints_override(self, tmp_path, monkeypatch):
        path = tmp_path / "settings.json"
        monkeypatch.setenv("TRADINGAGENTS_SETTINGS_PATH", str(path))
        result = runner.invoke(app, ["config", "path"])
        assert result.exit_code == 0
        assert str(path) in result.output.replace("\n", "")

    def test_config_show_without_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv(
            "TRADINGAGENTS_SETTINGS_PATH", str(tmp_path / "settings.json")
        )
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "No saved settings yet" in result.output

    def test_config_show_with_corrupt_file(self, tmp_path, monkeypatch):
        path = tmp_path / "settings.json"
        path.write_text("{broken", encoding="utf-8")
        monkeypatch.setenv("TRADINGAGENTS_SETTINGS_PATH", str(path))
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "couldn't be read" in result.output

    def test_config_show_and_reset_round_trip(self, tmp_path, monkeypatch):
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"llm_provider": "anthropic", "output_language": "Korean"}),
            encoding="utf-8",
        )
        monkeypatch.setenv("TRADINGAGENTS_SETTINGS_PATH", str(path))

        shown = runner.invoke(app, ["config", "show"])
        assert shown.exit_code == 0
        assert "anthropic" in shown.output
        assert "Korean" in shown.output

        reset = runner.invoke(app, ["config", "reset"])
        assert reset.exit_code == 0
        assert not path.exists()

        again = runner.invoke(app, ["config", "reset"])
        assert again.exit_code == 0
        assert "No saved settings" in again.output


@pytest.mark.unit
class TestDoctorCommand:
    def test_doctor_fails_on_missing_required_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv(
            "TRADINGAGENTS_SETTINGS_PATH", str(tmp_path / "settings.json")
        )
        for var in ("TRADINGAGENTS_LLM_PROVIDER", "OPENAI_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 1
        assert "OPENAI_API_KEY" in result.output

    def test_doctor_passes_with_key_present(self, tmp_path, monkeypatch):
        monkeypatch.setenv(
            "TRADINGAGENTS_SETTINGS_PATH", str(tmp_path / "settings.json")
        )
        monkeypatch.delenv("TRADINGAGENTS_LLM_PROVIDER", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("TRADINGAGENTS_RESULTS_DIR", str(tmp_path / "results"))
        import cli.doctor as doctor_module
        monkeypatch.setitem(
            doctor_module.DEFAULT_CONFIG, "results_dir", str(tmp_path / "results")
        )
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "Ready to run" in result.output
