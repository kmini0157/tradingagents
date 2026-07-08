"""Non-interactive hardening and crash-path behavior.

A scripted run must fail fast and loud (missing key), never hang on a hidden
prompt (announcements), and a checkpointed CLI stream must actually attach
the checkpointer — the historical --checkpoint flag was a silent no-op on
the CLI path because only propagate() recompiled the graph.
"""

import functools
import io
from typing import TypedDict
from unittest import mock

import pytest
from langgraph.graph import END, StateGraph

from tradingagents.graph.checkpointer import thread_id
from tradingagents.graph.trading_graph import TradingAgentsGraph


@pytest.mark.unit
class TestEnsureApiKeyNonInteractive:
    def test_missing_key_exits_instead_of_prompting(self, monkeypatch):
        from cli import utils

        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with mock.patch.object(utils.questionary, "password") as password_prompt, \
             pytest.raises(SystemExit) as ctx:
            utils.ensure_api_key("deepseek", interactive=False)
        password_prompt.assert_not_called()
        assert ctx.value.code == 1

    def test_existing_key_returned_without_prompt(self, monkeypatch):
        from cli import utils

        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-present")
        assert utils.ensure_api_key("deepseek", interactive=False) == "sk-present"

    def test_key_optional_provider_never_exits(self, monkeypatch):
        from cli import utils

        monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)
        assert utils.ensure_api_key("openai_compatible", interactive=False) is None


@pytest.mark.unit
class TestAnnouncementsNonInteractive:
    def test_require_attention_does_not_block_non_tty(self, monkeypatch):
        from rich.console import Console

        from cli import announcements

        fake_stdin = mock.Mock()
        fake_stdin.isatty.return_value = False
        monkeypatch.setattr(announcements.sys, "stdin", fake_stdin)
        with mock.patch.object(announcements.getpass, "getpass") as blocking_prompt:
            announcements.display_announcements(
                Console(file=io.StringIO()),
                {"announcements": ["Maintenance window"], "require_attention": True},
            )
        blocking_prompt.assert_not_called()

    def test_require_attention_still_blocks_on_tty(self, monkeypatch):
        from rich.console import Console

        from cli import announcements

        fake_stdin = mock.Mock()
        fake_stdin.isatty.return_value = True
        monkeypatch.setattr(announcements.sys, "stdin", fake_stdin)
        with mock.patch.object(announcements.getpass, "getpass") as blocking_prompt:
            announcements.display_announcements(
                Console(file=io.StringIO()),
                {"announcements": ["Maintenance window"], "require_attention": True},
            )
        blocking_prompt.assert_called_once()


def _graph_stub(tmp_path, enabled=True):
    """A MagicMock TradingAgentsGraph with real config/signature plumbing."""
    graph = mock.MagicMock()
    graph.config = {
        "checkpoint_enabled": enabled,
        "data_cache_dir": str(tmp_path),
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
    }
    graph.selected_analysts = ("market",)
    graph._run_signature = functools.partial(
        TradingAgentsGraph._run_signature, graph
    )
    return graph


@pytest.mark.unit
class TestCheckpointedRun:
    def test_disabled_yields_empty_session_and_leaves_graph(self, tmp_path):
        graph = _graph_stub(tmp_path, enabled=False)
        original = graph.graph
        with TradingAgentsGraph.checkpointed_run(graph, "NVDA", "2026-01-10") as session:
            assert session.config_patch == {}
            assert session.resume_step is None
            assert graph.graph is original
        graph.workflow.compile.assert_not_called()

    def test_enabled_attaches_checkpointer_and_restores(self, tmp_path):
        graph = _graph_stub(tmp_path, enabled=True)
        checkpointed = object()
        stateless = object()
        graph.workflow.compile.side_effect = lambda checkpointer=None: (
            checkpointed if checkpointer is not None else stateless
        )
        signature = graph._run_signature("stock")
        with TradingAgentsGraph.checkpointed_run(graph, "NVDA", "2026-01-10") as session:
            assert graph.graph is checkpointed
            assert session.resume_step is None  # fresh start, no prior checkpoint
            assert session.config_patch == {
                "configurable": {
                    "thread_id": thread_id("NVDA", "2026-01-10", signature)
                }
            }
        assert graph.graph is stateless

    def test_restores_graph_when_stream_raises(self, tmp_path):
        graph = _graph_stub(tmp_path, enabled=True)
        checkpointed = object()
        stateless = object()
        graph.workflow.compile.side_effect = lambda checkpointer=None: (
            checkpointed if checkpointer is not None else stateless
        )
        with pytest.raises(RuntimeError), \
             TradingAgentsGraph.checkpointed_run(graph, "NVDA", "2026-01-10"):
            raise RuntimeError("mid-run crash")
        assert graph.graph is stateless

    def test_clear_run_checkpoint_noop_when_disabled(self, tmp_path):
        graph = _graph_stub(tmp_path, enabled=False)
        # Must not touch the checkpoint store at all.
        TradingAgentsGraph.clear_run_checkpoint(graph, "NVDA", "2026-01-10")
        assert not (tmp_path / "checkpoints").exists()


@pytest.mark.unit
class TestExplicitSaveFailureExitCode:
    def test_failed_scripted_save_exits_one(self, tmp_path):
        """A --yes run whose requested report save fails must not exit 0 —
        scripts need the exit-code signal that the deliverable is missing."""
        import typer

        import cli.main as m

        blocker = tmp_path / "not-a-dir"
        blocker.write_text("file, not a directory", encoding="utf-8")
        graph = mock.MagicMock()
        graph.process_signal.return_value = "BUY"
        with pytest.raises(typer.Exit) as ctx:
            m._finish_run(
                {"final_trade_decision": "Rating: Buy"},
                {"ticker": "NVDA"},
                graph,
                tmp_path,
                wall_summary="",
                save_report=None,
                report_path=str(blocker / "reports"),
                show_report=None,
                non_interactive=True,
            )
        assert ctx.value.exit_code == 1


class _CountState(TypedDict):
    count: int


@pytest.mark.unit
class TestCheckpointResumeInput:
    """LangGraph resumes a checkpointed thread only when the input is None;
    streaming the initial state again restarts from the entry point and
    re-spends every completed node. These pin the None-on-resume rule for
    both the CLI stream path and propagate()'s _run_graph."""

    def _crashing_builder(self, runs, crash):
        def node_a(state):
            runs["a"] += 1
            return {"count": state["count"] + 1}

        def node_b(state):
            if crash["on"]:
                raise RuntimeError("simulated mid-run crash")
            runs["b"] += 1
            return {"count": state["count"] + 10}

        builder = StateGraph(_CountState)
        builder.add_node("analyst", node_a)
        builder.add_node("trader", node_b)
        builder.set_entry_point("analyst")
        builder.add_edge("analyst", "trader")
        builder.add_edge("trader", END)
        return builder

    def test_cli_stream_resume_does_not_rerun_completed_nodes(self, tmp_path):
        runs = {"a": 0, "b": 0}
        crash = {"on": True}
        stub = _graph_stub(tmp_path, enabled=True)
        stub.workflow = self._crashing_builder(runs, crash)

        # Run 1 (CLI shape): fresh session streams the initial state, crashes
        # at the second node with the first node's work checkpointed.
        with pytest.raises(RuntimeError), \
             TradingAgentsGraph.checkpointed_run(stub, "NVDA", "2026-01-10") as session:
            assert session.resume_step is None
            for _ in stub.graph.stream(
                {"count": 0}, config=session.config_patch, stream_mode="values"
            ):
                pass

        crash["on"] = False
        # Run 2: the session reports a resume, so the CLI streams None.
        with TradingAgentsGraph.checkpointed_run(stub, "NVDA", "2026-01-10") as session:
            assert session.resume_step is not None
            stream_input = None if session.resume_step is not None else {"count": 0}
            final = None
            for chunk in stub.graph.stream(
                stream_input, config=session.config_patch, stream_mode="values"
            ):
                final = chunk

        assert runs == {"a": 1, "b": 1}  # the completed node did NOT re-run
        assert final["count"] == 11

    def _run_graph_stub(self, tmp_path):
        graph = _graph_stub(tmp_path, enabled=True)
        graph.debug = False
        graph.config["results_dir"] = str(tmp_path)
        graph.propagator.get_graph_args.return_value = {}
        graph.graph.invoke.return_value = {"final_trade_decision": "Rating: Buy"}
        return graph

    def test_run_graph_resumes_with_none_input(self, tmp_path, monkeypatch):
        graph = self._run_graph_stub(tmp_path)
        monkeypatch.setattr(
            "tradingagents.graph.trading_graph.checkpoint_step",
            lambda *args, **kwargs: 3,
        )
        functools.partial(TradingAgentsGraph._run_graph, graph)("NVDA", "2026-01-10")
        assert graph.graph.invoke.call_args.args[0] is None

    def test_run_graph_fresh_start_passes_initial_state(self, tmp_path, monkeypatch):
        graph = self._run_graph_stub(tmp_path)
        monkeypatch.setattr(
            "tradingagents.graph.trading_graph.checkpoint_step",
            lambda *args, **kwargs: None,
        )
        functools.partial(TradingAgentsGraph._run_graph, graph)("NVDA", "2026-01-10")
        assert (
            graph.graph.invoke.call_args.args[0]
            is graph.propagator.create_initial_state.return_value
        )
