import datetime
import os
import sys
import time
from collections import deque
from functools import wraps
from pathlib import Path

import questionary
import typer
from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from cli.announcements import display_announcements, fetch_announcements
from cli.models import AnalystType
from cli.stats_handler import StatsCallbackHandler
from cli.user_settings import (
    clear_settings,
    has_core_settings,
    load_saved_settings,
    save_settings,
    settings_path,
)
from cli.utils import (
    ask_anthropic_effort,
    ask_gemini_thinking_config,
    ask_glm_region,
    ask_minimax_region,
    ask_openai_reasoning_effort,
    ask_output_language,
    ask_qwen_region,
    confirm_ollama_endpoint,
    depth_label,
    detect_asset_type,
    ensure_api_key,
    filter_analysts_for_asset_type,
    get_ticker,
    is_valid_ticker_input,
    known_provider_keys,
    normalize_ticker_symbol,
    parse_analysts_option,
    parse_depth_option,
    prompt_openai_compatible_url,
    resolve_backend_url,
    select_analysts,
    select_deep_thinking_agent,
    select_llm_provider,
    select_research_depth,
    select_shallow_thinking_agent,
)
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.analyst_execution import (
    AnalystWallTimeTracker,
    build_analyst_execution_plan,
    get_initial_analyst_node,
    sync_analyst_tracker_from_chunk,
)
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.model_catalog import get_model_options
from tradingagents.reporting import write_report_tree

console = Console()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # Enable shell completion
)


# Create a deque to store recent messages with a maximum length
class MessageBuffer:
    # Fixed teams that always run (not user-selectable)
    FIXED_AGENTS = {
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # Analyst name mapping
    ANALYST_MAPPING = {
        "market": "Market Analyst",
        "social": "Sentiment Analyst",
        "news": "News Analyst",
        "fundamentals": "Fundamentals Analyst",
    }

    # Report section mapping: section -> (analyst_key for filtering, finalizing_agent)
    # analyst_key: which analyst selection controls this section (None = always included)
    # finalizing_agent: which agent must be "completed" for this report to count as done
    REPORT_SECTIONS = {
        "market_report": ("market", "Market Analyst"),
        "sentiment_report": ("social", "Sentiment Analyst"),
        "news_report": ("news", "News Analyst"),
        "fundamentals_report": ("fundamentals", "Fundamentals Analyst"),
        "investment_plan": (None, "Research Manager"),
        "trader_investment_plan": (None, "Trader"),
        "final_trade_decision": (None, "Portfolio Manager"),
    }

    def __init__(self, max_length=100):
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report = None
        self.final_report = None  # Store the complete final report
        self.agent_status = {}
        self.current_agent = None
        self.report_sections = {}
        self.selected_analysts = []
        self._processed_message_ids = set()

    def init_for_analysis(self, selected_analysts):
        """Initialize agent status and report sections based on selected analysts.

        Args:
            selected_analysts: List of analyst type strings (e.g., ["market", "news"])
        """
        self.selected_analysts = [a.lower() for a in selected_analysts]

        # Build agent_status dynamically
        self.agent_status = {}

        # Add selected analysts
        for analyst_key in self.selected_analysts:
            if analyst_key in self.ANALYST_MAPPING:
                self.agent_status[self.ANALYST_MAPPING[analyst_key]] = "pending"

        # Add fixed teams
        for team_agents in self.FIXED_AGENTS.values():
            for agent in team_agents:
                self.agent_status[agent] = "pending"

        # Build report_sections dynamically
        self.report_sections = {}
        for section, (analyst_key, _) in self.REPORT_SECTIONS.items():
            if analyst_key is None or analyst_key in self.selected_analysts:
                self.report_sections[section] = None

        # Reset other state
        self.current_report = None
        self.final_report = None
        self.current_agent = None
        self.messages.clear()
        self.tool_calls.clear()
        self._processed_message_ids.clear()

    def get_completed_reports_count(self):
        """Count reports that are finalized (their finalizing agent is completed).

        A report is considered complete when:
        1. The report section has content (not None), AND
        2. The agent responsible for finalizing that report has status "completed"

        This prevents interim updates (like debate rounds) from counting as completed.
        """
        count = 0
        for section in self.report_sections:
            if section not in self.REPORT_SECTIONS:
                continue
            _, finalizing_agent = self.REPORT_SECTIONS[section]
            # Report is complete if it has content AND its finalizing agent is done
            has_content = self.report_sections.get(section) is not None
            agent_done = self.agent_status.get(finalizing_agent) == "completed"
            if has_content and agent_done:
                count += 1
        return count

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))

    def add_tool_call(self, tool_name, args):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name, content):
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        # For the panel display, only show the most recently updated section
        latest_section = None
        latest_content = None

        # Find the most recently updated section
        for section, content in self.report_sections.items():
            if content is not None:
                latest_section = section
                latest_content = content

        if latest_section and latest_content:
            # Format the current section for display
            section_titles = {
                "market_report": "Market Analysis",
                "sentiment_report": "Social Sentiment",
                "news_report": "News Analysis",
                "fundamentals_report": "Fundamentals Analysis",
                "investment_plan": "Research Team Decision",
                "trader_investment_plan": "Trading Team Plan",
                "final_trade_decision": "Portfolio Management Decision",
            }
            self.current_report = (
                f"### {section_titles[latest_section]}\n{latest_content}"
            )

        # Update the final complete report
        self._update_final_report()

    def _update_final_report(self):
        report_parts = []

        # Analyst Team Reports - use .get() to handle missing sections
        analyst_sections = ["market_report", "sentiment_report", "news_report", "fundamentals_report"]
        if any(self.report_sections.get(section) for section in analyst_sections):
            report_parts.append("## Analyst Team Reports")
            if self.report_sections.get("market_report"):
                report_parts.append(
                    f"### Market Analysis\n{self.report_sections['market_report']}"
                )
            if self.report_sections.get("sentiment_report"):
                report_parts.append(
                    f"### Social Sentiment\n{self.report_sections['sentiment_report']}"
                )
            if self.report_sections.get("news_report"):
                report_parts.append(
                    f"### News Analysis\n{self.report_sections['news_report']}"
                )
            if self.report_sections.get("fundamentals_report"):
                report_parts.append(
                    f"### Fundamentals Analysis\n{self.report_sections['fundamentals_report']}"
                )

        # Research Team Reports
        if self.report_sections.get("investment_plan"):
            report_parts.append("## Research Team Decision")
            report_parts.append(f"{self.report_sections['investment_plan']}")

        # Trading Team Reports
        if self.report_sections.get("trader_investment_plan"):
            report_parts.append("## Trading Team Plan")
            report_parts.append(f"{self.report_sections['trader_investment_plan']}")

        # Portfolio Management Decision
        if self.report_sections.get("final_trade_decision"):
            report_parts.append("## Portfolio Management Decision")
            report_parts.append(f"{self.report_sections['final_trade_decision']}")

        self.final_report = "\n\n".join(report_parts) if report_parts else None


message_buffer = MessageBuffer()


def create_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3), Layout(name="analysis", ratio=5)
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2), Layout(name="messages", ratio=3)
    )
    return layout


def format_tokens(n):
    """Format token count for display."""
    if n >= 1000:
        return f"{n/1000:.1f}k"
    return str(n)


def update_display(layout, spinner_text=None, stats_handler=None, start_time=None):
    # Header with welcome message
    layout["header"].update(
        Panel(
            "[bold green]Welcome to TradingAgents CLI[/bold green]\n"
            "[dim]© [Tauric Research](https://github.com/TauricResearch)[/dim]",
            title="Welcome to TradingAgents",
            border_style="green",
            padding=(1, 2),
            expand=True,
        )
    )

    # Progress panel showing agent status
    progress_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        box=box.SIMPLE_HEAD,  # Use simple header with horizontal lines
        title=None,  # Remove the redundant Progress title
        padding=(0, 2),  # Add horizontal padding
        expand=True,  # Make table expand to fill available space
    )
    progress_table.add_column("Team", style="cyan", justify="center", width=20)
    progress_table.add_column("Agent", style="green", justify="center", width=20)
    progress_table.add_column("Status", style="yellow", justify="center", width=20)

    # Group agents by team - filter to only include agents in agent_status
    all_teams = {
        "Analyst Team": [
            "Market Analyst",
            "Sentiment Analyst",
            "News Analyst",
            "Fundamentals Analyst",
        ],
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # Filter teams to only include agents that are in agent_status
    teams = {}
    for team, agents in all_teams.items():
        active_agents = [a for a in agents if a in message_buffer.agent_status]
        if active_agents:
            teams[team] = active_agents

    for team, agents in teams.items():
        # Add first agent with team name
        first_agent = agents[0]
        status = message_buffer.agent_status.get(first_agent, "pending")
        if status == "in_progress":
            spinner = Spinner(
                "dots", text="[blue]in_progress[/blue]", style="bold cyan"
            )
            status_cell = spinner
        else:
            status_color = {
                "pending": "yellow",
                "completed": "green",
                "error": "red",
            }.get(status, "white")
            status_cell = f"[{status_color}]{status}[/{status_color}]"
        progress_table.add_row(team, first_agent, status_cell)

        # Add remaining agents in team
        for agent in agents[1:]:
            status = message_buffer.agent_status.get(agent, "pending")
            if status == "in_progress":
                spinner = Spinner(
                    "dots", text="[blue]in_progress[/blue]", style="bold cyan"
                )
                status_cell = spinner
            else:
                status_color = {
                    "pending": "yellow",
                    "completed": "green",
                    "error": "red",
                }.get(status, "white")
                status_cell = f"[{status_color}]{status}[/{status_color}]"
            progress_table.add_row("", agent, status_cell)

        # Add horizontal line after each team
        progress_table.add_row("─" * 20, "─" * 20, "─" * 20, style="dim")

    layout["progress"].update(
        Panel(progress_table, title="Progress", border_style="cyan", padding=(1, 2))
    )

    # Messages panel showing recent messages and tool calls
    messages_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        expand=True,  # Make table expand to fill available space
        box=box.MINIMAL,  # Use minimal box style for a lighter look
        show_lines=True,  # Keep horizontal lines
        padding=(0, 1),  # Add some padding between columns
    )
    messages_table.add_column("Time", style="cyan", width=8, justify="center")
    messages_table.add_column("Type", style="green", width=10, justify="center")
    messages_table.add_column(
        "Content", style="white", no_wrap=False, ratio=1
    )  # Make content column expand

    # Combine tool calls and messages
    all_messages = []

    # Add tool calls
    for timestamp, tool_name, args in message_buffer.tool_calls:
        formatted_args = format_tool_args(args)
        all_messages.append((timestamp, "Tool", f"{tool_name}: {formatted_args}"))

    # Add regular messages
    for timestamp, msg_type, content in message_buffer.messages:
        content_str = str(content) if content else ""
        if len(content_str) > 200:
            content_str = content_str[:197] + "..."
        all_messages.append((timestamp, msg_type, content_str))

    # Sort by timestamp descending (newest first)
    all_messages.sort(key=lambda x: x[0], reverse=True)

    # Calculate how many messages we can show based on available space
    max_messages = 12

    # Get the first N messages (newest ones)
    recent_messages = all_messages[:max_messages]

    # Add messages to table (already in newest-first order)
    for timestamp, msg_type, content in recent_messages:
        # Format content with word wrapping
        wrapped_content = Text(content, overflow="fold")
        messages_table.add_row(timestamp, msg_type, wrapped_content)

    layout["messages"].update(
        Panel(
            messages_table,
            title="Messages & Tools",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Analysis panel showing current report
    if message_buffer.current_report:
        layout["analysis"].update(
            Panel(
                Markdown(message_buffer.current_report),
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        layout["analysis"].update(
            Panel(
                "[italic]Waiting for analysis report...[/italic]",
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )

    # Footer with statistics
    # Agent progress - derived from agent_status dict
    agents_completed = sum(
        1 for status in message_buffer.agent_status.values() if status == "completed"
    )
    agents_total = len(message_buffer.agent_status)

    # Report progress - based on agent completion (not just content existence)
    reports_completed = message_buffer.get_completed_reports_count()
    reports_total = len(message_buffer.report_sections)

    # Build stats parts
    stats_parts = [f"Agents: {agents_completed}/{agents_total}"]

    # LLM and tool stats from callback handler
    if stats_handler:
        stats = stats_handler.get_stats()
        stats_parts.append(f"LLM: {stats['llm_calls']}")
        stats_parts.append(f"Tools: {stats['tool_calls']}")

        # Token display with graceful fallback
        if stats["tokens_in"] > 0 or stats["tokens_out"] > 0:
            tokens_str = f"Tokens: {format_tokens(stats['tokens_in'])}\u2191 {format_tokens(stats['tokens_out'])}\u2193"
        else:
            tokens_str = "Tokens: --"
        stats_parts.append(tokens_str)

    stats_parts.append(f"Reports: {reports_completed}/{reports_total}")

    # Elapsed time
    if start_time:
        elapsed = time.time() - start_time
        elapsed_str = f"\u23f1 {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        stats_parts.append(elapsed_str)

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(" | ".join(stats_parts))

    layout["footer"].update(Panel(stats_table, border_style="grey50"))


_SOURCE_LABELS = {
    "flag": "command line",
    "env": "environment",
    "saved": "saved settings",
    "default": "default",
}

# Thinking/effort knobs: (config key, env var, provider they apply to).
_THINKING_KNOBS = (
    ("google_thinking_level", "TRADINGAGENTS_GOOGLE_THINKING_LEVEL", "google"),
    ("openai_reasoning_effort", "TRADINGAGENTS_OPENAI_REASONING_EFFORT", "openai"),
    ("anthropic_effort", "TRADINGAGENTS_ANTHROPIC_EFFORT", "anthropic"),
)


def _saved_for_provider(saved: dict, provider: str, key: str):
    """A saved value, honored only when it was saved for the same provider."""
    if saved.get("llm_provider") == provider:
        return saved.get(key)
    return None


def _catalog_default_models(provider: str) -> tuple[str | None, str | None]:
    """First real catalog option per mode, for --yes runs with no model choice.

    (None, None) when the provider has no fixed catalog (openrouter, azure)
    or only the custom-ID sentinel — those need an explicit model.
    """
    try:
        quick_opts = get_model_options(provider, "quick")
        deep_opts = get_model_options(provider, "deep")
    except KeyError:
        return None, None
    quick = next((value for _, value in quick_opts if value != "custom"), None)
    deep = next((value for _, value in deep_opts if value != "custom"), None)
    return quick, deep


def _note_non_trading_date(date_str: str, asset_type=None) -> None:
    """Soft heads-up for weekend dates (a common first-run surprise).

    Skipped for crypto, which trades around the clock.
    """
    if asset_type is not None and asset_type.value == "crypto":
        return
    try:
        parsed = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return
    if parsed.weekday() >= 5:
        console.print(
            f"[dim]Note: {date_str} is a {parsed.strftime('%A')}; markets are closed, "
            f"so price and indicator data will reflect the last trading day.[/dim]"
        )


def _resolve_run_settings(ov: dict, saved: dict, asset_type) -> tuple[dict | None, dict, str | None]:
    """Resolve every non-per-run selection without prompting.

    Per-field precedence: CLI flag > env var > saved settings > built-in
    default. Returns (settings, sources, problem); ``problem`` is a
    user-facing message when a field can't be resolved without a prompt
    (the --yes path exits on it, the fast path falls back to the wizard).
    """
    src: dict = {}
    problem = None

    # Output language
    if ov.get("lang"):
        language, src["language"] = ov["lang"], "flag"
    elif os.environ.get("TRADINGAGENTS_OUTPUT_LANGUAGE"):
        language, src["language"] = DEFAULT_CONFIG["output_language"], "env"
    elif saved.get("output_language"):
        language, src["language"] = saved["output_language"], "saved"
    else:
        language, src["language"] = DEFAULT_CONFIG["output_language"], "default"

    # Analysts (no env var exists; flag > saved > full team)
    if ov.get("analysts"):
        requested, src["analysts"] = list(ov["analysts"]), "flag"
    elif saved.get("analysts"):
        requested, src["analysts"] = (
            [AnalystType(a) for a in saved["analysts"]], "saved",
        )
    else:
        requested, src["analysts"] = list(AnalystType), "default"
    analysts = filter_analysts_for_asset_type(requested, asset_type)
    dropped = [a for a in requested if a not in analysts]
    if not analysts:
        problem = (
            "None of the selected analysts are available for this asset type "
            "(crypto has no fundamentals analyst) — pick at least one of "
            "market, social, news."
        )

    # Research depth
    depth_env = bool(os.environ.get("TRADINGAGENTS_MAX_DEBATE_ROUNDS")) and bool(
        os.environ.get("TRADINGAGENTS_MAX_RISK_ROUNDS")
    )
    if ov.get("depth"):
        depth, src["depth"] = ov["depth"], "flag"
    elif depth_env:
        depth, src["depth"] = DEFAULT_CONFIG["max_debate_rounds"], "env"
    elif saved.get("research_depth"):
        depth, src["depth"] = saved["research_depth"], "saved"
    else:
        depth, src["depth"] = DEFAULT_CONFIG["max_debate_rounds"], "default"

    # Provider
    if ov.get("provider"):
        provider, src["provider"] = ov["provider"].lower(), "flag"
    elif os.environ.get("TRADINGAGENTS_LLM_PROVIDER"):
        provider, src["provider"] = DEFAULT_CONFIG["llm_provider"].lower(), "env"
    elif saved.get("llm_provider"):
        provider, src["provider"] = saved["llm_provider"], "saved"
    else:
        provider, src["provider"] = DEFAULT_CONFIG["llm_provider"].lower(), "default"

    # Backend URL
    if ov.get("backend_url"):
        backend_url, src["backend"] = ov["backend_url"], "flag"
    else:
        backend_url = resolve_backend_url(
            provider,
            _saved_for_provider(saved, provider, "backend_url"),
            env_url=DEFAULT_CONFIG["backend_url"],
        )
        src["backend"] = (
            "env" if DEFAULT_CONFIG["backend_url"]
            else "saved" if _saved_for_provider(saved, provider, "backend_url")
            else "default"
        )
    if provider == "openai_compatible" and not backend_url:
        problem = problem or (
            "openai_compatible needs an endpoint URL: pass --backend-url or set "
            "TRADINGAGENTS_LLM_BACKEND_URL."
        )

    # Models (either env var pins both, mirroring the wizard's skip rule)
    models_env = bool(
        os.environ.get("TRADINGAGENTS_QUICK_THINK_LLM")
        or os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM")
    )
    catalog_quick, catalog_deep = (None, None)
    if not models_env:
        catalog_quick, catalog_deep = _catalog_default_models(provider)
    models = {}
    for mode, flag_key, config_key, saved_key, catalog_value in (
        ("quick", "quick_llm", "quick_think_llm", "quick_think_llm", catalog_quick),
        ("deep", "deep_llm", "deep_think_llm", "deep_think_llm", catalog_deep),
    ):
        if ov.get(flag_key):
            models[mode], src[mode] = ov[flag_key], "flag"
        elif models_env:
            models[mode], src[mode] = DEFAULT_CONFIG[config_key], "env"
        elif _saved_for_provider(saved, provider, saved_key):
            models[mode], src[mode] = _saved_for_provider(saved, provider, saved_key), "saved"
        elif catalog_value:
            models[mode], src[mode] = catalog_value, "default"
        else:
            models[mode], src[mode] = None, "default"
            problem = problem or (
                f"No default {mode}-thinking model for provider {provider!r}: "
                f"pass --{mode}-llm (or set TRADINGAGENTS_"
                f"{'QUICK' if mode == 'quick' else 'DEEP'}_THINK_LLM)."
            )

    # Provider-specific thinking/effort knobs
    knobs = {}
    for config_key, env_var, knob_provider in _THINKING_KNOBS:
        if os.environ.get(env_var):
            knobs[config_key] = DEFAULT_CONFIG[config_key]
        elif provider == knob_provider:
            knobs[config_key] = _saved_for_provider(saved, provider, config_key)
        else:
            knobs[config_key] = None

    settings = {
        "output_language": language,
        "analysts": analysts,
        "dropped_analysts": dropped,
        "research_depth": depth,
        "llm_provider": provider,
        "backend_url": backend_url,
        "shallow_thinker": models["quick"],
        "deep_thinker": models["deep"],
        **knobs,
    }
    return settings, src, problem


def _print_run_settings(settings: dict, src: dict, ticker: str, analysis_date: str) -> None:
    """Show the resolved run settings with where each value came from."""

    def tag(key: str) -> str:
        return f" [dim]({_SOURCE_LABELS.get(src.get(key, 'default'))})[/dim]"

    analyst_names = ", ".join(a.value for a in settings["analysts"]) or "none"
    lines = [
        f"[bold]Ticker:[/bold] {ticker}   [bold]Date:[/bold] {analysis_date}",
        f"[bold]Provider:[/bold] {settings['llm_provider']}{tag('provider')}",
    ]
    if settings.get("backend_url"):
        lines.append(f"[bold]Endpoint:[/bold] {settings['backend_url']}{tag('backend')}")
    lines.extend([
        f"[bold]Models:[/bold] quick={settings['shallow_thinker']}{tag('quick')} · "
        f"deep={settings['deep_thinker']}{tag('deep')}",
        f"[bold]Analysts:[/bold] {analyst_names}{tag('analysts')}",
        f"[bold]Research depth:[/bold] {depth_label(settings['research_depth'])}"
        f"{tag('depth')}",
        f"[bold]Language:[/bold] {settings['output_language']}{tag('language')}",
    ])
    for config_key, _, knob_provider in _THINKING_KNOBS:
        if settings["llm_provider"] == knob_provider and settings.get(config_key):
            lines.append(
                f"[bold]{knob_provider.title()} effort/thinking:[/bold] "
                f"{settings[config_key]}"
            )
    console.print(Panel("\n".join(lines), title="Run Settings", border_style="blue", padding=(1, 2)))
    if settings.get("dropped_analysts"):
        console.print(
            "[dim]Fundamentals analyst dropped: unavailable for crypto assets.[/dim]"
        )


def _selections_from(settings: dict, ticker: str, asset_type, analysis_date: str) -> dict:
    """Assemble the selections dict (get_user_selections' return shape)."""
    return {
        "ticker": ticker,
        "asset_type": asset_type.value,
        "analysis_date": analysis_date,
        "analysts": settings["analysts"],
        "research_depth": settings["research_depth"],
        "llm_provider": settings["llm_provider"],
        "backend_url": settings["backend_url"],
        "shallow_thinker": settings["shallow_thinker"],
        "deep_thinker": settings["deep_thinker"],
        "google_thinking_level": settings.get("google_thinking_level"),
        "openai_reasoning_effort": settings.get("openai_reasoning_effort"),
        "anthropic_effort": settings.get("anthropic_effort"),
        "output_language": settings["output_language"],
    }


def _autosave_settings(selections: dict) -> None:
    """Persist the run's selections for next time (best-effort, opt-out)."""
    if os.environ.get("TRADINGAGENTS_NO_SAVE_SETTINGS"):
        return
    payload = selections
    if selections.get("asset_type") == "crypto":
        # Crypto runs can't express a fundamentals preference (the analyst is
        # filtered out), so they must not rewrite the saved analyst team — one
        # BTC-USD run would otherwise silently drop fundamentals from every
        # future stock run.
        previous = load_saved_settings() or {}
        if previous.get("analysts"):
            payload = dict(
                selections,
                analysts=[AnalystType(a) for a in previous["analysts"]],
            )
    path = save_settings(payload)
    if path:
        console.print(
            f"[dim]Settings saved to {path} — next run offers them back, "
            f"or replay them with: tradingagents analyze --yes[/dim]"
        )


def _confirm_reuse_settings() -> bool:
    """One-keystroke fast path: Enter reuses the saved run settings."""
    choice = questionary.select(
        "Use these settings for this run?",
        choices=[
            questionary.Choice("Yes — run with these settings", value=True),
            questionary.Choice("No — customize for this run", value=False),
        ],
        instruction="\n- Press Enter to reuse, or pick customize",
        style=questionary.Style(
            [
                ("selected", "fg:yellow noinherit"),
                ("highlighted", "fg:yellow noinherit"),
                ("pointer", "fg:yellow noinherit"),
            ]
        ),
    ).ask()
    if choice is None:
        # Ctrl-C/Esc means abort, matching every other prompt in the wizard —
        # "customize" is an explicit menu choice, not the cancel behavior.
        console.print("\n[red]Cancelled. Exiting...[/red]")
        exit(1)
    return bool(choice)


def get_user_selections(overrides: dict | None = None, non_interactive: bool = False):
    """Get all user selections before starting the analysis display.

    ``overrides`` holds pre-answered selections from CLI flags (each skips its
    prompt); ``non_interactive`` (--yes) answers everything else from env vars,
    saved settings, and built-in defaults, in that order.
    """
    ov = overrides or {}
    saved = load_saved_settings() or {}
    is_tty = sys.stdin.isatty()
    interactive_key = is_tty and not non_interactive

    # Display ASCII art welcome message
    with open(Path(__file__).parent / "static" / "welcome.txt", encoding="utf-8") as f:
        welcome_ascii = f.read()

    # Create welcome box content
    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]TradingAgents: Multi-Agents LLM Financial Trading Framework - CLI[/bold green]\n\n"
    welcome_content += "[bold]Workflow Steps:[/bold]\n"
    welcome_content += "I. Analyst Team → II. Research Team → III. Trader → IV. Risk Management → V. Portfolio Management\n\n"
    welcome_content += (
        "[dim]Built by [Tauric Research](https://github.com/TauricResearch)[/dim]"
    )

    # Create and center the welcome box
    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Welcome to TradingAgents",
        subtitle="Multi-Agents LLM Financial Trading Framework",
    )
    console.print(Align.center(welcome_box))
    console.print()
    console.print()  # Add vertical space before announcements

    # Fetch and display announcements (silent on failure). Scripted --yes runs
    # skip the network round-trip; TRADINGAGENTS_NO_ANNOUNCEMENTS opts out.
    if not non_interactive and not os.environ.get("TRADINGAGENTS_NO_ANNOUNCEMENTS"):
        announcements = fetch_announcements()
        display_announcements(console, announcements)

    # Create a boxed questionnaire for each step
    def create_question_box(title, prompt, default=None):
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    def thinking_value_or_prompt(env_var, config_key, label, box_title, box_body, prompt_fn):
        """Return the env-configured reasoning/thinking value, or prompt for it.

        When ``env_var`` is set the interactive choice is skipped and the value
        the env overlay placed on DEFAULT_CONFIG is used — mirroring the
        env-precedence rule applied to the other selection steps.
        """
        if os.environ.get(env_var):
            value = DEFAULT_CONFIG[config_key]
            console.print(f"[green]✓ {label} from environment:[/green] {value}")
            return value
        console.print(create_question_box(box_title, box_body))
        return prompt_fn()

    # Step 1: Ticker symbol
    default_ticker = saved.get("last_ticker") or "SPY"
    if ov.get("ticker"):
        selected_ticker = ov["ticker"]
        console.print(f"[green]✓ Ticker from command line:[/green] {selected_ticker}")
    elif non_interactive:
        selected_ticker = default_ticker
        origin = "last run" if saved.get("last_ticker") else "default"
        console.print(f"[green]✓ Ticker ({origin}):[/green] {selected_ticker}")
    else:
        console.print(
            create_question_box(
                "Step 1: Ticker Symbol",
                "Enter the ticker, with exchange suffix when needed (e.g. SPY, 0700.HK, BTC-USD)",
                default_ticker,
            )
        )
        selected_ticker = get_ticker(saved.get("last_ticker"))
    asset_type = detect_asset_type(selected_ticker)
    # Only announce when it's not the default stock path, to avoid printing
    # "stock" on every run.
    if asset_type.value != "stock":
        console.print(
            f"[green]Detected asset type:[/green] {asset_type.value}"
        )

    # Step 2: Analysis date
    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    if ov.get("date"):
        analysis_date = ov["date"]
        console.print(f"[green]✓ Analysis date from command line:[/green] {analysis_date}")
    elif non_interactive:
        analysis_date = default_date
        console.print(f"[green]✓ Analysis date (today):[/green] {analysis_date}")
    else:
        console.print(
            create_question_box(
                "Step 2: Analysis Date",
                "Enter the analysis date (YYYY-MM-DD)",
                default_date,
            )
        )
        analysis_date = get_analysis_date()
    _note_non_trading_date(analysis_date, asset_type)

    # --yes: resolve everything else from flags > env > saved > defaults.
    if non_interactive:
        settings, src, problem = _resolve_run_settings(ov, saved, asset_type)
        if problem:
            console.print(f"[red]{problem}[/red]")
            raise typer.Exit(2)
        _print_run_settings(settings, src, selected_ticker, analysis_date)
        ensure_api_key(settings["llm_provider"], interactive=False)
        return _selections_from(settings, selected_ticker, asset_type, analysis_date)

    # Fast path: a returning user reuses the whole saved configuration with one
    # Enter instead of re-answering steps 3-8. Flags targeting those steps mean
    # the user is customizing, so the stepwise wizard runs instead.
    fast_path_flags = (
        "lang", "analysts", "depth", "provider", "quick_llm", "deep_llm", "backend_url",
    )
    if (
        is_tty
        and has_core_settings(saved)
        and not any(ov.get(key) for key in fast_path_flags)
    ):
        settings, src, problem = _resolve_run_settings({}, saved, asset_type)
        if problem is None:
            _print_run_settings(settings, src, selected_ticker, analysis_date)
            if _confirm_reuse_settings():
                ensure_api_key(settings["llm_provider"], interactive=interactive_key)
                selections = _selections_from(
                    settings, selected_ticker, asset_type, analysis_date
                )
                _autosave_settings(selections)
                return selections

    # Step 3: Output language (skipped when set via --lang or
    # TRADINGAGENTS_OUTPUT_LANGUAGE)
    if ov.get("lang"):
        output_language = ov["lang"]
        console.print(f"[green]✓ Output language from command line:[/green] {output_language}")
    elif os.environ.get("TRADINGAGENTS_OUTPUT_LANGUAGE"):
        output_language = DEFAULT_CONFIG["output_language"]
        console.print(
            f"[green]✓ Output language from environment:[/green] {output_language}"
        )
    else:
        console.print(
            create_question_box(
                "Step 3: Output Language",
                "Select the language for analyst reports and final decision"
            )
        )
        output_language = ask_output_language(saved.get("output_language"))

    # Step 4: Select analysts (skipped when set via --analysts)
    flag_analysts = None
    if ov.get("analysts"):
        requested_analysts = list(ov["analysts"])
        flag_analysts = filter_analysts_for_asset_type(requested_analysts, asset_type)
        if len(flag_analysts) < len(requested_analysts):
            console.print(
                "[dim]Fundamentals analyst dropped: unavailable for crypto assets.[/dim]"
            )
        if not flag_analysts:
            # e.g. `-a fundamentals` on a crypto ticker: nothing left to run,
            # so fall through to the interactive picker instead of crashing.
            console.print(
                "[red]None of the requested analysts are available for this "
                "asset type — pick at least one of market, social, news.[/red]"
            )
    if flag_analysts:
        selected_analysts = flag_analysts
        console.print(
            f"[green]✓ Analysts from command line:[/green] "
            f"{', '.join(analyst.value for analyst in selected_analysts)}"
        )
    else:
        console.print(
            create_question_box(
                "Step 4: Analysts Team", "Select your LLM analyst agents for the analysis"
            )
        )
        saved_analysts = [AnalystType(a) for a in saved.get("analysts", [])]
        selected_analysts = select_analysts(asset_type, preselected=saved_analysts or None)
        console.print(
            f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
        )

    # Step 5: Research depth (skipped via --depth, or when both round counts
    # are set via env).
    # Research depth maps to the debate + risk round counts; when both are
    # supplied through TRADINGAGENTS_MAX_DEBATE_ROUNDS / _MAX_RISK_ROUNDS we keep
    # the run non-interactive and honor the env values (#977).
    depth_from_env = bool(os.environ.get("TRADINGAGENTS_MAX_DEBATE_ROUNDS")) and bool(
        os.environ.get("TRADINGAGENTS_MAX_RISK_ROUNDS")
    )
    if ov.get("depth"):
        selected_research_depth = ov["depth"]
        console.print(
            f"[green]✓ Research depth from command line:[/green] "
            f"{depth_label(selected_research_depth)}"
        )
    elif depth_from_env:
        selected_research_depth = DEFAULT_CONFIG["max_debate_rounds"]
        console.print(
            f"[green]✓ Research depth from environment:[/green] "
            f"{DEFAULT_CONFIG['max_debate_rounds']} debate / "
            f"{DEFAULT_CONFIG['max_risk_discuss_rounds']} risk rounds"
        )
    else:
        console.print(
            create_question_box(
                "Step 5: Research Depth", "Select your research depth level"
            )
        )
        selected_research_depth = select_research_depth(saved.get("research_depth"))

    # Step 6: LLM Provider (skipped when set via --provider or
    # TRADINGAGENTS_LLM_PROVIDER).
    # The backend URL comes from --backend-url when given, then
    # TRADINGAGENTS_LLM_BACKEND_URL, otherwise the provider's default
    # endpoint — the same value the menu would have picked.
    provider_from_env = bool(os.environ.get("TRADINGAGENTS_LLM_PROVIDER"))
    provider_from_flag = bool(ov.get("provider"))
    if provider_from_flag:
        selected_llm_provider = ov["provider"].lower()
        backend_url = ov.get("backend_url") or resolve_backend_url(
            selected_llm_provider,
            _saved_for_provider(saved, selected_llm_provider, "backend_url"),
            env_url=DEFAULT_CONFIG["backend_url"],
        )
        console.print(f"[green]✓ LLM provider from command line:[/green] {selected_llm_provider}")
        if selected_llm_provider == "openai_compatible" and not backend_url:
            backend_url = prompt_openai_compatible_url()
        if backend_url:
            console.print(f"[green]✓ Backend URL:[/green] {backend_url}")
        if selected_llm_provider == "ollama":
            confirm_ollama_endpoint(backend_url)
        ensure_api_key(selected_llm_provider, interactive=interactive_key)
    elif provider_from_env:
        selected_llm_provider = DEFAULT_CONFIG["llm_provider"].lower()
        backend_url = ov.get("backend_url") or resolve_backend_url(
            selected_llm_provider, env_url=DEFAULT_CONFIG["backend_url"]
        )
        console.print(f"[green]✓ LLM provider from environment:[/green] {selected_llm_provider}")
        console.print(f"[green]✓ Backend URL:[/green] {backend_url}")
        # Still confirm/persist the API key so the run doesn't fail later.
        ensure_api_key(selected_llm_provider, interactive=interactive_key)
    else:
        console.print(
            create_question_box(
                "Step 6: LLM Provider", "Select your LLM provider"
            )
        )
        selected_llm_provider, backend_url = select_llm_provider(saved.get("llm_provider"))

        # Providers with regional endpoints prompt for the region as a secondary
        # step so the main dropdown stays clean (mainland China and international
        # accounts cannot share API keys).
        if selected_llm_provider == "qwen":
            selected_llm_provider, backend_url = ask_qwen_region()
        elif selected_llm_provider == "minimax":
            selected_llm_provider, backend_url = ask_minimax_region()
        elif selected_llm_provider == "glm":
            selected_llm_provider, backend_url = ask_glm_region()

        # Honor an explicit --backend-url or env backend URL even when the
        # provider was chosen interactively, so it isn't overwritten by the
        # menu default (#978).
        backend_url = ov.get("backend_url") or resolve_backend_url(
            selected_llm_provider, backend_url, env_url=DEFAULT_CONFIG["backend_url"]
        )

        # The generic OpenAI-compatible endpoint has no default; ask for it if
        # neither the menu nor the environment supplied one.
        if selected_llm_provider == "openai_compatible" and not backend_url:
            backend_url = prompt_openai_compatible_url(
                _saved_for_provider(saved, "openai_compatible", "backend_url")
            )

        # For Ollama, surface the resolved endpoint (OLLAMA_BASE_URL vs default)
        # before model selection so it's obvious where we're connecting.
        if selected_llm_provider == "ollama":
            confirm_ollama_endpoint(backend_url)

        # Confirm the provider's API key is present; prompt the user to paste
        # one and persist it to .env if it's missing, so the analysis run
        # doesn't fail later at the first API call.
        ensure_api_key(selected_llm_provider, interactive=interactive_key)

    # Step 7: Thinking agents. Each model resolves independently: --quick-llm /
    # --deep-llm wins, then the env pair (either TRADINGAGENTS_*_THINK_LLM var
    # pins both, mirroring the established skip rule), then a prompt defaulting
    # to the saved choice.
    models_from_env = bool(
        os.environ.get("TRADINGAGENTS_QUICK_THINK_LLM")
        or os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM")
    )
    needs_model_prompt = not models_from_env and not (
        ov.get("quick_llm") and ov.get("deep_llm")
    )
    if needs_model_prompt:
        console.print(
            create_question_box(
                "Step 7: Thinking Agents", "Select your thinking agents for analysis"
            )
        )
    if ov.get("quick_llm"):
        selected_shallow_thinker = ov["quick_llm"]
        console.print(
            f"[green]✓ Quick-thinking model from command line:[/green] "
            f"{selected_shallow_thinker}"
        )
    elif models_from_env:
        selected_shallow_thinker = DEFAULT_CONFIG["quick_think_llm"]
        console.print(
            f"[green]✓ Quick-thinking model from environment:[/green] "
            f"{selected_shallow_thinker}"
        )
    else:
        selected_shallow_thinker = select_shallow_thinking_agent(
            selected_llm_provider,
            _saved_for_provider(saved, selected_llm_provider, "quick_think_llm"),
        )
    if ov.get("deep_llm"):
        selected_deep_thinker = ov["deep_llm"]
        console.print(
            f"[green]✓ Deep-thinking model from command line:[/green] "
            f"{selected_deep_thinker}"
        )
    elif models_from_env:
        selected_deep_thinker = DEFAULT_CONFIG["deep_think_llm"]
        console.print(
            f"[green]✓ Deep-thinking model from environment:[/green] "
            f"{selected_deep_thinker}"
        )
    else:
        selected_deep_thinker = select_deep_thinking_agent(
            selected_llm_provider,
            _saved_for_provider(saved, selected_llm_provider, "deep_think_llm"),
        )

    # Step 8: Provider-specific reasoning/thinking configuration. Each knob is
    # settable via its TRADINGAGENTS_* env var; when that var is set (or the
    # provider itself came from env) the prompt is skipped and the configured
    # value is used — same env-precedence rule as the steps above. None = each
    # provider's own default.
    thinking_level = None
    reasoning_effort = None
    anthropic_effort = None

    provider_lower = selected_llm_provider.lower()
    if provider_from_env:
        thinking_level = DEFAULT_CONFIG["google_thinking_level"]
        reasoning_effort = DEFAULT_CONFIG["openai_reasoning_effort"]
        anthropic_effort = DEFAULT_CONFIG["anthropic_effort"]
    elif provider_lower == "google":
        thinking_level = thinking_value_or_prompt(
            "TRADINGAGENTS_GOOGLE_THINKING_LEVEL", "google_thinking_level",
            "Gemini thinking mode", "Step 8: Thinking Mode",
            "Configure Gemini thinking mode", ask_gemini_thinking_config,
        )
    elif provider_lower == "openai":
        reasoning_effort = thinking_value_or_prompt(
            "TRADINGAGENTS_OPENAI_REASONING_EFFORT", "openai_reasoning_effort",
            "Reasoning effort", "Step 8: Reasoning Effort",
            "Configure OpenAI reasoning effort level", ask_openai_reasoning_effort,
        )
    elif provider_lower == "anthropic":
        anthropic_effort = thinking_value_or_prompt(
            "TRADINGAGENTS_ANTHROPIC_EFFORT", "anthropic_effort",
            "Claude effort", "Step 8: Effort Level",
            "Configure Claude effort level", ask_anthropic_effort,
        )

    selections = {
        "ticker": selected_ticker,
        "asset_type": asset_type.value,
        "analysis_date": analysis_date,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "shallow_thinker": selected_shallow_thinker,
        "deep_thinker": selected_deep_thinker,
        "google_thinking_level": thinking_level,
        "openai_reasoning_effort": reasoning_effort,
        "anthropic_effort": anthropic_effort,
        "output_language": output_language,
    }
    # Remember the choices for the next session (interactive human runs only;
    # scripted and test runs never write).
    if interactive_key:
        _autosave_settings(selections)
    return selections


def get_analysis_date():
    """Get the analysis date from user input."""
    while True:
        date_str = typer.prompt(
            "", default=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        try:
            # Validate date format and ensure it's not in the future
            analysis_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            if analysis_date.date() > datetime.datetime.now().date():
                console.print("[red]Error: Analysis date cannot be in the future[/red]")
                continue
            return date_str
        except ValueError:
            console.print(
                "[red]Error: Invalid date format. Please use YYYY-MM-DD[/red]"
            )


def save_report_to_disk(final_state, ticker: str, save_path: Path):
    """Save the complete analysis report to disk (shared CLI/API writer)."""
    return write_report_tree(final_state, ticker, save_path)


def display_complete_report(final_state):
    """Display the complete analysis report sequentially (avoids truncation)."""
    console.print()
    console.print(Rule("Complete Analysis Report", style="bold green"))

    # I. Analyst Team Reports
    analysts = []
    if final_state.get("market_report"):
        analysts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts.append(("Sentiment Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analysts:
        console.print(Panel("[bold]I. Analyst Team Reports[/bold]", border_style="cyan"))
        for title, content in analysts:
            console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    # II. Research Team Reports
    if final_state.get("investment_debate_state"):
        debate = final_state["investment_debate_state"]
        research = []
        if debate.get("bull_history"):
            research.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("judge_decision"):
            research.append(("Research Manager", debate["judge_decision"]))
        if research:
            console.print(Panel("[bold]II. Research Team Decision[/bold]", border_style="magenta"))
            for title, content in research:
                console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    # III. Trading Team
    if final_state.get("trader_investment_plan"):
        console.print(Panel("[bold]III. Trading Team Plan[/bold]", border_style="yellow"))
        console.print(Panel(Markdown(final_state["trader_investment_plan"]), title="Trader", border_style="blue", padding=(1, 2)))

    # IV. Risk Management Team
    if final_state.get("risk_debate_state"):
        risk = final_state["risk_debate_state"]
        risk_reports = []
        if risk.get("aggressive_history"):
            risk_reports.append(("Aggressive Analyst", risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_reports.append(("Conservative Analyst", risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_reports.append(("Neutral Analyst", risk["neutral_history"]))
        if risk_reports:
            console.print(Panel("[bold]IV. Risk Management Team Decision[/bold]", border_style="red"))
            for title, content in risk_reports:
                console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

        # V. Portfolio Manager Decision
        if risk.get("judge_decision"):
            console.print(Panel("[bold]V. Portfolio Manager Decision[/bold]", border_style="green"))
            console.print(Panel(Markdown(risk["judge_decision"]), title="Portfolio Manager", border_style="blue", padding=(1, 2)))


def update_research_team_status(status):
    """Update status for research team members (not Trader)."""
    research_team = ["Bull Researcher", "Bear Researcher", "Research Manager"]
    for agent in research_team:
        message_buffer.update_agent_status(agent, status)


# Ordered list of analysts for status transitions
ANALYST_ORDER = ["market", "social", "news", "fundamentals"]
ANALYST_AGENT_NAMES = {
    "market": "Market Analyst",
    "social": "Sentiment Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}
ANALYST_REPORT_MAP = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}


def update_analyst_statuses(message_buffer, chunk, wall_time_tracker=None):
    """Update analyst statuses based on accumulated report state.

    Logic:
    - Store new report content from the current chunk if present
    - Check accumulated report_sections (not just current chunk) for status
    - Analysts with reports = completed
    - First analyst without report = in_progress
    - Remaining analysts without reports = pending
    - When all analysts done, set Bull Researcher to in_progress
    """
    selected = message_buffer.selected_analysts
    found_active = False

    if wall_time_tracker is not None:
        sync_analyst_tracker_from_chunk(wall_time_tracker, chunk)

    for analyst_key in ANALYST_ORDER:
        if analyst_key not in selected:
            continue

        agent_name = ANALYST_AGENT_NAMES[analyst_key]
        report_key = ANALYST_REPORT_MAP[analyst_key]

        # Capture new report content from current chunk
        if chunk.get(report_key):
            message_buffer.update_report_section(report_key, chunk[report_key])

        # Determine status from accumulated sections, not just current chunk
        has_report = bool(message_buffer.report_sections.get(report_key))

        if has_report:
            message_buffer.update_agent_status(agent_name, "completed")
        elif not found_active:
            message_buffer.update_agent_status(agent_name, "in_progress")
            found_active = True
        else:
            message_buffer.update_agent_status(agent_name, "pending")

    # When all analysts complete, transition research team to in_progress
    if (
        not found_active
        and selected
        and message_buffer.agent_status.get("Bull Researcher") == "pending"
    ):
        message_buffer.update_agent_status("Bull Researcher", "in_progress")

def extract_content_string(content):
    """Extract string content from various message formats.
    Returns None if no meaningful text content is found.
    """
    import ast

    def is_empty(val):
        """Check if value is empty using Python's truthiness."""
        if val is None or val == '':
            return True
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return True
            try:
                return not bool(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                return False  # Can't parse = real text
        return not bool(val)

    if is_empty(content):
        return None

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        text = content.get('text', '')
        return text.strip() if not is_empty(text) else None

    if isinstance(content, list):
        text_parts = [
            item.get('text', '').strip() if isinstance(item, dict) and item.get('type') == 'text'
            else (item.strip() if isinstance(item, str) else '')
            for item in content
        ]
        result = ' '.join(t for t in text_parts if t and not is_empty(t))
        return result if result else None

    return str(content).strip() if not is_empty(content) else None


def classify_message_type(message) -> tuple[str, str | None]:
    """Classify LangChain message into display type and extract content.

    Returns:
        (type, content) - type is one of: User, Agent, Data, Control
                        - content is extracted string or None
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    content = extract_content_string(getattr(message, 'content', None))

    if isinstance(message, HumanMessage):
        if content and content.strip() == "Continue":
            return ("Control", content)
        return ("User", content)

    if isinstance(message, ToolMessage):
        return ("Data", content)

    if isinstance(message, AIMessage):
        return ("Agent", content)

    # Fallback for unknown types
    return ("System", content)


def format_tool_args(args, max_length=80) -> str:
    """Format tool arguments for terminal display."""
    result = str(args)
    if len(result) > max_length:
        return result[:max_length - 3] + "..."
    return result

def _build_run_config(selections: dict, checkpoint: bool | None) -> dict:
    """Assemble the run config from interactive selections, honoring env precedence.

    Round counts and checkpoint follow "explicit env/flag wins": an env-applied
    value on DEFAULT_CONFIG is preserved unless the user overrode it on the CLI.
    """
    config = DEFAULT_CONFIG.copy()
    # Research depth sets both round counts, but an explicit env override
    # (TRADINGAGENTS_MAX_DEBATE_ROUNDS / _MAX_RISK_ROUNDS) wins over the
    # interactive selection — leave the env-applied value in place (#977).
    if not os.environ.get("TRADINGAGENTS_MAX_DEBATE_ROUNDS"):
        config["max_debate_rounds"] = selections["research_depth"]
    if not os.environ.get("TRADINGAGENTS_MAX_RISK_ROUNDS"):
        config["max_risk_discuss_rounds"] = selections["research_depth"]
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()
    # Provider-specific thinking configuration
    config["google_thinking_level"] = selections.get("google_thinking_level")
    config["openai_reasoning_effort"] = selections.get("openai_reasoning_effort")
    config["anthropic_effort"] = selections.get("anthropic_effort")
    config["output_language"] = selections.get("output_language", "English")
    # --checkpoint/--no-checkpoint overrides only when explicitly given; omitting
    # the flag preserves TRADINGAGENTS_CHECKPOINT_ENABLED / the default (#976).
    if checkpoint is not None:
        config["checkpoint_enabled"] = checkpoint
    return config


def run_analysis(
    checkpoint: bool | None = None,
    overrides: dict | None = None,
    non_interactive: bool = False,
    save_report: bool | None = None,
    report_path: Path | None = None,
    show_report: bool | None = None,
):
    # First get all user selections
    selections = get_user_selections(overrides, non_interactive)

    config = _build_run_config(selections, checkpoint)

    # Create stats callback handler for tracking LLM/tool calls
    stats_handler = StatsCallbackHandler()

    # Normalize analyst selection to predefined order (selection is a 'set', order is fixed)
    selected_set = {analyst.value for analyst in selections["analysts"]}
    selected_analyst_keys = [a for a in ANALYST_ORDER if a in selected_set]
    if not selected_analyst_keys:
        # Belt-and-braces: every selection path validates this, but an empty
        # team must never reach graph construction as a raw ValueError.
        console.print("[red]No analysts selected for this run — nothing to do.[/red]")
        raise typer.Exit(2)
    analyst_execution_plan = build_analyst_execution_plan(selected_analyst_keys)
    analyst_wall_time_tracker = AnalystWallTimeTracker(analyst_execution_plan)

    # Initialize message buffer with selected analysts
    message_buffer.init_for_analysis(selected_analyst_keys)

    # Track start time for elapsed display
    start_time = time.time()

    # Create result directory
    results_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)

    def save_message_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            content = content.replace("\n", " ")  # Replace newlines with spaces
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [{message_type}] {content}\n")
        return wrapper

    def save_tool_call_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, tool_name, args = obj.tool_calls[-1]
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
        return wrapper

    def save_report_section_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(section_name, content):
            func(section_name, content)
            if section_name in obj.report_sections and obj.report_sections[section_name] is not None:
                content = obj.report_sections[section_name]
                if content:
                    file_name = f"{section_name}.md"
                    text = "\n".join(str(item) for item in content) if isinstance(content, list) else content
                    with open(report_dir / file_name, "w", encoding="utf-8") as f:
                        f.write(text)
        return wrapper

    message_buffer.add_message = save_message_decorator(message_buffer, "add_message")
    message_buffer.add_tool_call = save_tool_call_decorator(message_buffer, "add_tool_call")
    message_buffer.update_report_section = save_report_section_decorator(message_buffer, "update_report_section")

    # Now start the display layout
    layout = create_layout()

    run_error: BaseException | None = None
    final_state: dict = {}
    try:
        # Construct the graph inside the guarded region: LLM-client creation
        # is the first place a bad key/model/endpoint surfaces, and it should
        # get the same clear failure report as a mid-stream error.
        graph = TradingAgentsGraph(
            selected_analyst_keys,
            config=config,
            debug=True,
            callbacks=[stats_handler],
        )

        # checkpointed_run recompiles the graph with a per-ticker saver (a
        # no-op session when checkpointing is off), so --checkpoint works on
        # the CLI's manual stream exactly like it does through propagate().
        with graph.checkpointed_run(
            selections["ticker"], selections["analysis_date"], selections["asset_type"]
        ) as checkpoint_session:
            final_state = _stream_analysis(
                graph,
                selections,
                layout,
                stats_handler,
                start_time,
                selected_analyst_keys,
                analyst_execution_plan,
                analyst_wall_time_tracker,
                checkpoint_session,
            )
    except KeyboardInterrupt as exc:
        run_error = exc
    except Exception as exc:
        run_error = exc

    if run_error is not None:
        _report_run_failure(run_error, config, results_dir)
        raise typer.Exit(130 if isinstance(run_error, KeyboardInterrupt) else 1)

    # A completed run's checkpoint must not shadow the next run (#1089).
    graph.clear_run_checkpoint(
        selections["ticker"], selections["analysis_date"], selections["asset_type"]
    )

    _finish_run(
        final_state,
        selections,
        graph,
        results_dir,
        wall_summary=analyst_wall_time_tracker.format_summary(),
        save_report=save_report,
        report_path=report_path,
        show_report=show_report,
        non_interactive=non_interactive,
    )


def _stream_analysis(
    graph,
    selections,
    layout,
    stats_handler,
    start_time,
    selected_analyst_keys,
    analyst_execution_plan,
    analyst_wall_time_tracker,
    checkpoint_session,
) -> dict:
    """Drive the graph stream under the Rich Live display.

    Returns the merged final state; exceptions (including Ctrl-C) propagate to
    run_analysis, which reports the failure point and any partial reports.
    """
    with Live(layout, refresh_per_second=4):
        # Initial display
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Add initial messages
        message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
        if selections["asset_type"] != "stock":
            message_buffer.add_message("System", f"Detected asset type: {selections['asset_type']}")
        message_buffer.add_message(
            "System", f"Analysis date: {selections['analysis_date']}"
        )
        message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(analyst.value for analyst in selections['analysts'])}",
        )
        if checkpoint_session.resume_step is not None:
            message_buffer.add_message(
                "System",
                f"Resuming from step {checkpoint_session.resume_step} "
                f"(saved checkpoint for this ticker and date)",
            )
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Update agent status to in_progress for the first analyst
        first_analyst = get_initial_analyst_node(analyst_execution_plan)
        message_buffer.update_agent_status(first_analyst, "in_progress")
        analyst_wall_time_tracker.mark_started(selected_analyst_keys[0])
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Create spinner text
        spinner_text = (
            f"Analyzing {selections['ticker']} on {selections['analysis_date']}..."
        )
        update_display(layout, spinner_text, stats_handler=stats_handler, start_time=start_time)

        # Initialize state and get graph args with callbacks.
        # Resolve the instrument identity once here so all agents anchor to
        # the real company (#814); the CLI builds state directly rather than
        # going through propagate(), so this must happen on the CLI path too.
        instrument_context = graph.resolve_instrument_context(
            selections["ticker"], selections["asset_type"]
        )
        init_agent_state = graph.propagator.create_initial_state(
            selections["ticker"],
            selections["analysis_date"],
            asset_type=selections["asset_type"],
            instrument_context=instrument_context,
        )
        # Pass callbacks to graph config for tool execution tracking
        # (LLM tracking is handled separately via LLM constructor)
        args = graph.propagator.get_graph_args(callbacks=[stats_handler])
        # Key the run to its checkpoint thread so a resume continues this
        # ticker+date instead of starting over.
        if checkpoint_session.config_patch:
            args["config"].update(checkpoint_session.config_patch)
        # LangGraph resumes a checkpointed thread only when the input is None;
        # passing state restarts from the entry point (and appends duplicate
        # messages onto the thread's persisted history).
        stream_input = (
            None if checkpoint_session.resume_step is not None else init_agent_state
        )

        # Stream the analysis
        trace = []
        for chunk in graph.graph.stream(stream_input, **args):
            # Process all messages in chunk, deduplicating by message ID
            for message in chunk.get("messages", []):
                msg_id = getattr(message, "id", None)
                if msg_id is not None:
                    if msg_id in message_buffer._processed_message_ids:
                        continue
                    message_buffer._processed_message_ids.add(msg_id)

                msg_type, content = classify_message_type(message)
                if content and content.strip():
                    message_buffer.add_message(msg_type, content)

                if hasattr(message, "tool_calls") and message.tool_calls:
                    for tool_call in message.tool_calls:
                        if isinstance(tool_call, dict):
                            message_buffer.add_tool_call(tool_call["name"], tool_call["args"])
                        else:
                            message_buffer.add_tool_call(tool_call.name, tool_call.args)

            # Update analyst statuses based on report state (runs on every chunk)
            update_analyst_statuses(
                message_buffer,
                chunk,
                wall_time_tracker=analyst_wall_time_tracker,
            )

            # Research Team - Handle Investment Debate State
            if chunk.get("investment_debate_state"):
                debate_state = chunk["investment_debate_state"]
                bull_hist = debate_state.get("bull_history", "").strip()
                bear_hist = debate_state.get("bear_history", "").strip()
                judge = debate_state.get("judge_decision", "").strip()

                # Only update status when there's actual content
                if bull_hist or bear_hist:
                    update_research_team_status("in_progress")
                if bull_hist:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Bull Researcher Analysis\n{bull_hist}"
                    )
                if bear_hist:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Bear Researcher Analysis\n{bear_hist}"
                    )
                if judge:
                    message_buffer.update_report_section(
                        "investment_plan", f"### Research Manager Decision\n{judge}"
                    )
                    update_research_team_status("completed")
                    message_buffer.update_agent_status("Trader", "in_progress")

            # Trading Team
            if chunk.get("trader_investment_plan"):
                message_buffer.update_report_section(
                    "trader_investment_plan", chunk["trader_investment_plan"]
                )
                if message_buffer.agent_status.get("Trader") != "completed":
                    message_buffer.update_agent_status("Trader", "completed")
                    message_buffer.update_agent_status("Aggressive Analyst", "in_progress")

            # Risk Management Team - Handle Risk Debate State
            if chunk.get("risk_debate_state"):
                risk_state = chunk["risk_debate_state"]
                agg_hist = risk_state.get("aggressive_history", "").strip()
                con_hist = risk_state.get("conservative_history", "").strip()
                neu_hist = risk_state.get("neutral_history", "").strip()
                judge = risk_state.get("judge_decision", "").strip()

                if agg_hist:
                    if message_buffer.agent_status.get("Aggressive Analyst") != "completed":
                        message_buffer.update_agent_status("Aggressive Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Aggressive Analyst Analysis\n{agg_hist}"
                    )
                if con_hist:
                    if message_buffer.agent_status.get("Conservative Analyst") != "completed":
                        message_buffer.update_agent_status("Conservative Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Conservative Analyst Analysis\n{con_hist}"
                    )
                if neu_hist:
                    if message_buffer.agent_status.get("Neutral Analyst") != "completed":
                        message_buffer.update_agent_status("Neutral Analyst", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Neutral Analyst Analysis\n{neu_hist}"
                    )
                if judge and message_buffer.agent_status.get("Portfolio Manager") != "completed":
                    message_buffer.update_agent_status("Portfolio Manager", "in_progress")
                    message_buffer.update_report_section(
                        "final_trade_decision", f"### Portfolio Manager Decision\n{judge}"
                    )
                    message_buffer.update_agent_status("Aggressive Analyst", "completed")
                    message_buffer.update_agent_status("Conservative Analyst", "completed")
                    message_buffer.update_agent_status("Neutral Analyst", "completed")
                    message_buffer.update_agent_status("Portfolio Manager", "completed")

            # Update the display
            update_display(layout, stats_handler=stats_handler, start_time=start_time)

            trace.append(chunk)

        # Streamed chunks are per-node deltas, not full state. Merge them
        # so every report field populated across the run is present.
        final_state = {}
        for chunk in trace:
            final_state.update(chunk)

        # Update all agent statuses to completed
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message(
            "System", f"Completed analysis for {selections['analysis_date']}"
        )
        message_buffer.add_message("System", analyst_wall_time_tracker.format_summary())

        # Update final report sections
        for section in message_buffer.report_sections:
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])

        update_display(layout, stats_handler=stats_handler, start_time=start_time)

    return final_state


def _report_run_failure(error: BaseException, config: dict, results_dir: Path) -> None:
    """Explain a mid-run failure: where it stopped, what's on disk, how to resume."""
    failing_agent = message_buffer.current_agent or "startup"
    if isinstance(error, KeyboardInterrupt):
        console.print(f"\n[yellow]Run interrupted at: {failing_agent}[/yellow]")
    else:
        console.print(
            f"\n[red]Run failed at {failing_agent}: "
            f"{type(error).__name__}: {error}[/red]"
        )
    report_dir = results_dir / "reports"
    finished = sorted(p.stem for p in report_dir.glob("*.md")) if report_dir.exists() else []
    if finished:
        console.print(
            f"[yellow]Completed sections already saved ({', '.join(finished)}):[/yellow] "
            f"{report_dir}"
        )
    if config.get("checkpoint_enabled"):
        console.print(
            "[yellow]Checkpointing is on — rerun with the same ticker and date "
            "to resume from the last completed step.[/yellow]"
        )
    else:
        console.print(
            "[dim]Tip: run with --checkpoint to make interrupted runs resumable.[/dim]"
        )


def _finish_run(
    final_state: dict,
    selections: dict,
    graph,
    results_dir: Path,
    *,
    wall_summary: str,
    save_report: bool | None,
    report_path: Path | None,
    show_report: bool | None,
    non_interactive: bool,
) -> None:
    """Post-analysis flow: save/show the report and print the final decision.

    The two historical prompts only appear on an interactive TTY with no
    deciding flag; scripted runs save to the default path and print a stable
    ``FINAL DECISION:`` line instead of blocking. Exits with code 2 when the
    run produced no final decision.
    """
    console.print("\n[bold cyan]Analysis Complete![/bold cyan]\n")
    console.print(f"[dim]{wall_summary}[/dim]")
    console.print(f"[dim]Live section reports: {results_dir / 'reports'}[/dim]")

    interactive = sys.stdin.isatty() and not non_interactive

    # Save the consolidated report tree
    do_save = save_report
    if do_save is None:
        if interactive:
            save_choice = typer.prompt("Save report?", default="Y").strip().upper()
            do_save = save_choice in ("Y", "YES", "")
        else:
            do_save = True
    save_failed = False
    if do_save:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = Path.cwd() / "reports" / f"{selections['ticker']}_{timestamp}"
        save_path = Path(report_path) if report_path else default_path
        if interactive and report_path is None:
            save_path = Path(
                typer.prompt(
                    "Save path (press Enter for default)", default=str(default_path)
                ).strip()
            )
        try:
            report_file = save_report_to_disk(final_state, selections["ticker"], save_path)
            console.print(f"\n[green]✓ Report saved to:[/green] {save_path.resolve()}")
            console.print(f"  [dim]Complete report:[/dim] {report_file.name}")
        except Exception as e:
            console.print(f"[red]Error saving report: {e}[/red]")
            save_failed = True

    # Extract and print the core decision — the same signal propagate() returns
    # to programmatic callers, and a stable line for scripts to grep.
    decision_text = final_state.get("final_trade_decision")
    if decision_text:
        try:
            decision = graph.process_signal(decision_text)
        except Exception:
            decision = None
        if decision:
            console.print(f"\n[bold green]FINAL DECISION: {decision}[/bold green]")

    # Display the full report
    do_show = show_report
    if do_show is None:
        if interactive:
            display_choice = (
                typer.prompt("\nDisplay full report on screen?", default="Y")
                .strip()
                .upper()
            )
            do_show = display_choice in ("Y", "YES", "")
        else:
            do_show = False
    if do_show:
        display_complete_report(final_state)

    if not decision_text:
        console.print(
            "[red]The run completed without a final trade decision — "
            "check the logs above.[/red]"
        )
        raise typer.Exit(2)

    # An explicitly requested (or scripted-default) save that failed is a
    # failed deliverable: scripts need a non-zero signal, not a red line lost
    # in the output. Interactive prompt-driven saves keep the soft behavior.
    if save_failed and (save_report is True or report_path or not interactive):
        console.print(
            "[red]The consolidated report could not be saved (see error above).[/red]"
        )
        raise typer.Exit(1)


def _clear_all_checkpoints_now() -> None:
    from tradingagents.graph.checkpointer import clear_all_checkpoints
    n = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
    console.print(f"[yellow]Cleared {n} checkpoint(s).[/yellow]")


def _validated_date(value: str) -> str:
    """Validate a --date flag value (same rules as the interactive prompt)."""
    value = value.strip()
    try:
        parsed = datetime.datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise typer.BadParameter(
            "Use YYYY-MM-DD, e.g. 2026-01-15.", param_hint="--date"
        ) from None
    if parsed.date() > datetime.datetime.now().date():
        raise typer.BadParameter(
            "Analysis date cannot be in the future.", param_hint="--date"
        )
    # strptime accepts non-padded parts ("2026-1-5"); normalize so downstream
    # string-keyed date comparisons see the canonical form.
    return parsed.strftime("%Y-%m-%d")


@app.callback(invoke_without_command=True)
def cli(
    ctx: typer.Context,
    checkpoint: bool | None = typer.Option(
        None,
        "--checkpoint/--no-checkpoint",
        help="Enable/disable checkpoint-resume (save state after each node so a "
        "crashed run can resume). Omit to honor TRADINGAGENTS_CHECKPOINT_ENABLED.",
    ),
    clear_checkpoints: bool = typer.Option(
        False,
        "--clear-checkpoints",
        help="Delete all saved checkpoints before running (force fresh start).",
    ),
):
    """TradingAgents CLI: Multi-Agents LLM Financial Trading Framework.

    Bare `tradingagents` launches the interactive wizard. `tradingagents
    analyze` accepts flags for one-line and scripted runs; `doctor` checks
    your setup; `config` manages saved settings.
    """
    if ctx.invoked_subcommand is not None:
        # `tradingagents --checkpoint analyze ...` parses at the root level;
        # stash the values so analyze can honor them instead of silently
        # dropping flags placed before the subcommand.
        ctx.obj = {"checkpoint": checkpoint, "clear_checkpoints": clear_checkpoints}
        return
    if clear_checkpoints:
        _clear_all_checkpoints_now()
    run_analysis(checkpoint=checkpoint)


@app.command()
def analyze(
    ctx: typer.Context,
    ticker: str | None = typer.Argument(
        None,
        help="Ticker symbol (e.g. NVDA, 0700.HK, BTC-USD). Prompted when omitted.",
    ),
    date: str | None = typer.Option(
        None, "--date", "-d",
        help="Analysis date, YYYY-MM-DD. Prompted when omitted (today with --yes).",
    ),
    analysts: str | None = typer.Option(
        None, "--analysts", "-a",
        help="'all' or a comma-separated set of: market, social (or sentiment), "
        "news, fundamentals.",
    ),
    depth: str | None = typer.Option(
        None, "--depth",
        help="Research depth: 1/3/5 or shallow/medium/deep. Round counts pinned "
        "via TRADINGAGENTS_MAX_*_ROUNDS env vars still win.",
    ),
    provider: str | None = typer.Option(
        None, "--provider",
        help="LLM provider key, e.g. openai, anthropic, google, ollama, qwen-cn.",
    ),
    quick_llm: str | None = typer.Option(
        None, "--quick-llm", help="Quick-thinking model ID."
    ),
    deep_llm: str | None = typer.Option(
        None, "--deep-llm", help="Deep-thinking model ID."
    ),
    backend_url: str | None = typer.Option(
        None, "--backend-url",
        help="LLM endpoint base URL (required for openai_compatible).",
    ),
    lang: str | None = typer.Option(
        None, "--lang", help="Report output language, e.g. Korean, Chinese, Spanish."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help="Run without prompts: unanswered choices come from env vars, then "
        "saved settings, then defaults.",
    ),
    save_report: bool | None = typer.Option(
        None, "--save-report/--no-save-report",
        help="Save (or skip saving) the consolidated report without asking.",
    ),
    report_path: str | None = typer.Option(
        None, "--report-path",
        help="Where to save the consolidated report (implies --save-report).",
    ),
    show_report: bool | None = typer.Option(
        None, "--show-report/--no-show-report",
        help="Print (or skip printing) the full report without asking.",
    ),
    checkpoint: bool | None = typer.Option(
        None,
        "--checkpoint/--no-checkpoint",
        help="Enable/disable checkpoint-resume (save state after each node so a "
        "crashed run can resume). Omit to honor TRADINGAGENTS_CHECKPOINT_ENABLED.",
    ),
    clear_checkpoints: bool = typer.Option(
        False,
        "--clear-checkpoints",
        help="Delete all saved checkpoints before running (force fresh start).",
    ),
):
    """Run an analysis. Anything omitted is prompted for interactively, or
    resolved from env vars, saved settings, and defaults with --yes.
    """
    overrides: dict = {}
    if ticker:
        value = ticker.strip()
        if not value or not is_valid_ticker_input(value):
            raise typer.BadParameter(
                f"Invalid ticker {ticker!r} — expected something like AAPL, "
                f"0700.HK, or BTC-USD.",
                param_hint="TICKER",
            )
        overrides["ticker"] = normalize_ticker_symbol(value)
    if date:
        overrides["date"] = _validated_date(date)
    if analysts:
        try:
            overrides["analysts"] = parse_analysts_option(analysts)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--analysts") from None
    if depth:
        try:
            overrides["depth"] = parse_depth_option(depth)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--depth") from None
    if provider:
        key = provider.strip().lower()
        if key not in known_provider_keys():
            raise typer.BadParameter(
                f"Unknown provider {provider!r}. Valid: "
                f"{', '.join(known_provider_keys())}.",
                param_hint="--provider",
            )
        overrides["provider"] = key
    if quick_llm:
        overrides["quick_llm"] = quick_llm.strip()
    if deep_llm:
        overrides["deep_llm"] = deep_llm.strip()
    if backend_url:
        if not backend_url.strip().startswith(("http://", "https://")):
            raise typer.BadParameter(
                "Backend URL must start with http:// or https://.",
                param_hint="--backend-url",
            )
        overrides["backend_url"] = backend_url.strip()
    if lang:
        overrides["lang"] = lang.strip()
    if report_path is not None and save_report is None:
        save_report = True

    # Honor root-level flags placed before the subcommand
    # (`tradingagents --checkpoint analyze ...`).
    root_flags = ctx.obj or {}
    if checkpoint is None:
        checkpoint = root_flags.get("checkpoint")
    clear_checkpoints = clear_checkpoints or root_flags.get("clear_checkpoints", False)

    if clear_checkpoints:
        _clear_all_checkpoints_now()
    run_analysis(
        checkpoint=checkpoint,
        overrides=overrides,
        non_interactive=yes,
        save_report=save_report,
        report_path=report_path,
        show_report=show_report,
    )


@app.command()
def doctor(
    ticker: str | None = typer.Argument(
        None, help="Optional ticker to resolve against live market data."
    ),
):
    """Check your setup (keys, endpoint, data vendors) before spending tokens."""
    from cli.doctor import run_doctor

    raise typer.Exit(run_doctor(ticker))


config_app = typer.Typer(help="Inspect or reset the saved CLI settings.")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show():
    """Print the saved settings and any env overrides that beat them."""
    path = settings_path()
    saved = load_saved_settings()
    console.print(f"[bold]Settings file:[/bold] {path}")
    if not saved:
        if path.exists():
            console.print(
                "[yellow]The file exists but couldn't be read — fix it or run "
                "`tradingagents config reset`.[/yellow]"
            )
        else:
            console.print(
                "[dim]No saved settings yet — they're written after each "
                "interactive run.[/dim]"
            )
        return
    for key, value in saved.items():
        console.print(f"  [cyan]{key}[/cyan]: {value}")
    from tradingagents.default_config import _ENV_OVERRIDES
    active_env = sorted(var for var in _ENV_OVERRIDES if os.environ.get(var))
    if active_env:
        console.print(
            f"[dim]Active env overrides (these beat saved settings): "
            f"{', '.join(active_env)}[/dim]"
        )


@config_app.command("path")
def config_path():
    """Print the settings file path."""
    console.print(str(settings_path()))


@config_app.command("reset")
def config_reset():
    """Delete the saved settings; the next run starts from a clean wizard."""
    if clear_settings():
        console.print(f"[green]Removed {settings_path()}[/green]")
    else:
        console.print("[dim]No saved settings to remove.[/dim]")


if __name__ == "__main__":
    app()
