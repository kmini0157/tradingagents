"""나만의 트레이딩 회사(Trading Firm) — 바로 실행용 진입점.

TradingAgents 프레임워크를 아래 취향에 맞춰 미리 세팅했습니다. `.env`에
API 키 한 줄만 넣으면(없으면 실행 중 붙여넣기) 곧바로 돌아갑니다.

  - 대상 시장 : 한국 종목 (예: 삼성전자 005930.KS) — KOSPI/KOSDAQ 벤치마크 자동
  - LLM       : Anthropic Claude (deep=Opus 4.8, quick=Sonnet 5, effort=high)
                └ .env 에서 TRADINGAGENTS_* 한 줄만 바꾸면 OpenAI 등으로 교체 가능
  - 리포트    : 한국어
  - 성향      : 공격형(aggressive) — 리스크/리서치 토론 라운드를 늘려
                공격적 시나리오까지 충분히 논쟁하도록 구성

빠른 시작 (3단계)
-----------------
    pip install -e .           # 1) 의존성 설치
    python my_company.py       # 2) 처음 실행하면 .env 템플릿을 만들고
                               #    ANTHROPIC_API_KEY 입력을 안내합니다
    # 3) 키를 넣은 뒤 다시 실행하면 분석이 시작됩니다.

사용 예
-------
    python my_company.py                       # 기본: 삼성전자, 최근 거래일
    python my_company.py 000660.KS             # SK하이닉스
    python my_company.py 035420.KS 2026-06-30  # 종목 + 분석 기준일 지정

OpenAI 로 바꾸려면 .env 에:
    OPENAI_API_KEY=sk-...
    TRADINGAGENTS_LLM_PROVIDER=openai
    TRADINGAGENTS_DEEP_THINK_LLM=gpt-5.5
    TRADINGAGENTS_QUICK_THINK_LLM=gpt-5.4-mini

선택 사항:
    FRED_API_KEY=...   # 거시지표(금리·물가) 분석. 없으면 해당 부분만 건너뜀.
                       # 무료 발급: https://fred.stlouisfed.org/docs/api/api_key.html

주의: 본 프레임워크는 연구용입니다. 투자/매매 자문이 아닙니다.
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

# tradingagents 를 import 하면 패키지 __init__ 이 python-dotenv 로 현재
# 작업 디렉터리의 .env 를 자동 로드한다 (override=False). 그래서 아래
# import 시점 이후로 os.getenv 는 .env 값을 본다.
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.api_key_env import get_api_key_env

# ── 나만의 회사 기본값 ──────────────────────────────────────────────
DEFAULT_TICKER = "005930.KS"   # 삼성전자
SELECTED_ANALYSTS = ("market", "social", "news", "fundamentals")

# 처음 실행 시 생성해 줄 .env 템플릿. 키만 채우면 된다.
_ENV_TEMPLATE = """\
# ── 나만의 트레이딩 회사 설정 ─────────────────────────────────
# 아래 줄의 <...> 를 실제 키로 바꾸고 저장한 뒤 `python my_company.py` 실행.

# 필수: Claude 사용 키 (기본 제공자)
ANTHROPIC_API_KEY=

# 선택: 거시지표(금리·물가) 분석용. 없으면 해당 분석만 건너뜁니다.
# 무료 발급 → https://fred.stlouisfed.org/docs/api/api_key.html
# FRED_API_KEY=

# 다른 LLM 제공자로 바꾸려면 주석을 풀고 채우세요 (예: OpenAI)
# OPENAI_API_KEY=
# TRADINGAGENTS_LLM_PROVIDER=openai
# TRADINGAGENTS_DEEP_THINK_LLM=gpt-5.5
# TRADINGAGENTS_QUICK_THINK_LLM=gpt-5.4-mini
"""


# 인증 실패로 보이는 예외를 식별하는 힌트 (제공자 SDK 종류와 무관하게 문자열로 판별).
_AUTH_ERROR_HINTS = (
    "authentication", "invalid api key", "invalid x-api-key", "invalid_api_key",
    "incorrect api key", "unauthorized", "401",
    "could not resolve authentication", "no api key",
)


def _looks_like_auth_error(exc: BaseException) -> bool:
    """예외가 API 키/인증 문제로 보이면 True (프리플라이트를 통과한 '틀린 키' 케이스)."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(hint in text for hint in _AUTH_ERROR_HINTS)


def _latest_business_day() -> str:
    """분석 기준일 기본값: 가장 최근 평일(주말이면 금요일로 back-off)."""
    d = date.today()
    while d.weekday() >= 5:      # 5=토, 6=일
        d -= timedelta(days=1)
    return d.isoformat()


def build_config() -> dict:
    """취향에 맞춘 설정. .env 의 TRADINGAGENTS_* 오버라이드는 그대로 존중된다."""
    config = DEFAULT_CONFIG.copy()

    # LLM: Anthropic Claude 를 기본 조합으로 세팅한다. 단, .env 에서
    # TRADINGAGENTS_LLM_PROVIDER 를 지정해 다른 제공자(OpenAI 등)로 바꾼
    # 경우에는 그 선택을 존중하고 건드리지 않는다.
    if not os.getenv("TRADINGAGENTS_LLM_PROVIDER"):
        config["llm_provider"] = "anthropic"
        config["deep_think_llm"] = "claude-opus-4-8"
        config["quick_think_llm"] = "claude-sonnet-5"
        config["anthropic_effort"] = "high"

    # 출력 언어: 한국어. 프레임워크는 이 값을 최종 리포트뿐 아니라
    # 애널리스트·강세/약세 리서처·리스크 토론(공격/중립/보수)·리서치
    # 매니저·트레이더·포트폴리오 매니저까지 전 단계에 적용하므로,
    # 파이프라인 산출물 전체가 한국어로 나온다.
    config["output_language"] = "Korean"

    # 공격형: 강세/약세 리서처 토론과 리스크팀(공격/중립/보수) 논쟁을
    # 각각 2라운드로 늘려 공격적 시나리오까지 충분히 압박·검증한다.
    config["max_debate_rounds"] = 2
    config["max_risk_discuss_rounds"] = 2

    # 크래시 나도 마지막 성공 단계부터 재개
    config["checkpoint_enabled"] = True

    return config


def _write_env_template() -> Path:
    """작업 디렉터리에 .env 템플릿을 만든다 (이미 있으면 건드리지 않음)."""
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        env_path.write_text(_ENV_TEMPLATE, encoding="utf-8")
    return env_path


def _prompt_and_save_key(env_var: str) -> bool:
    """대화형 터미널이면 키를 붙여넣게 받아 .env 에 저장한다. 저장 성공 시 True."""
    if not sys.stdin.isatty():
        return False
    import getpass

    try:
        key = getpass.getpass(f"  {env_var} 를 붙여넣고 Enter (건너뛰려면 그냥 Enter): ").strip()
    except (EOFError, KeyboardInterrupt):
        return False
    if not key:
        return False

    from dotenv import find_dotenv, set_key

    env_path = find_dotenv(usecwd=True) or str(Path.cwd() / ".env")
    Path(env_path).touch(exist_ok=True)
    set_key(env_path, env_var, key)
    os.environ[env_var] = key          # 현재 프로세스에도 즉시 반영
    print(f"  ✓ {env_var} 를 {env_path} 에 저장했습니다.\n")
    return True


def preflight(config: dict) -> None:
    """실행 전 점검: 필수 API 키 확인. 없으면 친절히 안내하고 종료.

    프레임워크는 키가 없어도 그래프를 조용히 만든 뒤 첫 API 호출에서야
    깊은 SDK 트레이스백으로 죽는다. 여기서 미리 막아 명확히 안내한다.
    """
    provider = config["llm_provider"]

    # 거시지표(FRED)는 선택 사항 — 없으면 해당 분석만 건너뛴다(치명적 아님).
    if not os.getenv("FRED_API_KEY"):
        print("ℹ️  FRED_API_KEY 미설정 — 거시지표(금리·물가) 분석은 건너뜁니다.")
        print("    무료 키: https://fred.stlouisfed.org/docs/api/api_key.html\n")

    env_var = get_api_key_env(provider)
    if env_var is None:
        return  # ollama 등 키가 필요 없는 제공자
    if os.getenv(env_var):
        return  # 키 있음 → 통과

    # 여기부터는 필수 키가 없는 경우.
    env_path = _write_env_template()
    print("=" * 64)
    print(f"⚠  {provider} 실행에 필요한 {env_var} 가 설정되지 않았습니다.")
    print("=" * 64)

    if _prompt_and_save_key(env_var):
        return  # 방금 입력받아 저장 → 계속 진행

    print(f"\n  {env_path} 파일을 열어 아래 줄을 채우고 다시 실행하세요:\n")
    print(f"      {env_var}=<당신의_키>\n")
    print("  그런 다음:  python my_company.py")
    print("=" * 64)
    sys.exit(1)


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TICKER
    trade_date = sys.argv[2] if len(sys.argv) > 2 else _latest_business_day()

    config = build_config()
    preflight(config)   # 키 없으면 여기서 명확히 안내하고 종료

    print(f"▶ 회사 가동: {ticker} / 기준일 {trade_date} "
          f"/ {config['llm_provider']}:{config['deep_think_llm']} / 성향=공격형\n")

    ta = TradingAgentsGraph(
        selected_analysts=SELECTED_ANALYSTS,
        debug=True,
        config=config,
    )

    try:
        _, decision = ta.propagate(ticker, trade_date)
    except Exception as exc:  # noqa: BLE001 — 인증 실패만 친절히 안내, 그 외는 그대로 전파
        if _looks_like_auth_error(exc):
            env_var = get_api_key_env(config["llm_provider"]) or "API 키"
            print(f"\n⚠  {config['llm_provider']} 인증에 실패했습니다 — {env_var} 값이 올바른지 확인하세요.")
            print(f"   .env 의 {env_var} 를 유효한 키로 바꾼 뒤 다시 실행: python my_company.py")
            sys.exit(1)
        raise

    print("\n===== 최종 결정 =====")
    print(decision)


if __name__ == "__main__":
    main()
