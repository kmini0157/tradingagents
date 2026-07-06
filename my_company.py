"""나만의 트레이딩 회사(Trading Firm) 실행 스크립트.

TradingAgents 프레임워크를, 아래 취향에 맞춰 미리 세팅한 진입점입니다.

  - 대상 시장 : 한국 종목 (예: 삼성전자 005930.KS)
  - LLM       : Anthropic Claude (deep=Opus 4.8, quick=Sonnet 5, effort=high)
                └ .env 에서 TRADINGAGENTS_* 한 줄만 바꾸면 OpenAI 등으로 교체 가능
  - 리포트    : 한국어
  - 성향      : 공격형(aggressive) — 리스크/리서치 토론 라운드를 늘려
                공격적 시나리오까지 충분히 논쟁하도록 구성

사용법
------
1) 의존성 설치:   pip install -e .
2) API 키 준비 :   프로젝트 루트에 .env 파일을 만들고 아래를 채웁니다.

       ANTHROPIC_API_KEY=sk-ant-...            # 필수 (Claude 사용 시)
       # FRED_API_KEY=...                      # 선택: 거시지표(금리·물가) 데이터

   OpenAI 로 바꾸려면 키와 함께 .env 에 다음을 추가:

       OPENAI_API_KEY=sk-...
       TRADINGAGENTS_LLM_PROVIDER=openai
       TRADINGAGENTS_DEEP_THINK_LLM=gpt-5.5
       TRADINGAGENTS_QUICK_THINK_LLM=gpt-5.4-mini

3) 실행:
       python my_company.py                    # 기본: 삼성전자, 최근 거래일
       python my_company.py 000660.KS          # SK하이닉스
       python my_company.py 035420.KS 2026-06-30  # 종목 + 분석 기준일 지정

주의: 본 프레임워크는 연구용입니다. 투자/매매 자문이 아닙니다.
"""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

# ── 나만의 회사 기본값 ──────────────────────────────────────────────
DEFAULT_TICKER = "005930.KS"   # 삼성전자


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

    # 리포트는 한국어로 (에이전트 내부 토론은 추론 품질을 위해 영어 유지)
    config["output_language"] = "Korean"

    # 공격형: 강세/약세 리서처 토론과 리스크팀(공격/중립/보수) 논쟁을
    # 각각 2라운드로 늘려 공격적 시나리오까지 충분히 압박·검증한다.
    config["max_debate_rounds"] = 2
    config["max_risk_discuss_rounds"] = 2

    # 크래시 나도 마지막 성공 단계부터 재개
    config["checkpoint_enabled"] = True

    return config


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TICKER
    trade_date = sys.argv[2] if len(sys.argv) > 2 else _latest_business_day()

    config = build_config()
    print(f"▶ 회사 가동: {ticker} / 기준일 {trade_date} "
          f"/ {config['llm_provider']}:{config['deep_think_llm']} / 성향=공격형")

    ta = TradingAgentsGraph(
        selected_analysts=("market", "social", "news", "fundamentals"),
        debug=True,
        config=config,
    )

    _, decision = ta.propagate(ticker, trade_date)
    print("\n===== 최종 결정 =====")
    print(decision)


if __name__ == "__main__":
    main()
