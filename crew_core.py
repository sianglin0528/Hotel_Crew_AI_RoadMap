import os
from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew

from data_utils import (
    load_occupancy_csv,
    fit_prophet_and_forecast, summarize_forecast,
    train_xgb_pricing_model, infer_price_range
)

# --------- 共用 LLM ---------
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

# --------- 定義 Agents ---------
customer_agent = Agent(
    role="Front Desk",
    goal="理解客人需求，轉交後台分析",
    backstory="你是專業櫃檯，熟悉飯店QA與流程",
    llm=MODEL
)

forecast_agent = Agent(
    role="Forecast Analyst",
    goal="用提供的真實預測數字，產出清楚的入住率摘要（中英各一段）",
    backstory="你是收益管理分析師，擅長把數字說人話",
    llm=MODEL
)

pricing_agent = Agent(
    role="Pricing Analyst",
    goal="用真實的建議價區間，寫出定價邏輯與風險緩衝",
    backstory="你是定價專家，懂競品、需求壓力與定位",
    llm=MODEL
)

response_agent = Agent(
    role="Bilingual Concierge",
    goal="整合結果，輸出一段中文+一段英文的對客回覆",
    backstory="你是多語客服，回覆專業簡潔、可直接貼給客人",
    llm=MODEL
)


# --------- 主流程 ---------
def run_crew(
    user_question: str,
    csv_path: str = "sample_data/occupancy_history.csv",
    comp_min: float = 120.0,
    comp_max: float = 180.0,
    event_boost: float = 0.0
):
    # 1) 真數字計算
    hist = load_occupancy_csv(csv_path)

    try:
        fcst = fit_prophet_and_forecast(hist, periods=7)
        fsum = summarize_forecast(fcst)
        avg_occ = round(fsum["avg_occ"], 1)
        forecast_window = f"{fsum['start']} ~ {fsum['end']}"
        min_occ = round(fsum["min_occ"], 1)
        max_occ = round(fsum["max_occ"], 1)
        occ_src = "Prophet"
    except Exception:
        # 最簡 fallback：近 14 天均值
        avg_occ = float(hist.sort_values("date").tail(14)["occupancy_pct"].mean())
        forecast_window = f"{hist['date'].max().date().isoformat()} ~ +7d"
        min_occ = max_occ = round(avg_occ, 1)
        occ_src = "Fallback(14天均值)"

    xgb_tuple = train_xgb_pricing_model(hist)  # CSV 有 price 才會生效
    price = infer_price_range(xgb_tuple, avg_occ, comp_min, comp_max)

    facts = {
        "question": user_question,
        "forecast_window": forecast_window,
        "avg_occ": round(avg_occ, 1),
        "min_occ": min_occ,
        "max_occ": max_occ,
        "occ_source": occ_src,
        "comp_min": comp_min,
        "comp_max": comp_max,
        "price_mid": round(price["price_mid"], 1),
        "price_lo": round(price["lo"], 1),
        "price_hi": round(price["hi"], 1),
        "pricing_basis": price["basis"],
    }

    # 2) 任務定義（全部加 expected_output）
    task1 = Task(
        description=(
            f"客人問題：{facts['question']}\n"
            "請用一兩句話重述客戶需求，準備轉交後台分析。"
        ),
        agent=customer_agent,
        expected_output="兩句以內的中文摘要（可附英文一句）。"
    )

    task2 = Task(
        description=(
            "請用以下真實預測資料，產出入住率摘要（中文+English）：\n"
            f"- 期間：{facts['forecast_window']}\n"
            f"- 平均入住率：{facts['avg_occ']}%\n"
            f"- 區間：{facts['min_occ']}% ~ {facts['max_occ']}%\n"
            f"- 來源：{facts['occ_source']}\n"
            "先中文一段、再 English 一段。"
        ),
        agent=forecast_agent,
        depends_on=[task1],
        expected_output="兩段文字：第一段中文摘要，第二段 English 摘要。"
    )

    task3 = Task(
        description=(
            "根據真實資料產出定價說明（條列）：\n"
            f"- 競品價：USD {facts['comp_min']} ~ {facts['comp_max']}\n"
            f"- 建議價區間：USD {facts['price_lo']} ~ {facts['price_hi']}（中位 {facts['price_mid']}）\n"
            f"- 方法：{facts['pricing_basis']}（xgboost 代表真模型，否則為 rule）\n"
            "- 請條列 3–5 點為什麼這樣定（需求壓力、競品定位、風險緩衝、敏感度）。"
        ),
        agent=pricing_agent,
        depends_on=[task2],
        expected_output="條列清單（3-5 點），最後一行再次標示區間與中位價（USD）。"
    )

    task4 = Task(
        description=(
            "整合上面內容，輸出對客回覆（中文 + 英文），包含：\n"
            f"1) 預測摘要（平均 {facts['avg_occ']}%，期間 {facts['forecast_window']}）\n"
            f"2) 建議價區間（USD {facts['price_lo']} ~ {facts['price_hi']}，中位 {facts['price_mid']}）\n"
            "3) 下一步需要的資訊（入住日期/人數/特殊需求）。語氣專業、簡潔。"
        ),
        agent=response_agent,
        depends_on=[task3],
        expected_output="兩段：第一段中文正式回覆，第二段 English 正式 reply。"
    )

    crew = Crew(
        agents=[customer_agent, forecast_agent, pricing_agent, response_agent],
        tasks=[task1, task2, task3, task4],
        verbose=False
    )

    final_output = crew.kickoff()
    return {
        "facts": facts,
        "final": final_output.raw if hasattr(final_output, "raw") else str(final_output)
    }

# --------- 測試入口 ---------
if __name__ == "__main__":
    out = run_crew(
        user_question="下週雙人房多少？",
        csv_path="sample_data/occupancy_history.csv",
        comp_min=120.0,
        comp_max=180.0,
        event_boost=0.0
    )
    print("\n=== FACTS ===")
    for k, v in out["facts"].items():
        print(f"{k}: {v}")
    print("\n=== FINAL ===\n")
    print(out["final"])
