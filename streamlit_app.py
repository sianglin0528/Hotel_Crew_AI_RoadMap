import os
import io
import pandas as pd
import streamlit as st
from dotenv import load_dotenv


from data_utils import load_occupancy_csv, simple_occupancy_forecast, dynamic_price_suggestion
from crew_core import run_crew
from data_utils import fit_prophet_and_forecast, summarize_forecast, infer_price_range


load_dotenv()


st.set_page_config(page_title="Hotel Crew AI Assistant", page_icon="🛎️")
st.title("Hotel CrewAI Assistant — RMS")
st.caption("多代理人｜入住率 × 動態定價｜雙語回覆")


# --- Sidebar: Data/Options ---
st.sidebar.header("📂 資料與參數")


uploaded = st.sidebar.file_uploader("上傳入住率 CSV (date, occupancy_pct)", type=["csv"])

if uploaded is not None:
    data_bytes = uploaded.read()
    df = pd.read_csv(io.BytesIO(data_bytes))
else:
    # 使用內建 sample
    sample_path = os.path.join("sample_data", "occupancy_history.csv")
if os.path.exists(sample_path):
        df = load_occupancy_csv(sample_path)
        st.sidebar.info("使用範例資料 sample_data/occupancy_history.csv")
else:
        df = pd.DataFrame(columns=["date", "occupancy_pct"])  # 空表




lookback = st.sidebar.slider("預測 lookback 天數", min_value=7, max_value=28, value=14, step=1)
boost = st.sidebar.slider("活動/季節加成 (％)", min_value=-10.0, max_value=20.0, value=0.0, step=0.5)
comp_min = st.sidebar.number_input("競品價（低）USD", value=120.0, min_value=50.0, max_value=500.0, step=1.0)
comp_max = st.sidebar.number_input("競品價（高）USD", value=180.0, min_value=50.0, max_value=800.0, step=1.0)


st.sidebar.divider()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
 st.sidebar.error("找不到 OPENAI_API_KEY，請在 .env 設定。")


# --- Quick Stats ---
# --- Quick Stats ---
st.subheader("入住率快覽")

# 空資料就先跳出（不用 else，避免語法卡住）
if df.empty:
    st.warning("尚未載入入住率資料，請上傳 CSV 或使用範例資料。")
    st.stop()

# 1) 顯示最近 10 筆
st.dataframe(df.tail(10), use_container_width=True)

# 2) 入住率預測（支援 lookback / boost）
fcst, summary = simple_occupancy_forecast(
    df,
    lookback_days=lookback,
    boost=boost,
)
est = float(summary["avg_occ"])
rationale = f"{summary['start']} → {summary['end']} 平均 {est:.1f}%"

# 3) 動態定價（強韌解包：不管回傳是 dict / 二值 / 三值都能吃）
try:
    result = dynamic_price_suggestion(df, comp_min, comp_max)
except TypeError:
    # 若函式其實吃的是平均入住率，就用 est 再試一次
    result = dynamic_price_suggestion(est, comp_min, comp_max)

if isinstance(result, dict):
    low = float(result["lo"])
    high = float(result["hi"])
    price_reason = (
        f"basis={result.get('basis','rule')}, "
        f"avg_occ={result.get('avg_occ', est):.1f}%"
    )
elif isinstance(result, tuple):
    if len(result) == 2 and isinstance(result[0], tuple):
        (low, high), price_reason = result
        low, high = float(low), float(high)
    elif len(result) == 3 and isinstance(result[1], tuple):
        _, (low, high), price_reason = result
        low, high = float(low), float(high)
    else:
        raise ValueError(f"Unexpected return from dynamic_price_suggestion: {result!r}")
else:
    raise ValueError(f"Unexpected type from dynamic_price_suggestion: {type(result)}")

# 4) 顯示結果（⚠️ 行尾不要加反斜線）
st.info(f"估計下週入住率：{est:.1f}%｜依據：{rationale}")
st.info(f"參考動態房價：USD {low:.1f} – {high:.1f}｜依據：{price_reason}")


# Chat
st.subheader("客人問題 / 問句")
user_q = st.text_input("例：下週雙人房多少錢？想要面山景。")

run = st.button("產生回覆（CrewAI）", type="primary")

if run:
    if not user_q.strip():
        st.warning("請先輸入一個問題")
        st.stop()

    facts = {
        "est_occupancy_pct": est if not df.empty else None,
        "occ_rationale": rationale if not df.empty else None,
        "event_boost_pct": boost,
        "comp_min_usd": float(comp_min),
        "comp_max_usd": float(comp_max),
        "occ_hint": "資料由 UI 提供；如為空則交由模型估算。",
    }

    with st.spinner("Crew 正在協作中…"):
        out = run_crew(user_q, facts)

    st.success("完成！")



    # 假設你已經讀了 csv、拿到 comp_min/comp_max、fcst / facts / price_info
    st.markdown("### 📈 真模型摘要 (Prophet / XGBoost)")
    c1, c2, c3 = st.columns(3)
    c1.metric("平均入住率(下週)", f"{facts['avg_occ']}%")
    c2.metric("建議價中位 (USD)", f"{facts['price_mid']}")
    c3.metric("方法", "XGBoost" if facts["pricing_basis"]=="xgboost" else "Rule-based")

    st.caption(f"區間 {facts['forecast_window']}｜價區間 USD {facts['price_lo']} ~ {facts['price_hi']}"
           + ("" if facts["xgb_mae"] is None else f"｜XGB MAE ≈ {facts['xgb_mae']}"))


    st.markdown("### 🧾 模型回覆（Final Output）")
    st.write(out)
