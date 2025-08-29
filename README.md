1. Business Understanding（業務理解）

飯店營運需要快速回答「下週房價多少？」這類問題，背後牽涉到：

下週入住率預測（活動/季節影響）

依入住率與競品價格做動態定價

最終以專業語氣回覆客人（雙語）

目標：把上述三段工作自動化，縮短回覆時間並提升一致性。

KPI（可延伸）：回覆時間、價格一致性、轉換率、人工節省時數。

2. Data Understanding（資料理解）

主要資料：sample_data/occupancy_history.csv

欄位：date, occupancy_pct

輔助參數：Lookback 天數、活動/季節加成（%）、競品價上下界（USD）

可擴充：多房型、多渠道、競品即時爬取、事件行事曆等。

3. Data Preparation（資料準備）

data_utils.py

load_occupancy_csv(path)：讀檔、日期處理、排序

simple_occupancy_forecast(df, lookback_days, boost)：近 N 天平均 + boost

dynamic_price_suggestion(occupancy, comp_min, comp_max)：價格區間 + 定位 + 理由

先用可解釋的簡易規則起步，之後可替換 Prophet / sktime / XGBoost。

4. Modeling（建模與流程）
4.1 Multi-Agent（CrewAI）

Customer Service Agent：理解需求、補齊資訊

Forecast Analyst：估下週入住率 + 理由

Pricing Analyst：依入住率 × 競品 → 建議區間與定位

🗣Response Agent：產出繁中 + English 回覆

Pipeline
graph TD
    T1[理解需求\n(Customer)] --> T2[估入住率\n(Forecast)]
    T2 --> T3[動態定價\n(Pricing)]
    T3 --> T4[最終回覆（中＋EN）\n(Response)]

4.2 系統架構
Hotel_Crew_AI_RoadMap/
├── crew_core.py        # Agents + Tasks + run_crew()
├── data_utils.py       # CSV / 預測 / 定價
├── streamlit_app.py    # Streamlit UI
├── sample_data/        # 範例資料
├── .env                # OPENAI_API_KEY=...
└── requirements.txt

graph LR
    U[User] --> UI[Streamlit App]
    UI --> DU[data_utils.py]
    UI --> CC[crew_core.py]
    CC --> OA[(OpenAI API)]
    DU --> UI
    CC --> UI

5. Evaluation（評估）

Baseline 檢查：

入住率估計值是否落在近 N 天平均 ± 合理範圍

價格區間是否落在競品帶附近（定位：略低/持平/略高）

回覆是否清楚、專業、行動呼籲（e.g., 盡早預訂 / 升等建議）

未來評估（可量化）：

回覆時間（秒）

轉換率 / 平均房價（ADR）提升

人工處理時數下降

6. Deployment（部署）
6.1 快速啟動
# 1) 安裝環境
python -m venv venv
source venv/bin/activate        # Windows 用 venv\Scripts\activate
pip install -r requirements.txt

# 2) 設定金鑰
echo "OPENAI_API_KEY=sk-xxxx" > .env

# 3) 跑起來
streamlit run streamlit_app.py

6.2 Git（可選）
printf "__pycache__/\nvenv/\n.env\n*.pyc\n.streamlit/\n" > .gitignore
git init && git add -A && git commit -m "init"
git branch -M main
git remote add origin https://github.com/<you>/Hotel_Crew_AI_RoadMap.git
git push -u origin main

7. UI 使用說明

左側 Sidebar 上傳 occupancy_history.csv 或使用 sample

調整參數：Lookback / 活動加成（%）/ 競品價上下界

看到入住率快覽與參考動態房價

輸入客人問題（例：下週雙人房多少錢？想要面山景）→ 點 🚀 產生回覆（CrewAI）

取得繁中 + 英文客服回覆（帶入住率/定價依據）

8. Roadmap（未來工作）

 更強的入住率模型（Prophet / sktime / XGBoost）

 多房型（標準/豪華/套房）同時輸出價格帶

 Upsell 建議（早餐、延退、升等）

 對話紀錄存檔（SQLite / Postgres）

 接 Slack / Notion webhook

 部署到 Streamlit Cloud / HF Spaces

9. 範例輸出（示意）

中文
親愛的客人您好… 建議價格區間為 USD 130–160。此建議基於當前入住率約 81.6%…

English
Dear Guest… Our suggested price range is USD 130–160 based on the current occupancy (~81.6%). …

10. 環境變數

.env

OPENAI_API_KEY=sk-xxxxx
OPENAI_MODEL=gpt-4o-mini  # 可選

11. 主要程式入口
# streamlit_app.py（節選）
from data_utils import load_occupancy_csv, simple_occupancy_forecast, dynamic_price_suggestion
from crew_core import run_crew

# …載入 CSV 與參數 → Quick Stats → 產生 facts →
out = run_crew(user_q, facts)   # 回傳最終文字

12. 授權與鳴謝

Code: MIT（可自訂）

感謝：OpenAI / Streamlit / CrewAI 社群
