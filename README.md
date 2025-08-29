# Hotel Crew AI Assistant — CrewAI × Streamlit

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![CrewAI](https://img.shields.io/badge/CrewAI-MultiAgent-green)](https://www.crewai.com/)
[![License](https://img.shields.io/badge/License-MIT-black)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/sianglin0528/Hotel_Crew_AI_RoadMap?style=social)](https://github.com/sianglin0528/Hotel_Crew_AI_RoadMap)

---

一個結合 **CrewAI 多智能體協作** 與 **Streamlit 前端互動** 的專案，用於模擬飯店收益管理場景：  
- 入住率預測 (Prophet / XGBoost / Rule-based baseline)  
- 動態定價建議 (競品價格 + 預測入住率)  
- AI Concierge：多代理人回答客戶詢問  
- 雲端部署 (Streamlit Cloud / GitHub Actions)  

---

> 多代理人（CrewAI）把**入住率預測 → 動態定價 → 客服回覆**串成一條龍。  
> 前端用 Streamlit，後端以 Python + OpenAI 模型。  
> 本 README 以 **CRISP-DM** 方法論梳理整個專案。

---

## 0. TL;DR

- **Agents**：Customer / Forecast / Pricing / Response  
- **Data**：`occupancy_history.csv (date, occupancy_pct)`  
- `streamlit run streamlit_app.py` → 輸入客人問題 → 產生**繁中 + 英文**回覆  
- 方法論：**CRISP-DM**（Business → Data → Prep → Modeling → Evaluation → Deployment）

---

## 1. Business Understanding（業務理解）

飯店營運需要快速回答「**下週房價多少？**」這類問題，背後牽涉到：
- 下週**入住率**預測（活動/季節影響）
- 依入住率與**競品價格**做**動態定價**
- 最終以**專業語氣**回覆客人（雙語）

**目標**：把上述三段工作自動化，縮短回覆時間並提升一致性。

**KPI**（可延伸）：回覆時間、價格一致性、轉換率、人工節省時數。

---

## 2. Data Understanding（資料理解）

- 主要資料：`sample_data/occupancy_history.csv`
  - 欄位：`date`, `occupancy_pct`
- 輔助參數：Lookback 天數、活動/季節加成（%）、競品價上下界（USD）

> 可擴充：多房型、多渠道、競品即時爬取、事件行事曆等。

---

## 3. Data Preparation（資料準備）

- `data_utils.py`
  - `load_occupancy_csv(path)`：讀檔、日期處理、排序
  - `simple_occupancy_forecast(df, lookback_days, boost)`：近 N 天平均 + boost
  - `dynamic_price_suggestion(occupancy, comp_min, comp_max)`：價格區間 + 定位 + 理由

> 先用可解釋的簡易規則起步，之後可替換 Prophet / sktime / XGBoost。

---

## 4. Modeling（建模與流程）

### 4.1 Multi-Agent（CrewAI）

- **Customer Service Agent**：理解需求、補齊資訊  
- **Forecast Analyst**：估下週入住率 + 理由  
- **Pricing Analyst**：依入住率 × 競品 → 建議區間與定位  
- **Response Agent**：產出**繁中 + English** 回覆

### 4.2 系統架構
```
Hotel_Crew_AI_RoadMap/
├── crew_core.py        # Agents + Tasks + run_crew()
├── data_utils.py       # CSV / 預測 / 定價
├── streamlit_app.py    # Streamlit UI
├── sample_data/        # 範例資料
├── .env                # OPENAI_API_KEY=...
└── requirements.txt
```

```mermaid
graph LR
    U[User] --> UI[Streamlit App]
    UI --> DU[data_utils.py]
    UI --> CC[crew_core.py]
    CC --> OA[(OpenAI API)]
    DU --> UI
    CC --> UI
```

---

## 5. Evaluation（評估）

Baseline 檢查：
- 入住率估計值是否落在近 N 天平均 ± 合理範圍
- 價格區間是否落在競品帶附近（定位：略低/持平/略高）
- 回覆是否**清楚、專業、行動呼籲**（e.g., 盡早預訂 / 升等建議）

未來評估（可量化）：
- 回覆時間（秒）
- 轉換率 / 平均房價（ADR）提升
- 人工處理時數下降

---

## 6. Deployment（部署）

### 6.1 快速啟動
```bash
# 1) 安裝環境
python -m venv venv
source venv/bin/activate        # Windows 用 venv\Scripts\activate
pip install -r requirements.txt

# 2) 設定金鑰
echo "OPENAI_API_KEY=sk-xxxx" > .env

# 3) 跑起來
streamlit run streamlit_app.py
```

### 6.2 Git（可選）
```bash
printf "__pycache__/\nvenv/\n.env\n*.pyc\n.streamlit/\n" > .gitignore
git init && git add -A && git commit -m "init"
git branch -M main
git remote add origin https://github.com/<you>/Hotel_Crew_AI_RoadMap.git
git push -u origin main
```

---

## 7. UI 使用說明

1. 左側 Sidebar 上傳 `occupancy_history.csv` 或使用 sample  
2. 調整參數：Lookback / 活動加成（%）/ 競品價上下界  
3. 看到**入住率快覽**與**參考動態房價**  
4. 輸入客人問題（例：*下週雙人房多少錢？想要面山景*）→ 點 **🚀 產生回覆（CrewAI）**  
5. 取得**繁中 + 英文**客服回覆（帶入住率/定價依據）

---

## 8. Roadmap（未來工作）

- [ ] 更強的入住率模型（Prophet / sktime / XGBoost）  
- [ ] 多房型（標準/豪華/套房）同時輸出價格帶  
- [ ] Upsell 建議（早餐、延退、升等）  
- [ ] 對話紀錄存檔（SQLite / Postgres）  
- [ ] 接 Slack / Notion webhook  
- [ ] 部署到 Streamlit Cloud / HF Spaces

---

## 9. 範例輸出（示意）

> **中文**  
> 親愛的客人您好… 建議價格區間為 **USD 130–160**。此建議基於當前入住率約 **81.6%**…  
>
> **English**  
> Dear Guest… Our suggested price range is **USD 130–160** based on the current occupancy (~**81.6%**). …

---

## 10. 環境變數

`.env`
```
OPENAI_API_KEY=sk-xxxxx
OPENAI_MODEL=gpt-4o-mini  # 可選
```

---

## 11. 主要程式入口

```python
# streamlit_app.py（節選）
from data_utils import load_occupancy_csv, simple_occupancy_forecast, dynamic_price_suggestion
from crew_core import run_crew

# …載入 CSV 與參數 → Quick Stats → 產生 facts →
out = run_crew(user_q, facts)   # 回傳最終文字
```

