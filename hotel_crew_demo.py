from crewai import Agent, Task, Crew
import os
from dotenv import load_dotenv

# 0) 載入 .env（需要 OPENAI_API_KEY）
load_dotenv()

# 1) 定義 Agents
customer_agent = Agent(
    role="Customer Service",
    goal="接收客人的需求，轉交給適合的 agent",
    backstory="你是飯店櫃檯人員，熟悉客房相關問題",
    llm="gpt-5",
    verbose=True,
)

forecast_agent = Agent(
    role="Forecast Analyst",
    goal="預測下週的住房率",
    backstory="你是飯店的收益管理師，熟悉 occupancy 預測",
    llm="gpt-5",
    verbose=True,
)

pricing_agent = Agent(
    role="Pricing Analyst",
    goal="根據住房率與競爭者價格，建議房價",
    backstory="你是飯店收益分析師，專門計算最適價格",
    llm="gpt-5",
    verbose=True,
)

response_agent = Agent(
    role="Response Agent",
    goal="用簡單的方式回答客人，並附加建議",
    backstory="你是飯店的客服專員，熟悉多語言溝通",
    llm="gpt-5",
    verbose=True,
)

# 2) 定義 Tasks（★ 每個 Task 都要有 expected_output）
task1 = Task(
    description="客人問：下週雙人房多少錢？請理解需求並整理成內部工作項目。",
    agent=customer_agent,
    expected_output=(
        "以簡短條列輸出：\n"
        "1) 客人問題重點\n"
        "2) 需要哪個分析（住房率、競品價）\n"
        "3) 需要交給哪個 agent"
    ),
)

task2 = Task(
    description="根據飯店情境，估算『下週住房率』並給出理由（季節性、活動、歷史趨勢）。",
    agent=forecast_agent,
    depends_on=[task1],
    expected_output=(
        "JSON 風格文字（非真正 JSON 也可）包含：\n"
        "- occupancy_rate_pct: 估計百分比\n"
        "- rationale: 估計依據（1-3 條）"
    ),
)

task3 = Task(
    description="使用 task2 的住房率，並假設競品雙人房價介於 120~180 USD，提出我們的動態房價建議。",
    agent=pricing_agent,
    depends_on=[task2],
    expected_output=(
        "條列輸出：\n"
        "1) 建議房價區間（USD）\n"
        "2) 與競品的定位（略低/持平/略高）\n"
        "3) 價格策略理由（含需求、入住率、活動）"
    ),
)

task4 = Task(
    description="整合前面任務，產出給客人的最終回覆（繁體中文、口吻專業友善）。",
    agent=response_agent,
    depends_on=[task3],
    expected_output=(
        "一段對客人的回覆：\n"
        "- 下週雙人房建議價格區間（USD）\n"
        "- 簡要理由（1-2 句）\n"
        "- 可行的下一步（是否要先預留、升等方案等）"
    ),
)

# 3) 建立 Crew
crew = Crew(
    agents=[customer_agent, forecast_agent, pricing_agent, response_agent],
    tasks=[task1, task2, task3, task4],
    verbose=True,
)

# 4) 啟動 Crew
result = crew.kickoff()
print("\n=== 最終回答 ===\n")
print(result)

