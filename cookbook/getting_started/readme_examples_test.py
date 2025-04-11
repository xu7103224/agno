"""Readme Examples
运行 `pip install openai duckduckgo-search yfinance lancedb tantivy pypdf agno` 来安装依赖项。"""

from agno.agent import Agent
from agno.embedder.openai import BaiLianEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.models.openai import QWQChat
from agno.team.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.vectordb.lancedb import LanceDb, SearchType

# 第零级：没有工具的智能体（基本推理任务）。
level_0_agent = Agent(
    model=QWQChat(),
    description="你是一位充满热情的新闻记者，擅长讲故事！",
    markdown=True,
)
level_0_agent.print_response(
    "告诉我一个来自纽约的突发新闻故事。", stream=True
)

# 第一级：具有工具的智能体，用于自主任务执行。
level_1_agent = Agent(
    model=QWQChat(),
    description="你是一位充满热情的新闻记者，擅长讲故事！",
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
)
level_1_agent.print_response(
    "告诉我一个来自纽约的突发新闻故事。", stream=True
)

# 第二级：具有知识的智能体，结合记忆和推理。
level_2_agent = Agent(
    model=QWQChat(),
    description="你是一位泰国美食专家！",
    instructions=[
        "搜索你的知识库中的泰国菜谱。",
        "如果问题更适合通过网络回答，请搜索网络来填补知识空白。",
        "优先使用知识库中的信息，而非网络搜索结果。",
    ],
    knowledge=PDFUrlKnowledgeBase(
        urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
        vector_db=LanceDb(
            uri="tmp/lancedb",
            table_name="recipes",
            search_type=SearchType.hybrid,
            embedder=BaiLianEmbedder(id="text-embedding-v3"),
        ),
    ),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
)

# 首次运行后注释掉
# if level_2_agent.knowledge is not None:
#     level_2_agent.knowledge.load()
level_2_agent.print_response(
    "如何制作椰奶鸡肉南姜汤", stream=True
)
level_2_agent.print_response("泰国咖喱的历史是什么？", stream=True)

# 第三级：多个智能体组成的团队协作处理复杂工作流程。
web_agent = Agent(
    name="网络智能体",
    role="搜索网络获取信息",
    model=QWQChat(),
    tools=[DuckDuckGoTools()],
    instructions="始终包含信息来源",
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="金融智能体",
    role="获取金融数据",
    model=QWQChat(),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)
    ],
    instructions="使用表格展示数据",
    show_tool_calls=True,
    markdown=True,
)

level_3_agent_team = Team(
    members=[web_agent, finance_agent],
    model=QWQChat(),
    mode="协调",
    success_criteria="一份全面的金融新闻报告，包含清晰的章节和数据驱动的洞察。",
    instructions=["始终包含信息来源", "使用表格展示数据"],
    show_tool_calls=True,
    markdown=True,
)
level_3_agent_team.print_response(
    "AI半导体公司的市场前景和财务表现如何？",
    stream=True,
)
