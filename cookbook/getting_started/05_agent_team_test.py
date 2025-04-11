"""🗞️ 多代理团队 - 您的专业新闻和金融团队！

此示例展示了如何创建一个强大的AI代理团队，共同合作提供全面的金融分析和新闻报道。
团队由以下成员组成：
1. 网络代理：搜索并分析最新新闻
2. 金融代理：分析金融数据和市场趋势
3. 主编：协调并结合两个代理的见解

运行：`pip install openai duckduckgo-search yfinance agno` 安装依赖
"""

from textwrap import dedent

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'libs', 'agno')))

from agno.agent import Agent
from agno.models.openai import QWQChat
from agno.team.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

web_agent = Agent(
    name="网络代理",
    role="搜索网络获取信息",
    model=QWQChat(),
    tools=[DuckDuckGoTools()],
    instructions=dedent("""\
        您是一位经验丰富的网络研究员和新闻分析师！ 🔍

        搜索信息时请遵循以下步骤：
        1. 从最近和最相关的来源开始
        2. 从多个来源交叉参考信息
        3. 优先考虑有声望的新闻机构和官方来源
        4. 始终引用带有链接的来源
        5. 专注于影响市场的新闻和重大发展

        您的风格指南：
        - 以清晰的新闻风格呈现信息
        - 使用项目符号列出关键要点
        - 在可用时包含相关引述
        - 为每条新闻指定日期和时间
        - 强调市场情绪和行业趋势
        - 以对整体叙述的简要分析结束
        - 特别关注监管新闻、收益报告和战略公告\
    """),
    show_tool_calls=True,
    markdown=True,
)

finance_agent = Agent(
    name="金融代理",
    role="获取金融数据",
    model=QWQChat(),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)
    ],
    instructions=dedent("""\
        您是一位熟练的金融分析师，专长于市场数据！ 📊

        分析金融数据时请遵循以下步骤：
        1. 从最新股价、交易量和日内范围开始
        2. 呈现详细的分析师建议和共识目标价格
        3. 包括关键指标：市盈率、市值、52周范围
        4. 分析交易模式和交易量趋势
        5. 与相关行业指数比较表现

        您的风格指南：
        - 使用表格呈现结构化数据
        - 为每个数据部分添加清晰的标题
        - 为技术术语添加简短解释
        - 用表情符号突出显著变化(📈 📉)
        - 使用项目符号快速洞见
        - 将当前值与历史平均值进行比较
        - 以数据驱动的金融展望结束\
    """),
    show_tool_calls=True,
    markdown=True,
)

agent_team = Team(
    members=[web_agent, finance_agent],
    model=QWQChat(),
    mode="coordinate",
    success_criteria=dedent("""\
        一份具有清晰章节和数据驱动洞见的全面金融新闻报告。
    """),
    instructions=dedent("""\
        您是一家著名金融新闻台的主编！ 📰

        您的角色：
        1. 协调网络研究员和金融分析师之间的工作
        2. 将他们的发现整合成引人入胜的叙述
        3. 确保所有信息都有适当的来源和验证
        4. 对新闻和数据提供平衡的视角
        5. 突出关键风险和机会

        您的风格指南：
        - 以引人注目的标题开始
        - 以有力的执行摘要开始
        - 先呈现金融数据，然后是新闻背景
        - 在不同类型的信息之间使用清晰的分隔
        - 在可用时包含相关图表或表格
        - 添加带有当前情绪的'市场情绪'部分
        - 在结尾包含'关键要点'部分
        - 在适当时以'风险因素'结束
        - 以'市场观察团队'和当前日期签名\
    """),
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    markdown=True,
    enable_agentic_context=True,
    show_members_responses=False,
)

# 使用多样化查询的示例用法
# agent_team.print_response(
#     message="总结分析师建议并分享NVDA的最新新闻",
#     stream=True,
# )
# agent_team.print_response(
#     message="AI半导体公司的市场前景和财务表现如何？",
#     stream=True,
# )
# agent_team.print_response(
#     message="分析TSLA的最新发展和财务表现",
#     stream=True,
# )

agent_team.print_response(
    message="分析宁德时代的最新发展和财务表现",
    stream=True,
)

# 更多要尝试的示例提示：
"""
要探索的高级查询：
1. "比较主要云提供商（AMZN、MSFT、GOOGL）的财务表现和最新新闻"
2. "最近美联储决策对银行股的影响如何？关注JPM和BAC"
3. "通过ATVI、EA和TTWO的表现分析游戏行业前景"
4. "社交媒体公司表现如何？比较META和SNAP"
5. "关于AI芯片制造商及其市场地位的最新情况是什么？"
"""
