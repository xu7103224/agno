"""🔍 AI研究代理 - 您的AI研究助手！

此示例展示了如何通过结合exa的搜索功能和学术写作技能创建高级研究代理，
以提供结构良好、基于事实的报告。

展示的主要功能：
- 使用Exa.ai进行学术和新闻搜索
- 带引用的结构化报告生成
- 自定义格式化和文件保存功能

要尝试的示例提示：
- "量子计算的最新发展是什么？"
- "研究人工意识的当前状态"
- "分析聚变能源的近期突破"
- "调查太空旅游对环境的影响"
- "探索长寿研究的最新发现"

运行 `pip install openai exa-py agno` 安装依赖。
"""

from datetime import datetime
from pathlib import Path
from textwrap import dedent

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'libs', 'agno')))

from agno.agent import Agent
from agno.models.openai import QWQChat
from agno.tools.exa import ExaTools

cwd = Path(__file__).parent.resolve()
tmp = cwd.joinpath("tmp")
if not tmp.exists():
    tmp.mkdir(exist_ok=True, parents=True)

today = datetime.now().strftime("%Y-%m-%d")

agent = Agent(
    model=QWQChat(),
    tools=[ExaTools(start_published_date=today, type="keyword")],
    description=dedent("""\
        您是Professor X-1000，一位杰出的AI研究科学家，擅长分析和综合复杂信息。
        您的专长在于创建引人入胜、基于事实的报告，结合学术严谨性和引人入胜的叙述。

        您的写作风格是：
        - 清晰权威
        - 引人入胜但专业
        - 以事实为中心，有适当引用
        - 对受过教育的非专业人士易于理解\
    """),
    instructions=dedent("""\
        从运行3个不同的搜索开始，以收集全面信息。
        分析并交叉参考来源的准确性和相关性。
        按照学术标准构建您的报告，但保持可读性。
        只包含可验证的事实，附带适当引用。
        创建引人入胜的叙述，引导读者了解复杂主题。
        以可行的要点和未来影响结束。\
    """),
    expected_output=dedent("""\
    一份以markdown格式呈现的专业研究报告：

    # {能够捕捉主题本质的引人注目标题}

    ## 执行摘要
    {关键发现和意义的简要概述}

    ## 引言
    {主题的背景和重要性}
    {研究/讨论的当前状态}

    ## 主要发现
    {重大发现或发展}
    {支持证据和分析}

    ## 影响
    {对领域/社会的影响}
    {未来方向}

    ## 关键要点
    - {要点1}
    - {要点2}
    - {要点3}

    ## 参考文献
    - [来源1](链接) - 关键发现/引用
    - [来源2](链接) - 关键发现/引用
    - [来源3](链接) - 关键发现/引用

    ---
    由Professor X-1000生成的报告
    高级研究系统部门
    日期：{current_date}\
    """),
    markdown=True,
    show_tool_calls=True,
    add_datetime_to_instructions=True,
    save_response_to_file=str(tmp.joinpath("{message}.md")),
)

# 使用示例
if __name__ == "__main__":
    # 生成关于前沿主题的研究报告
    agent.print_response(
        "研究脑机接口的最新发展", stream=True
    )

# 更多要尝试的示例提示：
"""
尝试这些研究主题：
1. "分析固态电池的当前状态"
2. "研究CRISPR基因编辑的最新突破"
3. "调查自动驾驶车辆的开发"
4. "探索量子机器学习的进展"
5. "研究人工智能对医疗保健的影响"
"""
