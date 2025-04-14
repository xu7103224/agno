"""🎓 高级研究工作流 - 您的AI研究助手！

此示例展示了如何构建一个结合以下功能的复杂研究工作流：
🔍 网络搜索功能，用于查找相关来源
📚 内容提取和处理
✍️ 学术风格报告生成
💾 智能缓存以提高性能

我们使用了以下工具，因为它们是免费提供的：
- DuckDuckGoTools：搜索网络上的相关文章
- Newspaper4kTools：抓取和处理文章内容

要尝试的示例研究主题：
- "量子计算的最新发展是什么？"
- "研究人工意识的当前状态"
- "分析聚变能源的近期突破"
- "调查太空旅游对环境的影响"
- "探索长寿研究的最新发现"

运行 `pip install openai duckduckgo-search newspaper4k lxml_html_clean sqlalchemy agno` 安装依赖。
"""

import json
from textwrap import dedent
from typing import Dict, Iterator, Optional

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'libs', 'agno')))

from agno.agent import Agent
from agno.models.openai import QwenMaxChat20250125
from agno.storage.sqlite import SqliteStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.utils.log import logger
from agno.utils.pprint import pprint_run_response
from agno.workflow import RunEvent, RunResponse, Workflow
from pydantic import BaseModel, Field


class Article(BaseModel):
    title: str = Field(..., description="文章标题。")
    url: str = Field(..., description="文章链接。")
    summary: Optional[str] = Field(
        ..., description="文章摘要（如果有）。"
    )


class SearchResults(BaseModel):
    articles: list[Article]


class ScrapedArticle(BaseModel):
    title: str = Field(..., description="文章标题。")
    url: str = Field(..., description="文章链接。")
    summary: Optional[str] = Field(
        ..., description="文章摘要（如果有）。"
    )
    content: Optional[str] = Field(
        ...,
        description="以markdown格式的文章内容（如果有）。如果内容不可用或没有意义，则返回None。",
    )


class ResearchReportGenerator(Workflow):
    description: str = dedent("""\
    生成全面的研究报告，结合学术严谨性和引人入胜的讲故事能力。
    此工作流程协调多个AI代理来搜索、分析和综合来自多种来源的信息，
    形成结构良好的报告。
    """)

    web_searcher: Agent = Agent(
        model=QwenMaxChat20250125(),
        tools=[DuckDuckGoTools()],
        description=dedent("""\
        您是ResearchBot-X，发现和评估学术和科学来源的专家。\
        """),
        instructions=dedent("""\
        您是一位一丝不苟的研究助手，擅长源评估！ 🔍
        搜索10-15个来源，并确定5-7个最权威和相关的来源。
        优先考虑：
        - 经同行评审的文章和学术出版物
        - 来自信誉良好机构的最新发展
        - 权威新闻来源和专家评论
        - 来自公认专家的多样化观点
        避免观点文章和非权威来源。\
        """),
        response_model=SearchResults,
        stream=True,
        use_json_mode=True,
    )

    article_scraper: Agent = Agent(
        model=QwenMaxChat20250125(),
        tools=[Newspaper4kTools()],
        description=dedent("""\
        您是ContentBot-X，提取和构建学术内容的专家。\
        """),
        instructions=dedent("""\
        您是一位精确的内容策展人，注重学术细节！ 📚
        在处理内容时：
           - 从文章中提取内容
           - 保留学术引用和参考
           - 在术语上保持技术准确性
           - 以清晰的章节逻辑地构建内容
           - 提取关键发现和方法细节
           - 优雅地处理付费墙内容
        以清晰的markdown格式呈现一切，以获得最佳可读性。\
        """),
        response_model=ScrapedArticle,
        stream=True,
        use_json_mode=True,
    )

    writer: Agent = Agent(
        model=QwenMaxChat20250125(),
        description=dedent("""\
        您是Professor X-2000，一位杰出的AI研究科学家，结合学术严谨性和引人入胜的叙事风格。\
        """),
        instructions=dedent("""\
        发挥世界级学术研究者的专业知识！
        🎯 分析阶段：
          - 评估来源可信度和相关性
          - 交叉引用多个来源的发现
          - 识别关键主题和突破
        💡 综合阶段：
          - 开发连贯的叙述框架
          - 连接不同的发现
          - 突出矛盾或差距
        ✍️ 写作阶段：
          - 以引人入胜的执行摘要开始，吸引读者
          - 清晰地呈现复杂思想
          - 用引用支持所有观点
          - 平衡深度和可访问性
          - 保持学术风格的同时确保可读性
          - 以影响和未来方向结束\
        """),
        expected_output=dedent("""\
        # {引人注目的学术标题}

        ## 执行摘要
        {关键发现和意义的简洁概述}

        ## 引言
        {研究背景}
        {领域当前状态}

        ## 方法
        {搜索和分析方法}
        {来源评估标准}

        ## 主要发现
        {重大发现和发展}
        {支持证据和分析}
        {对比观点}

        ## 分析
        {对发现的批判性评估}
        {多种视角的整合}
        {模式和趋势的识别}

        ## 影响
        {学术和实践意义}
        {未来研究方向}
        {潜在应用}

        ## 关键要点
        - {关键发现1}
        - {关键发现2}
        - {关键发现3}

        ## 参考文献
        {适当格式的学术引用}

        ---
        由Professor X-2000生成的报告
        高级研究部门
        日期：{current_date}\
        """),
        markdown=True,
        stream=True,
        use_json_mode=True,
    )

    def run(
        self,
        topic: str,
        use_search_cache: bool = True,
        use_scrape_cache: bool = True,
        use_cached_report: bool = True,
    ) -> Iterator[RunResponse]:
        """
        生成关于给定主题的综合新闻报告。

        此函数协调工作流程来搜索文章、抓取其内容并生成最终报告。
        它利用缓存机制来优化性能。

        参数：
            topic (str)：要生成新闻报告的主题。
            use_search_cache (bool, 可选)：是否使用缓存的搜索结果。默认为True。
            use_scrape_cache (bool, 可选)：是否使用缓存的抓取文章。默认为True。
            use_cached_report (bool, 可选)：是否返回之前在相同主题上生成的报告。默认为False。

        返回：
            Iterator[RunResponse]：包含生成报告或状态信息的流对象。

        步骤：
        1. 如果use_cached_report为True，检查缓存的报告。
        2. 搜索网络上关于该主题的文章：
            - 如果可用且use_search_cache为True，使用缓存的搜索结果。
            - 否则，执行新的网络搜索。
        3. 抓取每篇文章的内容：
            - 如果可用且use_scrape_cache为True，使用缓存的抓取文章。
            - 抓取不在缓存中的新文章。
        4. 使用抓取的文章内容生成最终报告。

        该函数利用`session_state`来存储和检索缓存数据。
        """
        logger.info(f"正在生成关于：{topic}的报告")

        # 如果use_cached_report为True，使用缓存的报告
        if use_cached_report:
            cached_report = self.get_cached_report(topic)
            if cached_report:
                yield RunResponse(
                    content=cached_report, event=RunEvent.workflow_completed
                )
                return

        # 搜索网络上关于该主题的文章
        search_results: Optional[SearchResults] = self.get_search_results(
            topic, use_search_cache
        )
        # 如果没有为该主题找到search_results，结束工作流程
        if search_results is None or len(search_results.articles) == 0:
            yield RunResponse(
                event=RunEvent.workflow_completed,
                content=f"抱歉，无法找到关于主题的任何文章：{topic}",
            )
            return

        # 抓取搜索结果
        scraped_articles: Dict[str, ScrapedArticle] = self.scrape_articles(
            search_results, use_scrape_cache
        )

        # 写研究报告
        yield from self.write_research_report(topic, scraped_articles)

    def get_cached_report(self, topic: str) -> Optional[str]:
        logger.info("检查是否存在缓存报告")
        return self.session_state.get("reports", {}).get(topic)

    def add_report_to_cache(self, topic: str, report: str):
        logger.info(f"保存主题的报告: {topic}")
        self.session_state.setdefault("reports", {})
        self.session_state["reports"][topic] = report
        # 将报告保存到存储中
        self.write_to_storage()

    def get_cached_search_results(self, topic: str) -> Optional[SearchResults]:
        logger.info("检查是否存在缓存搜索结果")
        return self.session_state.get("search_results", {}).get(topic)

    def add_search_results_to_cache(self, topic: str, search_results: SearchResults):
        logger.info(f"保存主题的搜索结果: {topic}")
        self.session_state.setdefault("search_results", {})
        self.session_state["search_results"][topic] = search_results.model_dump()
        # 将搜索结果保存到存储中
        self.write_to_storage()

    def get_cached_scraped_articles(
        self, topic: str
    ) -> Optional[Dict[str, ScrapedArticle]]:
        logger.info("检查是否存在缓存抓取文章")
        return self.session_state.get("scraped_articles", {}).get(topic)

    def add_scraped_articles_to_cache(
        self, topic: str, scraped_articles: Dict[str, ScrapedArticle]
    ):
        logger.info(f"保存主题的抓取文章: {topic}")
        self.session_state.setdefault("scraped_articles", {})
        self.session_state["scraped_articles"][topic] = scraped_articles
        # 将抓取文章保存到存储中
        self.write_to_storage()

    def get_search_results(
        self, topic: str, use_search_cache: bool, num_attempts: int = 3
    ) -> Optional[SearchResults]:
        # 如果use_search_cache为True，从会话状态获取缓存的search_results
        if use_search_cache:
            try:
                search_results_from_cache = self.get_cached_search_results(topic)
                if search_results_from_cache is not None:
                    search_results = SearchResults.model_validate(
                        search_results_from_cache
                    )
                    logger.info(
                        f"在缓存中找到{len(search_results.articles)}篇文章。"
                    )
                    return search_results
            except Exception as e:
                logger.warning(f"无法从缓存读取搜索结果: {e}")

        # 如果没有缓存的search_results，使用web_searcher查找最新文章
        for attempt in range(num_attempts):
            try:
                searcher_response: RunResponse = self.web_searcher.run(topic)
                if (
                    searcher_response is not None
                    and searcher_response.content is not None
                    and isinstance(searcher_response.content, SearchResults)
                ):
                    article_count = len(searcher_response.content.articles)
                    logger.info(
                        f"在第{attempt + 1}次尝试中找到{article_count}篇文章"
                    )
                    # 缓存搜索结果
                    self.add_search_results_to_cache(topic, searcher_response.content)
                    return searcher_response.content
                else:
                    logger.warning(
                        f"第{attempt + 1}/{num_attempts}次尝试失败：无效的响应类型"
                    )
            except Exception as e:
                logger.warning(f"第{attempt + 1}/{num_attempts}次尝试失败：{str(e)}")

        logger.error(f"在{num_attempts}次尝试后未能获取搜索结果")
        return None

    def scrape_articles(
        self, search_results: SearchResults, use_scrape_cache: bool
    ) -> Dict[str, ScrapedArticle]:
        scraped_articles: Dict[str, ScrapedArticle] = {}

        # 如果use_scrape_cache为True，从会话状态获取缓存的scraped_articles
        if use_scrape_cache:
            try:
                scraped_articles_from_cache = self.get_cached_scraped_articles(topic)
                if scraped_articles_from_cache is not None:
                    scraped_articles = scraped_articles_from_cache
                    logger.info(
                        f"在缓存中找到{len(scraped_articles)}篇抓取的文章。"
                    )
                    return scraped_articles
            except Exception as e:
                logger.warning(f"无法从缓存读取抓取的文章: {e}")

        # 抓取不在缓存中的文章
        for article in search_results.articles:
            if article.url in scraped_articles:
                logger.info(f"在缓存中找到抓取的文章: {article.url}")
                continue

            article_scraper_response: RunResponse = self.article_scraper.run(
                article.url
            )
            if (
                article_scraper_response is not None
                and article_scraper_response.content is not None
                and isinstance(article_scraper_response.content, ScrapedArticle)
            ):
                scraped_articles[article_scraper_response.content.url] = (
                    article_scraper_response.content
                )
                logger.info(f"抓取的文章: {article_scraper_response.content.url}")

        # 在会话状态中保存抓取的文章
        self.add_scraped_articles_to_cache(topic, scraped_articles)
        return scraped_articles

    def write_research_report(
        self, topic: str, scraped_articles: Dict[str, ScrapedArticle]
    ) -> Iterator[RunResponse]:
        logger.info("正在撰写研究报告")
        # 为写手准备输入
        writer_input = {
            "topic": topic,
            "articles": [v.model_dump() for v in scraped_articles.values()],
        }
        # 运行写手并生成响应
        yield from self.writer.run(json.dumps(writer_input, indent=4), stream=True)
        # 在缓存中保存研究报告
        self.add_report_to_cache(topic, self.writer.run_response.content)


# 如果直接执行脚本，运行工作流程
if __name__ == "__main__":
    from rich.prompt import Prompt

    # 示例研究主题
    example_topics = [
        "量子计算突破2024",
        "人工意识研究",
        "聚变能源发展",
        "太空旅游环境影响",
        "长寿研究进展",
    ]

    topics_str = "\n".join(
        f"{i + 1}. {topic}" for i, topic in enumerate(example_topics)
    )

    print(f"\n📚 示例研究主题：\n{topics_str}\n")

    # 从用户获取主题
    topic = Prompt.ask(
        "[bold]输入研究主题[/bold]\n✨",
        default="量子计算突破2024",
    )

    # 将主题转换为URL安全的字符串，用于session_id
    url_safe_topic = topic.lower().replace(" ", "-")

    # 初始化新闻报告生成器工作流程
    generate_research_report = ResearchReportGenerator(
        session_id=f"generate-report-on-{url_safe_topic}",
        storage=SqliteStorage(
            table_name="generate_research_report_workflow",
            db_file="tmp/workflows.db",
        ),
    )

    # 执行工作流程，启用缓存
    report_stream: Iterator[RunResponse] = generate_research_report.run(
        topic=topic,
        use_search_cache=True,
        use_scrape_cache=True,
        use_cached_report=True,
    )

    # 打印响应
    pprint_run_response(report_stream, markdown=True)
