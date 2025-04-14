"""🎨 AI图像记者 - 您的视觉分析和新闻伴侣！

此示例展示了如何创建一个能够分析图像并通过网络搜索将其与当前事件联系起来的AI代理。适用于：
1. 新闻报道和新闻业
2. 旅游和观光内容
3. 社交媒体分析
4. 教育演示
5. 活动报道

要尝试的示例图像：
- 著名地标（埃菲尔铁塔、泰姬陵等）
- 城市天际线
- 文化事件和节日
- 突发新闻场景
- 历史地点

运行 `pip install duckduckgo-search agno` 安装依赖。
"""

from textwrap import dedent

import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'libs', 'agno')))
from agno.agent import Agent
from agno.media import Image
from agno.models.openai import QwenV1MaxChat
from agno.tools.duckduckgo import DuckDuckGoTools

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'libs', 'agno')))

agent = Agent(
    model=QwenV1MaxChat(),
    description=dedent("""\
        您是一位世界级视觉记者和文化评论员，拥有通过讲故事让图像栩栩如生的天赋！ 📸✨ 
        凭借侦探般的观察技能和畅销书作者般的叙事风采，您将视觉分析转化为引人入胜的故事，既有信息性又能吸引人。\
    """),
    instructions=dedent("""\
        在分析图像和报道新闻时，请遵循以下原则：

        1. 视觉分析：
           - 以使用相关emoji的吸引人标题开始
           - 以专家精准度分解关键视觉元素
           - 注意其他人可能错过的细微细节
           - 将视觉元素与更广泛的上下文联系起来

        2. 新闻整合：
           - 研究并验证与图像相关的时事
           - 将历史背景与当代意义联系起来
           - 优先保证准确性同时保持吸引力
           - 在可用时包含相关统计或数据

        3. 讲故事风格：
           - 保持专业但引人入胜的语调
           - 使用生动、描述性的语言
           - 在相关时包含文化和历史参考
           - 以适合故事的难忘结语结束

        4. 报道指南：
           - 保持回答简洁但信息丰富（2-3段）
           - 平衡事实与人情趣味
           - 保持新闻诚信
           - 引用特定信息时注明来源

        将每张图像转化为引人入胜的新闻故事，既能提供信息又能启发！\
    """),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
    stream=True,
    use_json_mode=True,
)

# 使用著名地标的示例用法
agent.print_response(
    "告诉我关于这张图像的信息，并分享最新相关新闻。",
    images=[
        Image(
            #url="https://upload.wikimedia.org/wikipedia/commons/0/0c/GoldenGateBridge-001.jpg" 图片太大了，导致请求超时
            url="http://p1.img.cctvpic.com/photoAlbum/page/performance/img/2018/2/2/1517562079816_790.jpg"
        )
    ],
    stream=True,
)

# 更多要尝试的示例：
"""
要探索的示例提示：
1. "这个地点有什么历史意义？"
2. "这个地方随着时间如何变化？"
3. "这里发生什么文化活动？"
4. "这是什么建筑风格和影响？"
5. "有什么最近的发展影响这个区域？"

要分析的示例图像URL：
1. 埃菲尔铁塔："https://upload.wikimedia.org/wikipedia/commons/8/85/Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg"
2. 泰姬陵："https://upload.wikimedia.org/wikipedia/commons/b/bd/Taj_Mahal%2C_Agra%2C_India_edit3.jpg"
3. 金门大桥："https://upload.wikimedia.org/wikipedia/commons/0/0c/GoldenGateBridge-001.jpg"
"""

# 要获取变量中的响应：
# from rich.pretty import pprint
# response = agent.run(
#     "分析这个地标的建筑和最近的新闻。",
#     images=[Image(url="YOUR_IMAGE_URL")],
# )
# pprint(response.content)
