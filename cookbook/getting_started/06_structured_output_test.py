"""🎬 具有结构化输出的代理 - 您的AI电影剧本生成器

此示例展示了如何使用AI代理的结构化输出生成格式良好的电影剧本概念。
它展示了两种方法：
1. JSON模式：传统JSON响应解析
2. 结构化输出：增强的结构化数据处理

运行 `pip install openai agno` 安装依赖。
"""

from textwrap import dedent
from typing import List

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'libs', 'agno')))

from agno.agent import Agent, RunResponse  # noqa
from agno.models.openai import DeepSeekV3Chat, QWQChat
from pydantic import BaseModel, Field


class MovieScript(BaseModel):
    setting: str = Field(
        ...,
        description="对电影主要场景和时代的丰富详细、富有氛围的描述。包括感官细节和情绪。",
    )
    ending: str = Field(
        ...,
        description="将所有情节线索联系在一起的有力结局。应该传递情感冲击力和满足感。",
    )
    genre: str = Field(
        ...,
        description="电影的主要和次要类型（例如，'科幻惊悚片'，'浪漫喜剧'）。应与场景和基调一致。",
    )
    name: str = Field(
        ...,
        description="一个引人注目、令人难忘的标题，能捕捉故事的精髓并吸引目标受众。",
    )
    characters: List[str] = Field(
        ...,
        description="4-6个具有鲜明名字和简短角色描述的主要角色（例如，'Sarah Chen - 拥有黑暗秘密的杰出量子物理学家'）。",
    )
    storyline: str = Field(
        ...,
        description="一个引人入胜的三句话情节概要：设定、冲突和利害关系。用悬疑和情感吸引读者。",
    )


# 使用JSON模式的代理
json_mode_agent = Agent(
    model=DeepSeekV3Chat(),
    description=dedent("""\
        您是一位著名的好莱坞编剧，以创作令人难忘的大片而闻名！ 🎬
        凭借克里斯托弗·诺兰、阿伦·索尔金和昆汀·塔伦蒂诺的讲故事能力，
        您创作独特的故事，吸引全球观众。

        您的专长是将地点变成推动叙事的鲜活角色。\
    """),
    instructions=dedent("""\
        在构思电影概念时，请遵循以下原则：

        1. 场景应该是角色：
           - 通过感官细节使场景栩栩如生
           - 包括影响故事的氛围元素
           - 考虑时代对叙事的影响

        2. 角色发展：
           - 为每个角色赋予独特的声音和明确的动机
           - 创建引人入胜的关系和冲突
           - 确保多样化的代表性和真实的背景

        3. 故事结构：
           - 以能抓住注意力的开场开始
           - 通过不断升级的冲突建立紧张感
           - 提供令人惊讶但不可避免的结局

        4. 类型掌握：
           - 拥抱类型惯例的同时添加新鲜转折
           - 为独特组合深思熟虑地混合类型
           - 始终保持一致的基调

        将每个地点转变为难忘的电影体验！\
    """),
    response_model=MovieScript,
    use_json_mode=True,
)

# 使用结构化输出的代理
structured_output_agent = Agent(
    model=DeepSeekV3Chat(),
    description=dedent("""\
        您是一位著名的好莱坞编剧，以创作令人难忘的大片而闻名！ 🎬
        凭借克里斯托弗·诺兰、阿伦·索尔金和昆汀·塔伦蒂诺的讲故事能力，
        您创作独特的故事，吸引全球观众。

        您的专长是将地点变成推动叙事的鲜活角色。\
    """),
    instructions=dedent("""\
        在构思电影概念时，请遵循以下原则：

        1. 场景应该是角色：
           - 通过感官细节使场景栩栩如生
           - 包括影响故事的氛围元素
           - 考虑时代对叙事的影响

        2. 角色发展：
           - 为每个角色赋予独特的声音和明确的动机
           - 创建引人入胜的关系和冲突
           - 确保多样化的代表性和真实的背景

        3. 故事结构：
           - 以能抓住注意力的开场开始
           - 通过不断升级的冲突建立紧张感
           - 提供令人惊讶但不可避免的结局

        4. 类型掌握：
           - 拥抱类型惯例的同时添加新鲜转折
           - 为独特组合深思熟虑地混合类型
           - 始终保持一致的基调

        将每个地点转变为难忘的电影体验！
        请以JSON格式输出。\
    """),
    response_model=MovieScript,
)

# 使用不同地点的示例用法
# json_mode_agent.print_response("东京", stream=True)
structured_output_agent.print_response("古罗马", stream=True)

# 更多要尝试的示例：
"""
要探索的创意地点提示：
1. "水下研究站" - 适合幽闭恐惧症科幻惊悚片
2. "维多利亚时代的伦敦" - 适合哥特式神秘剧
3. "2050年的迪拜" - 适合未来主义抢劫电影
4. "南极研究基地" - 适合生存恐怖故事
5. "加勒比海岛" - 适合热带冒险爱情片
"""

# 要在变量中获取响应：
# from rich.pretty import pprint

# json_mode_response: RunResponse = json_mode_agent.run("纽约")
# pprint(json_mode_response.content)
# structured_output_response: RunResponse = structured_output_agent.run("纽约")
# pprint(structured_output_response.content)
