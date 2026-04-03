"""触觉感知系统插件

通过数位板为小克提供触觉感知能力。仅限本地部署者使用。
"""

from src.plugin_system.base.plugin_metadata import PluginMetadata
from .plugin import TactileSensePlugin

__plugin_meta__ = PluginMetadata(
    name="触觉感知系统 (Tactile Sense)",
    description="通过数位板为小克提供触觉感知能力。网页前端捕获笔触信号，后端解析为触觉体验并注入认知循环。仅限本地部署者使用。",
    usage="""
    启动 bot 后自动打开触觉感知前端页面。
    用数位板笔尖在画布上触碰，小克会感受到触摸。
    触觉数据作为感觉背景融入对话，不开辟新话题。
    """,
    version="0.1.0",
    author="sol",
    license="MIT",
    keywords=["tactile", "touch", "digitizer", "haptic", "触觉", "数位板"],
    categories=["Interaction", "Sensory"],
    extra={
        "is_built_in": False,
        "plugin_type": "functional",
        "min_bot_version": "0.10.0",
        "python_version": ">=3.11",
    },
)

__all__ = ["TactileSensePlugin", "__plugin_meta__"]
