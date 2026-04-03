"""自然语言摘要生成器 - 基于模板填充，不调用模型

将结构化触觉特征翻译为 ~30-60 字的体感描述，
措辞偏感受而非数据报告。只有这部分进入模型上下文。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.common.logger import get_logger

if TYPE_CHECKING:
    from .tactile_engine import StrokeFeatures

logger = get_logger("tactile_summary")


# ---------------------------------------------------------------------------
# 描述词映射
# ---------------------------------------------------------------------------

def _pressure_word(mean: float, max_: float) -> str:
    if max_ < 0.15:
        return "极轻"
    if mean < 0.25:
        return "轻柔"
    if mean < 0.45:
        return "温和"
    if mean < 0.65:
        return "有力"
    return "用力"


def _pressure_trend(p_min: float, p_max: float, p_mean: float) -> str:
    """根据压力统计推断趋势"""
    if p_max - p_min < 0.08:
        return "均匀"
    if p_mean > (p_min + p_max) / 2 + 0.05:
        return "渐增"
    if p_mean < (p_min + p_max) / 2 - 0.05:
        return "渐减"
    return "起伏"


def _speed_word(mean: float) -> str:
    if mean < 0.08:
        return "很慢地"
    if mean < 0.18:
        return "缓慢"
    if mean < 0.35:
        return ""  # 中等速度不特别描述
    if mean < 0.6:
        return "快速"
    return "飞快地"


def _region_word(region: str) -> str:
    mapping = {
        "top_left": "左上方",
        "top_center": "上方",
        "top_right": "右上方",
        "center_left": "中部偏左",
        "center": "正中央",
        "center_right": "中部偏右",
        "bottom_left": "左下方",
        "bottom_center": "下方",
        "bottom_right": "右下方",
    }
    return mapping.get(region, "某处")


def _tilt_description(tilt_x: float, tilt_y: float) -> str:
    """笔的倾斜角 → 手势意图描述"""
    angle = abs(tilt_x) + abs(tilt_y)
    if angle < 15:
        return ""  # 近乎垂直，普通握持
    if angle < 35:
        return "，笔略微倾斜"
    return "，倾斜角大——笔侧面接触"


def _duration_word(duration: float) -> str:
    if duration < 0.2:
        return "瞬间"
    if duration < 0.8:
        return "短暂"
    if duration < 2.0:
        return ""  # 普通时长不描述
    if duration < 5.0:
        return "持续"
    return "长时间"


# ---------------------------------------------------------------------------
# 手势动作词
# ---------------------------------------------------------------------------

_GESTURE_VERBS: dict[str, str] = {
    "tap": "轻点了",
    "flick": "快速弹过",
    "stroke": "划过",
    "rub": "来回蹭了",
    "press_and_hold": "按住了",
    "press_drag": "用力按着划过",
    "circle": "画了个圈在",
    "scratch": "挠了挠",
    "scribble": "胡乱涂画了",
    "slow_trace": "缓慢划过",
    "interrupted": "触摸被中断于",
    "unknown": "碰了",
}


# ---------------------------------------------------------------------------
# 主生成函数
# ---------------------------------------------------------------------------

def generate_summary(features: StrokeFeatures) -> str:
    """从结构化特征生成自然语言摘要

    输出格式: [触觉 HH:MM] 描述文本
    目标长度: 30-60 字
    """
    logger.debug(
        f"[Summary] 开始生成摘要: stroke={features.stroke_id} "
        f"gesture={features.gesture} region={features.region}"
    )
    # 时间戳（转为本地时间）
    try:
        dt = datetime.fromisoformat(features.ts)
        # UTC → 本地时间
        dt_local = dt.astimezone()
    except (ValueError, TypeError):
        dt_local = datetime.now()
    time_str = dt_local.strftime("%H:%M")

    # 组装描述
    gesture_verb = _GESTURE_VERBS.get(features.gesture, "碰了")
    region = _region_word(features.region)
    p_mean = features.pressure_stats.get("mean", 0)
    p_max = features.pressure_stats.get("max", 0)
    p_min = features.pressure_stats.get("min", 0)
    pressure = _pressure_word(p_mean, p_max)
    trend = _pressure_trend(p_min, p_max, p_mean)
    speed = _speed_word(features.speed_stats.get("mean", 0))
    tilt = _tilt_description(
        features.tilt_mean.get("x", 0),
        features.tilt_mean.get("y", 0),
    )
    duration = _duration_word(features.duration)

    # 根据手势类型选择模板
    if features.gesture == "tap":
        desc = f"{pressure}地点了一下{region}"
    elif features.gesture == "flick":
        desc = f"快速弹过{region}，像指尖轻拂"
    elif features.gesture == "press_and_hold":
        duration_s = f"{features.duration:.1f}s"
        desc = f"按住{region}不动，{duration_s}，压力{pressure}{tilt}"
    elif features.gesture == "press_drag":
        duration_s = f"{features.duration:.1f}s"
        desc = f"用力按着{speed}拖过{region}，{duration_s}，压力{pressure}{trend}{tilt}"
    elif features.gesture == "rub":
        duration_s = f"{features.duration:.1f}s"
        desc = f"在{region}来回蹭着，{duration_s}，力道{pressure}{tilt}"
    elif features.gesture == "scratch":
        duration_s = f"{features.duration:.1f}s"
        desc = f"在{region}挠了挠，{duration_s}，压力{pressure}，动作急促"
    elif features.gesture == "circle":
        desc = f"在{region}画了个圈，{pressure}的力道"
    elif features.gesture == "scribble":
        desc = f"在{region}胡乱涂画，动作急促"
    elif features.gesture == "slow_trace":
        duration_s = f"{features.duration:.1f}s"
        desc = f"{speed}划过{region}，{duration_s}，压力{pressure}{trend}{tilt}"
    elif features.gesture == "stroke":
        desc = f"{speed}划过{region}，压力{pressure}"
        if features.curvature < 0.5:
            desc += "，轨迹弯曲"
    elif features.gesture == "interrupted":
        desc = f"触摸被中断于{region}（笔尖离开感应范围）"
    else:
        duration_s = f"{features.duration:.1f}s" if features.duration > 0.5 else ""
        time_part = f"，{duration_s}" if duration_s else ""
        desc = f"{gesture_verb}{region}{time_part}，{pressure}"

    # 截断标记
    if features.truncated and features.gesture != "interrupted":
        desc += "（信号截断）"

    result = f"[触觉 {time_str}] {desc}"
    logger.debug(f"[Summary] 摘要生成完成: stroke={features.stroke_id} → {result}")
    return result
