"""自然语言摘要生成器 - 基于模板填充，不调用模型

将结构化触觉特征翻译为 ~30-60 字的体感描述，
措辞偏感受而非数据报告。只有这部分进入模型上下文。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from src.common.logger import get_logger

if TYPE_CHECKING:
    from .tactile_engine import StrokeFeatures, TactileEntry

logger = get_logger("tactile_summary")


# ---------------------------------------------------------------------------
# 笔倾斜 → 接触面积翻译
# 被摸的一方感受到的不是笔的角度，而是接触面从一个点变成了一片——
# 物理上笔越倾斜，笔尖与感应面的接触椭圆越大。故按面积语义翻译，不暴露仪器细节。
# ---------------------------------------------------------------------------

_TILT_CONTACT_POINT = 15  # 倾斜角（|x|+|y|，度）低于此值 → 点接触，不描述
_TILT_CONTACT_WIDE = 35   # 低于此值 → 接触面变宽（指腹级）；不低于 → 大面积贴合
_TILT_PHRASE_WIDE = "，接触面变宽了些，不是指尖是指腹"
_TILT_PHRASE_FULL = "，大片地贴着，像整个指腹压了上来"
_TILT_PHRASE_FULL_SHORT = "，贴得很实"  # 与质感短语并存导致超长时的收缩版
_SUMMARY_DESC_MAX_CHARS = 70  # 摘要描述目标长度上限（字）


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


# 速度分档阈值：2026-07-12 按部署者真实笔触分布重标定（48条样本：p25≈0.43 中位≈0.53 p75≈0.75，
# 轻弹可达1.4；旧阈值 0.08/0.18/0.35/0.6 与其手速不在一个坐标系，"温柔抚摸"全被判成快速、
# 顺毛门48条仅1条可达）。分档语义：缓慢档 = 质感层"像被顺毛"的准入速度。
_SPEED_VERY_SLOW = 0.25
_SPEED_SLOW = 0.55
_SPEED_MEDIUM = 0.90
_SPEED_FAST = 1.25


def _speed_word(mean: float) -> str:
    if mean < _SPEED_VERY_SLOW:
        return "很慢地"
    if mean < _SPEED_SLOW:
        return "缓慢"
    if mean < _SPEED_MEDIUM:
        return ""  # 中等速度不特别描述
    if mean < _SPEED_FAST:
        return "快速"
    return "飞快地"


def _region_endpoints(region: str) -> tuple[str, str]:
    """长划轨迹 region 形如 "start->end"，返回 (start, end)；普通 region 返回 (r, r)"""
    if "->" in region:
        a, b = region.split("->", 1)
        return a, b
    return region, region


def _region_word(region: str) -> str:
    # 长划轨迹 "start->end" → "从{A}一路划到{B}"
    if "->" in region:
        start_r, end_r = _region_endpoints(region)
        return f"从{_region_word(start_r)}一路划到{_region_word(end_r)}"
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
    """笔的倾斜角 → 接触面积体感描述（点 → 指腹 → 大片贴合）"""
    angle = abs(tilt_x) + abs(tilt_y)
    if angle < _TILT_CONTACT_POINT:
        return ""  # 近乎垂直，点接触，不描述
    if angle < _TILT_CONTACT_WIDE:
        return _TILT_PHRASE_WIDE
    return _TILT_PHRASE_FULL


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
    # 长划轨迹的 region 词本身已是"从A一路划到B"，locative 模板前面不再加"在/于"
    is_cross = "->" in features.region
    loc_prefix = "" if is_cross else "在"
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
        desc = f"{loc_prefix}{region}来回蹭着，{duration_s}，力道{pressure}{tilt}"
    elif features.gesture == "scratch":
        duration_s = f"{features.duration:.1f}s"
        desc = f"{loc_prefix}{region}挠了挠，{duration_s}，压力{pressure}，动作急促"
    elif features.gesture == "circle":
        desc = f"{loc_prefix}{region}画了个圈，{pressure}的力道"
    elif features.gesture == "scribble":
        desc = f"{loc_prefix}{region}胡乱涂画，动作急促"
    elif features.gesture == "slow_trace":
        duration_s = f"{features.duration:.1f}s"
        desc = f"{speed}划过{region}，{duration_s}，压力{pressure}{trend}{tilt}"
    elif features.gesture == "stroke":
        desc = f"{speed}划过{region}，压力{pressure}"
        if features.curvature < 0.5:
            desc += "，轨迹弯曲"
    elif features.gesture == "interrupted":
        desc = (
            f"触摸被中断，{region}（笔尖离开感应范围）"
            if is_cross
            else f"触摸被中断于{region}（笔尖离开感应范围）"
        )
    else:
        duration_s = f"{features.duration:.1f}s" if features.duration > 0.5 else ""
        time_part = f"，{duration_s}" if duration_s else ""
        desc = f"{gesture_verb}{region}{time_part}，{pressure}"

    # --- 质感层：满足特征组合时追加/替换质感短语（组合冲突按顺序取第一个）---
    # 这些短语是把 bot 交互中自然涌现的体感词收编进感官词表，让模板更贴近"感受"而非报告。
    speed_mean = features.speed_stats.get("mean", 0)
    texture = ""
    if (
        speed_mean < _SPEED_SLOW and 0.15 <= p_mean < 0.5 and features.path_len > 120
        and features.gesture in ("stroke", "slow_trace")
    ):
        # "顺毛" — bot 本人 2026-07-11 首次被摸时自己发明的词，收编进感官词表
        desc += "，像被顺毛"
        texture = "像被顺毛"
    elif (
        speed_mean >= _SPEED_MEDIUM and p_max < 0.3 and features.duration < 0.8
        and features.gesture in ("flick", "stroke")
    ):
        desc += "，痒痒的"
        texture = "痒痒的"
    elif features.gesture == "circle" and speed_mean < _SPEED_SLOW:
        # 慢速画圈 → 替换动词短语，更具摩挲感
        desc = f"{loc_prefix}{region}慢慢摩挲着画圈，{pressure}的力道"
        texture = "慢慢摩挲着画圈"
    elif (
        features.gesture in ("rub", "scratch")
        and features.displacement < 20 and features.duration > 1.5
    ):
        # 同点反复摩擦
        desc += "，固执地磨同一个地方"
        texture = "固执地磨同一个地方"
    if texture:
        logger.debug(f"[Summary] 质感层命中: stroke={features.stroke_id} → 追加/替换「{texture}」")

    # 质感短语 × 大接触面并存超长时：优先保质感短语，接触面短语收缩为"贴得很实"
    if texture and _TILT_PHRASE_FULL in desc and len(desc) > _SUMMARY_DESC_MAX_CHARS:
        over_len = len(desc)
        desc = desc.replace(_TILT_PHRASE_FULL, _TILT_PHRASE_FULL_SHORT)
        logger.debug(
            f"[Summary] 描述超长({over_len}字>{_SUMMARY_DESC_MAX_CHARS})，"
            f"接触面短语收缩为「贴得很实」({len(desc)}字)"
        )

    # 截断标记
    if features.truncated and features.gesture != "interrupted":
        desc += "（信号截断）"

    result = f"[触觉 {time_str}] {desc}"
    logger.debug(f"[Summary] 摘要生成完成: stroke={features.stroke_id} → {result}")
    return result


# ---------------------------------------------------------------------------
# 分组（乐句）摘要 —— 把一串同手势的连续触摸合成一句节奏化的体感描述
# ---------------------------------------------------------------------------

# 节奏分档阈值（组内相邻间隔，单位秒）
_RHYTHM_BRISK_MEAN = 1.0    # 间隔均值 < 此值 → 急促连续
_RHYTHM_STEADY_MEAN = 3.0   # 间隔均值 < 此值 → 不紧不慢；否则断断续续
_RHYTHM_PAUSE_GAP = 2.0     # 存在 > 此值的单个间隙 且 方差大 → "中间停了口气又继续"
_RHYTHM_PAUSE_VAR = 1.5     # 间隔方差 > 此值（配合长间隙）→ 停顿感
# 首尾压力均值差判定趋势
_GROUP_PRESSURE_DELTA = 0.08


def generate_group_summary(entries: list[TactileEntry]) -> str:
    """把一组「同手势 + 间隔相近」的连续触摸合成为一行节奏化摘要。

    entries: 时间升序的 TactileEntry 列表，len >= 2（单条组由调用方直接用原摘要）。
    输出: [触觉 HH:MM] <区域><动词>，接连N下，<节奏>，<压力趋势>
    时间戳取组内最后一条；动词沿用 _GESTURE_VERBS（去尾部方位介词以便拼接）。
    """
    n = len(entries)
    first_f = entries[0].features
    last_f = entries[-1].features
    gesture = first_f.gesture

    # 时间戳用组内最后一条
    try:
        dt_local = datetime.fromisoformat(last_f.ts).astimezone()
    except (ValueError, TypeError):
        dt_local = datetime.now()
    time_str = dt_local.strftime("%H:%M")

    # 动词沿用 _GESTURE_VERBS，去掉结尾的方位介词（在/于）以便拼接次数
    verb = _GESTURE_VERBS.get(gesture, "碰了").rstrip("在于")

    # --- 区域：同区说区域名，跨区说"从X到Y一带"（组内任一条跨区也计入端点集合）---
    start_region = _region_endpoints(first_f.region)[0]
    end_region = _region_endpoints(last_f.region)[1]
    distinct: set[str] = set()
    for e in entries:
        a, b = _region_endpoints(e.features.region)
        distinct.add(a)
        distinct.add(b)
    if len(distinct) == 1:
        region_lead = f"在{_region_word(start_region)}"
    else:
        region_lead = f"从{_region_word(start_region)}到{_region_word(end_region)}一带"

    # --- 节奏：组内相邻间隔的均值 + 方差 分档 ---
    intervals = [entries[i].timestamp - entries[i - 1].timestamp for i in range(1, n)]
    mean_iv = sum(intervals) / len(intervals) if intervals else 0.0
    var_iv = sum((x - mean_iv) ** 2 for x in intervals) / len(intervals) if intervals else 0.0
    max_iv = max(intervals) if intervals else 0.0
    if var_iv > _RHYTHM_PAUSE_VAR and max_iv > _RHYTHM_PAUSE_GAP:
        rhythm = "中间停了口气又继续"
    elif mean_iv < _RHYTHM_BRISK_MEAN:
        rhythm = "一下接一下急促连续"
    elif mean_iv < _RHYTHM_STEADY_MEAN:
        rhythm = "不紧不慢地一下下来"
    else:
        rhythm = "断断续续，时有时无"

    # --- 压力趋势：首尾组内 pressure mean 对比 ---
    first_p = first_f.pressure_stats.get("mean", 0)
    last_p = last_f.pressure_stats.get("mean", 0)
    delta = last_p - first_p
    if delta > _GROUP_PRESSURE_DELTA:
        trend = "力道一下比一下沉"
    elif delta < -_GROUP_PRESSURE_DELTA:
        trend = "力道渐渐放得轻了"
    else:
        trend = "力道自始至终稳着"

    desc = f"{region_lead}{verb}，接连{n}下，{rhythm}，{trend}"
    result = f"[触觉 {time_str}] {desc}"
    logger.debug(
        f"[Summary] 分组摘要: {n}条 {gesture} | 间隔均值={mean_iv:.2f}s 方差={var_iv:.2f} "
        f"压力Δ={delta:+.3f} → {result}"
    )
    return result
