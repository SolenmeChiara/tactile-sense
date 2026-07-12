"""触觉引擎核心 - stroke 处理、有效触摸过滤、滚动缓存、唤醒评分"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from src.common.logger import get_logger

logger = get_logger("tactile_engine")


@dataclass
class StrokeFeatures:
    """结构化触觉特征对象 (~200-400 bytes)，后端留档，不送模型"""

    stroke_id: str
    ts: str  # ISO 8601
    duration: float  # seconds
    sample_count: int
    path_len: float
    displacement: float
    curvature: float  # displacement / path_len, 0-1
    region: str  # e.g. "center_left"
    pressure_stats: dict[str, float] = field(default_factory=dict)  # min, max, mean, p75
    speed_stats: dict[str, float] = field(default_factory=dict)  # mean, max
    tilt_mean: dict[str, float] = field(default_factory=dict)  # x, y
    gesture: str = "unknown"
    truncated: bool = False

    def to_vector(self) -> list[float]:
        """用于新颖度距离计算的简化特征向量"""
        gesture_map = {
            "tap": 0.0, "flick": 0.1, "stroke": 0.2, "rub": 0.3,
            "press_and_hold": 0.4, "press_drag": 0.5, "circle": 0.6,
            "scratch": 0.7, "scribble": 0.8, "slow_trace": 1.0,
            "interrupted": 0.5, "unknown": 0.5,
        }
        region_map = {
            "top_left": 0.0, "top_center": 0.11, "top_right": 0.22,
            "center_left": 0.33, "center": 0.44, "center_right": 0.55,
            "bottom_left": 0.66, "bottom_center": 0.77, "bottom_right": 0.88,
        }
        # 长划轨迹 region 形如 "start->end"：取两格映射均值，避免 KeyError 回退 0.44 丢信息
        if "->" in self.region:
            start_r, end_r = self.region.split("->", 1)
            region_val = (region_map.get(start_r, 0.44) + region_map.get(end_r, 0.44)) / 2
        else:
            region_val = region_map.get(self.region, 0.44)
        return [
            gesture_map.get(self.gesture, 0.5),
            region_val,
            self.pressure_stats.get("mean", 0.0),
            self.speed_stats.get("mean", 0.0),
            min(self.duration / 10.0, 1.0),  # normalize to 0-1
        ]


@dataclass
class TactileEntry:
    """缓存中的一条触觉记录"""

    stroke_id: str
    features: StrokeFeatures
    summary: str  # 自然语言摘要
    timestamp: float  # unix timestamp


# ---------------------------------------------------------------------------
# 有效触摸过滤 —— 三选二
# ---------------------------------------------------------------------------

VALID_TOUCH_MIN_DURATION_MS = 200
VALID_TOUCH_MIN_PRESSURE = 0.1
VALID_TOUCH_MIN_DISTANCE_PX = 15


# ---------------------------------------------------------------------------
# 节奏聚合 / 缺席感知常量（读取侧聚合与缺席知觉，详见 get_recent_summaries）
# ---------------------------------------------------------------------------

# 相邻「同手势」触摸间隔 ≤ 此值（秒）时归入同一个「乐句」分组
GROUP_MAX_GAP_SECONDS = 8.0
# get_recent_summaries 最多输出的分组行数（取最近 N 组）
GROUP_OUTPUT_LIMIT = 5
# 触摸停止落在 [MIN, MAX] 区间（秒）才把「缺席」作为一种知觉汇报
ABSENCE_MIN_SECONDS = 180.0
# 触摸停止超过此值（秒）不再提（太久远，翻篇）
ABSENCE_MAX_SECONDS = 1800.0


# ---------------------------------------------------------------------------
# 回合快照常量（Fix A：同一回合内多次 prompt 构建复用同一份触觉内容）
# ---------------------------------------------------------------------------

# 快照存活时长（秒）：TTL 内的所有 prompt 构建复用同一快照，保证回合内一致
SNAPSHOT_TTL_SECONDS = 90.0
# 有新 stroke 时，距快照创建至少经过此值（秒）才换新快照——
# 既让同一回合的 planner+replyer 复用同一份内容，又把新触摸被隐藏的时长限制在约一个回合内
SNAPSHOT_MIN_REFRESH_SECONDS = 10.0


def is_valid_touch(stroke_data: dict[str, Any]) -> bool:
    """三选二判定：持续时间 > 200ms、压力峰值 > 0.1、移动距离 > 15px"""
    criteria_met = 0

    duration_ms = stroke_data.get("duration_ms", 0)
    if duration_ms > VALID_TOUCH_MIN_DURATION_MS:
        criteria_met += 1

    pressure_max = stroke_data.get("pressure_max", 0)
    if pressure_max > VALID_TOUCH_MIN_PRESSURE:
        criteria_met += 1

    total_distance = stroke_data.get("total_distance", 0)
    if total_distance > VALID_TOUCH_MIN_DISTANCE_PX:
        criteria_met += 1

    passed = criteria_met >= 2
    logger.debug(
        f"[TouchFilter] stroke={stroke_data.get('stroke_id', '?')} | "
        f"duration={duration_ms}ms pressure_max={pressure_max:.3f} distance={total_distance:.1f} | "
        f"criteria_met={criteria_met}/3 → {'PASS' if passed else 'REJECT'}"
    )
    return passed


# ---------------------------------------------------------------------------
# 区域分类
# ---------------------------------------------------------------------------

def classify_region(x: float, y: float, width: float, height: float) -> str:
    """将坐标映射到 3x3 九宫格区域"""
    col = "left" if x < width / 3 else ("right" if x > width * 2 / 3 else "center")
    row = "top" if y < height / 3 else ("bottom" if y > height * 2 / 3 else "center")

    if row == "center" and col == "center":
        region = "center"
    elif row == "center":
        region = f"center_{col}"
    elif col == "center":
        region = f"{row}_center"
    else:
        region = f"{row}_{col}"

    logger.debug(f"[Region] ({x:.0f}, {y:.0f}) in {width:.0f}x{height:.0f} → {region}")
    return region


# ---------------------------------------------------------------------------
# Stroke 处理：从前端原始数据 → StrokeFeatures
# ---------------------------------------------------------------------------

def process_stroke(stroke_data: dict[str, Any], canvas_width: float = 600, canvas_height: float = 400) -> StrokeFeatures:
    """将前端发来的 stroke 对象转为结构化特征"""
    from datetime import datetime, timezone

    stroke_id = stroke_data.get("stroke_id", f"s_{int(time.time())}_{0}")
    logger.debug(
        f"[StrokeProcess] 开始处理 stroke_id={stroke_id} | "
        f"samples={stroke_data.get('sample_count', '?')} gesture={stroke_data.get('gesture_hint', '?')} "
        f"canvas={canvas_width:.0f}x{canvas_height:.0f}"
    )

    # 区域判定：分别算 start 与 end 的九宫格
    # 同格 → 单格 region；异格 → "start->end"（长划轨迹，保留起止信息不被压成单格）
    start = stroke_data.get("start", {})
    end = stroke_data.get("end", {})
    start_region = classify_region(start.get("x", 0), start.get("y", 0), canvas_width, canvas_height)
    end_region = classify_region(end.get("x", 0), end.get("y", 0), canvas_width, canvas_height)
    if start_region == end_region:
        region = start_region
    else:
        region = f"{start_region}->{end_region}"
        logger.debug(f"[StrokeProcess] 长划轨迹: {start_region} → {end_region} (stroke_id={stroke_id})")

    duration_s = stroke_data.get("duration_ms", 0) / 1000.0
    total_distance = stroke_data.get("total_distance", 0)
    displacement = stroke_data.get("displacement", 0)
    curvature = displacement / total_distance if total_distance > 0 else 0

    features = StrokeFeatures(
        stroke_id=stroke_id,
        ts=datetime.now(timezone.utc).isoformat(),
        duration=round(duration_s, 3),
        sample_count=stroke_data.get("sample_count", 0),
        path_len=round(total_distance, 1),
        displacement=round(displacement, 1),
        curvature=round(curvature, 3),
        region=region,
        pressure_stats={
            "min": round(stroke_data.get("pressure_min", 0), 3),
            "max": round(stroke_data.get("pressure_max", 0), 3),
            "mean": round(stroke_data.get("pressure_mean", 0), 3),
            "p75": round(stroke_data.get("pressure_p75", 0), 3),
        },
        speed_stats={
            "mean": round(stroke_data.get("velocity_mean", 0), 3),
            "max": round(stroke_data.get("velocity_max", 0), 3),
        },
        tilt_mean={
            "x": round(stroke_data.get("tilt_mean_x", 0), 1),
            "y": round(stroke_data.get("tilt_mean_y", 0), 1),
        },
        gesture=stroke_data.get("gesture_hint", "unknown"),
        truncated=stroke_data.get("truncated", False),
    )
    logger.debug(
        f"[StrokeProcess] 特征提取完成 stroke_id={stroke_id} | "
        f"gesture={features.gesture} region={features.region} "
        f"duration={features.duration}s curvature={features.curvature} "
        f"pressure=[{features.pressure_stats.get('min', 0):.3f}, "
        f"{features.pressure_stats.get('mean', 0):.3f}, "
        f"{features.pressure_stats.get('max', 0):.3f}] "
        f"speed_mean={features.speed_stats.get('mean', 0):.4f} "
        f"tilt=({features.tilt_mean.get('x', 0):.1f}, {features.tilt_mean.get('y', 0):.1f})"
    )
    return features


# ---------------------------------------------------------------------------
# 新颖度计算
# ---------------------------------------------------------------------------

def compute_novelty(current: StrokeFeatures, recent: list[StrokeFeatures], compare_count: int = 3) -> float:
    """与最近 N 条触摸做欧氏距离，归一化到 0-1"""
    if not recent:
        logger.debug(f"[Novelty] stroke={current.stroke_id} | 无历史记录，新颖度=1.0")
        return 1.0  # 没有历史 = 完全新颖

    current_vec = current.to_vector()
    targets = recent[-compare_count:]
    distances = []

    for prev in targets:
        prev_vec = prev.to_vector()
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(current_vec, prev_vec)))
        distances.append(dist)

    avg_dist = sum(distances) / len(distances)
    # 特征向量各维 0-1，最大欧氏距离 = sqrt(5) ≈ 2.236
    novelty = min(avg_dist / 2.236, 1.0)
    logger.debug(
        f"[Novelty] stroke={current.stroke_id} | "
        f"compared_with={len(targets)} strokes | "
        f"distances={[round(d, 3) for d in distances]} avg={avg_dist:.3f} → novelty={novelty:.3f}"
    )
    return novelty


# ---------------------------------------------------------------------------
# 唤醒评分 (Phase 2 预留，Phase 1 不触发独立唤醒)
# ---------------------------------------------------------------------------

@dataclass
class WakeupFactors:
    duration: float = 0.0
    silence_gap: float = 0.0
    pressure: float = 0.0
    burst_count: float = 0.0
    novelty: float = 0.0
    text_gap: float = 0.0


def compute_wakeup_score(
    features: StrokeFeatures,
    novelty: float,
    seconds_since_last_touch: float,
    seconds_since_last_text: float,
    recent_burst_count: int,
) -> tuple[float, WakeupFactors]:
    """计算唤醒评分 (0-1)，返回 (score, factors)"""
    # 各因素权重
    w_duration = 0.20
    w_silence = 0.25
    w_pressure = 0.10
    w_burst = 0.10
    w_novelty = 0.25
    w_text_gap = 0.10

    # 持续时间分数：3s 以上大幅上升
    f_duration = min(features.duration / 5.0, 1.0)

    # 沉默间隔：越久越惊
    f_silence = min(seconds_since_last_touch / 300.0, 1.0)  # 5分钟满分

    # 压力
    f_pressure = min(features.pressure_stats.get("mean", 0) / 0.8, 1.0)

    # 连续触摸
    f_burst = min(recent_burst_count / 5.0, 1.0)

    # 新颖度
    f_novelty = novelty

    # 距上次文字消息
    f_text_gap = min(seconds_since_last_text / 600.0, 1.0) if seconds_since_last_text > 0 else 0.5

    factors = WakeupFactors(
        duration=round(f_duration * w_duration, 3),
        silence_gap=round(f_silence * w_silence, 3),
        pressure=round(f_pressure * w_pressure, 3),
        burst_count=round(f_burst * w_burst, 3),
        novelty=round(f_novelty * w_novelty, 3),
        text_gap=round(f_text_gap * w_text_gap, 3),
    )

    score = factors.duration + factors.silence_gap + factors.pressure + factors.burst_count + factors.novelty + factors.text_gap
    return round(score, 3), factors


# ---------------------------------------------------------------------------
# 回合快照
# ---------------------------------------------------------------------------

@dataclass
class _TactileSnapshot:
    """一轮注入的触觉快照。

    TTL 内的多次 prompt 构建（planner + replyer + 主动触发并行等）复用同一份内容，
    保证同一回合看到完全一致的触觉切片；换新快照时据 last_stroke_id 推进已读指针。
    """

    summaries: list[str]  # 注入用的最终摘要行（含"(已感知)"前缀 / 缺席行）
    created_at: float  # 快照创建时刻 (unix)
    last_stroke_id: str  # 创建时缓存末条 stroke_id（""=空缓存）；换新时据此推进已读指针
    stroke_count: int  # 创建时的累计有效 stroke 计数，用于判定"有无新 stroke"


# ---------------------------------------------------------------------------
# 触觉引擎主类
# ---------------------------------------------------------------------------

class TactileEngine:
    """触觉引擎单例 - 管理 stroke 处理、缓存、唤醒状态"""

    _instance: TactileEngine | None = None

    def __new__(cls) -> TactileEngine:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        # 滚动缓存：最近 15 条原始触觉记录（原始粒度保留，聚合在读取侧做）
        self._cache: deque[TactileEntry] = deque(maxlen=15)
        # 结构化特征存档（Phase 1 仅内存，Phase 2 可持久化）
        self._feature_log: deque[StrokeFeatures] = deque(maxlen=100)
        # 唤醒状态
        self._awake = False
        self._last_touch_time: float = 0.0
        self._last_text_time: float = 0.0
        self._last_wakeup_time: float = 0.0
        self._burst_timestamps: deque[float] = deque(maxlen=10)
        # 冷却期 (秒)
        self.cooldown_seconds = 10.0
        # 双阈值
        self.t_high = 0.65
        self.t_low = 0.39
        # 文字优先锁
        self._text_lock = False
        # canvas 尺寸（前端校准后发送）
        self.canvas_width = 600.0
        self.canvas_height = 400.0
        # 回调
        self._on_wakeup_callback: Any | None = None  # 状态切换到唤醒时
        self._on_stroke_callback: Any | None = None  # 每次有效 stroke 被接受时
        # 已读追踪：planner/replyer 消费过的最后一个 stroke_id
        self._last_read_stroke_id: str = ""
        # 缺席感知：本次触摸间歇是否已汇报过（新的有效 stroke 到达时复位）
        self._absence_reported: bool = False
        # 回合快照（Fix A）：TTL 内多次 prompt 构建复用同一份内容，换新时才推进已读
        self._snapshot: _TactileSnapshot | None = None
        # 累计有效 stroke 计数：单调递增，供快照判定"距上次快照是否有新 stroke"
        self._stroke_counter: int = 0
        # 触摸积累降阈值
        self._t_high_base: float = self.t_high  # 配置原值，唤醒后恢复到这里
        self.touch_threshold_drop: float = 0.05  # 每次有效触摸未唤醒时 t_high 降多少
        self.touch_threshold_floor: float = 0.20  # t_high 最低不低于此值
        # 锁
        self._lock = asyncio.Lock()

        logger.info("[TactileEngine] 触觉引擎初始化完成")

    async def process_incoming_stroke(self, stroke_data: dict[str, Any]) -> dict[str, Any]:
        """处理前端发来的 stroke 数据，返回处理结果"""
        from .summary_generator import generate_summary

        stroke_id = stroke_data.get("stroke_id", "?")
        logger.debug(f"[TactileEngine] 收到 stroke: {stroke_id}")
        logger.debug(f"[TactileEngine] stroke 原始数据: gesture={stroke_data.get('gesture_hint')} duration={stroke_data.get('duration_ms')}ms samples={stroke_data.get('sample_count')}")

        async with self._lock:
            # 有效触摸过滤
            if not is_valid_touch(stroke_data):
                logger.debug(f"[TactileEngine] stroke {stroke_id} 未通过有效触摸过滤，丢弃")
                return {"accepted": False, "reason": "invalid_touch"}

            # 处理 stroke → 结构化特征
            logger.debug(f"[TactileEngine] stroke {stroke_id} → 开始特征提取")
            features = process_stroke(stroke_data, self.canvas_width, self.canvas_height)
            self._feature_log.append(features)
            logger.debug(f"[TactileEngine] 特征日志大小: {len(self._feature_log)}/100")

            # 生成自然语言摘要
            summary = generate_summary(features)
            logger.debug(f"[TactileEngine] 摘要生成: {summary}")

            # 新颖度
            recent_features = [e.features for e in self._cache]
            novelty = compute_novelty(features, recent_features)

            # 更新缓存
            now = time.time()
            entry = TactileEntry(
                stroke_id=features.stroke_id,
                features=features,
                summary=summary,
                timestamp=now,
            )
            cache_was_full = len(self._cache) == self._cache.maxlen
            self._cache.append(entry)
            if cache_was_full:
                logger.debug(f"[TactileEngine] 缓存已满，最旧的一条被淘汰，当前缓存={len(self._cache)}")
            else:
                logger.debug(f"[TactileEngine] 缓存更新: {len(self._cache)}/{self._cache.maxlen}")

            # 累计有效 stroke 计数（供回合快照判定"距上次快照是否有新 stroke"）
            self._stroke_counter += 1

            # 新的有效触摸到达 → 复位缺席汇报标志（下一次静默期可再次汇报缺席）
            if self._absence_reported:
                logger.debug("[TactileEngine] 新触摸到达，复位缺席汇报标志")
            self._absence_reported = False

            # 更新连续触摸计数
            self._burst_timestamps.append(now)
            burst_count = sum(1 for t in self._burst_timestamps if now - t < 5.0)
            logger.debug(f"[TactileEngine] 5秒内连续触摸次数: {burst_count}")

            # 唤醒评分
            seconds_since_last_touch = now - self._last_touch_time if self._last_touch_time > 0 else 9999
            seconds_since_last_text = now - self._last_text_time if self._last_text_time > 0 else -1
            self._last_touch_time = now
            logger.debug(
                f"[TactileEngine] 时间间隔: 距上次触摸={seconds_since_last_touch:.1f}s "
                f"距上次文字={seconds_since_last_text:.1f}s"
            )

            score, factors = compute_wakeup_score(
                features, novelty, seconds_since_last_touch, seconds_since_last_text, burst_count,
            )

            # 双阈值状态切换 (Phase 1 不实际触发 planner，仅记录)
            prev_awake = self._awake
            in_cooldown = (now - self._last_wakeup_time) < self.cooldown_seconds

            state_changed = False
            if not self._awake and score >= self.t_high and not in_cooldown and not self._text_lock:
                self._awake = True
                self._last_wakeup_time = now
                state_changed = True
                logger.info(
                    f"[TactileEngine] *** 状态切换: 静默 → 唤醒 *** | "
                    f"score={score} >= t_high={self.t_high}"
                )
                # 唤醒后重置阈值到配置原值
                self._reset_threshold()
                # 主动触发回调
                if self._on_wakeup_callback is not None:
                    try:
                        summaries = [e.summary for e in self._cache]
                        asyncio.create_task(self._on_wakeup_callback(score, factors, summaries))
                    except Exception as e:
                        logger.error(f"[TactileEngine] 唤醒回调异常: {e}")
            elif self._awake and score <= self.t_low:
                self._awake = False
                state_changed = True
                logger.info(
                    f"[TactileEngine] 状态切换: 唤醒 → 静默 | "
                    f"score={score} <= t_low={self.t_low}"
                )
            elif not self._awake and score < self.t_high and not in_cooldown:
                # 未达阈值 → 自动降低 t_high，积累触摸最终触发
                old_t = self.t_high
                self.t_high = max(self.t_high - self.touch_threshold_drop, self.touch_threshold_floor)
                if self.t_high != old_t:
                    logger.debug(f"[TactileEngine] 触摸未达阈值，t_high 下降: {old_t:.2f} → {self.t_high:.2f}")
            elif not self._awake and score >= self.t_high and in_cooldown:
                logger.debug(f"[TactileEngine] 唤醒被冷却期阻止 (剩余 {self.cooldown_seconds - (now - self._last_wakeup_time):.1f}s)")
            elif not self._awake and score >= self.t_high and self._text_lock:
                logger.debug(f"[TactileEngine] 唤醒被文字优先锁阻止")

            # 全链路追踪日志
            log_entry = {
                "stroke_id": features.stroke_id,
                "stage": "wakeup_eval",
                "score": score,
                "threshold": {"high": self.t_high, "low": self.t_low},
                "prev_state": "awake" if prev_awake else "silent",
                "new_state": "awake" if self._awake else "silent",
                "cooldown_active": in_cooldown,
                "text_lock_active": self._text_lock,
                "novelty_score": round(novelty, 3),
                "factors": {
                    "duration": factors.duration,
                    "silence_gap": factors.silence_gap,
                    "pressure": factors.pressure,
                    "burst_count": factors.burst_count,
                    "novelty": factors.novelty,
                    "text_gap": factors.text_gap,
                },
            }
            logger.debug(f"[TactileEngine] 全链路追踪: {log_entry}")

            logger.debug(
                f"[TactileEngine] stroke {features.stroke_id} 处理完成 | "
                f"{features.gesture} @ {features.region} | "
                f"score={score} novelty={novelty:.3f} state={'唤醒' if self._awake else '静默'}"
            )

            # stroke 回调（写入记忆等）
            if self._on_stroke_callback is not None:
                try:
                    asyncio.create_task(self._on_stroke_callback(features, summary, score, novelty))
                except Exception as e:
                    logger.error(f"[TactileEngine] stroke 回调异常: {e}")

            return {
                "accepted": True,
                "stroke_id": features.stroke_id,
                "summary": summary,
                "gesture": features.gesture,
                "region": features.region,
                "wakeup_score": score,
                "state": "awake" if self._awake else "silent",
                "novelty": round(novelty, 3),
            }

    def notify_text_message(self) -> None:
        """通知引擎收到了文字消息（用于文字优先锁和合并策略）"""
        self._last_text_time = time.time()
        logger.debug("[TactileEngine] 收到文字消息通知，更新 last_text_time")

    def set_text_lock(self, locked: bool) -> None:
        """设置文字优先锁状态"""
        prev = self._text_lock
        self._text_lock = locked
        if prev != locked:
            logger.info(f"[TactileEngine] 文字优先锁: {'锁定' if locked else '解除'}")

    def get_recent_summaries(self) -> list[str]:
        """获取用于 prompt 注入的触觉摘要（回合快照语义，Fix A）。

        首次注入创建快照（内容 + 时间戳 + 缓存末条 id + stroke 计数）；TTL（SNAPSHOT_TTL_SECONDS）
        内的后续构建复用同一快照——保证同一回合的多次 prompt 构建（planner + replyer + 主动触发并行等）
        看到完全一致的触觉内容。快照过期，或「有新 stroke 且距快照创建 ≥ SNAPSHOT_MIN_REFRESH_SECONDS」
        时换新快照；换新的那一刻才把旧快照末条 id 推进为已读指针（"上一轮已随整轮注入完成 = 已感知"），
        因此已读指针绝不会在内容随一整轮注入完成之前抢跑。

        同步、无锁：本方法及 _compute_summaries 全程无 await，单次调用相对事件循环原子，
        stroke 计数在 process_incoming_stroke 的锁内自增，无需额外加锁。
        """
        now = time.time()
        snap = self._snapshot

        need_new = False
        reason = ""
        if snap is None:
            need_new = True
            reason = "首次"
        else:
            age = now - snap.created_at
            has_new_stroke = self._stroke_counter > snap.stroke_count
            if age >= SNAPSHOT_TTL_SECONDS:
                need_new = True
                reason = f"过期(age={age:.0f}s≥{SNAPSHOT_TTL_SECONDS:.0f}s)"
            elif has_new_stroke and age >= SNAPSHOT_MIN_REFRESH_SECONDS:
                need_new = True
                reason = f"新触摸(age={age:.0f}s≥{SNAPSHOT_MIN_REFRESH_SECONDS:.0f}s)"

        if not need_new and snap is not None:
            logger.debug(
                f"[TactileEngine] 复用触觉快照 (age={now - snap.created_at:.0f}s, 行数={len(snap.summaries)})"
            )
            return list(snap.summaries)

        # 换新快照：先把旧快照末条推进为已读（上一轮已随整轮注入完成 → 已感知）
        if snap is not None and snap.last_stroke_id and self._last_read_stroke_id != snap.last_stroke_id:
            prev = self._last_read_stroke_id
            self._last_read_stroke_id = snap.last_stroke_id
            logger.debug(
                f"[TactileEngine] 快照换新，已读指针推进: {prev or '(无)'} → {snap.last_stroke_id}"
            )

        summaries, last_stroke_id = self._compute_summaries()
        self._snapshot = _TactileSnapshot(
            summaries=list(summaries),
            created_at=now,
            last_stroke_id=last_stroke_id,
            stroke_count=self._stroke_counter,
        )
        logger.debug(
            f"[TactileEngine] 创建触觉快照 ({reason}): 行数={len(summaries)} 末条={last_stroke_id or '(空)'}"
        )
        return list(summaries)

    def _compute_summaries(self) -> tuple[list[str], str]:
        """计算当前触觉摘要，返回 (摘要行列表, 缓存末条 stroke_id)。仅由 get_recent_summaries 调用。

        读取侧完成三件事（同步、无锁、仅轻量内存运算，不做耗时 I/O）：
        1. 已读边界判定：marker 及之前 = 已感知（带 "(已感知) " 前缀），marker 之后 = 新；
           marker 被挤出缓存 → 缓存内全部晚于已读点 → 全部为新；从未读过 → 全部为新。
        2. 节奏聚合：把相邻「同手势 + 间隔 ≤ GROUP_MAX_GAP_SECONDS」的原始条目归为一个「乐句」分组，
           每组产出一行摘要；分组在已读边界处切开，保证 "(已感知)" 前缀语义精确。输出上限取最近若干组。
        3. 缺席感知：触摸停止落在 [ABSENCE_MIN, ABSENCE_MAX] 且本次间歇尚未汇报过时，末尾追加一行缺席提示。
        """
        from datetime import datetime

        from .summary_generator import generate_group_summary

        cache_list = list(self._cache)
        last_stroke_id = cache_list[-1].stroke_id if cache_list else ""

        # --- 1. 已读边界判定 ---
        if not cache_list:
            read_flags: list[bool] = []
        elif self._last_read_stroke_id == "":
            # 从未读过 → 全部为新
            read_flags = [False] * len(cache_list)
        else:
            marker_idx: int | None = None
            for i, entry in enumerate(cache_list):
                if entry.stroke_id == self._last_read_stroke_id:
                    marker_idx = i
                    break
            if marker_idx is None:
                # 已读点已被挤出缓存 → 缓存内全部晚于已读点 → 全部为新
                read_flags = [False] * len(cache_list)
                logger.debug(
                    f"[TactileEngine] 已读点 {self._last_read_stroke_id} 已被挤出缓存，缓存内全部视为新触觉"
                )
            else:
                # marker 及之前 = 已感知；marker 之后 = 新
                read_flags = [idx <= marker_idx for idx in range(len(cache_list))]

        # --- 2. 节奏聚合（分组，在已读边界处切开）---
        groups: list[tuple[list[TactileEntry], bool]] = []
        for entry, is_read in zip(cache_list, read_flags):
            if not groups:
                groups.append(([entry], is_read))
                continue
            prev_entries, prev_read = groups[-1]
            prev_entry = prev_entries[-1]
            same_gesture = entry.features.gesture == prev_entry.features.gesture
            gap = entry.timestamp - prev_entry.timestamp
            # 区域不同不阻断分组（跨区留给组内描述）；已读边界必须切开
            if same_gesture and gap <= GROUP_MAX_GAP_SECONDS and is_read == prev_read:
                prev_entries.append(entry)
            else:
                groups.append(([entry], is_read))

        if len(groups) > GROUP_OUTPUT_LIMIT:
            dropped = len(groups) - GROUP_OUTPUT_LIMIT
            groups = groups[-GROUP_OUTPUT_LIMIT:]
            logger.debug(f"[TactileEngine] 分组数超限，丢弃最旧 {dropped} 组，保留最近 {GROUP_OUTPUT_LIMIT} 组")

        summaries: list[str] = []
        for entries, is_read in groups:
            if len(entries) == 1:
                line = entries[0].summary
            else:
                line = generate_group_summary(entries)
            prefix = "(已感知) " if is_read else ""
            summaries.append(f"{prefix}{line}")

        logger.debug(
            f"[TactileEngine] 聚合完成: {len(cache_list)} 条原始 → {len(groups)} 组 → {len(summaries)} 行摘要"
        )

        # --- 3. 缺席感知（拉取式，无定时器；标志置位须在返回前完成）---
        if self._last_touch_time > 0 and not self._absence_reported:
            elapsed = time.time() - self._last_touch_time
            if ABSENCE_MIN_SECONDS <= elapsed <= ABSENCE_MAX_SECONDS:
                minutes = max(1, round(elapsed / 60.0))
                time_str = datetime.now().strftime("%H:%M")
                absence_line = f"[触觉 {time_str}] 笔尖已经离开一阵子了（约{minutes}分钟）"
                summaries.append(absence_line)
                self._absence_reported = True
                logger.debug(f"[TactileEngine] 缺席感知触发: 距上次触摸 {elapsed:.0f}s → {absence_line}")
            elif elapsed > ABSENCE_MAX_SECONDS:
                logger.debug(f"[TactileEngine] 触摸停止已超过 {ABSENCE_MAX_SECONDS:.0f}s，缺席翻篇，不再汇报")

        logger.debug(f"[TactileEngine] 摘要计算完成: 返回 {len(summaries)} 条")
        return summaries, last_stroke_id

    def mark_as_read(self) -> None:
        """标记当前缓存中所有触觉为已读。

        注意（Fix A 后）：已读推进已由 get_recent_summaries 的"换快照即推进"机制接管，
        正常注入路径不再调用此方法。保留作为手动兜底 / 向后兼容；直接调用会绕过快照，慎用。
        """
        if self._cache:
            self._last_read_stroke_id = self._cache[-1].stroke_id
            logger.debug(f"[TactileEngine] 标记已读到: {self._last_read_stroke_id}")

    def get_cache_status(self) -> dict[str, Any]:
        """获取缓存状态，用于调试和前端展示"""
        logger.debug(
            f"[TactileEngine] 状态查询: cache={len(self._cache)} "
            f"feature_log={len(self._feature_log)} awake={self._awake}"
        )
        return {
            "cache_size": len(self._cache),
            "entries": [
                {
                    "stroke_id": e.stroke_id,
                    "summary": e.summary,
                    "gesture": e.features.gesture,
                    "timestamp": e.timestamp,
                }
                for e in self._cache
            ],
            "awake": self._awake,
            "last_touch_time": self._last_touch_time,
            "feature_log_size": len(self._feature_log),
        }

    def set_wakeup_callback(self, callback) -> None:
        """注册唤醒回调。callback 签名: async def(score, factors, summaries)"""
        self._on_wakeup_callback = callback
        logger.info("[TactileEngine] 唤醒回调已注册")

    def _reset_threshold(self) -> None:
        """唤醒触发后重置 t_high 到配置原值"""
        if self.t_high != self._t_high_base:
            logger.debug(f"[TactileEngine] 唤醒后重置阈值: {self.t_high:.2f} → {self._t_high_base:.2f}")
            self.t_high = self._t_high_base

    def set_stroke_callback(self, callback) -> None:
        """注册 stroke 回调。callback 签名: async def(features, summary, score, novelty)"""
        self._on_stroke_callback = callback
        logger.info("[TactileEngine] stroke 回调已注册")

    def update_canvas_size(self, width: float, height: float) -> None:
        """更新 canvas 尺寸（前端发送）"""
        if width != self.canvas_width or height != self.canvas_height:
            logger.info(f"[TactileEngine] Canvas 尺寸更新: {self.canvas_width:.0f}x{self.canvas_height:.0f} → {width:.0f}x{height:.0f}")
        self.canvas_width = width
        self.canvas_height = height
