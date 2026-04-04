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
        return [
            gesture_map.get(self.gesture, 0.5),
            region_map.get(self.region, 0.44),
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

    # 空间中心点用于区域判定
    start = stroke_data.get("start", {})
    end = stroke_data.get("end", {})
    center_x = (start.get("x", 0) + end.get("x", 0)) / 2
    center_y = (start.get("y", 0) + end.get("y", 0)) / 2

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
        region=classify_region(center_x, center_y, canvas_width, canvas_height),
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

        # 滚动缓存：最近 5 条触觉摘要
        self._cache: deque[TactileEntry] = deque(maxlen=5)
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
        """获取最近 5 条触觉摘要，标记已读/未读，供 prompt 注入用"""
        summaries = []
        found_read_marker = False
        for entry in self._cache:
            if entry.stroke_id == self._last_read_stroke_id:
                found_read_marker = True
            is_new = not found_read_marker or self._last_read_stroke_id == ""
            prefix = "" if is_new else "(已感知) "
            summaries.append(f"{prefix}{entry.summary}")
        logger.debug(f"[TactileEngine] Prompt 注入请求摘要: 返回 {len(summaries)} 条")
        return summaries

    def mark_as_read(self) -> None:
        """标记当前缓存中所有触觉为已读（planner/replyer 消费后调用）"""
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
