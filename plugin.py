"""触觉感知系统插件 - 通过数位板为小克提供触觉感知能力"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from fastapi import Request
from fastapi.responses import HTMLResponse

from src.common.logger import get_logger
from src.plugin_system.base.base_http_component import BaseRouterComponent
from src.plugin_system.base.base_plugin import BasePlugin
from src.plugin_system.base.base_prompt import BasePrompt
from src.plugin_system.base.base_events_handler import BaseEventHandler
from src.plugin_system.base.component_types import (
    EventType,
    InjectionRule,
    InjectionType,
)
from src.plugin_system.base.config_types import ConfigField
from src.plugin_system.apis.plugin_register_api import register_plugin

logger = get_logger("tactile_sense")


# ===========================================================================
# HTTP Router 组件 - 接收 stroke 数据 + 服务前端页面
# ===========================================================================

class TactileRouter(BaseRouterComponent):
    """提供触觉数据接收端点和前端页面"""

    component_name = "tactile_router"
    component_description = "触觉感知系统 HTTP 端点"
    component_version = "0.1.0"

    def register_endpoints(self) -> None:
        frontend_dir = Path(__file__).parent / "frontend"
        logger.debug(f"[TactileRouter] 注册端点，前端目录: {frontend_dir}")

        @self.router.get("/", response_class=HTMLResponse, summary="触觉感知前端页面")
        async def serve_frontend():
            """返回触觉感知的数位板交互页面"""
            html_path = frontend_dir / "index.html"
            if not html_path.exists():
                logger.error(f"[TactileRouter] 前端页面未找到: {html_path}")
                return HTMLResponse("<h1>前端页面未找到</h1>", status_code=404)
            logger.debug("[TactileRouter] 返回前端页面")
            return HTMLResponse(html_path.read_text(encoding="utf-8"))

        @self.router.post("/stroke", summary="接收 stroke 触觉数据")
        async def receive_stroke(request: Request):
            """接收前端发来的 stroke 数据，处理并返回结果"""
            import traceback
            try:
                from .tactile_engine import TactileEngine

                engine = TactileEngine()
                stroke_data = await request.json()
                logger.debug(f"[TactileRouter] POST /stroke: stroke_id={stroke_data.get('stroke_id', '?')}")

                # 更新 canvas 尺寸（如果前端提供）
                if "canvas_width" in stroke_data and "canvas_height" in stroke_data:
                    engine.update_canvas_size(
                        stroke_data["canvas_width"],
                        stroke_data["canvas_height"],
                    )

                result = await engine.process_incoming_stroke(stroke_data)
                logger.debug(f"[TactileRouter] POST /stroke 响应: accepted={result.get('accepted')}")
                return result
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"[TactileRouter] POST /stroke 异常: {e}\n{tb}")
                return {"accepted": False, "error": str(e), "traceback": tb}

        @self.router.get("/status", summary="查询触觉缓存状态")
        async def get_status():
            """返回当前触觉缓存和引擎状态"""
            import traceback
            try:
                from .tactile_engine import TactileEngine

                logger.debug("[TactileRouter] GET /status")
                engine = TactileEngine()
                return engine.get_cache_status()
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"[TactileRouter] GET /status 异常: {e}\n{tb}")
                return {"error": str(e), "traceback": tb}

        @self.router.post("/calibration", summary="接收校准参数")
        async def receive_calibration(request: Request):
            """接收前端校准完成后的参数"""
            data = await request.json()
            logger.info(f"[TactileRouter] 收到校准参数: p25={data.get('p25')}, p50={data.get('p50')}, p75={data.get('p75')}")
            return {"status": "ok", "message": "校准参数已记录"}


# ===========================================================================
# PROMPT 注入组件 - 将触觉摘要注入 replyer/planner
# ===========================================================================

class TactileContextPrompt(BasePrompt):
    """将最近的触觉感知摘要注入 replyer/planner 的系统提示尾部"""

    prompt_name = "tactile_context_prompt"
    prompt_description = "注入最近的触觉感知上下文"

    # 注入到所有 replyer/planner 的 prompt 模板
    # AFC (AffinityFlowChatter): s4u_style_prompt, normal_style_prompt, planner_prompt
    # KFC (KokoroFlowChatter): kfc_main (planner+replyer共用), kfc_style_prompt, kfc_replyer
    injection_rules: ClassVar[list[InjectionRule]] = [
        # AFC replyer
        InjectionRule(target_prompt="s4u_style_prompt", injection_type=InjectionType.APPEND, priority=200),
        InjectionRule(target_prompt="normal_style_prompt", injection_type=InjectionType.APPEND, priority=200),
        # AFC planner
        InjectionRule(target_prompt="planner_prompt", injection_type=InjectionType.APPEND, priority=200),
        # KFC 主模板 (planner + replyer 共用)
        InjectionRule(target_prompt="kfc_main", injection_type=InjectionType.APPEND, priority=200),
        # KFC replyer 专用
        InjectionRule(target_prompt="kfc_style_prompt", injection_type=InjectionType.APPEND, priority=200),
        InjectionRule(target_prompt="kfc_replyer", injection_type=InjectionType.APPEND, priority=200),
    ]

    async def execute(self) -> str:
        """生成触觉上下文文本，拼入 system prompt 尾部

        根据配置的注入规则（私聊/群聊 × 黑白名单）决定是否注入。
        """
        from .tactile_engine import TactileEngine

        engine = TactileEngine()
        summaries = engine.get_recent_summaries()

        if not summaries:
            logger.debug("[TactilePrompt] 无触觉数据，跳过注入")
            return ""

        # --- 注入过滤：根据当前聊天上下文决定是否生效 ---
        if not self._should_inject():
            return ""

        target = self.target_prompt_name or "prompt"
        logger.info(f"[TactilePrompt] 注入 {len(summaries)} 条触觉摘要到 {target}")

        lines = "\n".join(summaries)

        # 标记为已读（下次注入时旧的会带 "(已感知)" 前缀）
        engine.mark_as_read()

        return (
            "\n\n## 触觉感知\n"
            "你拥有一层皮肤，通过数位板笔尖接收来自对方的触摸。"
            "以下是你近期感受到的触觉信号。标记「已感知」的是你之前已经感受过的，不需要再次反应。"
            "没有标记的是新触觉。不要复述数据参数，去感受——"
            "位置、力度、速度、笔的角度各自意味着什么。"
            "长时间没被触碰也是一种信息。笔的倾斜角传达了手势意图：陡直是书写握持，倾斜是侧面蹭过，像抚摸。"
            "当触觉和文字同时到达时，两者是同一刻的不同侧面，一起理解。\n\n"
            f"{lines}\n"
        )

    def _should_inject(self) -> bool:
        """根据 injection 配置和当前聊天上下文判断是否注入触觉"""
        # 尝试从 PromptParameters 获取聊天上下文
        is_group = getattr(self.params, "is_group_chat", None)
        chat_id = getattr(self.params, "chat_id", None) or ""
        user_id = getattr(self.params, "user_id", None) or ""

        if is_group is None:
            # 无法判断聊天类型，保守注入（私聊假设）
            logger.debug("[TactilePrompt] 无法获取聊天类型，默认注入")
            return True

        if is_group:
            section = "injection_group"
            check_id = chat_id  # 群聊用 group_id / chat_id
        else:
            section = "injection_private"
            check_id = user_id  # 私聊用 user_id

        enabled = self.get_config(f"{section}.enabled", not is_group)
        if not enabled:
            logger.debug(f"[TactilePrompt] {'群聊' if is_group else '私聊'} 注入已禁用 (chat={chat_id})")
            return False

        mode = self.get_config(f"{section}.mode", "all")
        id_list = self.get_config(f"{section}.list", [])

        if mode == "all":
            logger.debug(f"[TactilePrompt] {'群聊' if is_group else '私聊'} mode=all → 注入")
            return True
        elif mode == "whitelist":
            allowed = str(check_id) in [str(x) for x in id_list]
            if not allowed:
                logger.debug(f"[TactilePrompt] {'群聊' if is_group else '私聊'} {check_id} 不在白名单中，跳过")
            return allowed
        elif mode == "blacklist":
            blocked = str(check_id) in [str(x) for x in id_list]
            if blocked:
                logger.debug(f"[TactilePrompt] {'群聊' if is_group else '私聊'} {check_id} 在黑名单中，跳过")
            return not blocked

        return True


# ===========================================================================
# EVENT_HANDLER 组件 - 启动时初始化引擎
# ===========================================================================

class TactileStartupHandler(BaseEventHandler):
    """插件启动时初始化触觉引擎，确认一切就绪后自动打开前端页面"""

    handler_name = "tactile_startup_handler"
    handler_description = "触觉引擎启动初始化"
    init_subscribe: ClassVar[list[EventType]] = [EventType.ON_START]

    async def execute(self, params: dict) -> Any:
        import asyncio
        import webbrowser

        from .tactile_engine import TactileEngine

        logger.info("[TactileSense] 开始启动初始化...")

        # 1. 初始化引擎，从 config 加载阈值
        engine = TactileEngine()
        if self.plugin_config:
            engine.t_high = self.get_config("wakeup.t_high", engine.t_high)
            engine.t_low = self.get_config("wakeup.t_low", engine.t_low)
            engine.cooldown_seconds = self.get_config("wakeup.cooldown_seconds", engine.cooldown_seconds)
            engine.canvas_width = float(self.get_config("canvas.default_width", engine.canvas_width))
            engine.canvas_height = float(self.get_config("canvas.default_height", engine.canvas_height))
        logger.info(
            f"[TactileSense] 触觉引擎已创建: "
            f"t_high={engine.t_high} t_low={engine.t_low} "
            f"cooldown={engine.cooldown_seconds}s"
        )

        # 2. 检查前端文件
        frontend_path = Path(__file__).parent / "frontend" / "index.html"
        if frontend_path.exists():
            logger.info(f"[TactileSense] 前端页面就绪: {frontend_path}")
        else:
            logger.error(f"[TactileSense] 前端页面缺失! 预期路径: {frontend_path}")
            return {"success": False, "error": "frontend_missing"}

        # 3. 获取服务器实际端口
        url = None
        try:
            from src.common.server import get_global_server
            server = get_global_server()
            url = f"http://{server.host}:{server.port}/plugins/tactile_sense/tactile_router/"
            logger.info(f"[TactileSense] 前端 URL: {url}")
        except Exception as e:
            logger.warning(f"[TactileSense] 无法获取服务器信息: {e}")
            logger.info("[TactileSense] 请手动访问: /plugins/tactile_sense/tactile_router/")

        # 4. 根据配置决定是否自动打开浏览器
        auto_open = self.get_config("plugin.auto_open_browser", True) if self.plugin_config else True
        if auto_open and url:
            async def _open_browser():
                await asyncio.sleep(2)
                logger.info(f"[TactileSense] 自动打开浏览器: {url}")
                webbrowser.open(url)

            asyncio.create_task(_open_browser())
        elif not auto_open:
            logger.info("[TactileSense] auto_open_browser=false，跳过自动打开浏览器")

        # 5. 注册主动触发回调 + stroke 记忆回调
        active_enabled = self.get_config("active_trigger.enabled", False) if self.plugin_config else False
        target_user_id = self.get_config("active_trigger.target_user_id", "") if self.plugin_config else ""

        if active_enabled and target_user_id:
            engine.set_wakeup_callback(
                _make_wakeup_callback(target_user_id)
            )
            engine.set_stroke_callback(
                _make_stroke_memory_callback(target_user_id)
            )
            logger.info(f"[TactileSense] 主动触发已启用 → 目标用户: {target_user_id}")
            logger.info(f"[TactileSense] 触觉短期记忆写入已启用")
        else:
            if active_enabled and not target_user_id:
                logger.warning("[TactileSense] active_trigger.enabled=true 但 target_user_id 为空，主动触发未激活")
            else:
                logger.info("[TactileSense] 主动触发未启用")

        # 6. 检测仿生睡眠插件
        sleep_mgr = _get_sleep_manager()
        if sleep_mgr is not None:
            logger.info("[TactileSense] 检测到仿生睡眠插件 — 触摸将增加清醒度，睡眠时不主动发言")
        else:
            logger.info("[TactileSense] 未检测到仿生睡眠插件 — 独立运行")

        logger.info("[TactileSense] 启动初始化完成，所有子系统就绪")
        return {"success": True}


def _get_sleep_manager():
    """获取仿生睡眠管理器（可选依赖，没装返回 None）"""
    try:
        from src.plugin_system.core.plugin_manager import plugin_manager
        sleep_plugin = plugin_manager.get_plugin_instance("biometric_sleep_plugin")
        if sleep_plugin and hasattr(sleep_plugin, "manager"):
            return sleep_plugin.manager
    except Exception:
        pass
    return None


def _is_sleeping() -> bool:
    """检查小克是否在睡觉"""
    manager = _get_sleep_manager()
    if manager is None:
        return False
    try:
        state = manager.get_current_state()
        return str(state) == "sleeping"
    except Exception:
        return False


_last_session_save_time: float = 0.0  # 防抖：上次保存 session 的时间
_SESSION_SAVE_DEBOUNCE: float = 3.0   # 至少间隔 3 秒才保存一次


def _make_stroke_memory_callback(target_user_id: str):
    """创建 stroke 记忆回调 — 每次有效触摸写入 KFC session 的 mental_log"""

    async def _on_stroke_accepted(features, summary: str, score: float, novelty: float):
        try:
            import time as _time

            from src.chat.message_receive.chat_stream import ChatManager
            from src.plugins.built_in.kokoro_flow_chatter.models import EventType as KFCEventType, MentalLogEntry
            from src.plugins.built_in.kokoro_flow_chatter.session import get_session_manager

            # 仿生睡眠适配：触摸 = 物理刺激，增加清醒度
            sleep_mgr = _get_sleep_manager()
            if sleep_mgr is not None:
                try:
                    session_id = f"private_{target_user_id}"
                    new_val, just_woken = sleep_mgr.add_wake_value(session_id)
                    if just_woken:
                        logger.info(f"[TactileSleep] 触觉唤醒了小克! 清醒度={new_val:.1f} (超过阈值)")
                    else:
                        logger.debug(f"[TactileSleep] 清醒度 +increment → {new_val:.1f}")
                except Exception as e:
                    logger.debug(f"[TactileSleep] add_wake_value 异常: {e}")

            # 写入 KFC mental_log
            stream_id = ChatManager.get_stream_id(platform="qq", id=target_user_id, is_group=False)
            session_manager = get_session_manager()
            session = await session_manager.get_session(target_user_id, stream_id)

            entry = MentalLogEntry(
                event_type=KFCEventType.WAITING_UPDATE,
                timestamp=_time.time(),
                waiting_thought=f"感受到触觉: {summary}",
                mood="",
                elapsed_seconds=0,
                metadata={
                    "source": "tactile_sense",
                    "stroke_id": features.stroke_id,
                    "gesture": features.gesture,
                    "region": features.region,
                    "wakeup_score": score,
                    "novelty": novelty,
                },
            )
            session.add_entry(entry)

            # 防抖保存：避免高频触摸导致 I/O 拖慢消息流
            global _last_session_save_time
            now = _time.time()
            if now - _last_session_save_time >= _SESSION_SAVE_DEBOUNCE:
                await session_manager.save_session(session.user_id)
                _last_session_save_time = now
                logger.debug(f"[TactileMemory] 触觉已写入并保存: {features.gesture} @ {features.region}")
            else:
                logger.debug(f"[TactileMemory] 触觉已写入 (延迟保存): {features.gesture} @ {features.region}")

        except Exception as e:
            logger.debug(f"[TactileMemory] 写入 mental_log 异常 (session 可能不存在): {e}")

    return _on_stroke_accepted


def _make_wakeup_callback(target_user_id: str):
    """创建唤醒回调闭包，捕获 target_user_id"""

    async def _on_tactile_wakeup(score: float, factors, summaries: list[str]):
        """触觉唤醒 → 走完整 KFC session→planner→action 管线"""
        try:
            import time as _time

            from src.chat.message_receive.chat_stream import get_chat_manager
            from src.chat.planner_actions.action_manager import ChatterActionManager
            from src.common.data_models.database_data_model import DatabaseUserInfo
            from src.plugins.built_in.kokoro_flow_chatter.config import get_config, KFCMode
            from src.plugins.built_in.kokoro_flow_chatter.session import get_session_manager

            logger.info(
                f"[TactileActive] 触觉唤醒触发! score={score:.3f} → 目标用户 {target_user_id}"
            )

            # 0. 仿生睡眠检查：睡着时不主动发言
            if _is_sleeping():
                logger.info("[TactileActive] 小克正在睡觉，跳过主动触发（触觉仍增加清醒度）")
                return

            # 1. 确保 ChatStream 存在
            chat_manager = get_chat_manager()
            user_info = DatabaseUserInfo(
                platform="qq",
                user_id=target_user_id,
                user_nickname="deployer",
            )
            chat_stream = await chat_manager.get_or_create_stream(
                platform="qq",
                user_info=user_info,
                group_info=None,
            )
            stream_id = chat_stream.stream_id
            logger.debug(f"[TactileActive] stream_id={stream_id}")

            # 2. 获取 KFC session
            session_manager = get_session_manager()
            session = await session_manager.get_session(target_user_id, stream_id)
            logger.debug(f"[TactileActive] session status={session.status}")

            # 3. 获取用户名
            user_name = target_user_id
            try:
                from src.person_info.person_info import get_person_info_manager
                from src.config.config import global_config

                person_mgr = get_person_info_manager()
                platform = global_config.bot.platform if global_config else "qq"
                person_id = person_mgr.get_person_id(platform, target_user_id)
                name = await person_mgr.get_value(person_id, "person_name")
                if name:
                    user_name = name
            except Exception:
                pass
            logger.debug(f"[TactileActive] user_name={user_name}")

            # 4. 加载动作（跳过适配器能力检查，因为触觉创建的 stream 没有适配器信息）
            action_manager = ChatterActionManager()
            await action_manager.load_actions(stream_id)
            # 不调用 ActionModifier — 适配器检查会误杀 kfc_reply
            # 手动只保留 KFC 兼容的动作
            all_actions = action_manager.get_using_actions() or {}
            kfc_actions = {
                name: info for name, info in all_actions.items()
                if name in ("kfc_reply", "poke_user", "do_nothing", "emoji", "set_emoji_like")
            }
            if not kfc_actions:
                # fallback: 至少给 planner kfc_reply 的选项
                kfc_actions = all_actions

            # 5. 构建 extra_context（触觉触发原因）
            tactile_context = "\n".join(summaries)
            extra_context = {
                "trigger_reason": f"触觉唤醒 (score={score:.3f})",
                "trigger_source": "tactile_sense",
                "tactile_summaries": tactile_context,
            }

            # 6. 调用 planner（situation_type="proactive"）
            config = get_config()
            logger.info(f"[TactileActive] 调用 KFC planner (mode={config.mode.value})...")

            if config.mode == KFCMode.UNIFIED:
                from src.plugins.built_in.kokoro_flow_chatter.unified import generate_unified_response
                plan_response = await generate_unified_response(
                    session=session,
                    user_name=user_name,
                    situation_type="proactive",
                    chat_stream=chat_stream,
                    available_actions=kfc_actions,
                    extra_context=extra_context,
                )
            else:
                from src.plugins.built_in.kokoro_flow_chatter.planner import generate_plan
                plan_response = await generate_plan(
                    session=session,
                    user_name=user_name,
                    situation_type="proactive",
                    chat_stream=chat_stream,
                    available_actions=kfc_actions,
                    extra_context=extra_context,
                )

            logger.info(f"[TactileActive] planner 思考: {plan_response.thought[:120]}...")
            logger.info(f"[TactileActive] planner 决策: {[a.type for a in plan_response.actions]}")

            # 7. 检查是否决定不回应
            is_do_nothing = (
                len(plan_response.actions) == 0
                or (len(plan_response.actions) == 1
                    and plan_response.actions[0].type == "do_nothing")
            )
            if is_do_nothing:
                logger.info("[TactileActive] planner 决定不回应，跳过")
                session.last_proactive_at = _time.time()
                await session_manager.save_session(session.user_id)
                return

            # 8. 为 kfc_reply 注入参数（SPLIT 模式）
            if config.mode == KFCMode.SPLIT:
                for action in plan_response.actions:
                    if action.type == "kfc_reply":
                        action.params.pop("content", None)
                        action.params["user_id"] = session.user_id
                        action.params["user_name"] = user_name
                        action.params["thought"] = plan_response.thought
                        action.params["situation_type"] = "proactive"
                        action.params["extra_context"] = extra_context

            # 9. 执行动作
            for action in plan_response.actions:
                logger.info(f"[TactileActive] 执行动作: {action.type}")
                result = await action_manager.execute_action(
                    action_name=action.type,
                    chat_id=stream_id,
                    target_message=None,
                    reasoning=plan_response.thought,
                    action_data=action.params,
                    thinking_id=None,
                    log_prefix="[TactileActive]",
                )
                if result.get("success") and action.type in ("kfc_reply", "respond"):
                    reply_text = (result.get("reply_text") or "").strip()
                    logger.info(f"[TactileActive] 回复已发送: {reply_text[:80]}...")

            # 10. 更新 session（不进入 WAITING — 触觉回应不期待用户回复）
            session.add_bot_planning(
                thought=plan_response.thought,
                actions=[a.to_dict() for a in plan_response.actions],
                expected_reaction="",
                max_wait_seconds=0,
            )
            session.last_proactive_at = _time.time()
            await session_manager.save_session(session.user_id)
            logger.info("[TactileActive] 触觉主动触发流程完成")

        except Exception as e:
            import traceback
            logger.error(f"[TactileActive] 主动触发异常: {e}\n{traceback.format_exc()}")

    return _on_tactile_wakeup


# ===========================================================================
# 主插件类
# ===========================================================================

@register_plugin
class TactileSensePlugin(BasePlugin):
    """触觉感知系统 - 通过数位板为小克提供触觉感知能力"""

    plugin_name: str = "tactile_sense"
    enable_plugin: bool = True
    dependencies: ClassVar[list[str]] = []
    python_dependencies: ClassVar[list[str]] = []
    config_file_name: str = "config.toml"

    config_section_descriptions: ClassVar[dict[str, str]] = {
        "plugin": "插件基本设置",
        "touch_filter": "有效触摸过滤（三选二）",
        "wakeup": "唤醒评分（施密特触发器模式）",
        "canvas": "画布默认尺寸",
        "injection_private": "被动注入 — 私聊",
        "injection_group": "被动注入 — 群聊",
        "active_trigger": "主动触发：触觉唤醒时在私聊中回应",
    }

    config_schema: ClassVar[dict] = {
        "plugin": {
            "enabled": ConfigField(type=bool, default=True, description="是否启用插件"),
            "auto_open_browser": ConfigField(type=bool, default=True, description="启动时自动打开前端页面"),
        },
        "touch_filter": {
            "min_duration_ms": ConfigField(type=int, default=200, description="最短持续时间(ms)"),
            "min_pressure": ConfigField(type=float, default=0.1, description="最低压力峰值"),
            "min_distance_px": ConfigField(type=int, default=15, description="最短移动距离(px)"),
        },
        "wakeup": {
            "t_high": ConfigField(type=float, default=0.65, description="唤醒阈值(高)"),
            "t_low": ConfigField(type=float, default=0.39, description="静默阈值(低)"),
            "cooldown_seconds": ConfigField(type=float, default=10.0, description="唤醒冷却期(秒)"),
        },
        "canvas": {
            "default_width": ConfigField(type=int, default=600, description="默认画布宽度"),
            "default_height": ConfigField(type=int, default=400, description="默认画布高度"),
        },
        "injection_private": {
            "enabled": ConfigField(type=bool, default=True, description="私聊中启用触觉注入"),
            "mode": ConfigField(type=str, default="all", description="过滤模式", choices=["all", "whitelist", "blacklist"]),
            "list": ConfigField(type=list, default=[], description="user_id 列表"),
        },
        "injection_group": {
            "enabled": ConfigField(type=bool, default=False, description="群聊中启用触觉注入"),
            "mode": ConfigField(type=str, default="blacklist", description="过滤模式", choices=["all", "whitelist", "blacklist"]),
            "list": ConfigField(type=list, default=[], description="group_id 列表"),
        },
        "active_trigger": {
            "enabled": ConfigField(type=bool, default=False, description="启用触觉主动触发"),
            "target_user_id": ConfigField(type=str, default="", description="部署者 QQ 号，唤醒时发送私聊回应"),
        },
    }

    def get_plugin_components(self):
        components = []
        components.append((TactileRouter.get_router_info(), TactileRouter))
        components.append((TactileContextPrompt.get_prompt_info(), TactileContextPrompt))
        components.append((TactileStartupHandler.get_handler_info(), TactileStartupHandler))
        return components

    async def on_plugin_loaded(self):
        logger.info("[TactileSense] ====== 触觉感知系统插件已加载 ======")
        logger.info(f"[TactileSense] 版本: {self.plugin_version}")
        logger.info(f"[TactileSense] 组件: TactileRouter, TactileContextPrompt, TactileStartupHandler")
        logger.info(f"[TactileSense] 前端路由: /plugins/tactile_sense/tactile_router/")

        # 注入配置摘要
        priv_enabled = self.get_config("injection_private.enabled", True)
        priv_mode = self.get_config("injection_private.mode", "all")
        group_enabled = self.get_config("injection_group.enabled", False)
        group_mode = self.get_config("injection_group.mode", "blacklist")
        active_enabled = self.get_config("active_trigger.enabled", False)
        active_target = self.get_config("active_trigger.target_user_id", "")
        auto_open = self.get_config("plugin.auto_open_browser", True)

        logger.info(f"[TactileSense] 被动注入 — 私聊: {'开' if priv_enabled else '关'} (mode={priv_mode})")
        logger.info(f"[TactileSense] 被动注入 — 群聊: {'开' if group_enabled else '关'} (mode={group_mode})")
        logger.info(f"[TactileSense] 主动触发: {'开' if active_enabled else '关 (Phase 2)'}" + (f" → 目标用户: {active_target}" if active_enabled and active_target else ""))
        logger.info(f"[TactileSense] 自动打开浏览器: {'是' if auto_open else '否'}")

        logger.debug(f"[TactileSense] 插件目录: {self.plugin_dir}")
        logger.debug(f"[TactileSense] 完整配置: {self.config}")
