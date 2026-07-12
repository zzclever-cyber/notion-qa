"""
结构化日志模块
基于 Loguru，支持控制台彩色输出 + JSON 文件持久化

上下文绑定示例：
    from utils.logger import log
    log.bind(request_id="abc123").info("处理请求")
"""
import sys
from pathlib import Path
from loguru import logger as _logger
from config.settings import settings


# Loguru 全局实例
log = _logger


def setup_logger():
    """初始化全局日志配置"""
    _logger.remove()

    # ── 控制台输出（彩色） ──
    _logger.add(
        sys.stderr,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[request_id]:<16}</cyan> | "
            "<level>{message}</level>"
        ),
        level=settings.log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # ── 文件输出（JSON 格式，机器可读） ──
    log_dir = settings.project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    _logger.add(
        log_dir / "rag_agent_{time:YYYY-MM-DD}.log",
        format=(
            '{{"time": "{time:YYYY-MM-DD HH:mm:ss.SSS}", '
            '"level": "{level}", '
            '"request_id": "{extra[request_id]}", '
            '"session_id": "{extra[extra_session_id]}", '
            '"module": "{name}", '
            '"function": "{function}", '
            '"message": "{message}"}}'
        ),
        rotation="00:00",
        retention="30 days",
        level="DEBUG",
        encoding="utf-8",
        serialize=False,  # JSON 字符串格式
    )

    # 设置默认 extra 字段
    _logger.configure(extra={
        "request_id": "-",
        "extra_session_id": "-",
    })

    return _logger
