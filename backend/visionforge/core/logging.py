"""VisionForge Professional Logging System."""

import logging
import sys

from visionforge.core.config import get_settings


class ColoredConsoleFormatter(logging.Formatter):
    """Custom log formatter with ANSI color codes for interactive terminal output."""

    COLOR_CODES = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[1;31m",  # Bold Red
    }
    RESET_CODE = "\033[0m"
    BOLD_CODE = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with level-specific color and timestamp."""
        color = self.COLOR_CODES.get(record.levelno, self.RESET_CODE)
        levelname = record.levelname.ljust(8)

        # Standard log message components
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        record_name = f"{self.BOLD_CODE}{record.name}{self.RESET_CODE}"

        formatted_msg = (
            f"[{timestamp}] {color}{levelname}{self.RESET_CODE} "
            f"[{record_name}] {record.getMessage()}"
        )

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            formatted_msg += f"\n{record.exc_text}"

        return formatted_msg


def setup_logging(level: str | None = None) -> logging.Logger:
    """Configure structured console logging for VisionForge backend."""
    settings = get_settings()
    log_level = level or settings.log_level

    logger = logging.getLogger("visionforge")
    logger.setLevel(log_level.upper())

    # Avoid adding duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level.upper())
        formatter = ColoredConsoleFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


def get_logger(name: str = "visionforge") -> logging.Logger:
    """Get a logger instance scoped under the visionforge hierarchy."""
    if not name.startswith("visionforge"):
        name = f"visionforge.{name}"
    return logging.getLogger(name)
