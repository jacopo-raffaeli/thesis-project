import logging
from pathlib import Path


def setup_logging(log_file: Path | None = None, level: str = "INFO") -> logging.Logger:
    """Configure logging to console and optionally to file."""

    pkg_logger_name = "thesis_project.importance_mi"
    logger = logging.getLogger(pkg_logger_name)
    logger.setLevel(getattr(logging, level))

    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)-23s | %(levelname)-8s | %(filename)-20s:%(lineno)-4d | %(message)s"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(fmt)
        logger.addHandler(console_handler)

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)

    logger.propagate = False

    return logger


def get_logger() -> logging.Logger:
    """Get the logger."""
    return logging.getLogger("thesis_project.importance_mi")
