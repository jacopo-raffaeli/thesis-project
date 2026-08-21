import logging
from pathlib import Path

DEFAULT_LOG_FORMAT = "%(asctime)-23s | " "%(levelname)-8s | " "%(name)-35s | " "%(message)s"


def setup_logging(
    log_file: Path | None = None,
    level: str = "INFO",
) -> None:
    """Configure project-wide logging.

    ## Args
    * log_file:
        Optional path to a log file. If provided, logs are written both
        to the console and to this file.

    * level:
        Minimum logging level. Supported values are the standard logging
        levels, e.g. DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """

    root_logger = logging.getLogger()

    root_logger.handlers.clear()
    root_logger.setLevel(level.upper())

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
