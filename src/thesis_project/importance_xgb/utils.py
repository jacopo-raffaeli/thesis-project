import json
import logging
from pathlib import Path

from thesis_project import config as global_config
from thesis_project.importance_xgb import config, data_paths

logger = logging.getLogger(__name__)


def setup_logging(log_file: Path | None = None, level: str = "INFO") -> logging.Logger:
    """Configure logging to console and optionally to file."""

    pkg_logger_name = "thesis_project.importance_xgb"
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
    return logging.getLogger("thesis_project.importance_xgb")


def generate_output_path(config_obj: config.AnalysisConfig) -> Path:
    output_path = global_config.DATA_INT_DIR / config_obj.ticker / config_obj.base_dir / "runs"
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path


def generate_run_path(config_obj: config.AnalysisConfig) -> Path:
    output_path = generate_output_path(config_obj)
    run_number = generate_run_number(config_obj)
    run_path = output_path / f"run_{run_number:03d}"
    run_path.mkdir(parents=True, exist_ok=True)

    return run_path


def generate_run_number(config_obj: config.AnalysisConfig) -> int:
    output_path = generate_output_path(config_obj)

    existing_runs = []
    for item in output_path.iterdir():
        if item.is_dir():
            try:
                item_num = item.name.split("_")[-1]
                existing_runs.append(int(item_num))
            except ValueError:
                pass

    return max(existing_runs) + 1 if existing_runs else 1


def save_config_snapshot(
    config: config.AnalysisConfig,
    data_paths: data_paths.DataPathsConfig,
    run_dir: Path,
) -> None:
    """Save config and data paths to JSON for reproducibility."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # Convert config to dict
    data_path_dict = {
        "ticker": data_paths.ticker,
        "target": data_paths.target,
        "features": {k: str(v) for k, v in data_paths.features.items()},
    }

    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    data_path = run_dir / "data_paths.json"
    with open(data_path, "w") as f:
        json.dump(data_path_dict, f, indent=2)

    logger.info("Configuration settings saved: %s", config_path.relative_to(global_config.ROOT))
