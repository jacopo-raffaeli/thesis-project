from dataclasses import dataclass, field

from thesis_project import config as global_config


@dataclass(frozen=True)
class DatasetConfig:
    ticker: str

    min_time: str
    max_time: str

    offsets: dict[str, tuple[int, int]] = field(default_factory=dict[str, tuple[int, int]])

    def __post_init__(self):
        if self.ticker not in global_config.VALID_TICKERS:
            raise ValueError("Ticker is not valid")
