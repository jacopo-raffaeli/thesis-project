from dataclasses import dataclass
from pathlib import Path

from thesis_project import config


@dataclass
class ExcludedDates:
    ticker: config.FutTicker
    role: config.AssetRole

    ROOT: Path = config.DATA_PRO_DIR
    FILENAME: str = "excluded_dates.?"

    def __post_init__(self):
        if not self.path.exists():
            # TODO: Create the file
            ...

    @property
    def path(self) -> Path:
        return self.ROOT / self.ticker / self.role / self.FILENAME

    def load_dates(self): ...
