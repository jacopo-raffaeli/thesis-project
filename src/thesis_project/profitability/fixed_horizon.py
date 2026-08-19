import datetime
from dataclasses import dataclass

from thesis_project import config
from thesis_project.profitability import settings


@dataclass(frozen=True)
class FixedHorizonConfig:
    min_time: datetime.time
    max_time: datetime.time
    ticker: config.FutTicker
    ctd_contracts: int
    horizons: list[int]
    tolerance: int
    analyses: list[settings.AnalysisConfig]
