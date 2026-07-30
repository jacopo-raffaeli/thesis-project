from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class RLDataset:
    features: pd.DataFrame = field(default_factory=pd.DataFrame)
    basis: pd.Series = field(default_factory=pd.Series)
    ctd_spread: pd.Series | None = None
    fut_spread: pd.Series | None = None


@dataclass(frozen=True)
class EnvConfig:
    include_cost: bool = True
