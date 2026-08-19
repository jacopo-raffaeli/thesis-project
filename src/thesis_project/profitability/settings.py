from dataclasses import dataclass
from typing import Literal

import pandas as pd

from thesis_project import config

CTD_FACE_VALUE = config.ASSETS["btp"].contract_size
FUT_FACE_VALUE = config.ASSETS["fbtp"].contract_size
SAMPLING_INTERVAL = pd.Timedelta(config.LOB.freq)

PriceMode = Literal["mid", "quoted"]
FutContractMode = Literal["frac", "round"]
LiquidityMode = Literal["ignore", "level", "lob"]


@dataclass(frozen=True)
class AnalysisConfig:
    price_mode: PriceMode
    fut_contract_mode: FutContractMode
    liquidity_mode: LiquidityMode

    @property
    def name(self) -> str:
        return f"{self.price_mode}_price_{self.fut_contract_mode}_contract_{self.liquidity_mode}_liquidity"
