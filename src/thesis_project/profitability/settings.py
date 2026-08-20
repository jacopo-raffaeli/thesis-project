from dataclasses import dataclass
from typing import Literal

import pandas as pd

from thesis_project import config

CTD_FACE_VALUE = config.ASSETS["btp"].contract_size
FUT_FACE_VALUE = config.ASSETS["fbtp"].contract_size
SAMPLING_INTERVAL = pd.Timedelta(config.LOB.freq)

FutContractMode = Literal["frac", "round"]
PriceMode = Literal["mid", "quoted"]
VolumeMode = Literal["ignore", "level", "lob"]


@dataclass(frozen=True)
class AnalysisConfig:
    price_mode: PriceMode
    volume_modes: tuple[VolumeMode, ...]
    fut_contract_mode: FutContractMode

    @property
    def key(self) -> tuple[PriceMode, FutContractMode]:
        return (self.price_mode, self.fut_contract_mode)
