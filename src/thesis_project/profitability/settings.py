from dataclasses import dataclass

import pandas as pd

from thesis_project import config
from thesis_project.config import ContractMode, PriceMode, VolumeMode

CTD_FACE_VALUE = config.ASSETS["btp"].contract_size
FUT_FACE_VALUE = config.ASSETS["fbtp"].contract_size
SAMPLING_INTERVAL = pd.Timedelta(config.LOB.freq)


@dataclass(frozen=True)
class AnalysisConfig:
    price_mode: PriceMode
    volume_modes: tuple[VolumeMode, ...]
    fut_contract_mode: ContractMode

    @property
    def key(self) -> tuple[PriceMode, ContractMode]:
        return (self.price_mode, self.fut_contract_mode)
