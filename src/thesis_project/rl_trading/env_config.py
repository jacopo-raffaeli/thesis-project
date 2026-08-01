from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class EnvConfig:
    include_cost: bool
    persistence_min: int

    seed: int = 42
    obs_dtype: type = np.float32
    position_encoding: Literal["int", "ohe"] = "int"

    def __post_init__(self):
        if not 1 <= self.persistence_min <= 60:
            raise ValueError(f"Invalid persistence value: '{self.persistence_min}'")

    @property
    def persistence_sec(self) -> int:
        return self.persistence_min * 60
