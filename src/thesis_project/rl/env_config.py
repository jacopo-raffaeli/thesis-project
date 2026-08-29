from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class EnvConfig:
    mode: Literal["random", "serial"]
    include_cost: bool
    persistence_min: int

    seed: int = 42
    obs_dtype: type = np.float32
    position_encoding: Literal["int", "ohe"] = "int"
    trajectory_min: int | None = None

    def __post_init__(self):
        if not 1 <= self.persistence_min <= 60:
            raise ValueError(f"Invalid persistence value: '{self.persistence_min}'")

        if self.mode == "random" and self.trajectory_min is not None:
            raise ValueError("In random mode trajectory offset must not be specified")

        if self.mode == "serial" and self.trajectory_min is None:
            raise ValueError("In serial mode trajectory offset must be specified")

        if self.mode == "serial" and self.trajectory_min is not None:
            if not 0 <= self.trajectory_min < self.persistence_min:
                raise ValueError(
                    f"In serial mode trajectory offset must be between 0 and {self.persistence_min}"
                )

    @property
    def persistence_sec(self) -> int:
        return self.persistence_min * 60
