from dataclasses import dataclass
from typing import Literal

import numpy as np

from thesis_project import config

SEED = 42
OBS_DTYPE = np.float32


@dataclass(frozen=True)
class BasisTradingEnvConfig:
    reset_mode: Literal["random", "serial"]
    price_mode: config.PriceMode
    persistence_min: int

    seed: int = SEED
    obs_dtype: type = OBS_DTYPE
    position_encoding: Literal["int", "ohe"] = "int"
    trajectory_min: int | None = None

    def __post_init__(self):
        if not 1 <= self.persistence_min <= 60:
            raise ValueError(f"Invalid persistence value: '{self.persistence_min}'")

        if self.reset_mode == "random" and self.trajectory_min is not None:
            raise ValueError("In random mode trajectory offset must not be specified")

        if self.reset_mode == "serial":
            if self.trajectory_min is None:
                raise ValueError("In serial mode trajectory offset must be specified")

            if self.trajectory_min is not None:
                if not 0 <= self.trajectory_min < self.persistence_min:
                    raise ValueError(
                        f"In serial mode trajectory offset must be between 0 and {self.persistence_min}"
                    )

    @property
    def persistence_sec(self) -> int:
        return self.persistence_min * 60


@dataclass(frozen=True)
class ExecutionEnvConfig:
    reset_mode: Literal["random", "serial"]
    horizon_min: int
    step_sec: int
    max_n_tick: int
    tick_size: float

    seed: int = SEED
    obs_dtype: type = OBS_DTYPE

    def __post_init__(self):
        if self.horizon_min <= 0:
            raise ValueError(f"Invalid execution episode length: '{self.horizon_min}'")

        if self.step_sec <= 0:
            raise ValueError(f"Invalid execution step length: '{self.step_sec}'")

        if self.episode_sec % self.step_sec != 0:
            raise ValueError("Execution episode length must be divisible by execution step length")

        if self.max_n_tick < 1:
            raise ValueError(f"Invalid maximum quote distance: '{self.max_n_tick}'")

        if self.tick_size <= 0:
            raise ValueError(f"Invalid tick size: '{self.tick_size}'")

    @property
    def episode_sec(self) -> int:
        return self.horizon_min * 60

    @property
    def n_steps(self) -> int:
        return self.episode_sec // self.step_sec

    @property
    def n_actions(self) -> int:
        return self.max_n_tick + 1
