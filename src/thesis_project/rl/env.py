import os
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from thesis_project import config
from thesis_project.rl.dataset import DatasetConfig
from thesis_project.rl.env_config import EnvConfig

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


@dataclass(frozen=True)
class RLDataset:
    # Features
    market_features: pd.DataFrame
    calendar_features: pd.DataFrame
    calendar_enc_features: pd.DataFrame

    # Market
    ctd_mid: pd.Series = field(default_factory=pd.Series)
    fut_mid: pd.Series = field(default_factory=pd.Series)
    ctd_contracts: float = field(default_factory=float)
    fut_contracts: pd.Series = field(default_factory=pd.Series)
    ctd_spread: pd.Series = field(default_factory=pd.Series)
    fut_spread: pd.Series = field(default_factory=pd.Series)

    # Temporal
    dates: list[pd.Timestamp] = field(default_factory=list[pd.Timestamp])
    date_to_slice: dict[pd.Timestamp, slice] = field(default_factory=dict[pd.Timestamp, slice])

    def __post_init__(self):
        # Check series consistency
        assert isinstance(self.market_features.index, pd.DatetimeIndex)
        assert isinstance(self.calendar_features.index, pd.DatetimeIndex)
        assert isinstance(self.calendar_enc_features.index, pd.DatetimeIndex)
        assert isinstance(self.ctd_mid.index, pd.DatetimeIndex)
        assert isinstance(self.fut_mid.index, pd.DatetimeIndex)
        assert isinstance(self.ctd_spread.index, pd.DatetimeIndex)
        assert isinstance(self.fut_spread.index, pd.DatetimeIndex)
        assert self.market_features.index.equals(self.calendar_features.index)
        assert self.market_features.index.equals(self.calendar_enc_features.index)
        assert self.market_features.index.equals(self.ctd_mid.index)
        assert self.market_features.index.equals(self.fut_mid.index)
        assert self.market_features.index.equals(self.ctd_spread.index)
        assert self.market_features.index.equals(self.fut_spread.index)

        # Find unique dates
        dates_idx = self.market_features.index.tz_localize(None).floor("D")
        dates = dates_idx.unique()
        object.__setattr__(self, "dates", list(dates))

        # Map dates to indexes
        date_to_slice = {}
        for date in dates:
            rows = np.flatnonzero(dates_idx == date)
            date_to_slice[date] = slice(rows[0], rows[-1] + 1)
        object.__setattr__(self, "date_to_slice", date_to_slice)

    @property
    def n_market_features(self) -> int:
        return len(self.market_features.columns)

    @property
    def n_calendar_features(self) -> int:
        return len(self.calendar_features.columns)

    @property
    def n_calendar_enc_features(self) -> int:
        return len(self.calendar_enc_features.columns)

    @property
    def n_features(self) -> int:
        return self.n_market_features + self.n_calendar_features + self.n_calendar_enc_features

    @property
    def n_dates(self) -> int:
        return len(self.dates)

    @property
    def ctd_bid(self) -> pd.Series:
        return self.ctd_mid - (self.ctd_spread / 2)

    @property
    def ctd_ask(self) -> pd.Series:
        return self.ctd_mid + (self.ctd_spread / 2)

    @property
    def fut_bid(self) -> pd.Series:
        return self.fut_mid - (self.fut_spread / 2)

    @property
    def fut_ask(self) -> pd.Series:
        return self.fut_mid + (self.fut_spread / 2)


@dataclass(frozen=True)
class EpDataset:
    # Features
    market_features: pd.DataFrame
    calendar_features: pd.DataFrame
    calendar_enc_features: pd.DataFrame

    # Market
    ctd_mid: pd.Series = field(default_factory=pd.Series)
    fut_mid: pd.Series = field(default_factory=pd.Series)
    ctd_contracts: float = field(default_factory=float)
    fut_contracts: float = field(default_factory=float)
    ctd_spread: pd.Series = field(default_factory=pd.Series)
    fut_spread: pd.Series = field(default_factory=pd.Series)

    def __post_init__(self):
        # Check series consistency
        assert isinstance(self.market_features.index, pd.DatetimeIndex)
        assert isinstance(self.calendar_features.index, pd.DatetimeIndex)
        assert isinstance(self.calendar_enc_features.index, pd.DatetimeIndex)
        assert isinstance(self.ctd_mid.index, pd.DatetimeIndex)
        assert isinstance(self.fut_mid.index, pd.DatetimeIndex)
        assert isinstance(self.ctd_spread.index, pd.DatetimeIndex)
        assert isinstance(self.fut_spread.index, pd.DatetimeIndex)
        assert self.market_features.index.equals(self.calendar_features.index)
        assert self.market_features.index.equals(self.calendar_enc_features.index)
        assert self.market_features.index.equals(self.ctd_mid.index)
        assert self.market_features.index.equals(self.fut_mid.index)
        assert self.market_features.index.equals(self.ctd_spread.index)
        assert self.market_features.index.equals(self.fut_spread.index)

    @property
    def n_market_features(self) -> int:
        return len(self.market_features.columns)

    @property
    def n_calendar_features(self) -> int:
        return len(self.calendar_features.columns)

    @property
    def n_calendar_enc_features(self) -> int:
        return len(self.calendar_enc_features.columns)

    @property
    def n_features(self) -> int:
        return self.n_market_features + self.n_calendar_features + self.n_calendar_enc_features

    @property
    def episode_length(self) -> int:
        return len(self.market_features)

    @property
    def ctd_bid(self) -> pd.Series:
        return self.ctd_mid - (self.ctd_spread / 2)

    @property
    def ctd_ask(self) -> pd.Series:
        return self.ctd_mid + (self.ctd_spread / 2)

    @property
    def fut_bid(self) -> pd.Series:
        return self.fut_mid - (self.fut_spread / 2)

    @property
    def fut_ask(self) -> pd.Series:
        return self.fut_mid + (self.fut_spread / 2)


@dataclass(frozen=True, slots=True)
class StepReward:
    reward: float
    gross_reward: float
    cost: float


class BasisTradingEnv(gym.Env):
    # positions
    _N_ACTIONS = 3
    _SHORT_POSITION = -1
    _FLAT_POSITION = 0
    _LONG_POSITION = +1

    # action -> direction dictionary
    _ACT_TO_POS = {
        0: _SHORT_POSITION,  # Short
        1: _FLAT_POSITION,  # Flat
        2: _LONG_POSITION,  # Long
    }

    # direction -> action dictionary
    _POS_TO_ACT = {
        _SHORT_POSITION: 0,  # Short
        _FLAT_POSITION: 1,  # Flat
        _LONG_POSITION: 2,  # Long
    }

    # actions masking
    _FULL_MASK = np.array([True, True, True], dtype=bool)
    _FLAT_MASK = np.array([False, True, False], dtype=bool)

    def __init__(self, dataset: RLDataset, env_config: EnvConfig):
        super().__init__()

        self.date: pd.Timestamp
        self._date_idx: int = -1
        self.trajectory_min: int
        self.t: int

        self.prev_position: int
        self.curr_position: int
        self.reward: float
        self.gross_reward: float
        self.cost: float

        self.terminated: bool
        self.truncated: bool

        self.dataset = dataset
        self.ep_dataset: EpDataset | None = None

        self.config = env_config

        # Define observation spaces
        spaces = {
            "market": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(dataset.n_market_features,),
                dtype=env_config.obs_dtype,
            ),
            "position": self._position_space,
        }

        if dataset.n_calendar_features > 0:
            spaces["temporal"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(dataset.n_calendar_features,),
                dtype=env_config.obs_dtype,
            )

        if dataset.n_calendar_enc_features > 0:
            spaces["temporal_enc"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(dataset.n_calendar_enc_features,),
                dtype=env_config.obs_dtype,
            )

        # Define observation space
        self.observation_space = gym.spaces.Dict(spaces)

        # Define action space
        self.action_space = gym.spaces.Discrete(self._N_ACTIONS)

    # Main
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)

        # Reset arguments
        self.prev_position = 0
        self.curr_position = 0
        self.reward = 0
        self.gross_reward = 0
        self.cost = 0
        self.terminated = False
        self.truncated = False

        # Reset date and trajectory
        self.date = self._reset_date()
        self.trajectory_min = self._reset_trajectory()
        self.t = self._trajectory_sec
        self.ep_dataset = self._build_episode_rl_dataset()

        if self.t >= self.ep_dataset.episode_length:
            raise ValueError("Sampled trajectory exceeds episode length")

        # Get data
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int):
        # Update positions
        self.prev_position = self.curr_position
        self.curr_position = self._act_to_pos(action)

        # Compute reward
        step_reward = self._compute_reward(
            self.t, self._next_t, self.curr_position, self.prev_position
        )

        # Update reward
        self.reward = step_reward.reward
        self.gross_reward = step_reward.gross_reward
        self.cost = step_reward.cost

        # Update index
        self.t = self._next_t
        self.terminated = self._is_last_mrkt_t

        # Get data
        observation = self._get_observation()
        info = self._get_info()

        return observation, self.reward, self.terminated, self.truncated, info

    def close(self):
        pass

    def render(self):
        pass

    def action_masks(self) -> np.ndarray:
        if self._is_last_agent_t:
            return self._FLAT_MASK.copy()

        return self._FULL_MASK.copy()

    # Observation
    def _get_observation(self) -> dict[str, np.ndarray]:
        assert isinstance(self.ep_dataset, EpDataset)

        observation = {
            "market": self.ep_dataset.market_features.iloc[self.t].to_numpy(
                dtype=self.config.obs_dtype
            ),
            "position": self._encoded_position,
        }

        if self.ep_dataset.n_calendar_features > 0:
            observation["temporal"] = self.ep_dataset.calendar_features.iloc[self.t].to_numpy(
                dtype=self.config.obs_dtype
            )

        if self.ep_dataset.n_calendar_enc_features > 0:
            observation["temporal_enc"] = self.ep_dataset.calendar_enc_features.iloc[
                self.t
            ].to_numpy(dtype=self.config.obs_dtype)

        return observation

    # Info
    def _get_info(self) -> dict[str, Any]:
        assert isinstance(self.ep_dataset, EpDataset)
        return {
            # Episode
            "timestamp": self._timestamp,
            "step": self.t,
            "steps_to_go": self._steps_to_go,
            "terminal": self.terminated,
            # Portfolio
            "allocation": self.prev_position,
            "position": self.curr_position,
            # Reward
            "reward": self.reward,
            "gross_reward": self.gross_reward,
            "cost": self.cost,
            # CTD
            "ctd_mid": self._compute_ctd_mid(self.t),
            "ctd_spread": self._compute_ctd_spread(self.t),
            "ctd_contracts": self.ep_dataset.ctd_contracts,
            # FUT
            "fut_mid": self._compute_fut_mid(self.t),
            "fut_spread": self._compute_fut_spread(self.t),
            "fut_contracts": self.ep_dataset.fut_contracts,
            # Basis
            "basis_mid": self._compute_basis_mid(self.t),
            "basis_spread": self._compute_basis_spread(self.t),
        }

    # Reward
    def _compute_ctd_mid(self, t: int) -> float:
        assert isinstance(self.ep_dataset, EpDataset)

        return (config.BTP.contract_size / 100) * self.ep_dataset.ctd_mid.iloc[t]

    def _compute_fut_mid(self, t: int) -> float:
        assert isinstance(self.ep_dataset, EpDataset)

        return (config.FBTP.contract_size / 100) * self.ep_dataset.fut_mid.iloc[t]

    def _compute_ctd_spread(self, t: int) -> float:
        assert isinstance(self.ep_dataset, EpDataset)

        return (config.BTP.contract_size / 100) * self.ep_dataset.ctd_spread.iloc[t]

    def _compute_fut_spread(self, t: int) -> float:
        assert isinstance(self.ep_dataset, EpDataset)

        return (config.FBTP.contract_size / 100) * self.ep_dataset.fut_spread.iloc[t]

    def _compute_basis_mid(self, t: int) -> float:
        assert isinstance(self.ep_dataset, EpDataset)

        ctd_value = self.ep_dataset.ctd_contracts * self._compute_ctd_mid(t)

        fut_value = self.ep_dataset.fut_contracts * self._compute_fut_mid(t)

        return ctd_value - fut_value

    def _compute_basis_spread(self, t: int) -> float:
        assert isinstance(self.ep_dataset, EpDataset)

        ctd_spread = self.ep_dataset.ctd_contracts * self._compute_ctd_spread(t)

        fut_spread = self.ep_dataset.fut_contracts * self._compute_fut_spread(t)

        return ctd_spread + fut_spread

    def _compute_gross_reward(self, t: int, next_t: int, position: int) -> float:
        assert isinstance(self.ep_dataset, EpDataset)
        delta = self._compute_basis_mid(next_t) - self._compute_basis_mid(t)

        return position * delta

    def _compute_cost(self, t: int, curr_position: int, prev_position: int) -> float:
        spread = 0.5 * self._compute_basis_spread(t)
        size = abs(curr_position - prev_position)

        return size * spread

    def _compute_reward(
        self, t: int, next_t: int, curr_position: int, prev_position: int
    ) -> StepReward:
        gross_reward = self._compute_gross_reward(t, next_t, curr_position)
        cost = 0.0
        if self.config.price_mode == "quoted":
            cost = self._compute_cost(t, curr_position, prev_position)
        reward = gross_reward - cost

        return StepReward(reward, gross_reward, cost)

    # action <-> position
    def _act_to_pos(self, action: int) -> int:
        return self._ACT_TO_POS[action]

    def _pos_to_act(self, position: int) -> int:
        return self._POS_TO_ACT[position]

    # Episode
    def _build_episode_rl_dataset(self) -> EpDataset:
        rows = self.dataset.date_to_slice[self.date]
        market_features = self.dataset.market_features.iloc[rows]
        calendar_features = self.dataset.calendar_features.iloc[rows]
        calendar_enc_features = self.dataset.calendar_enc_features.iloc[rows]
        ctd_mid = self.dataset.ctd_mid.iloc[rows]
        fut_mid = self.dataset.fut_mid.iloc[rows]
        ctd_contracts = self.dataset.ctd_contracts
        fut_contracts = self.dataset.fut_contracts[self.date]
        ctd_spread = self.dataset.ctd_spread.iloc[rows]
        fut_spread = self.dataset.fut_spread.iloc[rows]

        return EpDataset(
            market_features=market_features,
            calendar_features=calendar_features,
            calendar_enc_features=calendar_enc_features,
            ctd_mid=ctd_mid,
            fut_mid=fut_mid,
            ctd_contracts=ctd_contracts,
            fut_contracts=fut_contracts,
            ctd_spread=ctd_spread,
            fut_spread=fut_spread,
        )

    # Reset date
    def _reset_date_random(self) -> pd.Timestamp:
        return self.np_random.choice(np.array(self.dataset.dates, dtype="datetime64[ns]"))

    def _reset_date_serial(self) -> pd.Timestamp:
        self._date_idx = (self._date_idx + 1) % self.dataset.n_dates
        return self.dataset.dates[self._date_idx]

    def _reset_date(self) -> pd.Timestamp:
        match self.config.reset_mode:
            case "random":
                return self._reset_date_random()

            case "serial":
                return self._reset_date_serial()

            case _:
                raise ValueError(f"Invalid evaluation mode: '{self.config.reset_mode}'")

    def init_serial_date(self):
        if self.config.reset_mode != "serial":
            raise ValueError("This method can be called only in serial mode")

        self._date_idx = -1

    # Reset trajectory
    def _reset_trajectory_random(self) -> int:
        return int(self.np_random.integers(self.config.persistence_min))

    def _reset_trajectory_serial(self) -> int:
        assert isinstance(self.config.trajectory_min, int)
        return self.config.trajectory_min

    def _reset_trajectory(self) -> int:
        match self.config.reset_mode:
            case "random":
                return self._reset_trajectory_random()

            case "serial":
                return self._reset_trajectory_serial()

            case _:
                raise ValueError(f"Invalid evalutation mode: '{self.config.reset_mode}'")

    # Actions encoding utilities
    @property
    def _encoded_position(self):
        match self.config.position_encoding:
            case "int":
                return np.array([self.curr_position], dtype=self.config.obs_dtype)

            case "ohe":
                return np.eye(self._N_ACTIONS, dtype=self.config.obs_dtype)[
                    self._pos_to_act(self.curr_position)
                ]

            case _:
                raise ValueError(f"Invalid position encoding '{self.config.position_encoding}'")

    @property
    def _position_space(self) -> gym.Space:
        match self.config.position_encoding:
            case "int":
                return gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(1,),
                    dtype=self.config.obs_dtype,
                )

            case "ohe":
                return gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self._N_ACTIONS,),
                    dtype=self.config.obs_dtype,
                )

            case _:
                raise ValueError(f"Invalid position encoding '{self.config.position_encoding}'")

    # General purpose utilities
    @property
    def _next_t(self) -> int:
        return min(self.t + self.config.persistence_sec, self._last_mrkt_t)

    @property
    def _timestamp(self) -> pd.Timestamp:
        assert isinstance(self.ep_dataset, EpDataset)
        return self.ep_dataset.market_features.index[self.t]

    @property
    def _trajectory_sec(self) -> int:
        return self.trajectory_min * 60

    @property
    def _episode_length(self) -> int:
        assert isinstance(self.ep_dataset, EpDataset)
        return len(self.ep_dataset.market_features)

    # Terminal index
    @property
    def _last_mrkt_t(self) -> int:
        return self._episode_length - 1

    @property
    def _last_agent_t(self) -> int:
        n_steps = (self._last_mrkt_t - self._trajectory_sec - 1) // self.config.persistence_sec

        return self._trajectory_sec + n_steps * self.config.persistence_sec

    @property
    def _is_last_mrkt_t(self) -> bool:
        return self.t == self._last_mrkt_t

    @property
    def _is_last_agent_t(self) -> bool:
        return self.t == self._last_agent_t

    @property
    def _steps_to_go(self) -> int:
        steps_to_go = (self._last_agent_t - self.t) // self.config.persistence_sec
        return max(0, steps_to_go)


if __name__ == "__main__":
    import warnings

    from gymnasium.utils.env_checker import check_env

    from thesis_project.rl.dataset import build_rl_dataset

    warnings.filterwarnings(
        "ignore",
        message=r"WARN: A Box observation space minimum value is -infinity.*",
        category=UserWarning,
    )

    warnings.filterwarnings(
        "ignore",
        message=r"WARN: A Box observation space maximum value is infinity.*",
        category=UserWarning,
    )

    dataset_config = DatasetConfig(
        ticker="fbtp",
        contract_mode="round",
        market_set="xgb_cls",
        calendar_set="default",
        calendar_enc_set="default",
    )

    env_config = EnvConfig(
        price_mode="quoted",
        reset_mode="random",
        persistence_min=10,
    )

    rl_dataset = build_rl_dataset(dataset_config)

    env = BasisTradingEnv(rl_dataset, env_config)
    check_env(env, skip_render_check=True, skip_close_check=True)
    print("Env checked!")
