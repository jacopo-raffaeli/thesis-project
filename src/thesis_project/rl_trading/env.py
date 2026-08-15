from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

from thesis_project.rl_trading.dataset_config import DatasetConfig
from thesis_project.rl_trading.env_config import EnvConfig
from thesis_project.rl_trading.features import FEATURES


@dataclass(frozen=True)
class RLDataset:
    features: pd.DataFrame = field(default_factory=pd.DataFrame)
    basis: pd.Series = field(default_factory=pd.Series)
    ctd_spread: pd.Series | None = None
    fut_spread: pd.Series | None = None

    dates: list[pd.Timestamp] = field(default_factory=list[pd.Timestamp])
    date_to_slice: dict[pd.Timestamp, slice] = field(default_factory=dict[pd.Timestamp, slice])

    def __post_init__(self):
        assert isinstance(self.features.index, pd.DatetimeIndex)
        assert isinstance(self.basis.index, pd.DatetimeIndex)
        assert self.features.index.equals(self.basis.index)

        if self.ctd_spread is not None:
            assert isinstance(self.ctd_spread.index, pd.DatetimeIndex)
            assert self.features.index.equals(self.ctd_spread.index)

        if self.fut_spread is not None:
            assert isinstance(self.fut_spread.index, pd.DatetimeIndex)
            assert self.features.index.equals(self.fut_spread.index)

        dates = self.features.index.floor(freq="D").tz_localize(None)
        unique_dates = dates.unique()
        object.__setattr__(self, "dates", list(unique_dates))

        date_to_slice = {}
        for date in unique_dates:
            rows = np.flatnonzero(dates == date)
            date_to_slice[date] = slice(rows[0], rows[-1] + 1)
        object.__setattr__(self, "date_to_slice", date_to_slice)

    @property
    def n_features(self) -> int:
        return len(self.features.columns)


@dataclass(frozen=True)
class EpDataset:
    features: pd.DataFrame = field(default_factory=pd.DataFrame)
    basis: pd.Series = field(default_factory=pd.Series)
    ctd_spread: pd.Series | None = None
    fut_spread: pd.Series | None = None

    def __post_init__(self):
        assert isinstance(self.features.index, pd.DatetimeIndex)
        assert isinstance(self.basis.index, pd.DatetimeIndex)
        assert self.features.index.equals(self.basis.index)

        if self.ctd_spread is not None:
            assert isinstance(self.ctd_spread.index, pd.DatetimeIndex)
            assert self.features.index.equals(self.ctd_spread.index)

        if self.fut_spread is not None:
            assert isinstance(self.fut_spread.index, pd.DatetimeIndex)
            assert self.features.index.equals(self.fut_spread.index)

    @property
    def n_features(self) -> int:
        return len(self.features.columns)


@dataclass(frozen=True, slots=True)
class StepReward:
    reward: float
    gross_reward: float
    cost: float


class BasisTradingEnv(gym.Env):
    # positions
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

    def __init__(self, dataset: RLDataset, config: EnvConfig):
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

        self.config = config

        # Define observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_space_size,), dtype=self.config.obs_dtype
        )

        # Define action space
        self.n_actions = 3
        self.action_space = gym.spaces.Discrete(self.n_actions)

    # Main
    def reset(self, *, seed: int | None = None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.prev_position = 0
        self.curr_position = 0
        self.reward = 0
        self.gross_reward = 0
        self.cost = 0
        self.terminated = False
        self.truncated = False

        self.date = self._reset_date()
        self.trajectory_min = self._reset_trajectory_min()
        self.t = self.trajectory_sec
        self.ep_dataset = self._build_episode_rl_dataset()

        if self.t >= len(self.ep_dataset.features):
            raise ValueError("Sampled trajectory exceeds episode length")

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.prev_position = self.curr_position
        self.curr_position = self._act_to_pos(action)
        step_reward = self._compute_reward(
            self.t, self.next_t, self.curr_position, self.prev_position
        )

        self.reward = step_reward.reward
        self.gross_reward = step_reward.gross_reward
        self.cost = step_reward.cost
        self.t = self.next_t

        info = {}
        if self._is_last_mrkt_t:
            info["liquidation"] = self._liquidation()

        observation = self._get_observation()
        info.update(self._get_info())

        return observation, self.reward, self.terminated, self.truncated, info

    def close(self):
        pass

    def render(self):
        pass

    # Observation
    def _get_observation(self) -> np.ndarray:
        if not self.terminated:
            observation = self._get_regular_observation()
        else:
            observation = self._get_terminal_observation()

        return observation

    def _get_regular_observation(self) -> np.ndarray:
        assert isinstance(self.ep_dataset, EpDataset)
        features = self.ep_dataset.features.iloc[self.t].to_numpy(dtype=self.config.obs_dtype)
        position = self._encoded_position
        observation = np.concatenate([features, position]).astype(
            dtype=self.config.obs_dtype, copy=False
        )

        return observation

    def _get_terminal_observation(self) -> np.ndarray:
        return np.zeros(self.obs_space_size, dtype=self.config.obs_dtype)

    # Info
    def _get_info(self) -> dict[str, Any]:
        return {
            "step": self.t,
            "time": self.timestamp,
            "date": self.date,
            "steps_to_go": self.steps_to_go,
            "allocation": self.prev_position,
            "position": self.curr_position,
            "reward": self.reward,
            "gross_reward": self.gross_reward,
            "cost": self.cost,
            "terminal": self.terminated,
        }

    def _get_terminal_info(self, closing_reward: StepReward) -> dict[str, Any]:
        return {
            "terminal_allocation": self.curr_position,
            "terminal_position": self._FLAT_POSITION,
            "agent_reward": self.reward,
            "agent_gross_reward": self.gross_reward,
            "agent_cost": self.cost,
            "closing_reward": closing_reward.reward,
            "closing_gross_reward": closing_reward.gross_reward,
            "closing_cost": closing_reward.cost,
        }

    # Rewards
    def _compute_gross_reward(self, t: int, next_t: int, position: int) -> float:
        assert isinstance(self.ep_dataset, EpDataset)
        basis = self.ep_dataset.basis
        delta = basis.iloc[next_t] - basis.iloc[t]

        return position * delta

    def _compute_cost(self, t: int, position: int, allocation: int) -> float:
        assert isinstance(self.ep_dataset, EpDataset)
        ctd_spread = self.ep_dataset.ctd_spread
        fut_spread = self.ep_dataset.fut_spread
        assert isinstance(ctd_spread, pd.Series)
        assert isinstance(fut_spread, pd.Series)

        size = abs(position - allocation)
        spread = 0.5 * (ctd_spread.iloc[t] + fut_spread.iloc[t])

        return size * spread

    def _compute_reward(self, t: int, next_t: int, position: int, allocation: int) -> StepReward:
        gross_reward = self._compute_gross_reward(t, next_t, position)
        cost = 0.0
        if self.config.include_cost:
            cost = self._compute_cost(t, position, allocation)
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
        features = self.dataset.features.iloc[rows]
        basis = self.dataset.basis.iloc[rows]

        ctd_spread = None
        fut_spread = None
        if self.config.include_cost:
            assert isinstance(self.dataset.ctd_spread, pd.Series)
            assert isinstance(self.dataset.fut_spread, pd.Series)
            ctd_spread = self.dataset.ctd_spread.iloc[rows]
            fut_spread = self.dataset.fut_spread.iloc[rows]

        return EpDataset(
            features=features, basis=basis, ctd_spread=ctd_spread, fut_spread=fut_spread
        )

    # Terminal
    def _liquidation(self) -> dict[str, Any]:
        closing_reward = self._compute_reward(
            self.t, self.mrkt_close_t, self._FLAT_POSITION, self.curr_position
        )
        closing_info = self._get_terminal_info(closing_reward)
        self.reward += closing_reward.reward
        self.gross_reward += closing_reward.gross_reward
        self.cost += closing_reward.cost
        self.terminated = True

        return closing_info

    # Reset
    def _reset_date(self) -> pd.Timestamp:
        match self.config.mode:
            case "random":
                return self._random_date()

            case "serial":
                return self._serial_date()

            case _:
                raise ValueError(f"Invalid evaluation mode: '{self.config.mode}'")

    def _random_date(self) -> pd.Timestamp:
        return self.np_random.choice(np.array(self.dataset.dates, dtype="datetime64[ns]"))

    def _serial_date(self) -> pd.Timestamp:
        self._date_idx += 1
        if self._date_idx == len(self.dataset.dates):
            raise StopIteration("Evaluation complete")

        return self.dataset.dates[self._date_idx]

    def _reset_trajectory_min(self) -> int:
        match self.config.mode:
            case "random":
                return self._random_trajectory()

            case "serial":
                return self._serial_trajectory()

            case _:
                raise ValueError(f"Invalid evalutation mode: '{self.config.mode}'")

    def _random_trajectory(self) -> int:
        return int(self.np_random.integers(self.config.persistence_min))

    def _serial_trajectory(self) -> int:
        assert isinstance(self.config.trajectory_min, int)
        return self.config.trajectory_min

    @property
    def _encoded_position(self):
        match self.config.position_encoding:
            case "int":
                return np.array([self.curr_position], dtype=self.config.obs_dtype)

            case "ohe":
                return np.eye(self.n_actions, dtype=self.config.obs_dtype)[
                    self._pos_to_act(self.curr_position)
                ]

            case _:
                raise ValueError(f"Invalid position encoding '{self.config.position_encoding}'")

    @property
    def obs_space_size(self) -> int:
        n = self.dataset.n_features

        match self.config.position_encoding:
            case "int":
                n += 1

            case "ohe":
                n += self.n_actions

            case _:
                raise ValueError(f"Invalid position encoding '{self.config.position_encoding}'")

        return n

    @property
    def trajectory_sec(self) -> int:
        return self.trajectory_min * 60

    @property
    def next_t(self) -> int:
        return self.t + self.config.persistence_sec

    @property
    def offset_to_close_min(self) -> int:
        return self.config.persistence_min - self.trajectory_min

    @property
    def offset_to_close_sec(self) -> int:
        return self.offset_to_close_min * 60

    @property
    def mrkt_close_t(self) -> int:
        assert isinstance(self.ep_dataset, EpDataset)
        return self.episode_length - 1

    @property
    def last_mrkt_t(self) -> int:
        assert isinstance(self.ep_dataset, EpDataset)
        return self.mrkt_close_t - self.offset_to_close_sec

    @property
    def last_agent_t(self) -> int:
        assert isinstance(self.ep_dataset, EpDataset)
        return self.last_mrkt_t - self.config.persistence_sec

    @property
    def _is_last_mrkt_t(self) -> bool:
        return self.t == self.last_mrkt_t

    @property
    def timestamp(self):
        assert isinstance(self.ep_dataset, EpDataset)
        return self.ep_dataset.features.index[self.t]

    @property
    def next_timestamp(self):
        assert isinstance(self.ep_dataset, EpDataset)
        return self.ep_dataset.features.index[self.next_t]

    @property
    def episode_length(self) -> int:
        assert isinstance(self.ep_dataset, EpDataset)
        return len(self.ep_dataset.features)

    @property
    def steps_to_go(self) -> int:
        return self.last_agent_t - self.t


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    from thesis_project.rl_trading.dataset import build_rl_dataset

    dataset_config = DatasetConfig(
        ticker="fbtp",
    )

    env_config = EnvConfig(mode="random", include_cost=True, persistence_min=10)

    rl_dataset = build_rl_dataset(dataset_config, env_config, FEATURES)

    env = BasisTradingEnv(rl_dataset, env_config)
    check_env(env, skip_render_check=True, skip_close_check=True)
    print("Env checked!")
