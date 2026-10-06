import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import pandas as pd

from thesis_project import config
from thesis_project.rl.dataset import DatasetConfig, EpDataset, RLDataset
from thesis_project.rl.env_config import ExecutionEnvConfig

ExecutionStatus = Literal[
    "market",
    "limit",
    "forced",
]


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    side: config.LobSide
    date: datetime.date
    time: datetime.time
    quantity: float

    def __post_init__(self):
        if self.quantity <= 0:
            raise ValueError(f"Execution quantity must be positive: '{self.quantity}'")


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    side: config.LobSide
    quantity: float
    status: ExecutionStatus
    limit_price: float | None
    execution_price: float
    execution_time: pd.Timestamp


class ExecutionEnv(gym.Env, ABC):
    def __init__(
        self,
        dataset: RLDataset,
        env_config: ExecutionEnvConfig,
    ):
        super().__init__()

        self._opening_time: pd.Timestamp | None = None
        self._current_time: pd.Timestamp | None = None

        self.t: int = 0
        self.steps_remaining: int = 0
        self.quantity: float = 0.0

        self.benchmark_price: float = 0.0
        self.limit_price: float | None = None
        self.last_action: int | None = None

        self.terminated: bool = False
        self.truncated: bool = False

        self._result: ExecutionResult | None = None
        self.ep_dataset: EpDataset | None = None

        self.dataset = dataset
        self.config = env_config

        # Build valid opening timestamps from random mode
        self._random_openings: list[pd.Timestamp]
        if self.config.reset_mode == "random":
            self._random_openings = self._build_random_openings()

        # Build valid opening timestamps from serial mode
        self._serial_openings: list[pd.Timestamp]
        self._serial_index: int
        if self.config.reset_mode == "serial":
            self._serial_openings = self._build_serial_openings()
            self._serial_index = -1

        # Define observation spaces
        spaces: dict[str, gym.spaces.Space] = {
            "market": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(dataset.n_market_features,),
                dtype=self.config.obs_dtype,
            ),
            "market_price": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=self.config.obs_dtype,
            ),
            "benchmark_distance": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=self.config.obs_dtype,
            ),
            "steps_remaining": gym.spaces.Box(
                low=0,
                high=self.config.n_steps,
                shape=(1,),
                dtype=self.config.obs_dtype,
            ),
        }

        if dataset.n_calendar_features > 0:
            spaces["temporal"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(dataset.n_calendar_features,),
                dtype=self.config.obs_dtype,
            )

        if dataset.n_calendar_enc_features > 0:
            spaces["temporal_enc"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(dataset.n_calendar_enc_features,),
                dtype=self.config.obs_dtype,
            )

        self.observation_space = gym.spaces.Dict(spaces)

        # Define action space
        self.action_space = gym.spaces.Discrete(self.config.n_actions)

    # Main
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)

        # Reset state
        self.t = 0
        self.steps_remaining = self.config.n_steps
        self.quantity = 0

        self.limit_price = None
        self.last_action = None

        self.terminated = False
        self.truncated = False

        self._result = None

        # Reset opening and episode
        self._opening_time = self._select_opening_time(options)
        self._current_time = self._get_opening_time()

        self.ep_dataset = self._build_episode_rl_dataset()
        self.quantity = self.ep_dataset.ctd_contracts

        # Reset benchmark
        self.benchmark_price = self._get_market_price(self._opening_time)
        if not np.isfinite(self.benchmark_price):
            raise ValueError(f"Invalid benchmark price at '{self._opening_time}'")

        # Get data
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid execution action: '{action}'")
        self.last_action = int(action)

        current_time = self._get_current_time()

        # Market order
        if action == 0:
            return self._step_market_order(current_time)

        # Limit order
        return self._step_limit_order(action, current_time)

    def close(self):
        pass

    def render(self):
        pass

    # Observation
    def _get_observation(self) -> dict[str, np.ndarray]:
        ep_dataset = self._get_ep_dataset()
        current_time = self._get_current_time()
        market_price = self._get_market_price(current_time)
        benchmark_distance = self._get_benchmark_distance(market_price)

        observation = {
            "market": ep_dataset.market_features.loc[current_time].to_numpy(
                dtype=self.config.obs_dtype
            ),
            "market_price": np.asarray(
                [self._get_market_price(current_time)],
                dtype=self.config.obs_dtype,
            ),
            "benchmark_distance": np.asarray([benchmark_distance], dtype=self.config.obs_dtype),
            "steps_remaining": np.asarray(
                [self.steps_remaining],
                dtype=self.config.obs_dtype,
            ),
        }

        if ep_dataset.n_calendar_features > 0:
            observation["temporal"] = ep_dataset.calendar_features.loc[current_time].to_numpy(
                dtype=self.config.obs_dtype
            )

        if ep_dataset.n_calendar_enc_features > 0:
            observation["temporal_enc"] = ep_dataset.calendar_enc_features.loc[
                current_time
            ].to_numpy(dtype=self.config.obs_dtype)

        return observation

    # Info
    def _get_info(self) -> dict[str, Any]:
        current_time = self._get_current_time()

        return {
            # Episode
            "timestamp": current_time,
            "step": self.t,
            "steps_remaining": self.steps_remaining,
            "terminal": self.terminated,
            # Execution
            "market_price": self._get_market_price(current_time),
            "limit_price": self.limit_price,
            "benchmark_price": self.benchmark_price,
            "action": self.last_action,
            "status": (self.result.status if self.result is not None else None),
            "execution_price": (self.result.execution_price if self.result is not None else None),
            "execution_time": (self.result.execution_time if self.result is not None else None),
            "quantity": self.quantity,
        }

    def _step_market_order(
        self,
        timestamp: pd.Timestamp,
    ):
        self._execute(
            status="market",
            timestamp=timestamp,
        )

        self.terminated = True

        observation = self._get_observation()
        reward = self._get_reward()
        info = self._get_info()

        return observation, reward, self.terminated, self.truncated, info

    def _step_limit_order(
        self,
        action: int,
        current_time: pd.Timestamp,
    ):
        self.limit_price = self._action_to_limit_price(action)

        interval_closing = current_time + pd.Timedelta(seconds=self.config.step_sec)
        closing_time = self._closing_time

        if interval_closing > closing_time:
            raise RuntimeError(
                "Execution decision interval extends beyond " "the execution episode closing time"
            )

        # Check for limit order fill
        fill_time = self._find_fill_time(
            current_time,
            interval_closing,
            self.limit_price,
        )

        if fill_time is not None:
            return self._step_limit_fill(fill_time)

        # Forced market order
        if interval_closing >= closing_time:
            return self._step_forced_execution(closing_time)

        # Cancel the current order and advance to the next decision time
        self._current_time = interval_closing
        self.t += 1
        self.steps_remaining = self.config.n_steps - self.t
        self.limit_price = None

        observation = self._get_observation()
        reward = 0.0
        info = self._get_info()

        return observation, reward, self.terminated, self.truncated, info

    def _step_limit_fill(
        self,
        fill_time: pd.Timestamp,
    ):
        # The limit price is the price proposed by the agent.
        # The execution price is the observed opposite quote at the fill time.
        execution_price = self._get_market_execution_price(fill_time)

        self._execute(
            status="limit",
            timestamp=fill_time,
            execution_price=execution_price,
        )

        self.terminated = True

        observation = self._get_observation()
        reward = self._get_reward()
        info = self._get_info()

        return observation, reward, self.terminated, self.truncated, info

    def _step_forced_execution(
        self,
        timestamp: pd.Timestamp,
    ):
        self._execute(
            status="forced",
            timestamp=timestamp,
        )

        self.terminated = True

        observation = self._get_observation()
        reward = self._get_reward()
        info = self._get_info()

        return observation, reward, self.terminated, self.truncated, info

    # Order mechanics
    def _action_to_limit_price(self, action: int) -> float:
        current_time = self._get_current_time()

        market_price = self._get_market_price(current_time)

        limit_price = self._get_limit_price(
            market_price,
            action,
        )

        if not np.isfinite(limit_price):
            raise ValueError(f"Invalid limit price: '{limit_price}'")

        return limit_price

    def _execute(
        self,
        status: ExecutionStatus,
        timestamp: pd.Timestamp,
        execution_price: float | None = None,
    ) -> None:
        if status == "limit":
            if execution_price is None:
                raise ValueError("Limit execution requires an execution price")
        else:
            if execution_price is not None:
                raise ValueError(
                    f"Execution price must not be provided for " f"'{status}' execution"
                )

            execution_price = self._get_market_execution_price(timestamp)

        if not np.isfinite(execution_price):
            raise ValueError(f"Invalid execution price at '{timestamp}'")

        quantity = self._get_quantity()

        self._current_time = timestamp
        self.steps_remaining = 0

        self._result = ExecutionResult(
            side=self._execution_side,
            quantity=quantity,
            status=status,
            limit_price=self.limit_price,
            execution_price=float(execution_price),
            execution_time=timestamp,
        )

    def _select_opening_time(
        self,
        options: dict[str, Any] | None,
    ) -> pd.Timestamp:
        if options is not None:
            if not isinstance(options, dict):
                raise TypeError("Execution reset options must be a dictionary")

            if options:
                return self._options_opening_time(options)

        match self.config.reset_mode:
            case "random":
                return self._random_opening_time()

            case "serial":
                return self._serial_opening_time()

            case None:
                raise ValueError(
                    "Execution reset requires 'opening_time' in options " "when reset_mode is None"
                )

            case _:
                raise ValueError(f"Invalid reset mode: '{self.config.reset_mode}'")

    def _random_opening_time(self) -> pd.Timestamp:
        if len(self._random_openings) == 0:
            raise ValueError("Dataset contains no valid execution opening times")

        index = self.np_random.integers(len(self._random_openings))

        opening_time = self._random_openings[index]
        self._validate_opening_time(opening_time)

        return opening_time

    def _serial_opening_time(self) -> pd.Timestamp:
        self._serial_index = (self._serial_index + 1) % len(self._serial_openings)
        return self._serial_openings[self._serial_index]

    def _options_opening_time(
        self,
        options: dict[str, Any],
    ) -> pd.Timestamp:
        if "opening_time" not in options:
            raise ValueError("Execution reset options must contain 'opening_time'")

        if not isinstance(options["opening_time"], pd.Timestamp):
            raise TypeError("Execution reset option 'opening_time' must be a pd.Timestamp")

        self._validate_opening_time(options["opening_time"])

        return options["opening_time"]

    # Episode
    def _build_episode_rl_dataset(self) -> EpDataset:
        opening = self._get_opening_time()
        closing = self._closing_time

        market_features = self.dataset.market_features.loc[opening:closing]
        calendar_features = self.dataset.calendar_features.loc[opening:closing]
        calendar_enc_features = self.dataset.calendar_enc_features.loc[opening:closing]
        ctd_mid = self.dataset.ctd_mid.loc[opening:closing]
        fut_mid = self.dataset.fut_mid.loc[opening:closing]
        ctd_spread = self.dataset.ctd_spread.loc[opening:closing]
        fut_spread = self.dataset.fut_spread.loc[opening:closing]

        episode_index = self.dataset.ctd_mid.index[
            (self.dataset.ctd_mid.index >= opening) & (self.dataset.ctd_mid.index <= closing)
        ]

        expected_length = self.config.episode_sec + 1

        if len(episode_index) != expected_length:
            raise ValueError(
                f"Invalid execution episode length: expected "
                f"{expected_length} rows, got {len(episode_index)}"
            )

        date = opening.tz_localize(None).floor("D")
        fut_contracts = self.dataset.fut_contracts.loc[date]

        return EpDataset(
            market_features=market_features,
            calendar_features=calendar_features,
            calendar_enc_features=calendar_enc_features,
            ctd_mid=ctd_mid,
            fut_mid=fut_mid,
            ctd_contracts=self.dataset.ctd_contracts,
            fut_contracts=fut_contracts,
            ctd_spread=ctd_spread,
            fut_spread=fut_spread,
        )

    def _build_random_openings(self) -> list[pd.Timestamp]:
        valid_openings: list[pd.Timestamp] = []

        for _, rows in self.dataset.date_to_slice.items():
            day_index = self.dataset.ctd_mid.index[rows]

            if len(day_index) == 0:
                continue

            latest_opening = day_index[-1] - pd.Timedelta(seconds=self.config.episode_sec)

            valid_openings.extend(day_index[day_index <= latest_opening])

        if not valid_openings:
            raise ValueError("Dataset contains no valid execution opening times")

        return valid_openings

    def _build_serial_openings(self) -> list[pd.Timestamp]:
        serial_openings: list[pd.Timestamp] = []

        for _, rows in self.dataset.date_to_slice.items():
            day_index = self.dataset.ctd_mid.index[rows]

            if len(day_index) == 0:
                continue

            first_opening = day_index[0]
            latest_opening = day_index[-1] - pd.Timedelta(seconds=self.config.episode_sec)

            opening_time = first_opening

            while opening_time <= latest_opening:
                serial_openings.append(opening_time)
                opening_time += pd.Timedelta(seconds=self.config.episode_sec)

        if not serial_openings:
            raise ValueError("Dataset contains no valid execution evaluation openings")

        return serial_openings

    @property
    def n_serial_openings(self) -> int:
        if self.config.reset_mode != "serial":
            raise RuntimeError("Serial openings are available only in serial mode")

        return len(self._serial_openings)

    def _validate_opening_time(
        self,
        timestamp: pd.Timestamp,
    ) -> None:
        if timestamp.nanosecond != 0 or timestamp.microsecond != 0:
            raise ValueError("Execution opening time must be aligned to the 1-second data grid")

        assert isinstance(self.dataset.ctd_mid.index, pd.DatetimeIndex)
        dataset_tz = self.dataset.ctd_mid.index.tz

        if timestamp.tz != dataset_tz:
            raise ValueError(
                f"Execution opening time is not in the expected tz: " f"'{timestamp.tz}'"
            )

        if timestamp not in self.dataset.ctd_mid.index:
            raise ValueError(f"Execution opening time is not present in dataset: " f"'{timestamp}'")

        closing_time = timestamp + pd.Timedelta(seconds=self.config.episode_sec)

        if closing_time not in self.dataset.ctd_mid.index:
            raise ValueError(
                f"Insufficient data for execution horizon opening at " f"'{timestamp}'"
            )

        if timestamp.normalize() != closing_time.normalize():
            raise ValueError(f"Execution horizon crosses a trading day boundary: " f"'{timestamp}'")

    def init_serial(self):
        if self.config.reset_mode != "serial":
            raise ValueError("This method can be called only in serial mode")

        self._serial_index = -1

    # Side specific
    @property
    @abstractmethod
    def _execution_side(self) -> config.LobSide:
        """Return the market side on which the passive order is placed."""

    @abstractmethod
    def _get_market_price(self, timestamp: pd.Timestamp) -> float:
        """Return the immediate executable market price for the execution side."""

    @abstractmethod
    def _get_limit_price(
        self,
        market_price: float,
        quote_distance: int,
    ) -> float:
        """Return the passive limit price."""

    @abstractmethod
    def _find_fill_time(
        self,
        opening_time: pd.Timestamp,
        closing_time: pd.Timestamp,
        limit_price: float,
    ) -> pd.Timestamp | None:
        """Return the first fill time in [opening_time, closing_time)."""

    @abstractmethod
    def _get_market_execution_price(
        self,
        timestamp: pd.Timestamp,
    ) -> float:
        """Return the executable market price at the given timestamp."""

    @abstractmethod
    def _get_reward(self) -> int:
        """Return the reward for the completed execution."""

    @abstractmethod
    def _get_benchmark_distance(self, market_price: float) -> float:
        """Return the current market price relative to the opening benchmark,
        measured in ticks and expressed as execution improvement.
        """

    # Properties
    @property
    def _closing_time(self) -> pd.Timestamp:
        opening_time = self._get_opening_time()

        return opening_time + pd.Timedelta(seconds=self.config.episode_sec)

    @property
    def result(self) -> ExecutionResult | None:
        return self._result

    # General purpose utilities
    def _get_opening_time(self) -> pd.Timestamp:
        if self._opening_time is None:
            raise RuntimeError("Environment must be reset before requesting the opening time")

        return self._opening_time

    def _get_current_time(self) -> pd.Timestamp:
        if self._current_time is None:
            raise RuntimeError("Environment must be reset before requesting the current time")

        return self._current_time

    def _get_ep_dataset(self) -> EpDataset:
        if self.ep_dataset is None:
            raise RuntimeError("Environment must be reset before requesting the episode dataset")

        return self.ep_dataset

    def _get_quantity(self) -> float:
        if self.quantity <= 0:
            raise RuntimeError("Execution quantity is not available")

        return self.quantity


class BidExecutionEnv(ExecutionEnv):
    """Execution environment for buying the CTD using passive bid orders."""

    @property
    def _execution_side(self) -> config.LobSide:
        return "bid"

    def _get_market_price(self, timestamp: pd.Timestamp) -> float:
        ep_dataset = self._get_ep_dataset()

        return float(ep_dataset.ctd_ask.loc[timestamp])

    def _get_limit_price(
        self,
        market_price: float,
        quote_distance: int,
    ) -> float:
        return market_price - quote_distance * self.config.tick_size

    def _find_fill_time(
        self,
        opening_time: pd.Timestamp,
        closing_time: pd.Timestamp,
        limit_price: float,
    ) -> pd.Timestamp | None:
        ep_dataset = self._get_ep_dataset()

        timestamps = ep_dataset.ctd_mid.index[
            (ep_dataset.ctd_mid.index >= opening_time) & (ep_dataset.ctd_mid.index < closing_time)
        ]

        ask = ep_dataset.ctd_ask.loc[timestamps]

        hit_mask = ask <= limit_price
        hit_timestamps = timestamps[hit_mask]

        if len(hit_timestamps) == 0:
            return None

        return hit_timestamps[0]

    def _get_market_execution_price(
        self,
        timestamp: pd.Timestamp,
    ) -> float:
        return self._get_market_price(timestamp)

    def _get_reward(self) -> int:
        result = self.result

        if result is None:
            raise RuntimeError("Execution result is not available")

        if result.status == "limit":
            if result.limit_price is None:
                raise RuntimeError("Limit execution result does not contain a limit price")

            execution_price = result.limit_price
        else:
            execution_price = result.execution_price

        return round((self.benchmark_price - execution_price) / self.config.tick_size)

    def _get_benchmark_distance(self, market_price: float) -> float:
        return (self.benchmark_price - market_price) / self.config.tick_size


class AskExecutionEnv(ExecutionEnv):
    """Execution environment for selling the CTD using passive ask orders."""

    @property
    def _execution_side(self) -> config.LobSide:
        return "ask"

    def _get_market_price(self, timestamp: pd.Timestamp) -> float:
        ep_dataset = self._get_ep_dataset()

        return float(ep_dataset.ctd_bid.loc[timestamp])

    def _get_limit_price(
        self,
        market_price: float,
        quote_distance: int,
    ) -> float:
        return market_price + quote_distance * self.config.tick_size

    def _find_fill_time(
        self,
        opening_time: pd.Timestamp,
        closing_time: pd.Timestamp,
        limit_price: float,
    ) -> pd.Timestamp | None:
        ep_dataset = self._get_ep_dataset()

        timestamps = ep_dataset.ctd_mid.index[
            (ep_dataset.ctd_mid.index >= opening_time) & (ep_dataset.ctd_mid.index < closing_time)
        ]

        bid = ep_dataset.ctd_bid.loc[timestamps]

        hit_mask = bid >= limit_price
        hit_timestamps = timestamps[hit_mask]

        if len(hit_timestamps) == 0:
            return None

        return hit_timestamps[0]

    def _get_market_execution_price(
        self,
        timestamp: pd.Timestamp,
    ) -> float:
        return self._get_market_price(timestamp)

    def _get_reward(self) -> int:
        result = self.result

        if result is None:
            raise RuntimeError("Execution result is not available")

        if result.status == "limit":
            if result.limit_price is None:
                raise RuntimeError("Limit execution result does not contain a limit price")

            execution_price = result.limit_price
        else:
            execution_price = result.execution_price

        return round((execution_price - self.benchmark_price) / self.config.tick_size)

    def _get_benchmark_distance(self, market_price: float) -> float:
        return (market_price - self.benchmark_price) / self.config.tick_size


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
        market_set="dummy",
        calendar_set="default",
        calendar_enc_set="default",
    )

    env_config = ExecutionEnvConfig(
        reset_mode="random",
        horizon_min=10,
        step_sec=10,
        max_n_tick=50,
        tick_size=0.01,
    )

    rl_dataset = build_rl_dataset(dataset_config)

    env = BidExecutionEnv(rl_dataset, env_config)
    check_env(env, skip_render_check=True, skip_close_check=True)
    print("Bid env checked!")

    env = AskExecutionEnv(rl_dataset, env_config)
    check_env(env, skip_render_check=True, skip_close_check=True)
    print("Ask env checked!")

    print("Both envs checked!")
