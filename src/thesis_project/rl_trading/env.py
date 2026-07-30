from logging import config

import gymnasium as gym
import numpy as np
import pandas as pd

from thesis_project.rl_trading.env_config import EnvConfig, RLDataset


class BasisTradingEnv(gym.Env):
    def __init__(self, dataset: RLDataset, config: EnvConfig):
        super().__init__()

        self.t: int
        self.position: int

        self.dataset = dataset
        self.config = config

        # Define observation space
        self.n_features = len(dataset.features.columns)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features,),
            dtype=np.float32,
        )

        # Define action space
        self.n_actions = 3
        self.action_space = gym.spaces.Discrete(self.n_actions)

        # action -> direction dictionary
        self._act_to_pos_dict = {
            0: -1,  # Short
            1: 0,  # Flat
            2: +1,  # Long
        }

        # direction -> action dictionary
        self._pos_to_act_dict = {
            -1: 0,  # Short
            0: 1,  # Flat
            +1: 2,  # Long
        }

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.t = 0
        self.position = 0

        obs = self._get_observation()
        info = {}

        return obs, info

    def step(self, action):
        new_position = self._act_to_pos(action)

        reward = self._compute_reward(new_position)

        self.position = new_position
        self.t += 1

        terminated = self.t >= len(self.dataset.features) - 1
        truncated = False

        observation = self._get_observation()
        info = {}

        return observation, reward, terminated, truncated, info

    def close(self):
        pass

    def render(self):
        pass

    def _get_observation(self):
        obs = self.dataset.features.iloc[self.t]
        return obs.to_numpy(dtype=np.float32)

    def _compute_return(self, position) -> float:
        basis = self.dataset.basis
        delta = basis.iloc[self.t + 1] - basis.iloc[self.t]

        return position * delta

    def _compute_cost(self, position: int) -> float:
        ctd_spread = self.dataset.ctd_spread
        fut_spread = self.dataset.fut_spread
        assert isinstance(ctd_spread, pd.Series)
        assert isinstance(fut_spread, pd.Series)

        size = abs(position - self.position)
        spread = 0.5 * (ctd_spread.iloc[self.t] + fut_spread.iloc[self.t])

        return size * spread

    def _compute_reward(self, position) -> float:
        reward = self._compute_return(position)
        if self.config.include_cost:
            reward -= self._compute_cost(position)

        return reward

    def _act_to_pos(self, action):
        return self._act_to_pos_dict[action]

    def _pos_to_act(self, position):
        return self._pos_to_act_dict[position]


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    dataset = RLDataset()
    config = EnvConfig()
    env = BasisTradingEnv(dataset, config)

    try:
        check_env(env)
        print("Environment check passed.")

    except Exception as e:
        print(f"Environment check failed: {e}")
