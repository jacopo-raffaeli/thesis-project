import gymnasium as gym
import numpy as np
import pandas as pd


class BasisTradingEnv(gym.Env):
    def __init__(self, data: pd.DataFrame, include_cost: bool):
        # Initialize parent class
        super().__init__()

        # Initialize arguments
        self.data = data
        self.include_cost = include_cost

        # Define observation space
        self.n_features = len(data.columns)
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
        self._action_to_position_dict = {
            0: -1,  # Short
            1: 0,  # Flat
            2: +1,  # Long
        }

        # direction -> action dictionary
        self._position_to_action_dict = {
            -1: 0,  # Short
            0: 1,  # Flat
            +1: 2,  # Long
        }

        self.reset()

    def reset(self):
        pass

    def step(self, action):
        pass

    def close(self):
        pass

    def render(self):
        pass

    def _compute_pnl(self):
        pass

    def _compute_cost(self):
        pass

    def _action_to_position(self, action):
        return self._action_to_position_dict[action]

    def _position_to_action(self, position):
        return self._action_to_position_dict[position]


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    data = pd.DataFrame()
    include_cost = True
    env = BasisTradingEnv(data, include_cost)

    try:
        check_env(env)
        print("Environment check passed.")

    except Exception as e:
        print(f"Environment check failed: {e}")
