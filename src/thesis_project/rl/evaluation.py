from typing import Any, Protocol

import gymnasium as gym
import numpy as np
import pandas as pd
from sb3_contrib.common.maskable.utils import get_action_masks

from thesis_project import rl


class Predictor(Protocol):
    def predict(
        self,
        observation: Any,
        *,
        deterministic: bool = True,
        action_masks: np.ndarray | None = None,
    ) -> tuple[Any, Any]: ...


class FixedActionPolicy:
    def __init__(self, action: int, fallback_action: int = 1):
        self.action = action
        self.fallback_action = fallback_action

    def predict(
        self,
        observation: Any,
        *,
        deterministic: bool = True,
        action_masks: np.ndarray | None = None,
    ) -> tuple[int, None]:
        if action_masks is None:
            raise ValueError("FixedActionPolicy requires action_masks.")

        if action_masks[self.action]:
            return self.action, None

        if not action_masks[self.fallback_action]:
            raise ValueError(
                f"Neither action {self.action} nor fallback action "
                f"{self.fallback_action} is valid."
            )

        return self.fallback_action, None


class FixedLongPolicy(FixedActionPolicy):
    def __init__(self):
        super().__init__(action=2)


class FixedFlatPolicy(FixedActionPolicy):
    def __init__(self):
        super().__init__(action=1)


class FixedShortPolicy(FixedActionPolicy):
    def __init__(self):
        super().__init__(action=0)


class RandomPolicy:
    def predict(
        self,
        observation: Any,
        *,
        deterministic: bool = True,
        action_masks: np.ndarray | None = None,
    ) -> tuple[int, None]:
        if action_masks is None:
            raise ValueError("RandomPolicy requires action_masks.")

        valid_actions = np.flatnonzero(action_masks)
        return int(np.random.choice(valid_actions)), None


def evaluate_episode(
    predictor: Predictor,
    env: gym.Env,
    episode: int | None = None,
) -> list[dict[str, Any]]:
    records = []
    obs, info = env.reset()

    while True:
        action, _ = predictor.predict(
            obs,
            deterministic=True,
            action_masks=get_action_masks(env),
        )

        obs, _, terminated, truncated, info = env.step(int(action))

        record = info.copy()
        if episode is not None:
            record["episode"] = episode
        records.append(record)

        if terminated or truncated:
            break

    return records


def evaluate_model(
    predictor: Predictor,
    env: rl.env.BasisTradingEnv,
) -> pd.DataFrame:
    if env.config.mode != "serial":
        raise ValueError(
            "Model evaluation is not intended for environments configured in random mode"
        )

    records = []
    for episode in range(env.dataset.n_dates):
        records.extend(evaluate_episode(predictor, env, episode + 1))

    env.close()

    return pd.DataFrame(records)
