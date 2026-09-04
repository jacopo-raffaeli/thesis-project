from typing import Any

import gymnasium as gym
import pandas as pd
from sb3_contrib.common.maskable.utils import get_action_masks

from thesis_project.rl.policies import Predictor


def evaluate_episode(
    predictor: Predictor,
    env: gym.Env,
    episode: int | None = None,
) -> list[dict[str, Any]]:
    records = []
    obs, info = env.reset()

    record = info.copy()
    if episode is not None:
        record["episode"] = episode
    records.append(record)

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
    env: gym.Env,
) -> pd.DataFrame:
    config = env.get_wrapper_attr("config")
    dataset = env.get_wrapper_attr("dataset")

    if config.mode != "serial":
        raise ValueError(
            "Model evaluation is not intended for environments configured in random mode"
        )

    records = []
    for episode in range(dataset.n_dates):
        records.extend(evaluate_episode(predictor, env, episode + 1))

    env.close()

    return pd.DataFrame(records)
