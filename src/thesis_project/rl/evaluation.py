from typing import Any

import gymnasium as gym
import pandas as pd
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import VecEnv

from thesis_project.rl.policies import Predictor


def evaluate_episode_gym(
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


def evaluate_model_gym(
    predictor: Predictor,
    env: gym.Env,
) -> pd.DataFrame:
    config = env.get_wrapper_attr("config")
    dataset = env.get_wrapper_attr("dataset")

    if config.reset_mode != "serial":
        raise ValueError(
            "Model evaluation is not intended for environments configured in random mode"
        )

    records = []

    for episode in range(dataset.n_dates):
        records.extend(
            evaluate_episode_gym(
                predictor,
                env,
                episode + 1,
            )
        )

    env.close()

    return pd.DataFrame(records)


def evaluate_episode_sb3(
    predictor: Predictor,
    env: VecEnv,
    episode: int | None = None,
) -> list[dict[str, Any]]:
    records = []

    obs = env.reset()

    # DummyVecEnv stores the reset info separately.
    info = env.reset_infos[0]

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

        obs, _, dones, infos = env.step(action)

        info = infos[0]

        record = info.copy()
        if episode is not None:
            record["episode"] = episode
        records.append(record)

        if dones[0]:
            break

    return records


def evaluate_model_sb3(
    predictor: Predictor,
    env: VecEnv,
) -> pd.DataFrame:
    config = env.get_attr("config")[0]
    dataset = env.get_attr("dataset")[0]

    if config.reset_mode != "serial":
        raise ValueError(
            "Model evaluation is not intended for environments configured in random mode"
        )

    records = []

    for episode in range(dataset.n_dates):
        records.extend(
            evaluate_episode_sb3(
                predictor,
                env,
                episode + 1,
            )
        )

    env.close()

    return pd.DataFrame(records)
