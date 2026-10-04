from typing import Any

import gymnasium as gym
import pandas as pd
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from thesis_project.rl.policies import Predictor


def _get_reset_info(env: VecEnv) -> dict[str, Any]:
    if isinstance(env, VecNormalize) and isinstance(env.venv, VecEnv):
        return env.venv.reset_infos[0].copy()

    return env.reset_infos[0].copy()


def evaluate_episode_gym(
    predictor: Predictor,
    env: gym.Env,
    episode: int | None = None,
) -> list[dict[str, Any]]:
    """
    Evaluate one episode using a standard Gymnasium environment.

    The environment is explicitly reset at the beginning of each episode.
    Unlike SB3 VecEnvs, standard Gymnasium environments do not automatically
    reset after termination.

    Args:
        * predictor: Policy used to select actions.
        * env: Gymnasium environment to evaluate.
        * episode: Optional episode identifier stored in each record.

    Returns:
        * records: A list of trajectory records, including the initial reset record and
        the final terminal record.
    """
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
    """
    Evaluate a predictor over all serial execution openings using a
    Gymnasium environment.

    The environment is reset once per episode by
    ``evaluate_episode_gym``. Evaluation requires serial reset mode so that
    episodes correspond to the valid execution openings in chronological
    order.

    Args:
        * predictor: Policy used to select actions.
        * env: Gymnasium environment to evaluate.

    Returns:
        * records: DataFrame containing the concatenated trajectory records for all
        evaluation episodes.
    """
    config = env.get_wrapper_attr("config")
    n_episodes = env.get_wrapper_attr("n_serial_openings")

    if config.reset_mode != "serial":
        raise ValueError(
            "Model evaluation is not intended for environments configured in random mode"
        )

    records = []

    for episode in range(n_episodes):
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
    obs: Any,
    reset_info: dict[str, Any],
    episode: int | None = None,
) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
    """
    Evaluate one episode using an SB3 VecEnv.

    The environment must already have been reset before the first episode.
    SB3 VecEnvs automatically reset after ``done=True``, so this function must
    not call ``env.reset()``. The observation returned by the terminal
    ``step`` is therefore the initial observation of the next episode and is
    returned to the caller for continued evaluation.

    For ``n_envs=1``, ``infos[0]`` and ``dones[0]`` refer to the single
    underlying environment. SB3-generated ``TimeLimit.truncated`` and
    ``terminal_observation`` entries are excluded from the trajectory records.

    Args:
        * predictor: Policy used to select actions.
        * env: SB3 vectorized environment to evaluate.
        * obs: Current vectorized observation. For the first episode this must
            come from ``env.reset()``; afterwards it must be the observation
            returned by the previous terminal ``step``.
        * reset_info: Info from the previous episode reset.
        * episode: Optional episode identifier stored in each record.

    Returns:
        * records: A tuple containing the episode trajectory records, the current
        observation, and the reset info for the next episode.
    """
    records = []

    # Record reset info.
    record = reset_info.copy()
    if episode is not None:
        record["episode"] = episode
    records.append(record)

    while True:
        action, _ = predictor.predict(
            obs,
            deterministic=True,
        )

        obs, _, dones, infos = env.step(action)

        info = infos[0].copy()
        info.pop("TimeLimit.truncated", None)
        info.pop("terminal_observation", None)

        if episode is not None:
            info["episode"] = episode

        records.append(info)

        if dones[0]:
            # DummyVecEnv has already reset the underlying environment.
            reset_info = _get_reset_info(env)
            break

    return records, obs, reset_info


def evaluate_model_sb3(
    predictor: Predictor,
    env: VecEnv,
) -> pd.DataFrame:
    """
    Evaluate a predictor over all serial execution openings using an SB3
    VecEnv.

    The VecEnv is explicitly reset only once at the beginning. SB3
    automatically resets the environment after each terminal step, so the
    observation returned by one episode is passed directly into the next.
    Evaluation requires serial reset mode so that episodes correspond to the
    valid execution openings in chronological order.

    Args:
        * predictor: Policy used to select actions.
        * env: SB3 vectorized environment to evaluate.

    Returns:
        * records: DataFrame containing the concatenated trajectory records for all
        evaluation episodes.
    """

    config = env.get_attr("config")[0]
    n_episodes = env.get_attr("n_serial_openings")[0]

    if config.reset_mode != "serial":
        raise ValueError(
            "Model evaluation is not intended for environments configured in random mode"
        )

    records = []

    obs = env.reset()
    reset_info = _get_reset_info(env)

    for episode in range(n_episodes):
        episode_records, obs, reset_info = evaluate_episode_sb3(
            predictor,
            env,
            obs,
            reset_info,
            episode + 1,
        )
        records.extend(episode_records)

    env.close()

    return pd.DataFrame(records)
