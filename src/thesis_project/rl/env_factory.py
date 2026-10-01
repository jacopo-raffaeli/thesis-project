from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize

from thesis_project.rl.env import RLDataset
from thesis_project.rl.env_config import BasisTradingEnvConfig


def make_gym_env(
    dataset: RLDataset,
    env_config: BasisTradingEnvConfig,
) -> gym.Env:
    env = gym.make(
        "BasisTradingEnv-v0",
        dataset=dataset,
        env_config=env_config,
    )

    return Monitor(env)


def make_vec_env(
    dataset: RLDataset,
    env_config: BasisTradingEnvConfig,
) -> DummyVecEnv:
    return DummyVecEnv([lambda: make_gym_env(dataset, env_config)])


def make_training_env(
    dataset: RLDataset,
    env_config: BasisTradingEnvConfig,
    *,
    normalize: bool,
) -> VecEnv:
    vec_env = make_vec_env(dataset, env_config)

    if not normalize:
        return vec_env

    return VecNormalize(
        vec_env,
        training=True,
        norm_obs=True,
        norm_reward=False,
        norm_obs_keys=["market"],
    )


def make_evaluation_env(
    dataset: RLDataset,
    env_config: BasisTradingEnvConfig,
    *,
    normalize: bool,
    path: Path | None = None,
) -> VecEnv:
    vec_env = make_vec_env(dataset, env_config)

    if not normalize:
        return vec_env

    if path is None:
        raise ValueError("normalize_stats_path is required when market normalization is enabled")

    vec_env = VecNormalize.load(
        str(path),
        vec_env,
    )

    vec_env.training = False
    vec_env.norm_obs = True
    vec_env.norm_reward = False
    vec_env.norm_obs_keys = ["market"]

    return vec_env
