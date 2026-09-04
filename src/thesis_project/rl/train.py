import json
import logging
from dataclasses import asdict, is_dataclass
from itertools import product
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from thesis_project import config, rl, utils
from thesis_project.rl.evaluation import evaluate_model

logger = logging.getLogger(__name__)


BATCH_SIZES = [64, 128, 256, 512]
CLIP_RANGES = [0.1, 0.2, 0.3]

N_SEED = 5
TOTAL_TIMESTEPS = 1_000_000


def generate_seeds(n: int, low: int = 0, high: int = 9999) -> list[int]:
    rng = np.random.default_rng(np.random.randint(low, high))
    return rng.integers(low, high, size=n).tolist()


def dump_configs(path: Path, **configs: Any) -> None:
    data = {
        name: asdict(value)  # type: ignore
        if is_dataclass(value)
        else value
        for name, value in configs.items()
    }

    path.write_text(
        json.dumps(data, indent=4, default=str),
        encoding="utf-8",
    )


def make_random_env(
    dataset: rl.env.RLDataset,
    env_config: rl.env.EnvConfig,
) -> gym.Env:
    env = gym.make(
        "BasisTradingEnv-v0",
        dataset=dataset,
        env_config=env_config,
    )

    return Monitor(env)


def make_serial_env(
    dataset: rl.env.RLDataset,
    env_config: rl.env.EnvConfig,
) -> gym.Env:
    env = gym.make(
        "BasisTradingEnv-v0",
        dataset=dataset,
        env_config=env_config,
    )

    return Monitor(env)


def train_model(
    env: gym.Env,
    *,
    batch_size: int,
    clip_range: float,
    seed: int,
    total_timesteps: int,
    tensorboard_log: str,
    tb_log_name: str,
) -> MaskablePPO:
    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        batch_size=batch_size,
        clip_range=clip_range,
        tensorboard_log=tensorboard_log,
    )

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        tb_log_name=tb_log_name,
    )

    return model


def main():
    dataset_config = rl.dataset.DatasetConfig(
        ticker="fbtp",
        n_jobs=4,
    )

    env_config_random = rl.env.EnvConfig(
        mode="random",
        price_mode="mid",
        contract_mode="round",
        persistence_min=10,
    )

    env_config_serial = rl.env.EnvConfig(
        mode="serial",
        price_mode="mid",
        contract_mode="round",
        persistence_min=10,
        trajectory_min=0,
    )

    logger.debug(f"""
    RL analysis:

    - Dataset:
        - {"ticker:":<15} {dataset_config.ticker}
        - {"n-jobs:":<15} {dataset_config.n_jobs}

    - Env:
        - {"Price:":<15} {env_config_random.price_mode}
        - {"Contract:":<15} {env_config_random.contract_mode}
        - {"Persistence:":<15} {env_config_random.persistence_min} min
    """)

    dataset = rl.dataset.build_rl_dataset(
        dataset_config,
        env_config_random,
        specs=rl.features.MARKET_FEATURES,
        cal_feature=rl.features.CALENDAR_FEATURES,
        cal_enc_feature=rl.features.CALENDAR_FEATURES_ENCODED,
    )

    logger.debug(f"""
    Dataset:

    - {"Number of features:":<15} {len(dataset.features.columns)}
        - {"Number of market features:":<15} {len(rl.features.MARKET_FEATURES)}
        - {"Number of calendar features:":<15} {len(rl.features.CALENDAR_FEATURES)}
        - {"Number of calendar encoded features:":<15} {len(rl.features.CALENDAR_FEATURES_ENCODED)}
    """)

    dataset_train, dataset_test = rl.dataset.split_rl_dataset(
        dataset,
        "2023-01",
    )

    del dataset

    env_train = make_random_env(
        dataset_train,
        env_config_random,
    )

    root = config.RES_EXP_DIR / "fbtp" / "rl"
    path = utils.io.create_run_path(root)
    seeds = generate_seeds(N_SEED)

    dump_configs(
        path / "config.json",
        dataset=dataset_config,
        env_train=env_config_random,
        env_test=env_config_serial,
        batch_sizes=BATCH_SIZES,
        clip_ranges=CLIP_RANGES,
        seeds=seeds,
        total_timesteps=TOTAL_TIMESTEPS,
    )

    for batch_size, clip_range in product(
        BATCH_SIZES,
        CLIP_RANGES,
    ):
        for seed in seeds:
            filename = f"bs_{batch_size}_cr_{clip_range}_{seed}"

            # Train model
            model = train_model(
                env_train,
                batch_size=batch_size,
                clip_range=clip_range,
                seed=seed,
                total_timesteps=TOTAL_TIMESTEPS,
                tensorboard_log=str(path / "tensorboard"),
                tb_log_name=filename,
            )
            name = f"{filename}.zip"
            model.save(path / name)

            # Evaluate the model on train set
            records = evaluate_model(
                model,
                dataset_train,
                env_config_serial,
            )
            name = f"{filename}_train.parquet"
            records.to_parquet(path / name)

            # Evaluate the model on test set
            records = evaluate_model(
                model,
                dataset_test,
                env_config_serial,
            )
            name = f"{filename}_test.parquet"
            records.to_parquet(path / name)

            del model

    env_train.close()


if __name__ == "__main__":
    main()
