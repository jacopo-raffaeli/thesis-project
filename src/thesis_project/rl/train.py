import json
import logging
from dataclasses import asdict, is_dataclass
from itertools import product
from pathlib import Path
from typing import Any

import gymnasium as gym
import pandas as pd
import stable_baselines3 as sb3
from stable_baselines3.common.monitor import Monitor

from thesis_project import config, rl, utils

logger = logging.getLogger(__name__)


BATCH_SIZES = [32, 64, 128]
CLIP_RANGES = [0.1, 0.2, 0.3]


def dump_configs(path: Path, **configs: Any) -> None:
    data = {
        name: asdict(config)  # type: ignore
        if is_dataclass(config)
        else config
        for name, config in configs.items()
    }

    path.write_text(
        json.dumps(data, indent=4, default=str),
        encoding="utf-8",
    )


def evaluation(model: sb3.PPO, env: gym.Env) -> pd.DataFrame:
    obs, info = env.reset()

    records = [info]

    while True:
        action, _ = model.predict(obs, deterministic=True)

        obs, _, terminated, truncated, info = env.step(int(action))
        records.append(info)

        if terminated or truncated:
            break

    return pd.DataFrame(records)


def main():
    dataset_config = rl.dataset.DatasetConfig(ticker="fbtp", n_jobs=4)

    env_config_train = rl.env.EnvConfig(
        mode="random",
        price_mode="mid",
        contract_mode="round",
        persistence_min=10,
    )

    env_config_test = rl.env.EnvConfig(
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
        - {"Price:":<15} {env_config_train.price_mode}
        - {"Contract:":<15} {env_config_train.contract_mode}
        - {"Persistence:":<15} {env_config_train.persistence_min} min
    """)

    dataset = rl.dataset.build_rl_dataset(
        dataset_config,
        env_config_train,
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

    dataset_train, dataset_test = rl.dataset.split_rl_dataset(dataset, "2023-01")

    del dataset

    env_train = gym.make("BasisTradingEnv-v0", dataset=dataset_train, env_config=env_config_train)
    env_train = Monitor(env_train)

    env_test = gym.make("BasisTradingEnv-v0", dataset=dataset_test, env_config=env_config_test)
    env_test = Monitor(env_test)

    root = config.RES_EXP_DIR / "fbtp" / "rl"
    path = utils.io.create_run_path(root)

    dump_configs(
        path / "config.json",
        dataset=dataset_config,
        env_train=env_config_train,
        env_test=env_config_test,
        batch_sizes=BATCH_SIZES,
        clip_ranges=CLIP_RANGES,
        seed=42,
        total_timesteps=1_000_000,
    )

    for batch_size, clip_range in product(BATCH_SIZES, CLIP_RANGES):
        model = sb3.PPO(
            policy="MlpPolicy",
            env=env_train,
            seed=42,
            batch_size=batch_size,
            clip_range=clip_range,
            tensorboard_log=str(path / "tensorboard"),
        )
        filename = f"bs_{batch_size}_cr_{clip_range}"

        model.learn(
            total_timesteps=1_000_000,
            progress_bar=True,
            tb_log_name=filename,
        )

        name = (path / filename).with_suffix(".zip")
        model.save(name)

        records = evaluation(model, env_test)
        name = (path / f"{filename}_evaluation").with_suffix(".parquet")
        records.to_parquet(name)

        del model


if __name__ == "__main__":
    main()
