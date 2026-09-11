import logging
from itertools import product
from pathlib import Path

import gymnasium as gym
from joblib import Parallel, delayed
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from thesis_project import config, rl, utils
from thesis_project.rl.evaluation import evaluate_model
from thesis_project.utils.io import dump_configs
from thesis_project.utils.misc import generate_seeds

logger = logging.getLogger(__name__)

# TODO:
# - Study VecNormalize usage
# - Address how to avoid normalizing everything
#   - We can do this if the observation space is of Dict type
#   - In such case it is enough to specify the keys of the dict to normalize
# - Address how to use the test normalization in testing
# - Check that the data collected in input are not normalized
#   - The actual env is completely agnostic of VecNormalize so this is not a problem


BATCH_SIZES = [64, 128, 256, 512]
CLIP_RANGES = [0.1, 0.2, 0.3]

N_SEED = 5
TOTAL_TIMESTEPS = 1_000_000


def make_env(
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
    tensorboard_log: str | None = None,
    tb_log_name: str = "MaskablePPO",
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


def train_evaluate(
    dataset_train: rl.env.RLDataset,
    dataset_test: rl.env.RLDataset,
    env_config_random: rl.env.EnvConfig,
    env_config_serial: rl.env.EnvConfig,
    path: Path,
    batch_size: int,
    clip_range: float,
    seed: int,
):
    filename = f"bs_{batch_size}_cr_{clip_range}_{seed}"

    env_train_random = make_env(
        dataset_train,
        env_config_random,
    )

    # Train model
    model = train_model(
        env_train_random,
        batch_size=batch_size,
        clip_range=clip_range,
        seed=seed,
        total_timesteps=TOTAL_TIMESTEPS,
        tensorboard_log=str(path / "tensorboard"),
        tb_log_name=filename,
    )
    env_train_random.close()

    # Save trained model
    model.save(path / f"{filename}.zip")

    # Evaluate the model on train set
    env_train_serial = make_env(
        dataset_train,
        env_config_serial,
    )

    records = evaluate_model(
        model,
        env_train_serial,
    )
    records.to_parquet(path / f"{filename}_train.parquet")

    # Evaluate the model on test set
    env_test_serial = make_env(
        dataset_test,
        env_config_serial,
    )

    records = evaluate_model(
        model,
        env_test_serial,
    )
    records.to_parquet(path / f"{filename}_test.parquet")

    del model


def main():
    dataset_config = rl.dataset.DatasetConfig(
        ticker="fbtp",
        n_jobs=4,
        contract_mode="round",
        market=rl.features.MARKET_FEATURES_XGB_CLS,
        calendar=rl.features.CALENDAR_FEATURES,
        calendar_enc=rl.features.CALENDAR_FEATURES_ENCODED,
    )

    env_config_random = rl.env.EnvConfig(
        reset_mode="random",
        price_mode="quoted",
        persistence_min=10,
    )

    env_config_serial = rl.env.EnvConfig(
        reset_mode="serial",
        price_mode="quoted",
        persistence_min=10,
        trajectory_min=0,
    )

    logger.debug(f"""
    RL analysis:

    - Settings:
        - {"Ticker:":<15} {dataset_config.ticker}
        - {"Price:":<15} {env_config_random.price_mode}
        - {"Contract:":<15} {dataset_config.contract_mode}

    - Dataset:
        - {"Features:":<15} {dataset_config.n_features}
            - {"Market features:":<15} {dataset_config.n_market_features}
            - {"Calendar features:":<15} {dataset_config.n_calendar_features}
            - {"Calendar features (encoded):":<15} {dataset_config.n_calendar_enc_features}

        - {"Jobs:":<15} {dataset_config.n_jobs}

    - Env:
        - {"Reset Mode:":<15} {env_config_random.reset_mode}
        - {"Persistence:":<15} {env_config_random.persistence_min} min
        - {"Position Encoding:":<15} {env_config_random.position_encoding}
    """)

    dataset = rl.dataset.build_rl_dataset(
        dataset_config,
    )

    dataset_train, dataset_test = rl.dataset.split_rl_dataset(
        dataset,
        "2023-01",
    )

    del dataset

    root = config.RES_EXP_DIR / "fbtp" / "rl"
    path = utils.io.create_run_path(root)
    seeds = generate_seeds(N_SEED)

    dump_configs(
        path / "config.json",
        dataset=dataset_config,
        env_random=env_config_random,
        env_serial=env_config_serial,
        batch_sizes=BATCH_SIZES,
        clip_ranges=CLIP_RANGES,
        seeds=seeds,
        total_timesteps=TOTAL_TIMESTEPS,
    )

    for batch_size, clip_range in product(
        BATCH_SIZES,
        CLIP_RANGES,
    ):
        Parallel(
            n_jobs=len(seeds),
            backend="loky",
        )(
            delayed(train_evaluate)(
                dataset_train,
                dataset_test,
                env_config_random,
                env_config_serial,
                path,
                batch_size,
                clip_range,
                seed,
            )
            for seed in seeds
        )


if __name__ == "__main__":
    main()
