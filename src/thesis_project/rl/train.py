import logging
from itertools import product
from pathlib import Path

from joblib import Parallel, delayed
from sb3_contrib import MaskablePPO

from thesis_project import config, rl, utils
from thesis_project.rl.env_factory import (
    make_evaluation_env,
    make_training_env,
)
from thesis_project.rl.evaluation import evaluate_model_sb3
from thesis_project.rl.experiment_config import ExperimentConfig
from thesis_project.utils.io import dump_configs
from thesis_project.utils.misc import generate_seeds

logger = logging.getLogger(__name__)


def train_model(
    env,
    *,
    batch_size: int,
    clip_range: float,
    seed: int,
    total_timesteps: int,
    tensorboard_log: str | None = None,
    tb_log_name: str = "MaskablePPO",
) -> MaskablePPO:
    model = MaskablePPO(
        policy="MultiInputPolicy",
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
    experiment: ExperimentConfig,
    path: Path,
    batch_size: int,
    clip_range: float,
    seed: int,
) -> None:
    filename = f"bs_{batch_size}_cr_{clip_range}_{seed}"

    # Training
    env_train = make_training_env(
        dataset_train,
        env_config_random,
        normalize=experiment.normalize_market_obs,
    )

    model = train_model(
        env_train,
        batch_size=batch_size,
        clip_range=clip_range,
        seed=seed,
        total_timesteps=experiment.total_timesteps,
        tensorboard_log=str(path / "tensorboard"),
        tb_log_name=filename,
    )

    # Save model
    model_path = path / f"{filename}.zip"
    model.save(model_path)

    # Save EnvNormalize stats if enabled
    path_vec_norm = None
    if experiment.normalize_market_obs:
        path_vec_norm = path / f"{filename}_vecnormalize.pkl"
        env_train.save(str(path_vec_norm))  # type: ignore

    env_train.close()

    # Train evaluation
    env_train_eval = make_evaluation_env(
        dataset_train,
        env_config_serial,
        normalize=experiment.normalize_market_obs,
        path=path_vec_norm,
    )

    records = evaluate_model_sb3(
        model,
        env_train_eval,
    )
    records.to_parquet(path / f"{filename}_train.parquet")

    # Test evaluation
    env_test_eval = make_evaluation_env(
        dataset_test,
        env_config_serial,
        normalize=experiment.normalize_market_obs,
        path=path_vec_norm,
    )

    records = evaluate_model_sb3(
        model,
        env_test_eval,
    )
    records.to_parquet(path / f"{filename}_test.parquet")

    del model


def main() -> None:
    experiment_config = ExperimentConfig(
        split_month="2023-01",
    )

    dataset_config = rl.dataset.DatasetConfig(
        ticker="fbtp",
        contract_mode="round",
        market_set="xgb_cls",
        calendar_set="default",
        calendar_enc_set="default",
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

    - Experiment:
        - {"Split month:":<20} {experiment_config.split_month}
        - {"Timesteps:":<20} {experiment_config.total_timesteps}
        - {"Seeds:":<20} {experiment_config.n_seeds}
        - {"Normalize market:":<20} {experiment_config.normalize_market_obs}

    - Dataset:
        - {"Ticker:":<20} {dataset_config.ticker}
        - {"Features:":<20} {dataset_config.n_features}
            - {"Market:":<20} {dataset_config.n_market_features}
            - {"Calendar:":<20} {dataset_config.n_calendar_features}
            - {"Calendar enc:":<20} {dataset_config.n_calendar_enc_features}
        - {"Contract mode:":<20} {dataset_config.contract_mode}
        - {"Market jobs:":<20} {dataset_config.n_jobs_market}

    - Environment:
        - {"Price mode:":<20} {env_config_random.price_mode}
        - {"Persistence:":<20} {env_config_random.persistence_min} min
        - {"Position encoding:":<20} {env_config_random.position_encoding}
    """)

    dataset = rl.dataset.build_rl_dataset(dataset_config)

    dataset_train, dataset_test = rl.dataset.split_rl_dataset(
        dataset,
        experiment_config.split_month,
    )

    del dataset

    root = config.RES_EXP_DIR / dataset_config.ticker / "rl"
    path = utils.io.create_run_path(root)

    seeds = generate_seeds(
        experiment_config.n_seeds,
        seed=experiment_config.seed,
    )

    dump_configs(
        path / "config.json",
        experiment=experiment_config,
        dataset=dataset_config,
        env_random=env_config_random,
        env_serial=env_config_serial,
        seeds=seeds,
    )

    for batch_size, clip_range in product(
        experiment_config.batch_sizes, experiment_config.clip_ranges
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
                experiment_config,
                path,
                batch_size,
                clip_range,
                seed,
            )
            for seed in seeds
        )


if __name__ == "__main__":
    main()
