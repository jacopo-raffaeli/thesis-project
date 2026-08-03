from dataclasses import dataclass, field
from typing import Any, Dict, List

from thesis_project import config as global_config
from thesis_project.importance_xgb import registry


@dataclass
class TransformSpec:
    name: str
    params: Dict[str, Any]
    apply_to: List[str] | None = None
    lags: List[int] | None = None


@dataclass
class AnalysisConfig:
    """
    Configuration for feature importance analysis based on xgboost.
    """

    # General
    analysis: str
    ticker: str = "fbtp"
    seed: int = 42
    base_dir: str = "importance-xgb"

    # Market specific
    min_time: str = "09:00:00"
    max_time: str = "17:00:00"

    # Sampling
    sampling_strategy: str = "uniform_v1"
    sampling_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "sample_ratio": 0.05,
        }
    )

    # Date filtering
    date_filters_offsets: Dict[str, List[int]] = field(
        default_factory=lambda: {
            "fut_last_trading_days": [5, 0],
        }
    )

    # Target
    horizon: int = 600
    use_base_target: bool = True

    target_transform: TransformSpec = field(
        default_factory=lambda: TransformSpec(
            name="delta",
            params={
                "deltas": [
                    600,
                ]
            },
        )
    )

    feature_transforms: List[TransformSpec] = field(
        default_factory=lambda: [
            TransformSpec(
                name="delta",
                apply_to=["basis"],
                params={
                    "deltas": [
                        60,
                    ],
                },
                lags=list(range(0, (20 + 1) * 60, 60)),
            ),
            TransformSpec(
                name="rolling",
                params={
                    "windows": [
                        300,
                        1800,
                    ],
                    "stats": [
                        "mean",
                        "std",
                        "min",
                        "max",
                    ],
                },
            ),
            TransformSpec(
                name="ratio",
                params={
                    "references": [
                        "hour",
                        "day",
                    ]
                },
            ),
        ]
    )

    # Calendar features
    calendar_features: List[str] = field(
        default_factory=lambda: [
            # "year",
            "month",
            # "week_of_year",
            # "day_of_year",
            "day_of_month",
            "day_of_week",
            "hour_of_day",
            "minute_of_day",
            "minute_of_hour",
            "second_of_day",
            "second_of_hour",
            "second_of_minute",
        ]
    )

    calendar_features_encoded: List[str] = field(
        default_factory=lambda: [
            # "hour_of_day",
            # "minute_of_day",
            # "minute_of_hour",
            # "second_of_day",
            # "second_of_hour",
            # "second_of_minute",
        ]
    )

    # Event based features
    event_based_features: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "fut_last_trading_day": {
                "file": "fut_metadata.csv",
                "column": "Last Trading Date",
                "transforms": ["days_to_next", "days_to_prev"],
            },
            # "fut_delivery_day": {
            #     "file": "fut_metadata.csv",
            #     "column": "Delivery Date",
            #     "transforms": ["days_to_next", "days_to_prev"],
            # },
            # "FUT Option Maturity Day": {
            #     "file": None,
            #     "column": None,
            #     "transforms": [
            #        "days_to_next",
            #        "days_to_prev"
            #    ]
            # },
            # "CTD Auction Day": {
            #     "file": None,
            #     "column": None,
            #     "transforms": [
            #        "days_to_next",
            #        "days_to_prev"
            #    ]
            # }
        }
    )

    daily_event_based_features: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "us_mrkt_open": {
                "event_time": "09:30:00",
                "event_tz": "America/New_York",
                "units": ["h", "m"],
            }
        }
    )

    # Parallelization
    n_jobs_preprocessing: int = 4
    n_jobs_xgb: int = -1
    n_jobs_seed: int = 5

    # Training
    perc_train: float = 0.7
    perc_val: float = 0.2
    n_splits_tscv: int = 5
    use_tscv: bool = False

    # XGBoost
    keep_features_nan: bool = True
    n_quantile: int = 3
    xgb_importance = "gain"

    # Optuna
    optuna_n_seeds: int = 5
    optuna_n_trials: int = 150
    optuna_sampler_n_startup_trials: int = 20
    optuna_pruner_n_startup_trials: int = 25
    optuna_pruner_n_warmup_steps: int = 30

    # SHAP Analysis
    shap_n_samples = 2500

    # Feature selection
    patience = 5
    tol = 1e-3


def create_config(**overrides) -> AnalysisConfig:
    """
    Create config with CLI overrides.
    """
    sample_ratio = overrides.pop("sample_ratio", None)

    config_obj = AnalysisConfig(**overrides)

    if sample_ratio is not None:
        config_obj.sampling_params["sample_ratio"] = sample_ratio
    config_obj.sampling_params["seed"] = config_obj.seed

    validate_config(config_obj)

    return config_obj


def get_xgb_fixed_params(config_obj: AnalysisConfig):
    common = {
        "tree_method": "hist",
        "verbosity": 0,
        "n_jobs": config_obj.n_jobs_xgb,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
    }

    if config_obj.analysis == "importance_cls":
        return {
            **common,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "num_class": config_obj.n_quantile,
        }

    elif config_obj.analysis == "importance_reg":
        return {
            **common,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
        }

    elif config_obj.analysis == "selection":
        return {
            **common,
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "n_estimators": 3000,
            "early_stopping_rounds": 50,
            "learning_rate": 1e-2,
            "max_depth": 11,
            "min_child_weight": 25,
            "lambda": 10,
            "alpha": 1e-3,
            "gamma": 1e-3,
        }

    raise ValueError(f"Unknown analysis '{config_obj.analysis}'")


def validate_config(config_obj: AnalysisConfig):
    _validate_ticker(config_obj)
    _validate_sampling_strategy(config_obj)
    _validate_samplig_params(config_obj)
    _validate_training_params(config_obj)
    _validate_horizon(config_obj)
    _validate_n_jobs(config_obj.n_jobs_xgb)
    _validate_n_jobs(config_obj.n_jobs_preprocessing)
    _validate_n_quantile(config_obj)
    _validate_n_splits(config_obj)
    _validate_target(config_obj)
    _validate_features(config_obj)


def _validate_ticker(config_obj: AnalysisConfig):
    if config_obj.ticker not in global_config.VALID_TICKERS:
        raise ValueError(
            f"Invalid ticker '{config_obj.ticker}', "
            f"expected one of {sorted(global_config.VALID_TICKERS)}"
        )


def _validate_sampling_strategy(config_obj: AnalysisConfig):
    if config_obj.sampling_strategy not in registry.SAMPLING_STRATEGIES_REGISTRY.keys():
        raise ValueError(
            f"Invalid sampling strategy '{config_obj.sampling_strategy}, "
            f"expected one of {sorted(registry.SAMPLING_STRATEGIES_REGISTRY.keys())}"
        )


def _validate_samplig_params(config_obj: AnalysisConfig):
    if config_obj.sampling_strategy == "uniform_v1":
        if not (0 < config_obj.sampling_params["sample_ratio"] <= 1):
            raise ValueError(
                f"Invalid sample ratio '{config_obj.sampling_params['sample_ratio']}, "
                "expected a value in (0,1]"
            )


def _validate_training_params(config_obj: AnalysisConfig):
    if not (0 < config_obj.perc_train < 1):
        raise ValueError(
            f"Invalid sample ratio '{config_obj.perc_train}', " "expected a value in (0,1)"
        )

    if not (0 < config_obj.perc_val < 1):
        raise ValueError(
            f"Invalid sample ratio '{config_obj.perc_val}', " "expected a value in (0,1)"
        )

    if config_obj.perc_train + config_obj.perc_val >= 1:
        raise ValueError(
            f"Invalid test + validation ratio '{config_obj.perc_train + config_obj.perc_val}', "
            "expected test + validation ratio must be < 1"
        )


def _validate_horizon(config_obj: AnalysisConfig):
    if config_obj.horizon <= 0:
        raise ValueError(
            f"Invalid analysis horizon '{config_obj.horizon}', " "expected a positive integer"
        )


def _validate_n_jobs(n: int):
    if n < -1 or n == 0:
        raise ValueError(f"Invalid number of jobs '{n}', " "expected a positive integer or -1")


def _validate_n_quantile(config_obj: AnalysisConfig):
    if config_obj.n_quantile < 2:
        raise ValueError(
            f"Invalid number of quantiles '{config_obj.n_quantile}', " "expected a number >= 2"
        )


def _validate_n_splits(config_obj: AnalysisConfig):
    if config_obj.n_splits_tscv < 2:
        raise ValueError(
            f"Invalid number of quantiles '{config_obj.n_splits_tscv}', " "expected a number >= 2"
        )


def _validate_non_empty_list(params):
    for name, values in params.items():
        if isinstance(values, List) and len(values) == 0:
            raise ValueError(f"Parameter '{name}' can not be empty")


def _validate_target_length(params):
    for name, values in params.items():
        if isinstance(values, List) and len(values) != 1:
            raise ValueError(
                f"Parameter '{name}' must contain exactly one value for target transform"
            )


def _validate_transform_name(name):
    if name not in registry.TRANSFORM_REGISTRY:
        raise ValueError(
            f"Invalid transform '{name}', "
            f"valid transforms [{', '.join(sorted(registry.TRANSFORM_REGISTRY.keys()))}]"
        )


def _validate_transform_params(name, params):
    if set(params.keys()) != registry.TRANSFORM_PARAMETER_REGISTRY[name]:
        raise ValueError(
            f"Invalid target parameters '{set(params.keys())}', "
            f"valid transforms [{', '.join(sorted(registry.TRANSFORM_PARAMETER_REGISTRY[name]))}]"
        )


# def _validate_transform_apply_to(apply_to):
#     if apply_to is None:
#         return

#     if not isinstance(apply_to, list):
#         raise ValueError("'apply_to' must be a list of feature names or None")

#     if len(apply_to) == 0:
#         raise ValueError("'apply_to' cannot be an empty list")

#     if any(type(x) is not str for x in apply_to):
#         raise ValueError("'apply_to' must contain only strings")

#     if len(set(apply_to)) != len(apply_to):
#         raise ValueError("'apply_to' contains duplicated feature names")

#     invalid = sorted(set(apply_to) - set(data_paths.get_data_paths()))
#     if invalid:
#         raise ValueError(
#             f"Unknown feature(s) in apply_to: {invalid}. "
#             f"Valid features are {sorted(global_config.BASE_FEATURES)}"
#         )


def _validate_transform_delta(params):
    deltas = params["deltas"]
    invalid = [str(delta) for delta in deltas if type(delta) is not int or delta <= 0]
    if invalid:
        raise ValueError(
            f"Invalid delta values [{', '.join(delta for delta in invalid)}], "
            f"expected positive integers"
        )


def _validate_transform_rolling(params):
    windows = params["windows"]
    stats = params["stats"]

    invalid_w = [str(w) for w in windows if type(w) is not int or w <= 0]
    if invalid_w:
        raise ValueError(
            f"Invalid rolling window values [{', '.join(invalid_w)}], "
            f"expected positive integers"
        )

    invalid_s = [s for s in stats if s not in registry.TRANSFORM_ROLLING_REGISTRY]
    if invalid_s:
        raise ValueError(
            f"Invalid rolling stat values [{', '.join(invalid_s)}], "
            f"valid transforms  {', '.join(s for s in sorted(registry.TRANSFORM_ROLLING_REGISTRY))}"
        )


def _validate_transform_ratio(params):
    references = params["references"]

    invalid = [r for r in references if r not in registry.TRANSFORM_RATIO_REGISTRY]
    if invalid:
        raise ValueError(
            f"Invalid ratio references values [{', '.join(invalid)}], "
            f"valid transforms {', '.join(r for r in sorted(registry.TRANSFORM_RATIO_REGISTRY))}"
        )


def _validate_transform_values(name, params):
    validate_transform_dict = {
        "delta": _validate_transform_delta,
        "rolling": _validate_transform_rolling,
        "ratio": _validate_transform_ratio,
    }

    validate_transform_dict[name](params)


def _validate_target(config_obj: AnalysisConfig):
    transform = config_obj.target_transform
    _validate_transform_name(transform.name)
    _validate_transform_params(transform.name, transform.params)
    _validate_non_empty_list(transform.params)
    _validate_target_length(transform.params)
    _validate_transform_values(transform.name, transform.params)


def _validate_market_features(config_obj: AnalysisConfig):
    for transform in config_obj.feature_transforms:
        _validate_transform_name(transform.name)
        _validate_transform_params(transform.name, transform.params)
        _validate_non_empty_list(transform.params)
        _validate_transform_values(transform.name, transform.params)


def _validate_calendar_features(config_obj: AnalysisConfig):
    for calendar_feature in config_obj.calendar_features:
        if calendar_feature not in registry.CALENDAR_FEATURES_REGISTRY.keys():
            raise ValueError(
                f"Invalid calendar feature '{calendar_feature}', "
                f"expected one of {sorted(registry.CALENDAR_FEATURES_REGISTRY)}"
            )


def _validate_calendar_encoded_features(config_obj: AnalysisConfig):
    for calendar_feature in config_obj.calendar_features_encoded:
        if calendar_feature not in registry.CALENDAR_FEATURES_ENCODED_REGISTRY.keys():
            raise ValueError(
                f"Invalid calendar feature '{calendar_feature}', "
                f"expected one of {sorted(registry.CALENDAR_FEATURES_REGISTRY)}"
            )


def _validate_event_based_features(config_obj: AnalysisConfig):
    for name, metadata in config_obj.event_based_features.items():
        for transform in metadata["transforms"]:
            if transform not in registry.EVENT_BASED_FEATURES_REGISTRY.keys():
                raise ValueError(
                    f"Invalid event based feature transform '{name}: {transform}', "
                    f"expected one of {sorted(registry.EVENT_BASED_FEATURES_REGISTRY.keys())}"
                )


def _validate_temporal_features(config_obj: AnalysisConfig):
    _validate_calendar_features(config_obj)
    _validate_calendar_encoded_features(config_obj)
    _validate_event_based_features(config_obj)


def _validate_features(config_obj: AnalysisConfig):
    _validate_market_features(config_obj)
    _validate_temporal_features(config_obj)


if __name__ == "__main__":
    overrides = {}
    config = create_config(**overrides)
