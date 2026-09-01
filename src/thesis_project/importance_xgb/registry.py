from thesis_project.importance_xgb import (
    sampling,
    transforms,
)

SAMPLING_STRATEGIES_REGISTRY = {"uniform_v1": sampling._uniform_v1}

TRANSFORM_REGISTRY = {
    "delta": transforms.DeltaTransform,
    "rolling": transforms.RollingTransform,
    "ratio": transforms.RatioTransform,
}

TRANSFORM_PARAMETER_REGISTRY = {
    "delta": {"deltas"},
    "rolling": {"windows", "stats"},
    "ratio": {"references"},
}

TRANSFORM_ROLLING_REGISTRY = {"mean", "std", "max", "min"}

TRANSFORM_RATIO_REGISTRY = {"day", "hour"}

XGB_IMPORTANCES_REGISTRY = [
    "weight",  # the number of times a feature is used to split the data across all trees.
    "gain",  # the average gain across all splits the feature is used in.
    "cover",  # the average coverage across all splits the feature is used in.
    "total_gain",  # the total gain across all splits the feature is used in.
    "total_cover",  # the total coverage across all splits the feature is used in.
]
