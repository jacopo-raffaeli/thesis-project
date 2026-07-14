from thesis_project.importance_xgb import (
    sampling,
    temporal,
    transforms,
)

CALENDAR_FEATURES_REGISTRY = {
    "year": temporal.compute_year,
    "month": temporal.compute_month,
    "week_of_year": temporal.compute_week_of_year,
    "day_of_year": temporal.compute_day_of_year,
    "day_of_month": temporal.compute_day_of_month,
    "day_of_week": temporal.compute_day_of_week,
    "hour_of_day": temporal.compute_hour_of_day,
    "minute_of_day": temporal.compute_minute_of_day,
    "minute_of_hour": temporal.compute_minute_of_hour,
    "second_of_day": temporal.compute_second_of_day,
    "second_of_hour": temporal.compute_second_of_hour,
    "second_of_minute": temporal.compute_second_of_minute,
}

CALENDAR_FEATURES_ENCODED_REGISTRY = {
    "hour_of_day": temporal.encode_hour_of_day,
    "minute_of_day": temporal.encode_minute_of_day,
    "minute_of_hour": temporal.encode_minute_of_hour,
    "second_of_day": temporal.encode_second_of_day,
    "second_of_hour": temporal.encode_second_of_hour,
    "second_of_minute": temporal.encode_second_of_minute,
}

EVENT_BASED_FEATURES_REGISTRY = {
    "days_to_next": temporal.days_to_next_event,
    "days_to_prev": temporal.days_to_prev_event,
}

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
