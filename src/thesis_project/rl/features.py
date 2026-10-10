from thesis_project.dataset.features_market import FeatureSpec
from thesis_project.dataset.features_temporal import CalEncFeatType, CalFeatType
from thesis_project.dataset.features_transforms import (
    Delta,
    Identity,
    Ratio,
    Rolling,
)

# fmt: off
BASELINE_TRADING: list[FeatureSpec] = [
    FeatureSpec(
        base_id="basis",
        transforms=[
            Identity(
                lags=[60, 300],
            ),
            Delta(
                deltas=[60],
                lags=[60, 300],
            ),
        ],
    ),
    FeatureSpec(
        base_id="ctd_mid_price",
        transforms=[
            Identity(),
            Delta(
                deltas=[60],
                lags=[60, 300],
            ),
        ],
    ),
    FeatureSpec(
        base_id="fut_mid_price",
        transforms=[
            Identity(),
            Delta(
                deltas=[60],
                lags=[60, 300],
            ),
        ],
    ),
    FeatureSpec(
        base_id="ctd_obi_1",
        transforms=[
            Identity(),
        ],
    ),
    FeatureSpec(
        base_id="fut_obi_1",
        transforms=[
            Identity(),
        ],
    ),
    FeatureSpec(
        base_id="ctd_bof_1",
        transforms=[
            Identity(),
        ],
    ),
    FeatureSpec(
        base_id="fut_bof_1",
        transforms=[
            Identity(),
        ],
    ),
    FeatureSpec(
        base_id="ctd_aof_1",
        transforms=[
            Identity(),
        ],
    ),
    FeatureSpec(
        base_id="fut_aof_1",
        transforms=[
            Identity(),
        ],
    ),
]


# fmt: off
BASELINE_EXECUTION: list[FeatureSpec] = [
    FeatureSpec(
        base_id="ctd_mid_price",
        transforms=[
            Identity(
                lags=[60, 300],
                keep_original=False,
            ),
            Delta(
                deltas=[60],
                lags=[60, 300],
            ),
        ],
    ),
    FeatureSpec(
        base_id="ctd_obi_1",
        transforms=[
            Identity(),
        ],
    ),
    FeatureSpec(
        base_id="ctd_bof_1",
        transforms=[
            Identity(),
        ],
    ),
    FeatureSpec(
        base_id="ctd_aof_1",
        transforms=[
            Identity(),
        ],
    ),
]


MARKET_FEATURES_XGB_CLS: list[FeatureSpec] = [
    FeatureSpec(
        base_id="basis",
        transforms=[
            Identity(),
            Delta(
                deltas=[60],
                lags=[60],
            ),
            Rolling(
                stats=["std"],
                windows=[300, 1800],
            ),
            Rolling(
                stats=["mean"],
                windows=[1800],
            ),
            Ratio(
                references=["hour", "day"],
            ),
        ],
    ),
    FeatureSpec(
        base_id="irr",
        transforms=[
            Identity(),
            Rolling(
                stats=["mean"],
                windows=[300, 1800],
            ),
            Ratio(
                references=["hour", "day" ],
            ),
        ],
    ),
    FeatureSpec(
        base_id="ctd_obi_1",
        transforms=[
            Identity(),
        ],
    ),
    FeatureSpec(
        base_id="ctd_spread",
        transforms=[
            Identity(),
        ],
    ),
]

MARKET_FEATURES_XGB_REG: list[FeatureSpec] = [
    FeatureSpec(
        base_id="basis",
        transforms=[
            Identity(),
            Delta(
                deltas=[60],
            ),
            Rolling(
                stats=["min"],
                windows=[300, 1800],
            ),
            Ratio(
                references=["hour", "day"],
            ),
        ],
    ),
    FeatureSpec(
        base_id="irr",
        transforms=[
            Identity(),
            Ratio(
                references=["hour", "day"],
            ),
        ],
    ),
    FeatureSpec(
        base_id="ctd_spread",
        transforms=[
            Identity(),
            Ratio(
                references=["hour", "day"]
            ),
        ],
    ),
    FeatureSpec(
        base_id="ctd_obi_1",
        transforms=[
            Identity(),
        ],
    ),
    FeatureSpec(
        base_id="ctd_obi_3",
        transforms=[
            Identity(),
        ],
    ),
    FeatureSpec(
        base_id="ctd_ofi_3",
        transforms=[
            Identity(),
        ],
    ),
]

MARKET_FEATURES_XGB_REG_IT: list[FeatureSpec] = [
    FeatureSpec(
        base_id="basis",
        transforms=[
            Delta(
                deltas=[60],
                lags=[840]
            ),
            Rolling(
                stats=["min"],
                windows=[300, 1800],
            ),
        ],
    ),
    FeatureSpec(
        base_id="irr",
        transforms=[
            Ratio(
                references=["hour", "day"],
            ),
        ],
    ),
    FeatureSpec(
        base_id="ctd_spread",
        transforms=[
            Ratio(
                references=["day"]
            ),
        ],
    ),
    FeatureSpec(
        base_id="fut_ofi_1",
        transforms=[
            Identity(),
            Ratio(
                references=["hour"],
            ),
        ],
    ),
    FeatureSpec(
        base_id="fut_ofi_2",
        transforms=[
            Identity(),
            Ratio(
                references=["hour", "day"],
            ),
        ],
    ),
    FeatureSpec(
        base_id="fut_ofi_3",
        transforms=[
            Identity(),
            Ratio(
                references=["hour", "day"],
            ),
        ],
    ),
]

CALENDAR_FEATURES: list[CalFeatType] = [
    # "hour_of_day",
    # "minute_of_day",
    # "second_of_day",
]

CALENDAR_ENC_FEATURES: list[CalEncFeatType] = [
    "hour_of_day",
    "minute_of_hour",
    "second_of_minute",
]

MARKET_FEATURE_SETS: dict[str, list[FeatureSpec]] = {
    "dummy": [FeatureSpec(base_id="basis", transforms=[Identity()])],
    "baseline": BASELINE_TRADING,
    "xgb_cls": MARKET_FEATURES_XGB_CLS,
    "xgb_reg": MARKET_FEATURES_XGB_REG,
    "xgb_reg_it": MARKET_FEATURES_XGB_REG_IT,
}

CALENDAR_FEATURE_SETS: dict[str, list[CalFeatType]] = {
    "dummy": [],
    "default": CALENDAR_FEATURES,
}

CALENDAR_ENC_FEATURE_SETS: dict[str, list[CalEncFeatType]] = {
    "dummy": [],
    "default": CALENDAR_ENC_FEATURES,
}
# fmt: on


def get_market_feature_set(name: str) -> list[FeatureSpec]:
    try:
        return MARKET_FEATURE_SETS[name]
    except KeyError:
        raise ValueError(f"Unknown market feature set: {name!r}") from None


def get_calendar_feature_set(name: str) -> list[CalFeatType]:
    try:
        return CALENDAR_FEATURE_SETS[name]
    except KeyError:
        raise ValueError(f"Unknown calendar feature set: {name!r}") from None


def get_calendar_enc_feature_set(name: str) -> list[CalEncFeatType]:
    try:
        return CALENDAR_ENC_FEATURE_SETS[name]
    except KeyError:
        raise ValueError(f"Unknown calendar encoded feature set: {name!r}") from None
