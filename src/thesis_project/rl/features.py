from thesis_project.dataset.features_market import FeatureSpec
from thesis_project.dataset.features_temporal import CalFeatEncType, CalFeatType
from thesis_project.dataset.features_transforms import (
    Delta,
    Identity,
    Ratio,
    Rolling,
)

# fmt: off
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
                references=["hour", "day" ],
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

CALENDAR_FEATURES_ENCODED: list[CalFeatEncType] = [
    "hour_of_day",
    "minute_of_hour",
    "second_of_minute",
]
# fmt: on
