from thesis_project.dataset.features import FeatureSpec
from thesis_project.dataset.temporal import CalFeatEncType, CalFeatType
from thesis_project.dataset.transforms import (
    Delta,
    Identity,
    Ratio,
    Rolling,
)

# TODO:
# - Find a way to group features all together (e.g. a dict)
# - Add the necessary machinery to pass them to the datset builder

# TODO:
# - Build the standard set of features:
# - XGB classification
# - XGB regression
# - XGB regression iterative

# fmt: off
MARKET_FEATURES: list[FeatureSpec] = [
    FeatureSpec(
        base_id="basis",
        transforms=[
            Identity(),
            Delta(
                deltas=[60],
                lags=[60],
            ),
            Ratio(
                references=["day", "hour"],
            ),
            Rolling(
                stats=["std"],
                windows=[300, 1800],
            ),
            Rolling(
                stats=["mean"],
                windows=[1800],
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
                references=["day", "hour"],
            ),
        ],
    ),
    FeatureSpec(
        base_id="ctd_obi_1",
        transforms=[
            Identity()
        ],
    ),
    FeatureSpec(
        base_id="ctd_spread",
        transforms=[
            Identity()
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
    "minute_of_day",
    "second_of_day",
]
# fmt: on
