from thesis_project.dataset.features import FeatureSpec
from thesis_project.dataset.temporal import CalFeatType
from thesis_project.dataset.transforms import (
    Delta,
    Identity,
)

# fmt: off
MARKET_FEATURES: list[FeatureSpec] = [
    FeatureSpec(
        base_id="basis",
        transforms=[
            Identity(
                lags=60,
            ),
            Delta(
                deltas=[60],
                lags=[60, 120, 180],
            ),
        ]
    ),
    # FeatureSpec(
    #     base_id="ctd_obi_lvl_1",
    #     transforms=[
    #         Identity(),
    #         Rolling(
    #             stats=["mean"],
    #             windows=[60, 120],
    #         ),
    #         Rolling(
    #             stats=["std"],
    #             windows=[180],
    #             lags=60
    #         )
    #     ]
    # ),
    # FeatureSpec(
    #     base_id="fut_mid_price",
    #     transforms=[
    #         Ratio(
    #             references="day",
    #         )
    #     ]
    # )
]

CALENDAR_FEATURES: list[CalFeatType] = [
    "hour_of_day",
    "minute_of_day",
    "second_of_day",
]

# CALENDAR_FEATURES_ENCODED: list[CalFeatEncType] = [
#     "hour_of_day",
#     "minute_of_day",
#     "second_of_day",
# ]
# fmt: on
