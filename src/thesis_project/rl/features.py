from thesis_project.dataset.features import FeatureSpec
from thesis_project.dataset.transforms import (
    Delta,
    Identity,
)

# fmt: off
FEATURES: list[FeatureSpec] = [
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
# fmt: on
