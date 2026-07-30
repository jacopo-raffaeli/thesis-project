from collections import defaultdict
from dataclasses import dataclass, field

from natsort import natsorted

from thesis_project.rl_trading.data import BASE_FEATURES
from thesis_project.rl_trading.transforms import (
    BaseTransform,
    Delta,
    Identity,
    Ratio,
    Rolling,
)


@dataclass(frozen=True)
class FeatureSpec:
    base_id: str
    transforms: list[BaseTransform] = field(default_factory=list[BaseTransform])

    def __post_init__(self):
        if self.base_id not in BASE_FEATURES.keys():
            raise ValueError("Base feature is not in data.BASE_FEATURES")

        if len(self.transforms) == 0:
            raise ValueError("At least one transform must be specified")

        if len(set(self.output_names)) != len(self.output_names):
            raise ValueError(f"Duplicate outputs for {self.base_id}")

    @property
    def n_outputs(self) -> int:
        return sum(transform.n_outputs for transform in self.transforms)

    @property
    def output_names(self) -> list[str]:
        names = []
        for transform in self.transforms:
            names.extend(transform.output_names(self.base_id))

        return names


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
    FeatureSpec(
        base_id="ctd_obi_lvl_1",
        transforms=[
            Identity(),
            Rolling(
                stats=["mean"],
                windows=[60, 120],
            ),
            Rolling(
                stats=["std"],
                windows=[180],
                lags=60
            )
        ]
    ),
    FeatureSpec(
        base_id="fut_mid_price",
        transforms=[
            Ratio(
                references="day",
            )
        ]
    )
]
# fmt: on


def summary(features: list[FeatureSpec]) -> None:
    """
    Print a summary of the configured features.
    """

    merged: dict[str, list[str]] = defaultdict(list)

    for feature in features:
        merged[feature.base_id].extend(feature.output_names)

    total = sum(len(names) for names in merged.values())

    print(f"\nTotal number of features: {total}\n")

    print("Features by base:\n")
    for base in natsorted(merged):
        names = natsorted(merged[base])

        print(f"{base}: {len(names)}")
        for name in names:
            print(f"  - {name}")

        print()


if __name__ == "__main__":
    summary(FEATURES)
