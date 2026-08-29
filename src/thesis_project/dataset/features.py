from collections import defaultdict
from dataclasses import dataclass, field

from natsort import natsorted

from thesis_project.dataset.data import BASE_FEATURES
from thesis_project.dataset.transforms import (
    BaseTransform,
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

    @property
    def max_lag(self) -> int:
        max_lag = 0
        for transform in self.transforms:
            max_lag = max(max_lag, transform.max_lag)

        return max_lag

    @property
    def max_lookback(self) -> int:
        max_lookback = 0
        for transform in self.transforms:
            max_lookback = max(max_lookback, transform.max_lookback)

        return max_lookback


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


def validate_feature_specs(feature_specs: list[FeatureSpec]):
    """
    Check for duplicates in FeatureSpec base ids
    """
    seen = set()
    for spec in feature_specs:
        if spec.base_id in seen:
            raise ValueError(f"Duplicate FeatureSpec object '{spec.base_id}'")

        seen.add(spec.base_id)

    seen = set()
    for spec in feature_specs:
        for name in spec.output_names:
            if name in seen:
                raise ValueError(f"Duplicate feature '{name}'")

            seen.add(name)
