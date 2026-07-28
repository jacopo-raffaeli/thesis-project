from dataclasses import dataclass, field

from thesis_project.rl_trading.transforms import BaseTransform, Delta

# TODO: Define a set of features to test with
# TODO: Check that basis are in data.BASE_FEATURES
# TODO: Define utilities to count the number of features
# TODO: Define utilities to count the list the features


@dataclass(frozen=True)
class FeatureSpec:
    base: str
    transforms: list[BaseTransform] = field(default_factory=list[BaseTransform])


FEATURES: list[FeatureSpec] = [FeatureSpec(base="basis", transforms=[Delta(deltas=[60, 120, 180])])]
