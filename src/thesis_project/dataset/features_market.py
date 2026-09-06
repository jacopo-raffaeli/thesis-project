from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from natsort import natsorted

from thesis_project.dataset.data import BASE_FEATURES
from thesis_project.dataset.features_transforms import (
    TRANSFORM_PARSERS,
    BaseTransform,
    parse_identity,
    parse_lag_suffix,
)


@dataclass(frozen=True)
class FeatureSpec:
    base_id: str
    transforms: list[BaseTransform] = field(default_factory=list[BaseTransform])

    def __post_init__(self):
        if self.base_id not in BASE_FEATURES.keys():
            raise ValueError("Base feature is not in base features")

        if len(self.transforms) == 0:
            raise ValueError("At least one transform must be specified")

        if len(set(self.output_names)) != len(self.output_names):
            dups = [x for x in self.output_names if self.output_names.count(x) > 1]
            dups = set(dups)
            raise ValueError(f"Duplicate outputs for {self.base_id!r}: {dups!r}")

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

    @property
    def lookback(self) -> int:
        return self.max_lag + self.max_lookback


def summary(specs: list[FeatureSpec]) -> None:
    """
    Print a summary of the configured features.
    """

    merged: dict[str, list[str]] = defaultdict(list)

    for feature in specs:
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


def validate_feature_specs(specs: list[FeatureSpec]):
    """
    Check for duplicates in FeatureSpec base ids
    """
    seen = set()
    for spec in specs:
        if spec.base_id in seen:
            raise ValueError(f"Duplicate FeatureSpec object '{spec.base_id}'")

        seen.add(spec.base_id)

    seen = set()
    for spec in specs:
        for name in spec.output_names:
            if name in seen:
                raise ValueError(f"Duplicate feature '{name}'")

            seen.add(name)


# Parser
def extract_base_id(
    name: str,
) -> str:
    """
    Extract the base feature ID from an output feature name.
    The longest matching BASE_FEATURES prefix is selected.
    """
    if not isinstance(name, str):
        raise TypeError("Feature name must be a string")

    if not name:
        raise ValueError("Feature name must not be empty")

    if name in BASE_FEATURES:
        return name

    candidates = [base_id for base_id in BASE_FEATURES if name.startswith(f"{base_id}_")]

    if not candidates:
        raise ValueError(f"Could not extract base feature " f"from '{name}'")

    return max(
        candidates,
        key=len,
    )


def extract_transform_parser(
    name: str,
    base_id: str,
) -> Callable[[str, str], BaseTransform] | None:
    if name == base_id:
        return None

    prefix = f"{base_id}_"
    if not name.startswith(prefix):
        raise ValueError(f"Feature '{name}' does not belong to base feature '{base_id}'")

    suffix = name.removeprefix(prefix)
    transform_name, _ = parse_lag_suffix(suffix)

    # Only lag suffix remains -> Identity
    if not transform_name:
        return None

    marker = transform_name.split("_", 1)[0]

    parser = TRANSFORM_PARSERS.get(marker)
    if parser is None:
        raise ValueError(f"Unknown transform in feature name: '{name}'")

    return parser


def extract_transform_type(name: str, base_id: str) -> str | None:
    if name == base_id:
        return None

    prefix = f"{base_id}_"
    if not name.startswith(prefix):
        raise ValueError(f"Feature '{name}' does not belong to base feature '{base_id}'")

    suffix = name.removeprefix(prefix)
    transform_name, _ = parse_lag_suffix(suffix)

    # Only lag suffixes remain -> Identity
    if not transform_name:
        return None

    marker = transform_name.split("_", 1)[0]

    transform_type = TRANSFORM_PARSERS.get(marker)
    if transform_type is None:
        raise ValueError(f"Unknown transform in feature name: '{name}'")

    return transform_type


def extract_transform(name: str, base_id: str) -> BaseTransform:
    parser = extract_transform_parser(name, base_id)

    if parser is None:
        return parse_identity(name, base_id)

    return parser(name, base_id)


def parse_feature(
    name: str,
) -> tuple[str, BaseTransform]:
    """
    Reverse-engineer one output feature name.
    """
    base_id = extract_base_id(name)

    transform = extract_transform(
        name,
        base_id,
    )

    return base_id, transform


def validate_reverse_engineering(names: list[str], specs: list[FeatureSpec]):
    """
    Verify the consistency of the reverse-engineered features
    """
    reconstructed = [name for spec in specs for name in spec.output_names]

    expected = set(names)
    actual = set(reconstructed)

    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)

        raise ValueError(
            "Reverse engineering is not lossless.\n"
            f"Missing features: {missing}\n"
            f"Extra features: {extra}"
        )


def build_feature_specs(
    names: list[str],
) -> list[FeatureSpec]:
    """
    Reverse-engineer a collection of output feature names into FeatureSpec.
    """
    grouped: dict[
        str,
        list[BaseTransform],
    ] = defaultdict(list)

    for name in names:
        base_id, transform = parse_feature(name)

        grouped[base_id].append(transform)

    specs = [
        FeatureSpec(
            base_id=base_id,
            transforms=transforms,
        )
        for base_id, transforms in grouped.items()
    ]

    validate_reverse_engineering(names, specs)
    validate_feature_specs(specs)

    return specs


if __name__ == "__main__":
    input = [
        # ------------------------------------------------------------------
        # Identity
        # ------------------------------------------------------------------
        "basis",
        "irr",
        "fut_mid_price",
        # Lagged identity
        "basis_lag_1s",
        "basis_lag_60s",
        "fut_mid_price_lag_300s",
        "fut_mid_price_lag_1s",
        # ------------------------------------------------------------------
        # Delta
        # ------------------------------------------------------------------
        "basis_delta_1s",
        "basis_delta_60s",
        "basis_delta_300s",
        # Lagged delta
        "basis_delta_60s_lag_1s",
        "basis_delta_60s_lag_60s",
        "basis_delta_60s_lag_300s",
        # ------------------------------------------------------------------
        # Rolling
        # ------------------------------------------------------------------
        "basis_roll_mean_10s",
        "basis_roll_mean_30s",
        "basis_roll_mean_30s_lag_10s",
        "basis_roll_std_60s",
        "basis_roll_min_300s",
        "basis_roll_max_600s",
        # Multiple rolling parameters are represented by separate outputs
        "basis_roll_mean_10s_lag_60s",
        "basis_roll_std_60s_lag_300s",
        "basis_roll_min_300s_lag_60s",
        "basis_roll_max_600s_lag_60s",
        # ------------------------------------------------------------------
        # Ratio
        # ------------------------------------------------------------------
        "basis_day_ratio",
        "basis_hour_ratio",
        "basis_day_ratio_lag_60s",
        "basis_hour_ratio_lag_300s",
        # ------------------------------------------------------------------
        # Base features with underscores
        # ------------------------------------------------------------------
        "fut_bid_price_1",
        "fut_ask_price_5",
        "fut_bid_size_1",
        "fut_ask_size_10",
        "fut_bid_price_1_delta_60s",
        "fut_ask_price_5_delta_300s",
        "fut_bid_size_1_roll_mean_60s",
        "fut_ask_size_10_roll_std_300s",
        "fut_bid_price_1_lag_60s",
        "fut_ask_price_5_delta_60s_lag_300s",
        # ------------------------------------------------------------------
        # More complex existing base IDs
        # ------------------------------------------------------------------
        "fut_obi_3",
        "fut_obi_3_delta_60s",
        "fut_obi_3_lag_300s",
        "fut_bof_1",
        "fut_bof_1_delta_60s",
        "fut_ofi_2",
        "fut_ofi_2_roll_std_300s",
        "fut_ofi_2_lag_60s",
        "fut_mid_price_delta_1s",
        "fut_mid_price_roll_mean_60s",
        "fut_mid_price_roll_std_300s_lag_60s",
        "fut_micro_price",
        "fut_micro_price_delta_60s",
        "fut_micro_price_lag_300s",
        "fut_spread",
        "fut_spread_roll_mean_60s",
        "fut_spread_roll_max_300s_lag_60s",
        # ------------------------------------------------------------------
        # Same base + several independent transforms
        # ------------------------------------------------------------------
        "basis_roll_mean_60s",
        "basis_roll_mean_60s_lag_300s",
    ]

    print("Specs correctly reconstructed!")
