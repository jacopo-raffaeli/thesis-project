from collections import defaultdict
from typing import Callable, cast, get_args

from thesis_project.dataset.data import BASE_FEATURES
from thesis_project.dataset.features_market import FeatureSpec, validate_feature_specs
from thesis_project.dataset.features_temporal import CalFeatEncType, CalFeatType
from thesis_project.dataset.features_transforms import (
    BaseTransform,
    Delta,
    Identity,
    Ratio,
    Rolling,
    _validate_deltas,
    _validate_lags,
    _validate_references,
    _validate_stats,
    _validate_windows,
)


def _parse_int(
    value: str,
    suffix: str,
) -> int:
    if not value.endswith(suffix):
        raise ValueError(f"Expected suffix '{suffix}', got '{value}'")

    try:
        return int(value.removesuffix(suffix))
    except ValueError as exc:
        raise ValueError(f"Invalid integer in '{value}'") from exc


def parse_lag_suffix(
    name: str,
) -> tuple[str, list[int]]:
    """
    Split <transform_name>_lag_<n>s into <transform_name>, [lags]
    """
    parts = name.split("_")
    lags = []

    while len(parts) >= 2 and parts[-2] == "lag":
        lag = _parse_int(
            parts[-1],
            "s",
        )

        lags.append(lag)
        parts = parts[:-2]

    lags.reverse()

    if lags:
        _validate_lags(lags)

    return "_".join(parts), lags


def parse_identity(name: str, base_id: str) -> Identity:
    suffix = name.removeprefix(base_id)

    if suffix == "":
        lags = []
    else:
        if not suffix.startswith("_"):
            raise ValueError(f"Invalid Identity feature name: '{name}'")

        _, lags = parse_lag_suffix(suffix[1:])

    return Identity(
        lags=lags if bool(lags) else None,
        keep_original=False if bool(lags) else True,
    )


def parse_delta(name: str, base_id: str) -> Delta:
    suffix = name.removeprefix(f"{base_id}_")
    transform_name, lags = parse_lag_suffix(suffix)

    parts = transform_name.split("_")

    if len(parts) != 2 or parts[0] != "delta":
        raise ValueError(f"Invalid Delta feature name: '{name}'")

    delta = _parse_int(parts[1], "s")
    _validate_deltas([delta])

    return Delta(
        deltas=[delta],
        lags=lags if bool(lags) else None,
        keep_original=False if bool(lags) else True,
    )


def parse_rolling(name: str, base_id: str) -> Rolling:
    suffix = name.removeprefix(f"{base_id}_")
    transform_name, lags = parse_lag_suffix(suffix)

    parts = transform_name.split("_")

    if len(parts) != 3 or parts[0] != "roll":
        raise ValueError(f"Invalid Rolling feature name: '{name}'")

    stat = parts[1]
    window = _parse_int(parts[2], "s")

    _validate_stats([stat])
    _validate_windows([window])

    return Rolling(
        stats=[cast(Rolling.RollingStat, stat)],
        windows=[window],
        lags=lags if bool(lags) else None,
        keep_original=False if bool(lags) else True,
    )


def parse_ratio(name: str, base_id: str) -> Ratio:
    suffix = name.removeprefix(f"{base_id}_")
    transform_name, lags = parse_lag_suffix(suffix)

    parts = transform_name.split("_")

    if len(parts) != 2 or parts[1] != "ratio":
        raise ValueError(f"Invalid Ratio feature name: '{name}'")

    reference = parts[0]

    _validate_references([reference])

    return Ratio(
        references=[cast(Ratio.RatioReference, reference)],
        lags=lags if bool(lags) else None,
        keep_original=False if bool(lags) else True,
    )


TRANSFORM_PARSERS = {
    "delta": parse_delta,
    "roll": parse_rolling,
    "day": parse_ratio,
    "hour": parse_ratio,
}


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
):
    """
    Reverse-engineer a collection of output feature names.
    """
    calendar = set()
    calendar_enc = set()

    # Filter out calendar features first
    for name in names:
        if name in get_args(CalFeatType):
            calendar.add(name)
            names.remove(name)

        if name.replace("_sin", "") in get_args(CalFeatEncType):
            calendar_enc.add(name.replace("_sin", ""))
            names.remove(name)

        if name.replace("_cos", "") in get_args(CalFeatEncType):
            calendar_enc.add(name.replace("_cos", ""))
            names.remove(name)

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

    return specs, calendar, calendar_enc


if __name__ == "__main__":
    names = [
        # Identity
        "basis",
        "irr",
        "fut_mid_price",
        # Lagged identity
        "basis_lag_1s",
        "basis_lag_60s",
        "fut_mid_price_lag_300s",
        "fut_mid_price_lag_1s",
        # Delta
        "basis_delta_1s",
        "basis_delta_60s",
        "basis_delta_300s",
        # Lagged delta
        "basis_delta_60s_lag_1s",
        "basis_delta_60s_lag_60s",
        "basis_delta_60s_lag_300s",
        # Rolling
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
        # Ratio
        "basis_day_ratio",
        "basis_hour_ratio",
        "basis_day_ratio_lag_60s",
        "basis_hour_ratio_lag_300s",
        # Base features with underscores
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
        # More complex existing base IDs
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
        # Same base + several independent transforms
        "hour_of_day",
        "hour_of_day_sin",
        "hour_of_day_cos",
    ]
    specs = build_feature_specs(names)
    print("Features reversed successfully!")
