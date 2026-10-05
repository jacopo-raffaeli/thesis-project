import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from thesis_project import config

Records = dict[int, pd.DataFrame]
ExecutionRecords = dict[config.LobSide, Records]


_FIGSIZE_DEFAULT = (12, 6)


def load_records(
    path: Path,
    *,
    batch_size: int,
    clip_range: float,
    seeds: list[int],
    split: str,
    side: config.LobSide,
) -> Records:
    """Load execution records for one side."""
    records = {}

    for seed in seeds:
        file = path / f"{side}_bs_{batch_size}_cr_{clip_range}_{seed}_{split}.parquet"

        if not file.exists():
            raise FileNotFoundError(file)

        records[seed] = pd.read_parquet(file)

    return records


def load_paired_records(
    path: Path,
    *,
    batch_size: int,
    clip_range: float,
    seeds: list[int],
    split: str,
) -> ExecutionRecords:
    """Load bid and ask execution records."""
    return {
        side: load_records(
            path,
            batch_size=batch_size,
            clip_range=clip_range,
            seeds=seeds,
            split=split,
            side=side,
        )
        for side in ("bid", "ask")
    }


def load_metadata(
    path: Path,
    *,
    batch_size: int,
    clip_range: float,
    seeds: list[int],
    split: str,
    env: str = "serial",
) -> dict[str, Any]:
    """Load experiment, dataset, and execution-environment metadata."""
    config_file = path / "config.json"

    if not config_file.exists():
        raise FileNotFoundError(config_file)

    with config_file.open() as file:
        run_config = json.load(file)

    experiment = run_config["experiment"]
    dataset = run_config["dataset"]
    environment = run_config[f"env_{env}"]

    return {
        # Experiment
        "split": split,
        "split_month": experiment["split_month"],
        "total_timesteps": experiment["total_timesteps"],
        "batch_size": batch_size,
        "clip_range": clip_range,
        "normalize_market_obs": experiment["normalize_market_obs"],
        # Dataset
        "ticker": dataset["ticker"],
        "contract_mode": dataset["contract_mode"],
        "ctd_contracts": dataset["ctd_contracts"],
        "market_set": dataset["market_set"],
        "calendar_set": dataset["calendar_set"],
        "calendar_enc_set": dataset["calendar_enc_set"],
        # Environment
        "reset_mode": environment["reset_mode"],
        "horizon_min": environment["horizon_min"],
        "step_sec": environment["step_sec"],
        "max_n_tick": environment["max_n_tick"],
        "tick_size": environment["tick_size"],
        # Seeds
        "seeds": seeds,
    }


def _prepare_record(records: pd.DataFrame) -> pd.DataFrame:
    """Keep one terminal observation per execution episode."""
    data = records.copy()

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["execution_time"] = pd.to_datetime(data["execution_time"])

    data = data.sort_values(["episode", "timestamp"])

    terminal = data.loc[data["terminal"]].copy()

    if terminal["episode"].duplicated().any():
        raise ValueError("Multiple terminal observations found for an episode")

    opening_times = data.groupby("episode", sort=False)["timestamp"].first().rename("opening_time")

    terminal = terminal.join(opening_times, on="episode")

    terminal["execution_time_sec"] = (
        terminal["execution_time"] - terminal["opening_time"]
    ).dt.total_seconds()

    return terminal.reset_index(drop=True)


def prepare_records(records: Records) -> Records:
    return {seed: _prepare_record(data) for seed, data in records.items()}


def _execution_improvement(
    records: pd.DataFrame,
    *,
    side: config.LobSide,
    tick_size: float,
) -> pd.Series:
    """Execution improvement relative to the benchmark, in ticks."""
    if side == "bid":
        return (records["benchmark_price"] - records["execution_price"]) / tick_size
    else:
        return (records["execution_price"] - records["benchmark_price"]) / tick_size


def _passive_quote_distances(
    records: pd.DataFrame,
    *,
    side: config.LobSide,
) -> pd.Series:
    """Return submitted passive quote distances, in ticks.

    Action 0 is market execution. Positive actions are quote distances
    in ticks from the current market price.
    """
    return records.loc[records["action"] > 0, "action"].astype(float)


def run_metrics(
    records: pd.DataFrame,
    *,
    side: config.LobSide,
    tick_size: float,
) -> dict[str, float]:
    """Compute execution metrics for one seed."""
    if records.empty:
        raise ValueError("No terminal execution records found")

    improvement = _execution_improvement(records, side=side, tick_size=tick_size)

    status_counts = records["status"].value_counts()

    n_episodes = len(records)
    n_market = int(status_counts.get("market", 0))
    n_limit = int(status_counts.get("limit", 0))
    n_forced = int(status_counts.get("forced", 0))

    passive_attempts = n_limit + n_forced
    limit_fill_rate = n_limit / passive_attempts if passive_attempts != 0 else np.nan

    quote_distances = _passive_quote_distances(records, side=side)

    return {
        "episodes": float(n_episodes),
        "market": float(n_market),
        "limit": float(n_limit),
        "forced": float(n_forced),
        "market_pct": n_market / n_episodes,
        "limit_pct": n_limit / n_episodes,
        "forced_pct": n_forced / n_episodes,
        "limit_fill_rate": limit_fill_rate,
        "execution_improvement_mean": improvement.mean(),
        "execution_improvement_median": improvement.median(),
        "execution_time_mean": records["execution_time_sec"].mean(),
        "execution_time_median": records["execution_time_sec"].median(),
        "passive_quote_distance_mean": (
            quote_distances.mean() if not quote_distances.empty else np.nan
        ),
        "passive_quote_distance_median": (
            quote_distances.median() if not quote_distances.empty else np.nan
        ),
    }


def aggregate_metrics(
    metrics: dict[int, dict[str, float]],
) -> pd.DataFrame:
    data = pd.DataFrame.from_dict(
        metrics,
        orient="index",
    )
    data.index.name = "seed"

    return pd.DataFrame(
        {
            "mean": data.mean(),
            "std": data.std(),
        }
    )


def summarize(
    records: Records,
    *,
    side: config.LobSide,
    tick_size: float,
) -> pd.DataFrame:
    metrics = {
        seed: run_metrics(data, side=side, tick_size=tick_size) for seed, data in records.items()
    }

    summary = aggregate_metrics(metrics)
    summary.attrs["n_seeds"] = len(records)

    return summary


def _format_metric(
    metrics: pd.DataFrame,
    name: str,
    *,
    decimals: int = 2,
    suffix: str = "",
) -> str:
    mean = metrics.loc[name, "mean"]

    if metrics.attrs["n_seeds"] == 1:
        return f"{mean:,.{decimals}f}{suffix}"

    std = metrics.loc[name, "std"]

    if pd.isna(std):
        return f"{mean:,.{decimals}f}{suffix}"

    return f"{mean:,.{decimals}f} ± " f"{std:,.{decimals}f}{suffix}"


def _format_percentage(
    metrics: pd.DataFrame,
    name: str,
) -> str:
    mean = metrics.loc[name, "mean"] * 100

    if metrics.attrs["n_seeds"] == 1:
        return f"{mean:.2f}%"

    std = metrics.loc[name, "std"]

    if pd.isna(std):
        return f"{mean:.2f}%"

    return f"{mean:.2f} ± {std * 100:.2f}%"


def format_summary_table(
    summaries: dict[config.LobSide, pd.DataFrame],
) -> pd.DataFrame:
    """Format bid and ask summaries side by side."""
    rows = [
        ("Episodes", "episodes", "integer"),
        ("Market executions", "market_pct", "percentage"),
        ("Limit executions", "limit_pct", "percentage"),
        ("Forced executions", "forced_pct", "percentage"),
        ("Limit fill rate", "limit_fill_rate", "percentage"),
        (
            "Mean execution improvement",
            "execution_improvement_mean",
            "ticks",
        ),
        (
            "Median execution improvement",
            "execution_improvement_median",
            "ticks",
        ),
        ("Mean execution time", "execution_time_mean", "seconds"),
        ("Median execution time", "execution_time_median", "seconds"),
        (
            "Mean passive quote distance",
            "passive_quote_distance_mean",
            "ticks",
        ),
        (
            "Median passive quote distance",
            "passive_quote_distance_median",
            "ticks",
        ),
    ]

    formatted = pd.DataFrame(
        index=[row[0] for row in rows],
        columns=["Bid", "Ask"],
        dtype=object,
    )

    for label, metric, metric_type in rows:
        for side, column in (("bid", "Bid"), ("ask", "Ask")):
            summary = summaries[side]

            if metric_type == "percentage":
                value = _format_percentage(summary, metric)
            elif metric_type == "integer":
                value = _format_metric(
                    summary,
                    metric,
                    decimals=0,
                )
            elif metric_type == "seconds":
                value = _format_metric(
                    summary,
                    metric,
                    decimals=1,
                    suffix=" s",
                )
            else:
                value = _format_metric(
                    summary,
                    metric,
                    decimals=2,
                    suffix=" ticks",
                )

            formatted.loc[label, column] = value

    return formatted


def print_summary(
    summaries: dict[config.LobSide, pd.DataFrame],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    if metadata:
        print("\nSettings:")

        for key, value in metadata.items():
            print(f"- {key + ':':<30} {value}")

    print("\nExecution Performance:")
    display(format_summary_table(summaries))


def plot_execution_improvement(
    records: ExecutionRecords,
    *,
    tick_size: float,
    label: str | None = None,
) -> None:
    """Plot execution improvement for bid and ask side by side."""
    config.default_plt()

    prepared = {side: prepare_records(records[side]) for side in ("bid", "ask")}

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        sharey=True,
    )

    status_colors = {
        "market": "tab:blue",
        "limit": "tab:green",
        "forced": "tab:red",
    }

    for ax, side in zip(axes, ("bid", "ask")):
        data = pd.concat(
            [frame.assign(seed=seed) for seed, frame in prepared[side].items()],
            ignore_index=True,
        )

        data["execution_improvement"] = _execution_improvement(
            data,
            side=side,
            tick_size=tick_size,
        )

        data = data.sort_values(["seed", "episode"]).reset_index(drop=True)
        data["plot_episode"] = np.arange(1, len(data) + 1)

        for status, color in status_colors.items():
            subset = data.loc[data["status"] == status]

            ax.scatter(
                subset["plot_episode"],
                subset["execution_improvement"],
                s=10,
                alpha=0.6,
                color=color,
                label=status.capitalize(),
            )

        ax.axhline(
            0,
            linewidth=0.5,
            color="black",
            linestyle="--",
        )

        ax.set_xlabel("Episode")
        ax.set_title(side.capitalize())

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(loc="best")

    axes[0].set_ylabel("Execution improvement (ticks)")

    title = "Execution Improvement per Episode"
    if label is not None:
        title += f" - {label}"

    fig.suptitle(title)
    fig.tight_layout()

    plt.show()


def report(
    records: ExecutionRecords,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Generate the execution report for paired bid/ask records."""
    prepared = {side: prepare_records(records[side]) for side in ("bid", "ask")}

    if metadata is None or "tick_size" not in metadata:
        raise ValueError("Execution report requires 'tick_size' in metadata")

    tick_size = float(metadata["tick_size"])

    summaries = {
        side: summarize(
            prepared[side],
            side=side,
            tick_size=tick_size,
        )
        for side in ("bid", "ask")
    }

    label = None if metadata is None else metadata.get("split")

    print_summary(
        summaries,
        metadata=metadata,
    )

    plot_execution_improvement(
        prepared,
        tick_size=tick_size,
        label=label,
    )
