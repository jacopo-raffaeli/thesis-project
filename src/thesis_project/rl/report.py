from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from thesis_project import config

Records = dict[int, pd.DataFrame]


_FIGSIZE_DEFAULT = (12, 6)
_FIGSIZE_HEATMAP = (14, 8)


def load_records(
    path: Path,
    *,
    batch_size: int,
    clip_range: float,
    seeds: list[int],
    split: str,
) -> Records:
    records = {}

    for seed in seeds:
        file = path / f"bs_{batch_size}_cr_{clip_range}_{seed}_{split}.parquet"

        if not file.exists():
            raise FileNotFoundError(file)

        records[seed] = pd.read_parquet(file)

    return records


def remove_reset_observations(
    records: pd.DataFrame,
) -> pd.DataFrame:
    indices = records.groupby("episode", sort=False).head(1).index

    return records.drop(indices).copy()


def remove_terminal_observations(
    records: pd.DataFrame,
) -> pd.DataFrame:
    indeces = ~records["terminal"]

    return records.loc[indeces].copy()


def derive_records(records: pd.DataFrame) -> pd.DataFrame:
    data = records.copy()

    # Order data
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values(
        ["episode", "timestamp"],
    ).reset_index(drop=True)

    # Compute cumulative rewards
    data["cum_reward"] = data["reward"].cumsum()
    data["cum_gross_reward"] = data["gross_reward"].cumsum()
    data["cum_cost"] = data["cost"].cumsum()

    # Compute drawdowns
    data["cum_max_reward"] = data["cum_reward"].cummax()
    data["cum_max_gross_reward"] = data["cum_gross_reward"].cummax()
    data["drawdown"] = data["cum_reward"] - data["cum_max_reward"]
    data["gross_drawdown"] = data["cum_gross_reward"] - data["cum_max_gross_reward"]

    # Compute trades statistics
    data["trade"] = data["position"] != data["allocation"]
    data["turnover"] = (data["position"] - data["allocation"]).abs()

    data["profit"] = data["trade"] & (data["reward"] > 0)
    data["neutral"] = data["trade"] & (data["reward"] == 0)
    data["loss"] = data["trade"] & (data["reward"] < 0)

    return data


def daily_pnl(records: pd.DataFrame) -> pd.DataFrame:
    data = records.copy()

    daily = data.groupby("episode").agg(
        gross_pnl=("gross_reward", "sum"),
        net_pnl=("reward", "sum"),
        cost=("cost", "sum"),
    )

    return daily


def run_metrics(
    records: pd.DataFrame,
) -> dict[str, float]:
    daily = daily_pnl(records)

    daily_returns = daily["net_pnl"]

    daily_mean = daily_returns.mean()
    daily_std = daily_returns.std()

    sortino_target = 0.0
    downside = daily_returns[daily_returns < sortino_target] - sortino_target

    downside_std = np.sqrt(np.mean(downside.pow(2)))

    cumulative = daily_returns.cumsum()
    running_max = cumulative.cummax()
    daily_drawdown = cumulative - running_max

    action_counts = records["position"].value_counts()

    total_actions = len(records)
    total_trades = records["trade"].sum()

    traded_records = records.loc[records["trade"]]

    profitable_trades = traded_records["profit"].sum()
    neutral_trades = traded_records["neutral"].sum()
    losing_trades = traded_records["loss"].sum()

    return {
        "n_days": float(len(daily)),
        "n_steps": float(len(records)),
        "steps_per_day": float(len(records) / len(daily)),
        "gross_pnl": records["gross_reward"].sum(),
        "net_pnl": records["reward"].sum(),
        "cost": records["cost"].sum(),
        "gross_pnl_per_day": daily["gross_pnl"].mean(),
        "net_pnl_per_day": daily["net_pnl"].mean(),
        "cost_per_day": daily["cost"].mean(),
        "daily_pnl_mean": daily_mean,
        "daily_pnl_std": daily_std,
        "sharpe": (daily_mean / daily_std if daily_std != 0 else np.nan),
        "sortino": ((daily_mean - sortino_target) / downside_std if downside_std != 0 else np.nan),
        "max_drawdown": daily_drawdown.min(),
        "worst_day": daily_returns.min(),
        "best_day": daily_returns.max(),
        "profitable_days": (daily_returns > 0).mean(),
        "neutral_days": (daily_returns == 0).mean(),
        "losing_days": (daily_returns < 0).mean(),
        "short": float(action_counts.get(-1, 0)),
        "flat": float(action_counts.get(0, 0)),
        "long": float(action_counts.get(1, 0)),
        "short_pct": (action_counts.get(-1, 0) / total_actions),
        "flat_pct": (action_counts.get(0, 0) / total_actions),
        "long_pct": (action_counts.get(1, 0) / total_actions),
        "trades": float(total_trades),
        "turnover": records["turnover"].sum(),
        "profitable_trades": float(profitable_trades),
        "neutral_trades": float(neutral_trades),
        "losing_trades": float(losing_trades),
        "win_rate": (profitable_trades / total_trades if total_trades != 0 else np.nan),
        "positive_rewards": (records["reward"] > 0).mean(),
        "zero_rewards": (records["reward"] == 0).mean(),
        "negative_rewards": (records["reward"] < 0).mean(),
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


def plot_pnl(
    records: Records,
    *,
    label: str | None = None,
) -> None:
    data = pd.concat(
        [records[seed]["cum_reward"].rename(seed) for seed in records],
        axis=1,
    )

    gross = pd.concat(
        [records[seed]["cum_gross_reward"].rename(seed) for seed in records],
        axis=1,
    )

    net_mean = data.mean(axis=1)
    gross_mean = gross.mean(axis=1)

    config.default_plt()

    fig, ax = plt.subplots(figsize=_FIGSIZE_DEFAULT)

    ax.plot(
        gross_mean,
        label="Gross",
    )

    ax.plot(
        net_mean,
        label="Net",
    )

    if len(records) > 1:
        net_std = data.std(axis=1)
        gross_std = gross.std(axis=1)

        ax.fill_between(
            gross_mean.index,
            gross_mean - gross_std,
            gross_mean + gross_std,
            alpha=0.2,
        )

        ax.fill_between(
            net_mean.index,
            net_mean - net_std,
            net_mean + net_std,
            alpha=0.2,
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("€")
    ax.set_title("Cumulative PnL" if label is None else f"Cumulative PnL - {label}")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="upper left")

    fig.tight_layout()
    plt.show()


def plot_pnl_distribution(
    records: Records,
    *,
    label: str | None = None,
) -> None:
    daily = pd.concat(
        [daily_pnl(data)["net_pnl"].rename(seed) for seed, data in records.items()],
        axis=1,
    )

    config.default_plt()

    fig, ax = plt.subplots(figsize=_FIGSIZE_DEFAULT)

    ax.hist(
        daily.to_numpy().ravel(),
        bins=50,
    )

    ax.set_xlabel("Daily PnL")
    ax.set_ylabel("Count")
    ax.set_title("Daily PnL Distribution" if label is None else f"Daily PnL Distribution - {label}")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.show()


def plot_daily_pnl(
    records: Records,
    *,
    label: str | None = None,
) -> None:
    daily = pd.concat(
        [daily_pnl(data)["net_pnl"].rename(seed) for seed, data in records.items()],
        axis=1,
    )

    mean = daily.mean(axis=1)

    config.default_plt()

    fig, ax = plt.subplots(figsize=_FIGSIZE_DEFAULT)

    ax.plot(mean)

    if len(records) > 1:
        std = daily.std(axis=1)

        ax.fill_between(
            mean.index,
            mean - std,
            mean + std,
            alpha=0.2,
        )

    ax.axhline(
        0,
        linewidth=0.8,
    )

    ax.set_xlabel("Trading day")
    ax.set_ylabel("€")
    ax.set_title("Daily PnL" if label is None else f"Daily PnL - {label}")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.show()


def plot_drawdown(
    records: Records,
    *,
    label: str | None = None,
) -> None:
    data = pd.concat(
        [records[seed]["drawdown"].rename(seed) for seed in records],
        axis=1,
    )

    mean = data.mean(axis=1)

    config.default_plt()

    fig, ax = plt.subplots(figsize=_FIGSIZE_DEFAULT)

    if len(records) == 1:
        ax.plot(
            mean,
            label="Net",
        )
    else:
        std = data.std(axis=1)

        ax.plot(
            mean,
            label="Net",
        )
        ax.fill_between(
            mean.index,
            mean - std,
            mean + std,
            alpha=0.2,
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("€")
    ax.set_title("Drawdown" if label is None else f"Drawdown - {label}")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="lower right")

    fig.tight_layout()
    plt.show()


def plot_action_series(
    records: Records,
    *,
    label: str | None = None,
) -> None:
    config.default_plt()

    fig, ax = plt.subplots(figsize=_FIGSIZE_DEFAULT)

    for seed, data in records.items():
        ax.plot(
            data["position"],
            label=str(seed),
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Action")
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Short", "Flat", "Long"])
    ax.set_title("Agent's Actions" if label is None else f"Agent's Actions - {label}")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if len(records) > 1:
        ax.legend()

    fig.tight_layout()
    plt.show()


def _plot_heatmap(
    data: pd.DataFrame,
    *,
    value: str,
    persistence_min: int,
    date_tick_step: int,
    title: str,
) -> None:
    data = data.copy()

    data["timestamp"] = pd.to_datetime(data["timestamp"])

    close = data.loc[data["terminal"], "timestamp"].iloc[0]
    close_min = close.hour * 60 + close.minute

    data = data.loc[~data["terminal"]].copy()

    data["date"] = data["timestamp"].dt.normalize()

    data["time_min"] = (
        data["timestamp"].dt.hour * 60
        + data["timestamp"].dt.minute
        + data["timestamp"].dt.second / 60
    )

    values = data.pivot(
        index="date",
        columns="time_min",
        values=value,
    ).sort_index()

    assert isinstance(values.index, pd.DatetimeIndex)

    if values.empty:
        raise ValueError("Evaluation contains no non-terminal observations.")

    times = values.columns.to_numpy(float)

    edges = np.append(
        times,
        min(times[-1] + persistence_min, close_min),
    )

    if value == "position":
        matrix = values.to_numpy(float)

        colors = [
            "#ff0000",
            "#ffffff",
            "#008000",
        ]

        cmap = ListedColormap(colors)
        norm = BoundaryNorm(
            [-1.5, -0.5, 0.5, 1.5],
            cmap.N,
        )

        legend = [
            Patch(facecolor=colors[0], label="Short"),
            Patch(facecolor=colors[1], label="Flat"),
            Patch(facecolor=colors[2], label="Long"),
        ]

    elif value == "reward":
        matrix = values.to_numpy(float)
        matrix = np.where(
            matrix > 0,
            1,
            np.where(matrix < 0, -1, 0),
        )

        colors = [
            "#ff0000",
            "#ffffff",
            "#008000",
        ]

        cmap = ListedColormap(colors)
        norm = BoundaryNorm(
            [-1.5, -0.5, 0.5, 1.5],
            cmap.N,
        )

        legend = [
            Patch(facecolor=colors[0], label="Negative"),
            Patch(facecolor=colors[1], label="Zero"),
            Patch(facecolor=colors[2], label="Positive"),
        ]

    else:
        raise ValueError(f"Unsupported heatmap value: {value!r}")

    config.default_plt()

    fig, ax = plt.subplots(figsize=_FIGSIZE_HEATMAP)

    ax.pcolormesh(
        edges,
        np.arange(len(values) + 1),
        np.ma.masked_invalid(matrix),
        cmap=cmap,
        norm=norm,
        shading="flat",
    )

    ticks = np.arange(
        np.ceil(times[0] / 60) * 60,
        close_min + 1,
        60,
    )

    ax.set(
        xticks=ticks,
        xticklabels=[f"{int(t // 60):02d}:{int(t % 60):02d}" for t in ticks],
        xlim=(times[0], close_min),
        yticks=(
            np.arange(
                0,
                len(values),
                date_tick_step,
            )
            + 0.5
        ),
        yticklabels=(values.index[::date_tick_step].strftime("%Y-%m-%d")),
        ylim=(len(values), 0),
        xlabel="Time of day",
        ylabel="Date",
        title=title,
    )

    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(
        handles=legend,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )

    fig.tight_layout(
        rect=(0, 0, 0.90, 1),
    )

    plt.show()


def plot_action_heatmaps(
    records: Records,
    *,
    persistence_min: int,
    date_tick_step: int = 10,
    label: str | None = None,
) -> None:
    for seed, data in records.items():
        title = f"Trading Heatmap - Seed {seed}"

        if label is not None:
            title = f"{title} - {label}"

        _plot_heatmap(
            data,
            value="position",
            persistence_min=persistence_min,
            date_tick_step=date_tick_step,
            title=title,
        )


def plot_reward_heatmaps(
    records: Records,
    *,
    persistence_min: int,
    date_tick_step: int = 10,
    label: str | None = None,
) -> None:
    for seed, data in records.items():
        title = f"Reward Heatmap - Seed {seed}"

        if label is not None:
            title = f"{title} - {label}"

        _plot_heatmap(
            data,
            value="reward",
            persistence_min=persistence_min,
            date_tick_step=date_tick_step,
            title=title,
        )


def print_summary(
    metrics: pd.DataFrame,
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    n = 22
    n_seeds = metrics.attrs["n_seeds"]

    evaluation = {
        "n_days": metrics[next(iter(metrics))]["n_days"],
        "steps_per_day": metrics[next(iter(metrics))]["steps_per_day"],
        "n_steps": metrics[next(iter(metrics))]["n_steps"],
    }

    print("\nSummary:")

    if metadata:
        for key, value in metadata.items():
            print(f"- {key + ':':<{n}} {value}")

    print(f"- {'Seeds:':<{n}} " f"{n_seeds:,}")

    print("\nEvaluation:")
    print(f"- {'Trading days:':<{n}} " f"{evaluation['n_days']:.0f}")
    print(f"- {'Steps per day:':<{n}} " f"{evaluation['steps_per_day']:.0f}")
    print(f"- {'Total steps:':<{n}} " f"{evaluation['n_steps']:,.0f}")

    print("\nPnL:")
    print(f"- {'Gross PnL:':<{n}} " f"{_format_metric(metrics, 'gross_pnl')} €")
    print(f"- {'Net PnL:':<{n}} " f"{_format_metric(metrics, 'net_pnl')} €")
    print(f"- {'Cost:':<{n}} " f"{_format_metric(metrics, 'cost')} €")
    print(f"- {'Gross PnL / day:':<{n}} " f"{_format_metric(metrics, 'gross_pnl_per_day')} €")
    print(f"- {'Net PnL / day:':<{n}} " f"{_format_metric(metrics, 'net_pnl_per_day')} €")

    print("\nRisk:")
    print(f"- {'Daily volatility:':<{n}} " f"{_format_metric(metrics, 'daily_pnl_std')} €")
    print(f"- {'Sharpe ratio:':<{n}} " f"{_format_metric(metrics, 'sharpe')}")
    print(f"- {'Sortino ratio:':<{n}} " f"{_format_metric(metrics, 'sortino')}")
    print(f"- {'Maximum drawdown:':<{n}} " f"{_format_metric(metrics, 'max_drawdown')} €")
    print(f"- {'Worst day:':<{n}} " f"{_format_metric(metrics, 'worst_day')} €")
    print(f"- {'Best day:':<{n}} " f"{_format_metric(metrics, 'best_day')} €")
    print(f"- {'Profitable days:':<{n}} " f"{_format_percentage(metrics, 'profitable_days')}")
    print(f"- {'Losing days:':<{n}} " f"{_format_percentage(metrics, 'losing_days')}")

    print("\nActions:")
    print(f"- {'Short:':<{n}} " f"{_format_percentage(metrics, 'short_pct')}")
    print(f"- {'Flat:':<{n}} " f"{_format_percentage(metrics, 'flat_pct')}")
    print(f"- {'Long:':<{n}} " f"{_format_percentage(metrics, 'long_pct')}")

    print("\nTrades:")
    print(f"- {'Number of trades:':<{n}} " f"{_format_metric(metrics, 'trades')}")
    print(f"- {'Turnover:':<{n}} " f"{_format_metric(metrics, 'turnover')}")
    print(f"- {'Profitable trades:':<{n}} " f"{_format_metric(metrics, 'profitable_trades')}")
    print(f"- {'Neutral trades:':<{n}} " f"{_format_metric(metrics, 'neutral_trades')}")
    print(f"- {'Losing trades:':<{n}} " f"{_format_metric(metrics, 'losing_trades')}")
    print(f"- {'Win rate:':<{n}} " f"{_format_percentage(metrics, 'win_rate')}")

    print("\nRewards:")
    print(f"- {'Positive rewards:':<{n}} " f"{_format_percentage(metrics, 'positive_rewards')}")
    print(f"- {'Zero rewards:':<{n}} " f"{_format_percentage(metrics, 'zero_rewards')}")
    print(f"- {'Negative rewards:':<{n}} " f"{_format_percentage(metrics, 'negative_rewards')}")


def _format_metric(
    metrics: pd.DataFrame,
    name: str,
) -> str:
    mean = metrics.loc[name, "mean"]

    if metrics.attrs["n_seeds"] == 1:
        return f"{mean:.2f}"

    std = metrics.loc[name, "std"]

    if pd.isna(std):
        return f"{mean:.2f}"

    return f"{mean:.2f} ± {std:.2f}"


def _format_percentage(
    metrics: pd.DataFrame,
    name: str,
) -> str:
    mean = metrics.loc[name, "mean"] * 100  # type: ignore

    if metrics.attrs["n_seeds"] == 1:
        return f"{mean:.2f} %"

    std = metrics.loc[name, "std"] * 100  # type: ignore

    if pd.isna(std):
        return f"{mean:.2f} %"

    return f"{mean:.2f} ± {std:.2f} %"


def report(
    records: Records,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    metadata = {} if metadata is None else metadata

    label = metadata.get("set")
    persistence_min = metadata.get("persistence_min", 10)
    date_tick_step = metadata.get("date_tick_step", 10)

    records = {
        seed: derive_records(remove_reset_observations(data)) for seed, data in records.items()
    }

    metrics = {
        seed: run_metrics(
            data,
        )
        for seed, data in records.items()
    }

    aggregated = aggregate_metrics(metrics)
    aggregated.attrs["n_seeds"] = len(records)

    print_summary(
        aggregated,
        metadata=metadata,
    )

    plot_pnl(
        records,
        label=label,
    )

    plot_pnl_distribution(
        records,
        label=label,
    )

    plot_daily_pnl(
        records,
        label=label,
    )

    plot_drawdown(
        records,
        label=label,
    )

    # plot_action_series(
    #     records,
    #     label=label,
    # )

    print()
    print()
    print("Action Heatmap:")
    plot_action_heatmaps(
        records,
        persistence_min=persistence_min,
        date_tick_step=date_tick_step,
        label=label,
    )

    print()
    print()
    print("Reward Heatmap:")
    plot_reward_heatmaps(
        records,
        persistence_min=persistence_min,
        date_tick_step=date_tick_step,
        label=label,
    )
