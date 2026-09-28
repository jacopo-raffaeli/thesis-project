import json
from typing import Any

import pandas as pd
from pandas._libs.tslibs.nattype import NaTType

from thesis_project import config, rl, utils
from thesis_project.dataset.data import BASE_FEATURES
from thesis_project.utils.misc import get_dates_to_exclude

CTD_SCALE = config.BTP.scale
FUT_SCALE = config.FBTP.scale

OPENING_TIME = config.DEFAULT_OPENING_TIME
CLOSING_TIME = config.DEFAULT_CLOSING_TIME
TIME_WINDOW = (OPENING_TIME, CLOSING_TIME)


def load_market(
    ticker: config.FutTicker,
) -> pd.DataFrame:
    EXCLUDED = get_dates_to_exclude(
        ticker,
        config.DEFAULT_EXCLUDED_DATES,
    )

    ctd_bid_price_1 = (
        utils.io.load_filtered_parquet(
            BASE_FEATURES["ctd_bid_price_1"].path,
            time_window=TIME_WINDOW,
            dates_to_exclude=EXCLUDED,
        )
        * CTD_SCALE
    )

    ctd_ask_price_1 = (
        utils.io.load_filtered_parquet(
            BASE_FEATURES["ctd_ask_price_1"].path,
            time_window=TIME_WINDOW,
            dates_to_exclude=EXCLUDED,
        )
        * CTD_SCALE
    )

    fut_bid_price_1 = (
        utils.io.load_filtered_parquet(
            BASE_FEATURES["fut_bid_price_1"].path,
            time_window=TIME_WINDOW,
            dates_to_exclude=EXCLUDED,
        )
        * FUT_SCALE
    )

    fut_ask_price_1 = (
        utils.io.load_filtered_parquet(
            BASE_FEATURES["fut_ask_price_1"].path,
            time_window=TIME_WINDOW,
            dates_to_exclude=EXCLUDED,
        )
        * FUT_SCALE
    )

    ctd_mid_price = (
        utils.io.load_filtered_parquet(
            BASE_FEATURES["ctd_mid_price"].path,
            time_window=TIME_WINDOW,
            dates_to_exclude=EXCLUDED,
        )
        * CTD_SCALE
    )

    fut_mid_price = (
        utils.io.load_filtered_parquet(
            BASE_FEATURES["fut_mid_price"].path,
            time_window=TIME_WINDOW,
            dates_to_exclude=EXCLUDED,
        )
        * FUT_SCALE
    )

    ctd_spread = (
        utils.io.load_filtered_parquet(
            BASE_FEATURES["ctd_spread"].path,
            time_window=TIME_WINDOW,
            dates_to_exclude=EXCLUDED,
        )
        * CTD_SCALE
    )

    fut_spread = (
        utils.io.load_filtered_parquet(
            BASE_FEATURES["fut_spread"].path,
            time_window=TIME_WINDOW,
            dates_to_exclude=EXCLUDED,
        )
        * FUT_SCALE
    )

    assert ctd_bid_price_1.index.equals(ctd_ask_price_1.index)
    assert ctd_bid_price_1.index.equals(fut_bid_price_1.index)
    assert ctd_bid_price_1.index.equals(fut_ask_price_1.index)
    assert ctd_bid_price_1.index.equals(ctd_mid_price.index)
    assert ctd_bid_price_1.index.equals(fut_mid_price.index)
    assert ctd_bid_price_1.index.equals(ctd_spread.index)
    assert ctd_bid_price_1.index.equals(fut_spread.index)

    data = [
        ctd_bid_price_1,
        ctd_ask_price_1,
        fut_bid_price_1,
        fut_ask_price_1,
        ctd_mid_price,
        fut_mid_price,
        ctd_spread,
        fut_spread,
    ]

    df = pd.concat(data, axis=1)

    df = df.rename(columns={c: c.replace(".parquet", "") for c in df.columns})

    return df


def load_records(
    ticker: config.FutTicker,
    run: int,
    batch_size: int = 64,
    clip_range: float = 0.2,
) -> rl.report.Records:
    root = config.RES_EXP_DIR / ticker / "rl"
    folder = f"run-{run:03d}"
    path = root / folder

    with open(path / "config.json", "r") as f:
        settings = json.load(f)

    records_train = rl.report.load_records(
        path,
        batch_size=batch_size,
        clip_range=clip_range,
        seeds=settings["seeds"],
        split="train",
    )

    records_test = rl.report.load_records(
        path,
        batch_size=batch_size,
        clip_range=clip_range,
        seeds=settings["seeds"],
        split="test",
    )

    records: rl.report.Records = {}
    for seed in settings["seeds"]:
        record_train = records_train[seed]
        record_test = records_test[seed]
        offset = record_train["episode"].max()
        record_test["episode"] += offset
        record = [record_train, record_test]
        record = pd.concat(record)
        record = record.set_index("timestamp").sort_index()
        records[seed] = record

    return records


def get_trades(
    record_df: pd.DataFrame,
) -> pd.DataFrame:
    trades = []

    for opening_time, row in record_df.iterrows():
        prev_pos = row["allocation"]
        curr_pos = row["position"]

        if curr_pos == 0 or curr_pos == prev_pos:
            continue

        future = record_df.loc[record_df.index > opening_time]

        closing_mask = future["position"] != curr_pos

        if not closing_mask.any():
            continue

        closing_time = future.index[closing_mask][0]
        closing_pos = future.loc[closing_time, "position"]

        closing_future = record_df.loc[record_df.index >= closing_time]

        stopping_mask = closing_future["position"] != closing_pos

        if stopping_mask.any():
            first_different_time = stopping_mask[stopping_mask].index[0]
            stopping_time = closing_future.index[closing_future.index < first_different_time][-1]
        else:
            stopping_time = closing_future.index[-1]

        trades.append(
            {
                "opening_time": opening_time,
                "closing_time": closing_time,
                "stopping_time": stopping_time,
                "direction": curr_pos,
            }
        )

    return pd.DataFrame(trades)


def find_limit_hit(
    df: pd.DataFrame,
    limit_price: float,
    direction: int,
    start_time: pd.Timestamp,
) -> tuple[pd.Timestamp | NaTType, pd.Timedelta | NaTType]:
    match direction:
        case 1:
            # Place a bid limit order at 'limit_price'.
            # Check if the best ask reaches the limit order.
            reachable = df["ctd_ask_price_1"] <= limit_price

        case -1:
            # Place an ask limit order at 'limit_price'.
            # Check if the best bid reaches the limit order.
            reachable = df["ctd_bid_price_1"] >= limit_price

        case _:
            raise ValueError(direction)

    if not reachable.any():
        return pd.NaT, pd.NaT

    hit_time = reachable[reachable].index[0]

    return hit_time, hit_time - start_time


def get_original_trade(
    opening_record: pd.Series,
    closing_record: pd.Series,
    direction: int,
) -> dict[str, Any]:
    ctd_contracts = opening_record["ctd_contracts"]
    fut_contracts = opening_record["fut_contracts"]

    ctd_mid_op = opening_record["ctd_mid"]
    ctd_spread_op = opening_record["ctd_spread"]
    fut_mid_op = opening_record["fut_mid"]
    fut_spread_op = opening_record["fut_spread"]

    ctd_mid_cl = closing_record["ctd_mid"]
    ctd_spread_cl = closing_record["ctd_spread"]
    fut_mid_cl = closing_record["fut_mid"]
    fut_spread_cl = closing_record["fut_spread"]

    basis_ctd_side_op = ctd_mid_op * ctd_contracts
    basis_fut_side_op = fut_mid_op * fut_contracts
    basis_mid_op = basis_ctd_side_op - basis_fut_side_op

    basis_ctd_side_cl = ctd_mid_cl * ctd_contracts
    basis_fut_side_cl = fut_mid_cl * fut_contracts
    basis_mid_cl = basis_ctd_side_cl - basis_fut_side_cl

    cost_op = (ctd_spread_op * ctd_contracts + fut_spread_op * fut_contracts) / 2
    cost_cl = (ctd_spread_cl * ctd_contracts + fut_spread_cl * fut_contracts) / 2
    cost = cost_op + cost_cl

    gross_pnl = direction * (basis_mid_cl - basis_mid_op)
    net_pnl = gross_pnl - cost

    return {
        "ctd_contracts": ctd_contracts,
        "fut_contracts": fut_contracts,
        "ctd_mid_opening": ctd_mid_op,
        "ctd_spread_opening": ctd_spread_op,
        "fut_mid_opening": fut_mid_op,
        "fut_spread_opening": fut_spread_op,
        "basis_mid_opening": basis_mid_op,
        "ctd_mid_closing": ctd_mid_cl,
        "ctd_spread_closing": ctd_spread_cl,
        "fut_mid_closing": fut_mid_cl,
        "fut_spread_closing": fut_spread_cl,
        "basis_mid_closing": basis_mid_cl,
        "cost_opening": cost_op,
        "cost_closing": cost_cl,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
    }


def get_limit_trade(
    opening_record: pd.Series,
    closing_record: pd.Series,
    opening_market: pd.Series,
    closing_market: pd.Series,
    direction: int,
    opening_limit: bool,
    closing_limit: bool,
) -> dict[str, Any]:
    ctd_contracts = opening_record["ctd_contracts"]
    fut_contracts = opening_record["fut_contracts"]

    # Opening execution
    if opening_limit:
        ctd_mid_op = opening_record["ctd_mid"]
        ctd_spread_op = 0.0
    else:
        ctd_mid_op = opening_market["ctd_mid_price"]
        ctd_spread_op = opening_market["ctd_spread"]

    fut_mid_op = opening_market["fut_mid_price"]
    fut_spread_op = opening_market["fut_spread"]

    # Closing execution
    if closing_limit:
        ctd_mid_cl = closing_record["ctd_mid"]
        ctd_spread_cl = 0.0
    else:
        ctd_mid_cl = closing_market["ctd_mid_price"]
        ctd_spread_cl = closing_market["ctd_spread"]

    fut_mid_cl = closing_market["fut_mid_price"]
    fut_spread_cl = closing_market["fut_spread"]

    basis_ctd_side_op = ctd_mid_op * ctd_contracts
    basis_fut_side_op = fut_mid_op * fut_contracts
    basis_mid_op = basis_ctd_side_op - basis_fut_side_op

    basis_ctd_side_cl = ctd_mid_cl * ctd_contracts
    basis_fut_side_cl = fut_mid_cl * fut_contracts
    basis_mid_cl = basis_ctd_side_cl - basis_fut_side_cl

    cost_op = (ctd_spread_op * ctd_contracts + fut_spread_op * fut_contracts) / 2
    cost_cl = (ctd_spread_cl * ctd_contracts + fut_spread_cl * fut_contracts) / 2
    cost = cost_op + cost_cl

    gross_pnl = direction * (basis_mid_cl - basis_mid_op)
    net_pnl = gross_pnl - cost

    return {
        "ctd_mid_opening_limit": ctd_mid_op,
        "ctd_spread_opening_limit": ctd_spread_op,
        "fut_mid_opening_limit": fut_mid_op,
        "fut_spread_opening_limit": fut_spread_op,
        "basis_mid_opening_limit": basis_mid_op,
        "ctd_mid_closing_limit": ctd_mid_cl,
        "ctd_spread_closing_limit": ctd_spread_cl,
        "fut_mid_closing_limit": fut_mid_cl,
        "fut_spread_closing_limit": fut_spread_cl,
        "basis_mid_closing_limit": basis_mid_cl,
        "cost_opening_limit": cost_op,
        "cost_closing_limit": cost_cl,
        "gross_pnl_limit": gross_pnl,
        "net_pnl_limit": net_pnl,
    }


def analyze_trades(
    record_df: pd.DataFrame,
    market_df: pd.DataFrame,
    opening_limit: bool = True,
    closing_limit: bool = True,
) -> list[dict[str, Any]]:
    trades = get_trades(record_df)
    results = []

    for _, trade in trades.iterrows():
        opening_time = trade["opening_time"]
        closing_time = trade["closing_time"]
        stopping_time = trade["stopping_time"]
        direction = trade["direction"]

        opening_record = record_df.loc[opening_time]
        closing_record = record_df.loc[closing_time]

        ctd_contracts = opening_record["ctd_contracts"]
        fut_contracts = opening_record["fut_contracts"]

        # ---------------------------------------------------------
        # Original PPO trade
        # ---------------------------------------------------------

        original = get_original_trade(
            opening_record,
            closing_record,
            direction,
        )

        # ---------------------------------------------------------
        # Entry limit
        # ---------------------------------------------------------

        opening_market_data = market_df.between_time(
            opening_time.time(),
            closing_time.time(),
        )

        if opening_market_data.empty:
            continue

        if opening_limit:
            opening_limit_price = opening_record["ctd_mid"]

            opening_hit_time, opening_time_to_hit = find_limit_hit(
                opening_market_data,
                opening_limit_price,
                direction,
                opening_time,
            )
        else:
            opening_limit_price = pd.NaT
            opening_hit_time = opening_time
            opening_time_to_hit = pd.Timedelta(0)

        # ---------------------------------------------------------
        # Closing limit
        # ---------------------------------------------------------

        closing_market_data = market_df.between_time(
            closing_time.time(),
            stopping_time.time(),
        )

        if closing_market_data.empty:
            continue

        if closing_limit:
            closing_limit_price = closing_record["ctd_mid"]

            closing_hit_time, closing_time_to_hit = find_limit_hit(
                closing_market_data,
                closing_limit_price,
                -direction,
                closing_time,
            )
        else:
            closing_limit_price = pd.NaT
            closing_hit_time = closing_time
            closing_time_to_hit = pd.Timedelta(0)

        # ---------------------------------------------------------
        # Feasibility
        # ---------------------------------------------------------

        opening_limit_hit = pd.notna(opening_hit_time)
        closing_limit_hit = pd.notna(closing_hit_time)

        limit_trade_feasible = opening_limit_hit and closing_limit_hit

        result = {
            # Trade info
            "opening_time": opening_time,
            "closing_time": closing_time,
            "stopping_time": stopping_time,
            "direction": direction,
            "ctd_contracts": ctd_contracts,
            "fut_contracts": fut_contracts,
            # Entry limit
            "opening_limit_price": opening_limit_price,
            "opening_hit_time": opening_hit_time,
            "opening_time_to_hit": opening_time_to_hit,
            # Closing limit
            "closing_limit_price": closing_limit_price,
            "closing_hit_time": closing_hit_time,
            "closing_time_to_hit": closing_time_to_hit,
            # Feasibility
            "opening_limit_hit": opening_limit_hit,
            "closing_limit_hit": closing_limit_hit,
            "limit_trade_feasible": limit_trade_feasible,
            # Original trade
            **original,
        }

        # ---------------------------------------------------------
        # Limit trade PnL
        # ---------------------------------------------------------

        if not limit_trade_feasible:
            results.append(result)
            continue

        assert isinstance(opening_hit_time, pd.Timestamp)
        assert isinstance(closing_hit_time, pd.Timestamp)
        opening_market = market_df.loc[opening_hit_time]
        closing_market = market_df.loc[closing_hit_time]

        assert isinstance(opening_market, pd.Series)
        assert isinstance(closing_market, pd.Series)
        limit_trade = get_limit_trade(
            opening_record=opening_record,
            closing_record=closing_record,
            opening_market=opening_market,
            closing_market=closing_market,
            direction=direction,
            opening_limit=opening_limit,
            closing_limit=closing_limit,
        )

        result.update(limit_trade)
        results.append(result)

    return results


def main(
    ticker: config.FutTicker,
    run: int,
    batch_size: int = 64,
    clip_range: float = 0.2,
    opening_limit: bool = True,
    closing_limit: bool = True,
) -> dict[int, pd.DataFrame]:
    market_df = load_market(ticker)
    records = load_records(
        ticker,
        run,
        batch_size=batch_size,
        clip_range=clip_range,
    )

    results = {}

    for seed, record in records.items():
        seed_results = []

        for _, record_df in record.groupby("episode"):
            assert isinstance(record_df.index, pd.DatetimeIndex)

            date = record_df.index.date[0]
            market_day = market_df[str(date) : str(date)]

            trades = analyze_trades(
                record_df=record_df,
                market_df=market_day,
                opening_limit=opening_limit,
                closing_limit=closing_limit,
            )

            for trade in trades:
                trade["date"] = date
                seed_results.append(trade)

        if seed_results:
            seed_df = pd.DataFrame(seed_results)
            seed_df = seed_df.set_index("date").sort_index()
        else:
            seed_df = pd.DataFrame()

        results[seed] = seed_df

    return results
