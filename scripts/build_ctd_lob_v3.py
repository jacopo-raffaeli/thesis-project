# TODO:
# - Core Infrastructure - Logging, error handling, config validation
# - DST Handling - Investigation script first, then handling if needed
# - Sanity Checks - Input validation, output validation, quality metrics
# - Production Monitoring - Progress reporting, memory management, metadata
# - Code Quality - Testing, documentation, organization
# - Future - Conversion factors (deferred), advanced features
# - Add the futures ticker to config.py

import os
import sys
from datetime import datetime
from functools import partial

import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm

# Import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import build_ctd_lob_config as config


def generate_lob_columns(levels):
    """
    Generate LOB column names for given number of levels.
    """

    columns = []
    for level in range(1, levels + 1):
        columns.extend([f"L{level}-BidPrice", f"L{level}-BidSize"])
    for level in range(1, levels + 1):
        columns.extend([f"L{level}-AskPrice", f"L{level}-AskSize"])
    return columns


def load_futures_lob(date_str, freq):
    """
    Load and preprocess futures LOB data for a given date.

    Returns:
        pd.DataFrame or None: Resampled futures LOB, or None if error
    """

    # Build futures LOB file path
    future_file_path = os.path.normpath(
        os.path.join(config.FUTURE_DIR, config.FUTURE_LOB_SUBDIR, f"{date_str}-{date_str}.zip")
    )

    # Load futures LOB data
    try:
        futures_lob = pd.read_csv(future_file_path)
    except FileNotFoundError:
        print(f"Missing Future file: {date_str}")
        return None
    except Exception as e:
        print(f"Error loading future {date_str}: {e}")
        return None

    # Filter for BTP futures contract
    is_btp_contract = futures_lob["#RIC"] == "FBTPc1"
    futures_lob = futures_lob[is_btp_contract]

    # Timezone handling
    futures_lob["Date-Time"] = pd.to_datetime(futures_lob["Date-Time"])
    if futures_lob["Date-Time"].dt.tz is None:
        futures_lob["Date-Time"] = futures_lob["Date-Time"].dt.tz_localize("Europe/Berlin")
    else:
        futures_lob["Date-Time"] = futures_lob["Date-Time"].dt.tz_convert("Europe/Berlin")

    futures_lob["mid"] = (futures_lob["L1-BidPrice"] + futures_lob["L1-AskPrice"]) / 2

    # Resample to desired frequency
    futures_lob = futures_lob.set_index("Date-Time").resample(freq).last()

    return futures_lob


def get_ctd_isin(ctd_bonds, date_str):
    """
    Get CTD bond ISIN for a given date.

    Returns:
        str or None: CTD ISIN, or None if not found
    """

    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        date_matches = ctd_bonds["Date"].dt.date == target_date
        ctd_isin = ctd_bonds[date_matches]["ISIN"].item()
        return ctd_isin
    except (ValueError, KeyError):
        print(f"Missing CTD bond ISIN for {date_str}")
        return None
    except Exception as e:
        print(f"Error getting CTD bond ISIN {date_str}: {e}")
        return None


def load_bond_proposals(year, month_str, day, ctd_isin):
    """
    Load and preprocess bond proposals for a given date and CTD bond.

    Returns:
        pd.DataFrame or None: Filtered bond proposals, or None if error
    """

    # Build bond proposals file path
    date_str = f"{day[0:4]}-{day[4:6]}-{day[6:8]}"
    proposals_path = os.path.normpath(
        os.path.join(config.CASH_DIR, str(year), month_str, str(day), config.MTS_PROPOSALS_FILE)
    )

    # Load bond proposals data
    try:
        bond_proposals = pd.read_parquet(proposals_path)
    except FileNotFoundError:
        print(f"Missing MTS proposals: {date_str}")
        return None
    except Exception as e:
        print(f"Error loading proposals {date_str}: {e}")
        return None

    # Filter for CTD bond only
    is_ctd_bond = bond_proposals["BONDCODE"] == ctd_isin
    bond_proposals = bond_proposals[is_ctd_bond]

    # Filter valid orders
    is_status = bond_proposals.STATUS == 0
    is_logon = bond_proposals.CHECK_LOGON == 0
    bond_proposals = bond_proposals[is_status & is_logon]

    if bond_proposals.empty:
        print(f"No valid orders for {date_str}")
        return None

    # Timezone handling
    if bond_proposals["updateTS"].dt.tz is None:
        bond_proposals["updateTS"] = bond_proposals["updateTS"].dt.tz_localize("Europe/Berlin")
    else:
        bond_proposals["updateTS"] = bond_proposals["updateTS"].dt.tz_convert("Europe/Berlin")

    if bond_proposals["endTS"].dt.tz is None:
        bond_proposals["endTS"] = bond_proposals["endTS"].dt.tz_localize("Europe/Berlin")
    else:
        bond_proposals["endTS"] = bond_proposals["endTS"].dt.tz_convert("Europe/Berlin")

    return bond_proposals


def reconstruct_lob(bond_proposals, freq, levels, lob_columns):
    """
    Reconstruct CTD LOB from bond proposals.

    Returns:
        pd.DataFrame: Reconstructed LOB with mid prices
    """
    # Generate timestamp range
    # NOTE: Likely cause of the first row NaN issue in ctd lob
    # Maybe using .ceil() instead of floor() could help
    timestamp_start = bond_proposals["updateTS"].min().floor(freq)
    timestamp_end = bond_proposals["endTS"].max().floor(freq)

    date_range = pd.date_range(start=timestamp_start, end=timestamp_end, freq=freq)
    if date_range.tz is None:
        lob_timestamps = pd.DataFrame({"timestamp": date_range.tz_localize("Europe/Berlin")})
    else:
        lob_timestamps = pd.DataFrame({"timestamp": date_range})

    # Create IntervalIndex for order lifetimes
    intervals = pd.IntervalIndex.from_arrays(
        bond_proposals["updateTS"], bond_proposals["endTS"], closed="both"
    )

    # Prepare partial function
    count_events_partial = partial(
        count_active_events, bond_proposals=bond_proposals, intervals=intervals, levels=levels
    )

    # Parallelized LOB reconstruction
    ctd_lob = pd.DataFrame.from_records(
        lob_timestamps["timestamp"].parallel_apply(count_events_partial), columns=lob_columns
    )

    ctd_lob["timestamp"] = lob_timestamps["timestamp"]
    ctd_lob.set_index("timestamp", inplace=True)

    # Compute mid price
    ctd_lob["mid"] = (ctd_lob["L1-BidPrice"] + ctd_lob["L1-AskPrice"]) / 2

    return ctd_lob


def count_active_events(ts, bond_proposals, intervals, levels=10):
    """
    Reconstruct Limit Order Book at given timestamp.

    Finds all active orders at timestamp ts and aggregates by price level.
    Returns flattened list representing 10-level BID-ASK LOB.

    Args:
        ts: Timestamp to reconstruct LOB for
        bond_proposals: DataFrame with order data (BIDPRICE, ASKPRICE, BIDQTY, ASKQTY)
        intervals: IntervalIndex of order lifetimes (updateTS to endTS)
        levels: Number of LOB levels (default 10)

    Returns:
        List of 40 values representing 10-level BID-ASK LOB
    """
    # Import inside function for self-contained function required by pandarallel
    import numpy as np

    vals = []

    # BID SIDE: Find active bids, group by price, sort best bid first (descending)
    active_bids = (intervals.contains(ts)) & (bond_proposals["BIDPRICE"] != 0)
    bid_levels = (
        bond_proposals[active_bids].groupby("BIDPRICE")["BIDQTY"].sum().sort_index(ascending=False)
    )

    # Take top N levels and convert to list of (price, qty) tuples
    bid_items = list(bid_levels.head(levels).items())

    # Pad to exactly N levels if needed
    # NOTE: Likely cause of the first row NaN issue in ctd lob
    while len(bid_items) < levels:
        bid_items.append((np.nan, np.nan))

    # Flatten to vals
    for price, qty in bid_items:
        vals.extend([price, qty])

    # ASK SIDE: Find active asks, group by price, sort best ask first (ascending)
    active_asks = (intervals.contains(ts)) & (bond_proposals["ASKPRICE"] != 0)
    ask_levels = (
        bond_proposals[active_asks].groupby("ASKPRICE")["ASKQTY"].sum().sort_index(ascending=True)
    )

    # Take top N levels and convert to list of (price, qty) tuples
    ask_items = list(ask_levels.head(levels).items())

    # Pad to exactly N levels if needed
    while len(ask_items) < levels:
        ask_items.append((np.nan, np.nan))

    # Flatten to vals
    for price, qty in ask_items:
        vals.extend([price, qty])

    return vals


def process_single_day(day, year, month_str, ctd_bonds, freq, levels, lob_columns, output_dir):
    """
    Process a single day: load data, reconstruct LOB, save outputs.

    Returns:
        dict: Status with 'success' (bool), 'date' (str), 'error' (str or None)
    """
    date_str = f"{day[0:4]}-{day[4:6]}-{day[6:8]}"

    # Load futures LOB data
    futures_lob = load_futures_lob(date_str, freq)
    if futures_lob is None:
        return {"success": False, "date": date_str, "error": "futures_load_failed"}

    # Get CTD bond ISIN
    ctd_isin = get_ctd_isin(ctd_bonds, date_str)
    if ctd_isin is None:
        return {"success": False, "date": date_str, "error": "ctd_isin_not_found"}

    # Load bond proposals
    bond_proposals = load_bond_proposals(year, month_str, day, ctd_isin)
    if bond_proposals is None:
        return {"success": False, "date": date_str, "error": "bond_proposals_failed"}

    # Reconstruct LOB
    ctd_lob = reconstruct_lob(bond_proposals, freq, levels, lob_columns)

    # Save daily outputs
    day_str = day[6:8]
    ctd_filename = os.path.join(
        output_dir, f"ctd_lob_freq_{freq}_{year}_{month_str}_{day_str}.parquet"
    )
    futures_filename = os.path.join(
        output_dir, f"futures_lob_freq_{freq}_{year}_{month_str}_{day_str}.parquet"
    )

    ctd_lob.to_parquet(ctd_filename, engine="pyarrow")
    futures_lob.to_parquet(futures_filename, engine="pyarrow")

    return {"success": True, "date": date_str, "error": None}


if __name__ == "__main__":
    # Initialize pandarallel (progress_bar=False to avoid interference with tqdm)
    pandarallel.initialize(progress_bar=False, verbose=1)

    # Configuration from imported config
    freq = config.FREQ
    start_year = config.START_YEAR
    end_year = config.END_YEAR
    years = range(start_year, end_year + 1)
    levels = config.LEVELS

    # Load CTD bond series
    ctd_bonds_path = os.path.normpath(os.path.join(config.DATA_DIR, config.CTD_BONDS_FILE))
    ctd_bonds = pd.read_csv(ctd_bonds_path, delimiter=";")
    ctd_bonds["Date"] = pd.to_datetime(ctd_bonds["Date"], format="%d/%m/%Y")

    # Generate LOB column names
    LOB_COLUMNS = generate_lob_columns(levels)

    # Create output directory
    output_dir = config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    for year in years:
        for month in range(1, 13):
            month_str = f"{month:02d}"

            # Construct month directory path
            month_dir = os.path.normpath(os.path.join(config.CASH_DIR, str(year), month_str))

            try:
                day_list = os.listdir(month_dir)
            except FileNotFoundError:
                print(f"Month directory not found: {year}/{month_str}")
                continue
            except Exception as e:
                print(f"Error accessing month directory {year}/{month_str}: {e}")
                continue

            for day in tqdm(day_list, desc=f"{year}-{month_str}"):
                process_single_day(
                    day, year, month_str, ctd_bonds, freq, levels, LOB_COLUMNS, output_dir
                )

            print(f"Completed {year}-{month_str}")
