import pandas as pd


def swtich_dates(df: pd.DataFrame):
    # Extract CTD switch by ISIN
    switches_isin = df["ISIN"].ne(df["ISIN"].shift())
    results_isin = df[switches_isin]

    # Extract CTD switch by CUSIP
    switches_cusip = df["CUSIP"].ne(df["CUSIP"].shift())
    results_cusip = df[switches_cusip]

    if not results_isin.equals(results_cusip):
        raise ValueError("Error: ISIN and CUSIP switch results do not match.")

    return results_isin.index.to_frame(name="CTD Switch Date").reset_index(drop=True)


if __name__ == "__main__":
    from thesis_project import config as global_config

    TICKER = "fbtp"
    ctd_df = pd.read_csv(
        global_config.RAW_DIR / TICKER / "daily_cf.csv", parse_dates=["Date"], index_col="Date"
    )
    switch_dates_df = swtich_dates(ctd_df)
    switch_dates_df.to_csv(global_config.RAW_DIR / TICKER / "ctd_switch.csv", index=False)
