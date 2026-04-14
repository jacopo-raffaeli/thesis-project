from pathlib import Path

import pandas as pd


def save_figure(fig, fig_dir: Path, stem: str):
    fig.savefig(fig_dir / f"{stem}.pdf", dpi=300, bbox_inches="tight")


def save_table(df: pd.DataFrame, tab_dir: Path, stem: str):
    df.to_excel(tab_dir / f"{stem}.xlsx", index=True)


def save_table_image(df: pd.DataFrame, tab_dir: Path, stem: str):
    import dataframe_image as dfi

    dfi.export(df, tab_dir / f"{stem}.png", table_conversion="chrome")
