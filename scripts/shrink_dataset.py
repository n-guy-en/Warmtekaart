"""
Utility script to generate a slimmed-down parquet file for the Streamlit app.

It keeps only the columns the UI and analytics consume, casts them to compact
numeric types, and stores the result next to the original dataset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "data" / "data_kWh.parquet"
DEST_PATH = ROOT / "data" / "data_kWh_compact.parquet"

COLUMNS_KEEP: Sequence[str] = [
    "aantal_VBOs",
    "totale_oppervlakte",
    "woonplaats",
    "Energieklasse",
    "latitude",
    "longitude",
    "bouwjaar",
    "pandstatus",
    "kWh_per_m2",
    "gemiddeld_jaarverbruik",
    "gemiddeld_jaarverbruik_mWh",
    "Dataset",
]

FLOAT32_COLS = {
    "latitude",
    "longitude",
    "kWh_per_m2",
    "gemiddeld_jaarverbruik",
    "gemiddeld_jaarverbruik_mWh",
}

INT32_COLS = {
    "aantal_VBOs",
    "totale_oppervlakte",
    "bouwjaar",
}

CATEGORY_COLS = {
    "woonplaats",
    "Energieklasse",
    "Dataset",
}


def main() -> None:
    if not SRC_PATH.exists():
        raise FileNotFoundError(f"Bronbestand niet gevonden: {SRC_PATH}")

    df = pd.read_parquet(SRC_PATH, columns=list(COLUMNS_KEEP))

    for col in FLOAT32_COLS & set(df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    for col in INT32_COLS & set(df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")

    for col in CATEGORY_COLS & set(df.columns):
        df[col] = df[col].astype("category")

    if "pandstatus" in df.columns:
        df = df[df["pandstatus"] == "Pand in gebruik"].reset_index(drop=True)

    DEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DEST_PATH, engine="pyarrow", compression="snappy")

    orig_size = SRC_PATH.stat().st_size / (1024 * 1024)
    new_size = DEST_PATH.stat().st_size / (1024 * 1024)
    print(f"Originele dataset: {orig_size:.1f} MB")
    print(f"Compacte dataset: {new_size:.1f} MB")
    print(f"Kolommen behouden: {list(df.columns)}")


if __name__ == "__main__":
    main()
