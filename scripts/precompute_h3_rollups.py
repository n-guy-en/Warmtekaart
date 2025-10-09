"""
Bouw een pre-geaggregeerde dataset met H3-rollen.

Resultaat:
- data/precomputed/h3_res12_grouped.parquet
  Bevat per h3_r12 + filter-combinatie de gesommeerde waarden en parents
  voor resoluties 9–12 om runtime-berekeningen te minimaliseren.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h3
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SRC_PATH = DATA_DIR / "data_kWh.parquet"
OUT_DIR = DATA_DIR / "precomputed"
OUT_PATH = OUT_DIR / "h3_res12_grouped.parquet"

COLUMNS: Sequence[str] = [
    "aantal_VBOs",
    "totale_oppervlakte",
    "woonplaats",
    "Energieklasse",
    "latitude",
    "longitude",
    "bouwjaar",
    "kWh_per_m2",
    "gemiddeld_jaarverbruik",
    "gemiddeld_jaarverbruik_mWh",
    "Dataset",
]


def _load_source() -> pd.DataFrame:
    if not SRC_PATH.exists():
        raise FileNotFoundError(f"Bronbestand niet gevonden: {SRC_PATH}")
    df = pd.read_parquet(SRC_PATH, columns=list(COLUMNS))

    # Compacte dtypes
    for col in ["latitude", "longitude", "kWh_per_m2", "gemiddeld_jaarverbruik", "gemiddeld_jaarverbruik_mWh"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    for col in ["aantal_VBOs", "totale_oppervlakte", "bouwjaar"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")

    for col in ["woonplaats", "Energieklasse", "Dataset"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def _compute_h3_columns(df: pd.DataFrame) -> pd.DataFrame:
    lat_np = df["latitude"].astype("float32").to_numpy()
    lon_np = df["longitude"].astype("float32").to_numpy()
    base_res = 12
    df["h3_r12"] = [
        h3.latlng_to_cell(float(lat), float(lon), base_res)
        if pd.notna(lat) and pd.notna(lon) else None
        for lat, lon in zip(lat_np, lon_np)
    ]
    for res in (11, 10, 9):
        df[f"h3_r{res}"] = [
            h3.cell_to_parent(h, res) if h else None
            for h in df["h3_r12"]
        ]
    return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["cnt"] = 1
    df["sum_mwh"] = pd.to_numeric(df["gemiddeld_jaarverbruik_mWh"], errors="coerce").fillna(0).astype("float32")
    df["sum_area"] = pd.to_numeric(df["totale_oppervlakte"], errors="coerce").fillna(0).astype("float32")
    df["sum_kwh"] = pd.to_numeric(df["kWh_per_m2"], errors="coerce").fillna(0).astype("float32")
    df["sum_vbos"] = pd.to_numeric(df["aantal_VBOs"], errors="coerce").fillna(0).astype("float32")

    df["lat"] = pd.to_numeric(df["latitude"], errors="coerce").fillna(0).astype("float32")
    df["lon"] = pd.to_numeric(df["longitude"], errors="coerce").fillna(0).astype("float32")
    df["sum_bouwjaar"] = pd.to_numeric(df["bouwjaar"], errors="coerce").fillna(0).astype("float32")

    group_cols = [
        "h3_r12",
        "h3_r11",
        "h3_r10",
        "h3_r9",
        "woonplaats",
        "Energieklasse",
        "Dataset",
        "bouwjaar",
    ]

    agg = (
        df.groupby(group_cols, observed=True)
          .agg(
              sum_mwh=("sum_mwh", "sum"),
              sum_area=("sum_area", "sum"),
              sum_kwh=("sum_kwh", "sum"),
              sum_vbos=("sum_vbos", "sum"),
              cnt=("cnt", "sum"),
              sum_lat=("lat", "sum"),
              sum_lon=("lon", "sum"),
              sum_bouwjaar=("sum_bouwjaar", "sum"),
          )
          .reset_index()
    )

    # Gemiddelden voor UI (houd dezelfde kolomnamen als bron)
    with np.errstate(divide="ignore", invalid="ignore"):
        agg["kWh_per_m2"] = (agg["sum_kwh"] / agg["cnt"]).astype("float32")
        agg["gemiddeld_jaarverbruik_mWh"] = (agg["sum_mwh"] / agg["cnt"]).astype("float32")
        agg["totale_oppervlakte"] = (agg["sum_area"] / agg["cnt"]).astype("float32")
        agg["aantal_VBOs"] = (agg["sum_vbos"] / agg["cnt"]).astype("float32")
        agg["latitude"] = (agg["sum_lat"] / agg["cnt"]).astype("float32")
        agg["longitude"] = (agg["sum_lon"] / agg["cnt"]).astype("float32")

    for col in ["sum_mwh", "sum_area", "sum_kwh", "sum_vbos", "sum_lat", "sum_lon", "sum_bouwjaar"]:
        if col in agg.columns:
            agg[col] = agg[col].astype("float32")

    if "cnt" in agg.columns:
        agg["cnt"] = agg["cnt"].astype("int32")

    return agg


def main() -> None:
    df = _load_source()
    df = _compute_h3_columns(df)
    agg = _aggregate(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(OUT_PATH, engine="pyarrow", compression="snappy")

    print(f"Uitgangsbestand: {SRC_PATH}")
    print(f"Aantal records origineel: {len(df):,}")
    print(f"Geaggregeerd bestand: {OUT_PATH}")
    print(f"Aantal records geaggregeerd: {len(agg):,}")
    print(f"Bestandsgrootte: {OUT_PATH.stat().st_size / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()
