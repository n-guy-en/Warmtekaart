# core/h3agg.py
from __future__ import annotations

import h3
import pandas as pd
import streamlit as st

from .config import BASE_H3_RES, AVG_HA_BY_RES


# =============================
# ÉÉN KEER H3 OP HOGE RESOLUTIE
# =============================

def build_res12(df_src: pd.DataFrame) -> pd.DataFrame:
    """
    Maak één kolom h3_r12 op BASE_H3_RES (12) voor alle punten.
    Identiek aan je _build_res12 in de monolith.
    """
    lat_np = df_src["latitude"].astype("float32").to_numpy()
    lon_np = df_src["longitude"].astype("float32").to_numpy()
    h3_res12 = [h3.latlng_to_cell(float(la), float(lo), BASE_H3_RES)
                for la, lo in zip(lat_np, lon_np)]
    return df_src.assign(h3_r12=h3_res12)


def ensure_parent_series_for(df_with_h3_res12: pd.DataFrame, res: int, cache: dict) -> pd.Series:
    """
    Maak/haal de parent H3-serie voor een doelresolutie 'res'.
    Identiek aan je _ensure_parent_series_for, maar cache wordt van buiten meegegeven.
    """
    if res == BASE_H3_RES:
        return df_with_h3_res12["h3_r12"]
    if res in cache:
        return cache[res]
    parents = [h3.cell_to_parent(h, res) for h in df_with_h3_res12["h3_r12"]]
    ser = pd.Series(parents, index=df_with_h3_res12.index, name=f"h3_r{res}")
    cache[res] = ser
    return ser


# =============================
# Snelle aggregatie + roll-up
# =============================

@st.cache_data(show_spinner=False, max_entries=10)
def build_res12_agg(df_points_res12: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregatie op res12 met exact dezelfde kolommen als in je monolith.
    Verwacht dat df_points_res12 kolommen heeft:
      - h3_r12, kWh_per_m2, gemiddeld_jaarverbruik_mWh, totale_oppervlakte, bouwjaar, aantal_VBOs
    """
    tmp = df_points_res12.copy()
    tmp["kwh_sum"] = pd.to_numeric(tmp["kWh_per_m2"], errors="coerce").fillna(0).astype("float32")
    tmp["cnt"]     = 1
    res12 = (
        tmp.groupby("h3_r12", sort=False, observed=True)
           .agg(
               sum_mwh=("gemiddeld_jaarverbruik_mWh", "sum"),
               sum_area=("totale_oppervlakte", "sum"),
               sum_kwh=("kwh_sum", "sum"),
               cnt=("cnt", "sum"),
               mean_bouwjaar=("bouwjaar", "mean"),
               sum_vbos=("aantal_VBOs", "sum"),
           )
           .reset_index()
    )
    return res12


@st.cache_data(show_spinner=False, max_entries=10)
def rollup_to_resolution(res12_agg: pd.DataFrame, target_res: int, _cache_key: int = 0) -> pd.DataFrame:
    """
    Roll-up van res12 naar target_res (of identiek laten als 12),
    exact dezelfde berekeningen/kolomnamen als je monolith.
    """
    if target_res == BASE_H3_RES:
        out = res12_agg.copy()
        out = out.rename(columns={"h3_r12": "h3_index"})
    else:
        parents = res12_agg["h3_r12"].map(lambda h: h3.cell_to_parent(h, target_res))
        tmp = res12_agg.assign(h3_parent=parents)
        out = (
            tmp.groupby("h3_parent", sort=False, observed=True)
               .agg(
                   sum_mwh=("sum_mwh", "sum"),
                   sum_area=("sum_area", "sum"),
                   sum_kwh=("sum_kwh", "sum"),
                   cnt=("cnt", "sum"),
                   mean_bouwjaar=("mean_bouwjaar", "mean"),
                   sum_vbos=("sum_vbos", "sum"),
               )
               .reset_index()
               .rename(columns={"h3_parent": "h3_index"})
        )

    out["kWh_per_m2"]                  = (out["sum_kwh"] / out["cnt"]).round(0)
    out["aantal_huizen"]               = out["cnt"].astype(int)
    out["gemiddeld_jaarverbruik_mWh"]  = out["sum_mwh"].round(0)
    out["totale_oppervlakte"]          = out["sum_area"].round(0)
    out["bouwjaar"]                    = out["mean_bouwjaar"].round(0)
    out["aantal_VBOs"]                 = out["sum_vbos"].round(0).astype(int)

    return out[[
        "h3_index",
        "kWh_per_m2",
        "gemiddeld_jaarverbruik_mWh",
        "totale_oppervlakte",
        "aantal_huizen",
        "bouwjaar",
        "aantal_VBOs",
    ]]


# =============================
# Oppervlakte per resolutie
# =============================

def area_ha_for_res(res: int) -> float:
    """
    Gemiddelde hectare per cel voor de resolutie, exact zoals in je monolith.
    """
    return float(AVG_HA_BY_RES.get(res, 2.2))
