# core/h3agg.py
from __future__ import annotations

import h3
import pandas as pd
import streamlit as st

from .config import BASE_H3_RES, AVG_HA_BY_RES

H3_RES13_COL = f"h3_r{BASE_H3_RES}"

# =============================
# ÉÉN KEER H3 OP HOGE RESOLUTIE
# =============================

def build_res13(df_src: pd.DataFrame) -> pd.DataFrame:
    """
    Maak één kolom met de basisresolutie (BASE_H3_RES) voor alle punten.
    """
    lat_np = df_src["latitude"].astype("float32").to_numpy()
    lon_np = df_src["longitude"].astype("float32").to_numpy()
    h3_cells = [h3.latlng_to_cell(float(la), float(lo), BASE_H3_RES)
                for la, lo in zip(lat_np, lon_np)]
    return df_src.assign(**{H3_RES13_COL: h3_cells})


def ensure_parent_series_for(df_with_res13: pd.DataFrame, res: int, cache: dict) -> pd.Series:
    """
    Maak/haal de parent H3-serie voor een doelresolutie 'res'.
    """
    if res == BASE_H3_RES:
        return df_with_res13[H3_RES13_COL]
    if res in cache:
        return cache[res]
    parents = [h3.cell_to_parent(h, res) for h in df_with_res13[H3_RES13_COL]]
    ser = pd.Series(parents, index=df_with_res13.index, name=f"h3_r{res}")
    cache[res] = ser
    return ser


# =============================
# Snelle aggregatie + roll-up
# =============================

@st.cache_data(show_spinner=False, max_entries=10)
def build_res13_agg(df_points_res13: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregatie op de basisresolutie
    Verwacht dat df_points_base kolommen heeft:
      - H3_RES13_COL, kWh_per_m2, gemiddeld_jaarverbruik_mWh, totale_oppervlakte, bouwjaar, aantal_VBOs
    """
    tmp = df_points_res13.copy()
    tmp["kwh_sum"] = pd.to_numeric(tmp["kWh_per_m2"], errors="coerce").fillna(0).astype("float32")
    tmp["cnt"]     = 1
    res_base = (
        tmp.groupby(H3_RES13_COL, sort=False, observed=True)
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
    return res_base


@st.cache_data(show_spinner=False, max_entries=10)
def rollup_to_resolution(res13_agg: pd.DataFrame, target_res: int, _cache_key: int = 0) -> pd.DataFrame:
    """
    Roll-up van basisresolutie naar target_res (of laten als basis),
    """
    if target_res == BASE_H3_RES:
        out = res13_agg.copy()
        out = out.rename(columns={H3_RES13_COL: "h3_index"})
    else:
        parents = res13_agg[H3_RES13_COL].map(lambda h: h3.cell_to_parent(h, target_res))
        tmp = res13_agg.assign(h3_parent=parents)
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
    Gemiddelde hectare per cel voor de resolutie
    """
    return float(AVG_HA_BY_RES.get(res, AVG_HA_BY_RES[BASE_H3_RES]))
