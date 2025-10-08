# core/h3sites.py
from __future__ import annotations

import hashlib
import json
from typing import Iterable

import h3
import pandas as pd
import streamlit as st


# ============================================================
# Kleine helper
# ============================================================

def _hash_ids(ids: Iterable[str]) -> str:
    a = ",".join(sorted(map(str, ids)))
    return hashlib.md5(a.encode("utf-8")).hexdigest()


# ============================================================
# Neighbor pairs voor k-ring (gecacht)
# ============================================================

@st.cache_data(show_spinner=False, max_entries=6, ttl=300)
def build_neighbor_pairs(unique_cells: list[str], k: int) -> pd.DataFrame:
    """
    Bouw (center, neighbor)-paren voor alle cells in een k-ring.
    Identiek aan je monolith-functie.
    """
    pairs = []
    for h in unique_cells:
        for nh in h3.grid_disk(h, k):
            pairs.append((h, nh))
    return pd.DataFrame(pairs, columns=["center", "neighbor"])


# ============================================================
# k-ring aggregatie (cluster_MWh, cluster_buildings)
# ============================================================

@st.cache_data(show_spinner=False, max_entries=6, ttl=300)
def aggregate_clusters(df_hex: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Verwacht df_hex met kolommen:
      - h3_index, gemiddeld_jaarverbruik_mWh, aantal_huizen
    Retourneert per center:
      - cluster_MWh, cluster_buildings
    """
    cells = df_hex["h3_index"].astype(str).unique().tolist()
    pairs = build_neighbor_pairs(cells, k)

    base = df_hex.loc[:, ["h3_index", "gemiddeld_jaarverbruik_mWh", "aantal_huizen"]]
    base = base.rename(columns={
        "h3_index": "neighbor",
        "gemiddeld_jaarverbruik_mWh": "mwh",
        "aantal_huizen": "houses",
    })

    joined = pairs.merge(base, on="neighbor", how="left")
    sums = (
        joined.groupby("center", sort=False)
              .agg(cluster_MWh=("mwh", "sum"),
                   cluster_buildings=("houses", "sum"))
              .reset_index()
              .rename(columns={"center": "h3_index"})
    )
    sums["cluster_MWh"] = pd.to_numeric(sums["cluster_MWh"]).fillna(0).round(0).astype(int)
    sums["cluster_buildings"] = pd.to_numeric(sums["cluster_buildings"]).fillna(0).round(0).astype(int)
    return sums


# ============================================================
# Shortlist centers (sneller rekenen)
# ============================================================

def shortlist_centers(df_hex: pd.DataFrame, threshold_kwh_m2: float, top_frac: float = 0.6) -> pd.DataFrame:
    """
    Houd alleen hexen:
    - Boven drempel (kWh/m²)
    - Plus top (1 - top_frac) op MWh
    Retourneert DataFrame met enkel kolom h3_index.
    """
    s = df_hex[["h3_index", "kWh_per_m2", "gemiddeld_jaarverbruik_mWh"]].copy()
    s["kWh_per_m2"] = pd.to_numeric(s["kWh_per_m2"], errors="coerce").fillna(0)
    s["gemiddeld_jaarverbruik_mWh"] = pd.to_numeric(s["gemiddeld_jaarverbruik_mWh"], errors="coerce").fillna(0)

    hot = s["kWh_per_m2"] >= float(threshold_kwh_m2)
    q = s["gemiddeld_jaarverbruik_mWh"].quantile(1 - float(top_frac))
    big = s["gemiddeld_jaarverbruik_mWh"] >= q
    keep = hot | big
    return s.loc[keep, ["h3_index"]].drop_duplicates()


# ============================================================
# Fingerprint voor cache key van clusters
# ============================================================

def filters_fingerprint(params: dict, cell_ids: Iterable[str]) -> str:
    """
    Maak een stabiele vingerafdruk van parameters + set cell_ids.
    (Exacte string-samenstelling zoals in de monolith.)
    """
    blob = json.dumps(params, sort_keys=True) + f"|n={len(list(cell_ids))}|" + ",".join(sorted(map(str, cell_ids)))
    return hashlib.md5(blob.encode()).hexdigest()


# ============================================================
# Cached variant van cluster-aggregatie op shortlist
# ============================================================

@st.cache_data(show_spinner=False, max_entries=10, ttl=300)
def compute_clusters_cached(cache_key: str, df_hex_small: pd.DataFrame, k: int, ttl=1800) -> pd.DataFrame:
    """
    Cached variant van aggregate_clusters zonder externe bestanden.
    Verwacht df_hex_small met kolommen:
      - h3_index, gemiddeld_jaarverbruik_mWh, aantal_huizen
    """
    cells = df_hex_small["h3_index"].astype(str).unique().tolist()
    pairs = []
    for h in cells:
        for nh in h3.grid_disk(h, k):
            pairs.append((h, nh))
    pairs = pd.DataFrame(pairs, columns=["center", "neighbor"])

    base = df_hex_small.rename(columns={
        "h3_index": "neighbor",
        "gemiddeld_jaarverbruik_mWh": "mwh",
        "aantal_huizen": "houses",
    })[["neighbor", "mwh", "houses"]]

    joined = pairs.merge(base, on="neighbor", how="left")
    sums = (
        joined.groupby("center", sort=False)
              .agg(cluster_MWh=("mwh", "sum"),
                   cluster_buildings=("houses", "sum"))
              .reset_index()
              .rename(columns={"center": "h3_index"})
    )
    sums["cluster_MWh"] = pd.to_numeric(sums["cluster_MWh"]).fillna(0).round(0).astype(int)
    sums["cluster_buildings"] = pd.to_numeric(sums["cluster_buildings"]).fillna(0).round(0).astype(int)
    return sums


# ============================================================
# Selectie van sites (minimale afstand + capaciteit/benutting)
# ============================================================

@st.cache_data(show_spinner=False, max_entries=20)
def select_sites_from_clusters(
    cluster_df: pd.DataFrame,
    min_sep_cells: int,
    topk: int,
    cap_mwh: float,
    cap_buildings: int,
    ttl:1800,
) -> pd.DataFrame:
    """
    Neemt cluster_df met:
      - h3_index, cluster_MWh, cluster_buildings
    En produceert top-k locaties:
      - connected_MWh/buildings (geclipped op capaciteit)
      - utilization_pct
      - lat/lon (centroid van h3 cel)
    """
    out = cluster_df.copy()
    out["connected_MWh"] = out["cluster_MWh"].clip(upper=float(cap_mwh)).round(0).astype(int)
    out["connected_buildings"] = out["cluster_buildings"].clip(upper=int(cap_buildings)).astype(int)
    out["utilization"] = out["connected_MWh"] / max(float(cap_mwh), 1.0)

    cand = out.sort_values("connected_MWh", ascending=False).reset_index(drop=True)
    chosen = []
    blocked = set()
    for r in cand.itertuples():
        # sla over als binnen min. scheiding van reeds gekozen
        too_close = any(
            (h3.grid_distance(r.h3_index, b) is not None) and
            (h3.grid_distance(r.h3_index, b) <= int(min_sep_cells)) for b in blocked
        )
        if too_close:
            continue
        chosen.append(r)
        for nh in h3.grid_disk(r.h3_index, int(min_sep_cells)):
            blocked.add(nh)
        if len(chosen) >= int(topk):
            break

    rows = []
    for r in chosen:
        lat, lon = h3.cell_to_latlng(r.h3_index)
        rows.append({
            "h3_index": r.h3_index,
            "lat": float(lat),
            "lon": float(lon),
            "cluster_MWh": int(r.cluster_MWh),
            "cluster_buildings": int(r.cluster_buildings),
            "connected_MWh": int(r.connected_MWh),
            "connected_buildings": int(r.connected_buildings),
            "utilization_pct": int(round(100 * r.utilization, 0)),
            "cap_MWh": float(cap_mwh),
            "cap_buildings": int(cap_buildings),
        })
    return pd.DataFrame(rows)
