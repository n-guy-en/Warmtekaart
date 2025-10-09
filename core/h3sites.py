# core/h3sites.py
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Iterable, List

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

@st.cache_data(show_spinner=False, max_entries=4, ttl=300)
def build_neighbor_pairs(unique_cells: List[str], k: int) -> pd.DataFrame:
    """
    Bouw (center, neighbor)-paren voor alle cells in een k-ring.
    Retourneert DataFrame met kolommen: center, neighbor
    """
    rows = []
    K = int(k)
    for h in unique_cells:
        for nh in h3.grid_disk(h, K):
            rows.append((h, nh))
    if not rows:
        return pd.DataFrame(columns=["center", "neighbor"])
    out = pd.DataFrame(rows, columns=["center", "neighbor"])
    return out


@lru_cache(maxsize=16384)
def _grid_disk_cached(cell_id: str, k: int) -> tuple[str, ...]:
    """
    Kleine cache rond h3.grid_disk zodat we herhaalde neighbor-berekeningen vermijden.
    """
    return tuple(h3.grid_disk(cell_id, k))


def _sum_neighbors(compact_lookup: dict[str, tuple[float, int]], centers: List[str], k: int):
    """
    Lever (center, total_mwh, total_buildings) op zonder een volledige pairs DataFrame in het geheugen te houden.
    """
    K = int(k)
    for center in centers:
        total_mwh = 0.0
        total_buildings = 0
        for neighbor in _grid_disk_cached(center, K):
            vals = compact_lookup.get(neighbor)
            if vals is None:
                continue
            mwh, houses = vals
            total_mwh += mwh
            total_buildings += houses
        yield center, total_mwh, total_buildings


# ============================================================
# k-ring aggregatie (cluster_MWh, cluster_buildings)
# ============================================================

@st.cache_data(show_spinner=False, max_entries=4, ttl=300)
def aggregate_clusters(df_hex: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Verwacht df_hex met kolommen:
      - h3_index, gemiddeld_jaarverbruik_mWh, aantal_huizen
    Retourneert per center (h3_index):
      - cluster_MWh (int32), cluster_buildings (int32)
    """
    cells = df_hex["h3_index"].astype(str).unique().tolist()
    base = df_hex.loc[:, ["h3_index", "gemiddeld_jaarverbruik_mWh", "aantal_huizen"]].copy()

    base["gemiddeld_jaarverbruik_mWh"] = (
        pd.to_numeric(base["gemiddeld_jaarverbruik_mWh"], errors="coerce")
          .fillna(0)
          .astype("float32")
    )
    base["aantal_huizen"] = (
        pd.to_numeric(base["aantal_huizen"], errors="coerce")
          .fillna(0)
          .astype("int32")
    )

    lookup = {
        str(row.h3_index): (float(row.gemiddeld_jaarverbruik_mWh), int(row.aantal_huizen))
        for row in base.itertuples(index=False)
    }

    records = []
    for center, total_mwh, total_buildings in _sum_neighbors(lookup, cells, k):
        records.append(
            (
                center,
                int(round(total_mwh)),
                int(total_buildings),
            )
        )

    if not records:
        return pd.DataFrame(columns=["h3_index", "cluster_MWh", "cluster_buildings"])

    sums = pd.DataFrame(records, columns=["h3_index", "cluster_MWh", "cluster_buildings"])
    sums["cluster_MWh"] = sums["cluster_MWh"].astype("int32")
    sums["cluster_buildings"] = sums["cluster_buildings"].astype("int32")
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
    s = df_hex.loc[:, ["h3_index", "kWh_per_m2", "gemiddeld_jaarverbruik_mWh"]].copy()
    s["kWh_per_m2"] = pd.to_numeric(s["kWh_per_m2"], errors="coerce").fillna(0).astype("float32")
    s["gemiddeld_jaarverbruik_mWh"] = pd.to_numeric(s["gemiddeld_jaarverbruik_mWh"], errors="coerce").fillna(0).astype("float32")

    hot = s["kWh_per_m2"] >= float(threshold_kwh_m2)
    # top_frac = 0.6  -> neem top 40% op MWh mee
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
    """
    blob = json.dumps(params, sort_keys=True) + f"|n={len(list(cell_ids))}|" + ",".join(sorted(map(str, cell_ids)))
    return hashlib.md5(blob.encode()).hexdigest()


# ============================================================
# Cached variant van cluster-aggregatie op shortlist
# ============================================================

@st.cache_data(show_spinner=False, max_entries=8, ttl=300)
def compute_clusters_cached(cache_key: str, df_hex_small: pd.DataFrame, k: int, ttl: int = 1800) -> pd.DataFrame:
    """
    Cached variant van aggregate_clusters zonder externe bestanden.
    Verwacht df_hex_small met kolommen:
      - h3_index, gemiddeld_jaarverbruik_mWh, aantal_huizen
    """
    # Minimaliseer kolommen en dtypes
    base = df_hex_small.loc[:, ["h3_index", "gemiddeld_jaarverbruik_mWh", "aantal_huizen"]].copy()

    base["gemiddeld_jaarverbruik_mWh"] = (
        pd.to_numeric(base["gemiddeld_jaarverbruik_mWh"], errors="coerce")
          .fillna(0)
          .astype("float32")
    )
    base["aantal_huizen"] = (
        pd.to_numeric(base["aantal_huizen"], errors="coerce")
          .fillna(0)
          .astype("int32")
    )

    lookup = {
        str(row.h3_index): (float(row.gemiddeld_jaarverbruik_mWh), int(row.aantal_huizen))
        for row in base.itertuples(index=False)
    }
    cells = list(lookup.keys())

    records = []
    for center, total_mwh, total_buildings in _sum_neighbors(lookup, cells, k):
        records.append(
            (
                center,
                int(round(total_mwh)),
                int(total_buildings),
            )
        )

    if not records:
        return pd.DataFrame(columns=["h3_index", "cluster_MWh", "cluster_buildings"])

    sums = pd.DataFrame(records, columns=["h3_index", "cluster_MWh", "cluster_buildings"])
    sums["cluster_MWh"] = sums["cluster_MWh"].astype("int32")
    sums["cluster_buildings"] = sums["cluster_buildings"].astype("int32")
    return sums




# ============================================================
# Selectie van sites (minimale afstand + capaciteit/benutting)
# ============================================================

@st.cache_data(show_spinner=False, max_entries=12)
def select_sites_from_clusters(
    cluster_df: pd.DataFrame,
    min_sep_cells: int,
    topk: int,
    cap_mwh: float,
    cap_buildings: int,
    ttl: int = 1800,
) -> pd.DataFrame:
    """
    Neemt cluster_df met:
      - h3_index, cluster_MWh, cluster_buildings
    En produceert top-k locaties:
      - connected_MWh/buildings (geclipped op capaciteit)
      - utilization_pct
      - lat/lon (centroid van h3 cel)
    """
    # Compacte kopie met expliciete dtypes
    out = cluster_df.loc[:, ["h3_index", "cluster_MWh", "cluster_buildings"]].copy()
    out["cluster_MWh"] = pd.to_numeric(out["cluster_MWh"], errors="coerce").fillna(0).astype("int32")
    out["cluster_buildings"] = pd.to_numeric(out["cluster_buildings"], errors="coerce").fillna(0).astype("int32")

    cap_mwh_f = float(cap_mwh)
    cap_bld_i = int(cap_buildings)

    out["connected_MWh"] = out["cluster_MWh"].clip(upper=cap_mwh_f).astype("int32")
    out["connected_buildings"] = out["cluster_buildings"].clip(upper=cap_bld_i).astype("int32")
    out["utilization"] = out["connected_MWh"] / max(cap_mwh_f, 1.0)

    # Sorteer op grootste aansluiting, selecteer met minimale onderlinge afstand
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
        for nh in _grid_disk_cached(r.h3_index, int(min_sep_cells)):
            blocked.add(nh)
        if len(chosen) >= int(topk):
            break

    # Materialiseer eindresultaat (records) met centrum-coördinaten
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
            "cap_MWh": float(cap_mwh_f),
            "cap_buildings": int(cap_bld_i),
        })

    # compacte DataFrame terug
    return pd.DataFrame(rows)
