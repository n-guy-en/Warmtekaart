# app.py
from __future__ import annotations

# ========== Imports ==========
import time
import gc

import h3
import pandas as pd
import pydeck as pdk
import streamlit as st

# ---- interne modules (uit deze repo) ----
from core.config import (
    LAYER_CFG,
    BASEMAP_CFG,
    ENERGIEARMOEDE_PATH,
    KOOPWONINGEN_PATH,
    WOONCORPORATIE_PATH,
    SPOORDEEL_PATH,
    WATERDEEL_PATH,
)
from core.utils import (
    format_dutch_number,
    get_dynamic_line_width,
    get_dynamic_resolution,
    get_color,
    build_deck_tooltip,
)
from core.layers import (
    build_base_layers,
    create_layers_by_zoom,
    create_indicative_area_layer,
    create_site_layers,
    create_bodem_layers,
    filter_geojson_by_selection,
    create_extra_layers,
)
from core.h3sites import (
    shortlist_centers,
    filters_fingerprint,
    compute_clusters_cached,
    select_sites_from_clusters,
)
from core.io import load_geojson, load_data, load_precomputed_h3_grouped
from ui.sidebar import build_sidebar
from ui.kpis_and_tables import render_kpis, render_tabs

# (optioneel) live RAM-meting in sidebar
#try:
#    import psutil, os
#    mem = psutil.Process(os.getpid()).memory_info().rss / 1e6
#    st.sidebar.write(f"RAM-gebruik: {mem:.1f} MB")
#except Exception:
#    pass

# TODO_RAMDEBUG: verwijder deze helper zodra RAM-diagnose is afgerond.
def _log_ram(label: str) -> None:
    try:
        import psutil, os  # noqa: F401  # hergebruik bestaande import
    except Exception:
        return
    try:
        mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1e6
        print(f"[RAM_DEBUG] {label}: {mem_mb:.1f} MB")
    except Exception:
        pass

# ========== Eerste init (NIET cache leegmaken) ==========
if "app_initialized" not in st.session_state:
    st.session_state["app_initialized"] = True

# ========== Streamlit pagina setup ==========
st.set_page_config(page_title="Friese Warmtevraagkaart", layout="wide")
st.markdown('<h1 style="font-size: 35px;">Friese Warmtevraagkaart (Heat Demand)</h1>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 16px; margin-top: -10px;">De kaart laat het gemiddelde jaarverbruik in 2024 zien van gas in m3, omgerekend naar kWh en MWh.</p>', unsafe_allow_html=True)

# --- containers om de gewenste volgorde te forceren ---
kpi_container = st.container()
map_container = st.container()
tables_container = st.container()

# ========== GeoJSON / CSV laden ==========
_gj_common_props = ["buurtnaam", "gemeentenaam"]

gjson_energiearmoede = load_geojson(
    ENERGIEARMOEDE_PATH,
    keep_props=[LAYER_CFG["energiearmoede"]["prop_name"], *_gj_common_props],
    coord_precision=3
)
gjson_koopwoningen = load_geojson(
    KOOPWONINGEN_PATH,
    keep_props=[LAYER_CFG["koopwoningen"]["prop_name"], *_gj_common_props],
    coord_precision=3
)
gjson_corporatie = load_geojson(
    WOONCORPORATIE_PATH,
    keep_props=[LAYER_CFG["wooncorporatie"]["prop_name"], *_gj_common_props],
    coord_precision=3
)
gjson_spoordeel = load_geojson(
    SPOORDEEL_PATH,
    keep_props=[],
    coord_precision=3
)

df_raw = load_data()
_log_ram("after_load_data")

# ========== Sidebar / UI ==========
df_filtered_input, ui = build_sidebar(df_raw)
_log_ram("after_sidebar")

# ========== State init ==========
st.session_state.setdefault("show_map", False)
st.session_state.setdefault("sites", [])
st.session_state.setdefault("sites_costed", [])
st.session_state.setdefault("sites_ready", False)

st.session_state.setdefault("first_hint_shown", False)

# ===== Helpers voor stabiele vergelijkingen =====
def _as_sorted_list(x):
    if x is None: return []
    if isinstance(x, (list, tuple, set)): return sorted(list(x))
    return [x]

def _as_int(x, default=0):
    try: return int(x)
    except Exception: return default

def _as_float(x, default=0.0):
    try: return float(x)
    except Exception: return default

def _as_tuple_2(x, default=(0, 0)):
    try:
        a, b = x
        return (_as_int(a), _as_int(b))
    except Exception:
        return default

# ===== Filters-snapshot =====
def _build_filters_snapshot(ui: dict) -> dict:
    L = st.session_state.get("LAYER_CFG", LAYER_CFG)
    return {
        "zoom_level":          _as_int(ui.get("zoom_level")),
        "resolution":          _as_int(ui.get("resolution")),
        "extruded":            bool(ui.get("extruded")),
        "map_style":           ui.get("map_style", "light"),
        "hide_basemap":        bool(ui.get("hide_basemap", False)),
        "show_main_layer":     bool(ui.get("show_main_layer", True)),
        "show_indicative_layer": bool(ui.get("show_indicative_layer", True)),
        "threshold":           _as_float(ui.get("threshold", 50.0)),
        "woonplaats":          _as_sorted_list(ui.get("woonplaats_selectie")),
        "Energieklasse":       _as_sorted_list([str(x) for x in ui.get("energieklasse_selectie", [])]),
        "bouwjaar_range":      _as_tuple_2(ui.get("bouwjaar_range", (0, 3000))),
        "type_pand":           str(ui.get("pand_selectie", "")),
        L["energiearmoede"]["toggle_key"]:  bool(st.session_state.get(L["energiearmoede"]["toggle_key"], False)),
        L["koopwoningen"]["toggle_key"]:    bool(st.session_state.get(L["koopwoningen"]["toggle_key"], False)),
        L["wooncorporatie"]["toggle_key"]:  bool(st.session_state.get(L["wooncorporatie"]["toggle_key"], False)),
        "extra_opacity":       _as_float(ui.get("extra_opacity", 0.55)),
        L["spoordeel"]["toggle_key"]:       bool(st.session_state.get(L["spoordeel"]["toggle_key"], False)),
        L["waterdeel"]["toggle_key"]:       bool(st.session_state.get(L["waterdeel"]["toggle_key"], False)),
        "spoor_opacity":       _as_float(ui.get("spoor_opacity", 0.5)),
        "water_opacity":       _as_float(ui.get("water_opacity", 0.6)),
        "participatie":        _as_int(ui.get("participatie", st.session_state.get("participatie", 80))),
        "show_sites_layer":    bool(ui.get("show_sites_layer", False)),
        "kring_radius":        _as_int(ui.get("kring_radius", st.session_state.get("kring_radius", 3))),
        "min_sep":             _as_int(ui.get("min_sep", st.session_state.get("min_sep", 3))),
        "n_sites":             _as_int(ui.get("n_sites", st.session_state.get("n_sites", 3))),
        "cap_mwh":             _as_int(ui.get("cap_mwh", st.session_state.get("cap_mwh", 100_000))),
        "cap_buildings":       _as_int(ui.get("cap_buildings", st.session_state.get("cap_buildings", 1_000))),
        "fixed_cost":          _as_int(ui.get("fixed_cost", st.session_state.get("fixed_cost", 25_000))),
        "var_cost":            _as_int(ui.get("var_cost", st.session_state.get("var_cost", 35))),
        "opex_pct":            _as_int(ui.get("opex_pct", st.session_state.get("opex_pct", 10))),
    }

def _filters_without_zoom(ui: dict) -> dict:
    snap = _build_filters_snapshot(ui).copy()
    snap.pop("zoom_level", None)
    snap.pop("resolution", None)
    return snap

if "prev_filters" not in st.session_state:
    st.session_state.prev_filters = _build_filters_snapshot(ui)

current_filters = _build_filters_snapshot(ui)
filters_changed = current_filters != st.session_state.prev_filters

if filters_changed:
    st.session_state.prev_filters = current_filters
    st.session_state.show_map = False
    st.session_state["_map_dirty"] = True
    st.session_state["sites_ready"] = False
else:
    st.session_state["_map_dirty"] = False

if st.sidebar.button("Maak kaart"):
    st.session_state.show_map = True
    st.session_state["_map_dirty"] = False

if st.session_state.get("_map_dirty"):
    pass  # manier om toekomstige logica toe te voegen; melding verplaatst naar kaartcontainer

# ========== H3 indexering en aggregaties ==========
BASE_H3_RES = 12

def _build_res12(df_src: pd.DataFrame):
    lat_np = df_src["latitude"].astype("float32").to_numpy()
    lon_np = df_src["longitude"].astype("float32").to_numpy()
    h3_res12 = [h3.latlng_to_cell(float(la), float(lo), BASE_H3_RES) for la, lo in zip(lat_np, lon_np)]
    return df_src.assign(h3_r12=h3_res12)

@st.cache_data(show_spinner=False, max_entries=2, ttl=1800)
def _build_res12_cached(df_src: pd.DataFrame):
    return _build_res12(df_src)

@st.cache_data(show_spinner=False, max_entries=6, ttl=1800)
def _ensure_parent_series_for_cached(df_with_h3_res12: pd.DataFrame, res: int) -> pd.Series:
    if res == BASE_H3_RES:
        return df_with_h3_res12["h3_r12"]
    parents = [h3.cell_to_parent(h, res) for h in df_with_h3_res12["h3_r12"]]
    return pd.Series(parents, index=df_with_h3_res12.index, name=f"h3_r{res}")

# Kleiner cachevenster/TTL om RAM-opbouw te voorkomen
@st.cache_data(show_spinner=False, max_entries=2, ttl=240)
def build_res12_agg(df_points_res12: pd.DataFrame):
    tmp = df_points_res12.loc[:, ["h3_r12","kWh_per_m2","gemiddeld_jaarverbruik_mWh","totale_oppervlakte","bouwjaar","aantal_VBOs"]]
    kwh_sum = pd.to_numeric(tmp["kWh_per_m2"], errors="coerce").fillna(0).astype("float32")
    cnt = pd.Series(1, index=tmp.index, dtype="int32")
    res12 = (
        tmp.assign(kwh_sum=kwh_sum, cnt=cnt)
           .groupby("h3_r12", sort=False, observed=True)
           .agg(
               sum_mwh=("gemiddeld_jaarverbruik_mWh", "sum"),
               sum_area=("totale_oppervlakte", "sum"),
               sum_kwh=("kwh_sum", "sum"),
               cnt=("cnt", "sum"),
               mean_bouwjaar=("bouwjaar", "mean"),
               sum_vbos=("aantal_VBOs", "sum")
           )
           .reset_index()
    )
    return res12

@st.cache_data(show_spinner=False, max_entries=2, ttl=240)
def rollup_to_resolution(res12_agg: pd.DataFrame, target_res: int, _cache_key: int = 0):
    if target_res == 12:
        out = res12_agg.copy().rename(columns={"h3_r12": "h3_index"})
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
                   sum_vbos=("sum_vbos", "sum")
               )
               .reset_index()
               .rename(columns={"h3_parent": "h3_index"})
        )
    out["kWh_per_m2"] = (out["sum_kwh"] / out["cnt"]).round(0)
    out["aantal_huizen"] = out["cnt"].astype(int)
    out["gemiddeld_jaarverbruik_mWh"] = out["sum_mwh"].round(0)
    out["totale_oppervlakte"] = out["sum_area"].round(0)
    out["bouwjaar"] = out["mean_bouwjaar"].round(0)
    out["aantal_VBOs"] = out["sum_vbos"].round(0).astype(int)
    return out[[
        "h3_index","kWh_per_m2","gemiddeld_jaarverbruik_mWh","totale_oppervlakte",
        "aantal_huizen","bouwjaar","aantal_VBOs"
    ]]

AVG_HA_BY_RES = {9: 17.6, 10: 8.8, 11: 4.4, 12: 2.2}
def area_ha_for_res(res: int) -> float:
    return AVG_HA_BY_RES.get(res, 2.2)


def _build_map_dataframe(df_input: pd.DataFrame, res: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_map = _build_res12_cached(df_input).reindex(df_input.index)
    _log_ram("after_h3_res12_raw")

    if "h3_r12" in df_map.columns:
        missing = df_map["h3_r12"].isna()
        if missing.any():
            df_map.loc[missing, "h3_r12"] = [
                h3.latlng_to_cell(float(la), float(lo), BASE_H3_RES)
                for la, lo in zip(df_map.loc[missing, "latitude"], df_map.loc[missing, "longitude"])
            ]

    if res == BASE_H3_RES:
        df_map["h3_index"] = df_map["h3_r12"]
    else:
        parent_series = _ensure_parent_series_for_cached(df_map, res).rename(f"h3_r{res}")
        if f"h3_r{res}" not in df_map.columns:
            df_map = df_map.join(parent_series, how="left")
        df_map["h3_index"] = df_map[f"h3_r{res}"]

    for col, kind in [
        ("gemiddeld_jaarverbruik_mWh", "float32"),
        ("kWh_per_m2", "float32"),
        ("totale_oppervlakte", "float32"),
        ("bouwjaar", "float32"),
        ("aantal_VBOs", "int32"),
    ]:
        if col in df_map.columns:
            df_map[col] = pd.to_numeric(df_map[col], errors="coerce").astype(kind)

    df_extra_info = df_map.loc[:, ["h3_index", "woonplaats"]].drop_duplicates(subset=["h3_index"])

    res12_agg = build_res12_agg(
        df_map[["h3_r12", "kWh_per_m2", "gemiddeld_jaarverbruik_mWh",
                "totale_oppervlakte", "bouwjaar", "aantal_VBOs"]]
    )
    df_filtered = rollup_to_resolution(res12_agg, res, _cache_key=res)
    _log_ram("after_rollup_raw")
    del res12_agg

    return df_filtered, df_extra_info, df_map


# ========== Hoofdscherm ==========
if st.session_state.show_map:
    res = int(ui["resolution"])
    zoom_level = int(ui.get("zoom_level", 0))
    threshold = float(ui["threshold"])

    df_filtered, df_extra_info, df_view_source = _build_map_dataframe(df_filtered_input, res)

    # afronden/afgeleiden
    df_filtered["kWh_per_m2"] = df_filtered["kWh_per_m2"].round(0)
    df_filtered["gemiddeld_jaarverbruik_mWh"] = df_filtered["gemiddeld_jaarverbruik_mWh"].round(0)
    df_filtered["totale_oppervlakte"] = df_filtered["totale_oppervlakte"].round(0)
    df_filtered["bouwjaar"] = df_filtered["bouwjaar"].round(0)

    # Oppervlakte en dichtheid
    cell_area_ha = area_ha_for_res(res)
    df_filtered["area_ha"] = float(cell_area_ha)
    df_filtered["MWh_per_ha"] = df_filtered["gemiddeld_jaarverbruik_mWh"] / df_filtered["area_ha"]
    df_filtered["area_ha_r"] = df_filtered["area_ha"]
    df_filtered["MWh_per_ha_r"] = df_filtered["MWh_per_ha"].round(2)
    df_filtered["gemiddeld_jaarverbruik_mWh_r"] = df_filtered["gemiddeld_jaarverbruik_mWh"].round(0).astype(int)

    # Compacte dtypes vasthouden om RAM onder controle te houden
    for col, dtype in [
        ("kWh_per_m2", "float32"),
        ("gemiddeld_jaarverbruik_mWh", "float32"),
        ("totale_oppervlakte", "float32"),
        ("MWh_per_ha", "float32"),
        ("MWh_per_ha_r", "float32"),
        ("area_ha", "float32"),
        ("area_ha_r", "float32"),
    ]:
        if col in df_filtered.columns:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce").astype(dtype)

    for col in ["aantal_huizen", "aantal_VBOs", "gemiddeld_jaarverbruik_mWh_r", "bouwjaar"]:
        if col in df_filtered.columns:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors="coerce").fillna(0).astype("int32")

    # Kleuren en 3D hoogte
    df_filtered["color"] = df_filtered["kWh_per_m2"].apply(get_color)
    MAX_HEIGHT = max(df_filtered["kWh_per_m2"].max(), 1)
    threshold = float(ui["threshold"])
    df_filtered["scaled_elevation"] = ((df_filtered["kWh_per_m2"] - 10) / max((MAX_HEIGHT - 10), 1) * MAX_HEIGHT)
    df_filtered["scaled_elevation"] = df_filtered["scaled_elevation"].clip(lower=0, upper=threshold)

    # merge extra tooltip info
    df_filtered = df_filtered.merge(df_extra_info, on="h3_index", how="left")
    df_filtered = df_filtered[[
        "h3_index","kWh_per_m2","color","woonplaats","aantal_huizen","aantal_VBOs",
        "scaled_elevation","totale_oppervlakte","gemiddeld_jaarverbruik_mWh",
        "gemiddeld_jaarverbruik_mWh_r","bouwjaar",
        "MWh_per_ha","MWh_per_ha_r","area_ha","area_ha_r"
    ]]

    # --------- Warmtevoorziening (alleen als toggle aan) ---------

    # --------- Warmtevoorziening (alleen als toggle aan én woonplaats geselecteerd) ---------
    woonplaatsen_selected = [wp for wp in ui.get("woonplaats_selectie", []) if wp]
    allow_sites = ui.get("show_sites_layer") and zoom_level >= 11 and woonplaatsen_selected
    sites_records = []
    if allow_sites:
        compute_requested = ui.get("compute_sites", False)
        if compute_requested:
            shortlist_top_frac = 0.85
            threshold_kwh_m2 = float(ui["threshold"])
            k_val = int(st.session_state.kring_radius)

            centers_keep = shortlist_centers(df_filtered, threshold_kwh_m2=threshold_kwh_m2, top_frac=shortlist_top_frac)
            df_for_clusters = df_filtered.merge(centers_keep, on="h3_index", how="inner") if not centers_keep.empty else df_filtered

            cluster_params = {"k": k_val, "threshold": threshold_kwh_m2, "shortlist_frac": shortlist_top_frac}
            cache_key = filters_fingerprint(cluster_params, df_for_clusters["h3_index"].astype(str).unique())

            cluster_input = df_for_clusters.loc[:, ["h3_index", "gemiddeld_jaarverbruik_mWh", "aantal_huizen"]]
            clusters = compute_clusters_cached(cache_key, cluster_input, k_val)
            _log_ram("after_clusters")

            clusters = clusters.merge(
                df_filtered[["h3_index", "woonplaats", "kWh_per_m2", "aantal_VBOs", "gemiddeld_jaarverbruik_mWh"]],
                on="h3_index", how="left"
            )

            sites_df = select_sites_from_clusters(
                clusters,
                min_sep_cells=st.session_state.min_sep,
                topk=st.session_state.n_sites,
                cap_mwh=float(st.session_state.cap_mwh),
                cap_buildings=int(st.session_state.cap_buildings),
                ttl=1800,
            )

            records = []
            if sites_df is not None and not sites_df.empty:
                sites = sites_df.merge(
                    df_filtered[["h3_index", "woonplaats"]].drop_duplicates(),
                    on="h3_index", how="left"
                )
                sites["gebied_label"] = sites["woonplaats"].fillna("Onbekend")

                def _fmt0s(x): return format_dutch_number(int(x), 0)

                for rec in sites.itertuples(index=False):
                    record = {
                        "h3_index": rec.h3_index,
                        "woonplaats": rec.woonplaats,
                        "gebied_label": rec.gebied_label,
                        "cluster_buildings": int(rec.cluster_buildings),
                        "cap_buildings": int(rec.cap_buildings),
                        "connected_buildings": int(rec.connected_buildings),
                        "cluster_MWh": int(rec.cluster_MWh),
                        "cap_MWh": int(rec.cap_MWh),
                        "connected_MWh": int(rec.connected_MWh),
                        "utilization_pct": int(rec.utilization_pct),
                        "cluster_buildings_fmt": _fmt0s(rec.cluster_buildings),
                        "cap_buildings_fmt": _fmt0s(rec.cap_buildings),
                        "connected_buildings_fmt": _fmt0s(rec.connected_buildings),
                        "cluster_MWh_fmt": _fmt0s(rec.cluster_MWh),
                        "cap_MWh_fmt": _fmt0s(rec.cap_MWh),
                        "connected_MWh_fmt": _fmt0s(rec.connected_MWh),
                        "utilization_pct_fmt": f"{int(rec.utilization_pct)}",
                        "vaste_kosten": float(st.session_state.fixed_cost),
                        "opex": float(st.session_state.opex_pct) / 100.0 * float(st.session_state.fixed_cost),
                        "variabele_kosten": float(rec.connected_MWh) * float(st.session_state.var_cost),
                    }
                    record["indicatieve_kosten_site"] = int(round(
                        record["vaste_kosten"] + record["opex"] + record["variabele_kosten"]
                    ))
                    record["hex_section_display"] = "none"
                    record["site_section_display"] = "block"
                    record["geo_section_display"] = "none"
                    record["lat"], record["lon"] = h3.cell_to_latlng(rec.h3_index)
                    records.append(record)

            st.session_state.sites = records
            st.session_state.sites_costed = records
            st.session_state.sites_ready = bool(records)
            del cluster_input, clusters, sites_df
        elif not st.session_state.get("sites_ready"):
            st.session_state.sites = []
            st.session_state.sites_costed = []
        if st.session_state.get("sites_ready"):
            sites_records = st.session_state.sites
    else:
        st.session_state.sites = []
        st.session_state.sites_costed = []
        st.session_state.sites_ready = False

    if not st.session_state.get("sites_ready"):
        sites_records = []

    # ========== Kaartlagen ==========
    geojson_dict = {
        "energiearmoede": gjson_energiearmoede,
        "koopwoningen": gjson_koopwoningen,
        "corporatie": gjson_corporatie,
        "spoordeel": gjson_spoordeel,
    }

    extra_layers = []
    if any([
        st.session_state.get(LAYER_CFG["energiearmoede"]["toggle_key"]),
        st.session_state.get(LAYER_CFG["koopwoningen"]["toggle_key"]),
        st.session_state.get(LAYER_CFG["wooncorporatie"]["toggle_key"]),
    ]):
        extra_layers = create_extra_layers(geojson_dict, ui["woonplaats_selectie"], ui["zoom_level"], ui["extra_opacity"])

    bodem_layers = create_bodem_layers(geojson_dict)

    # H3 hoofdlaag(en) per zoom → géén records, direct DataFrame
    base_hex_cols = [
        "h3_index","color","scaled_elevation","woonplaats",
        "aantal_huizen","aantal_VBOs",
        "MWh_per_ha_r","gemiddeld_jaarverbruik_mWh_r",
        "area_ha_r","kWh_per_m2","totale_oppervlakte","bouwjaar"
    ]
    df_hex_view = df_filtered.loc[:, base_hex_cols].copy()

    def _fmt0_val(series):
        return series.astype("int64").map(lambda v: format_dutch_number(int(v), 0))

    def _fmt2_val(series):
        return series.astype("float32").map(lambda v: format_dutch_number(float(v), 2))

    df_hex_view["aantal_huizen_fmt"]                = _fmt0_val(df_hex_view["aantal_huizen"])
    df_hex_view["aantal_VBOs_fmt"]                  = _fmt0_val(df_hex_view["aantal_VBOs"])
    df_hex_view["MWh_per_ha_r_fmt"]                 = _fmt2_val(df_hex_view["MWh_per_ha_r"])
    df_hex_view["gemiddeld_jaarverbruik_mWh_r_fmt"] = _fmt0_val(df_hex_view["gemiddeld_jaarverbruik_mWh_r"])
    df_hex_view["area_ha_r_fmt"]                    = _fmt2_val(df_hex_view["area_ha_r"])
    df_hex_view["kWh_per_m2_fmt"]                   = _fmt0_val(df_hex_view["kWh_per_m2"])
    df_hex_view["totale_oppervlakte_fmt"]           = _fmt0_val(df_hex_view["totale_oppervlakte"])
    df_hex_view["bouwjaar_fmt"]                     = df_hex_view["bouwjaar"].astype("int64").map(lambda v: format_dutch_number(int(v), 0))

    df_hex_view["hex_section_display"]  = "block"
    df_hex_view["site_section_display"] = "none"
    df_hex_view["geo_section_display"]  = "none"

    cols_for_hex = [
        "h3_index","color","scaled_elevation","woonplaats",
        "aantal_huizen","aantal_VBOs",
        "MWh_per_ha_r","gemiddeld_jaarverbruik_mWh_r",
        "area_ha_r","kWh_per_m2","totale_oppervlakte","bouwjaar",
        "aantal_huizen_fmt","aantal_VBOs_fmt",
        "MWh_per_ha_r_fmt","gemiddeld_jaarverbruik_mWh_r_fmt",
        "area_ha_r_fmt","kWh_per_m2_fmt","totale_oppervlakte_fmt","bouwjaar_fmt",
        "hex_section_display","site_section_display","geo_section_display"
    ]
    df_hex_view = df_hex_view.loc[:, cols_for_hex]
    indic_mask = df_filtered["kWh_per_m2"] > threshold
    df_indicative = df_hex_view.loc[indic_mask, :]
    _log_ram("before_pydeck_layers")
    layers = create_layers_by_zoom(df_hex_view, ui["show_main_layer"], ui["extruded"], ui["zoom_level"])

    # Sites (bovenop)
    if allow_sites and sites_records:
        sites_costed_records = st.session_state.sites_costed if st.session_state.get("sites_ready") else None
        layers.extend(create_site_layers(sites_records, sites_costed_records))

    # Indicatieve aandachtslaag
    if ui["show_indicative_layer"] and not df_indicative.empty:
        layers.append(create_indicative_area_layer(df_indicative, ui["extruded"], ui["zoom_level"]))

    # Basemap
    hide_bg = bool(ui.get("hide_basemap"))
    base_layers = build_base_layers(ui.get("map_style"), hide_bg)

    # Volgorde: basemap → bodem → woonlagen → H3/indicatief/sites
    all_layers = base_layers + bodem_layers + extra_layers + layers

    # ========== ViewState ==========
    def _view_for_selection(df_full, woonplaatsen_geselecteerd):
        FRIESLAND_CENTER = (53.125, 5.75); FRIESLAND_ZOOM = 8
        MIN_ZOOM, MAX_ZOOM = 8, 12.0
        if not woonplaatsen_geselecteerd:
            return FRIESLAND_CENTER[0], FRIESLAND_CENTER[1], FRIESLAND_ZOOM
        df_sel = df_full[df_full["woonplaats"].isin(woonplaatsen_geselecteerd)]
        if df_sel.empty:
            return FRIESLAND_CENTER[0], FRIESLAND_CENTER[1], FRIESLAND_ZOOM
        lat_center = float(df_sel["latitude"].mean()); lon_center = float(df_sel["longitude"].mean())
        if len(woonplaatsen_geselecteerd) == 1:
            return lat_center, lon_center, 12.0
        lat_min, lat_max = float(df_sel["latitude"].min()), float(df_sel["latitude"].max())
        lon_min, lon_max = float(df_sel["longitude"].min()), float(df_sel["longitude"].max())
        lat_span = max(0.0001, lat_max - lat_min); lon_span = max(0.0001, lon_max - lon_min)
        span = max(lat_span, lon_span)
        if   span > 2.0:  zoom = 8.0
        elif span > 1.0:  zoom = 8.0
        elif span > 0.5:  zoom = 9.0
        elif span > 0.25: zoom = 9.0
        elif span > 0.12: zoom = 10.0
        else:             zoom = 11.0
        zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
        return lat_center, lon_center, zoom

    lat, lon, zoom = _view_for_selection(df_view_source, ui["woonplaats_selectie"])
    st.session_state.view_state = pdk.ViewState(
        longitude=lon, latitude=lat, zoom=zoom,
        min_zoom=7.5, max_zoom=14, pitch=0, bearing=0,
    )

    # ========== KPI ==========
    with kpi_container:
        render_kpis(df_filtered, st.session_state.participatie)

    # ========== Kaart render + cleanup ==========
    deck_kwargs = {"map_style": "blank" if hide_bg else None}

    with map_container:
        deck = pdk.Deck(
            layers=all_layers,
            initial_view_state=st.session_state.view_state,
            tooltip=build_deck_tooltip(),
            **deck_kwargs,
        )
        st.pydeck_chart(deck, use_container_width=True)

    # Opruimen om RAM-pieken terug te geven
    del deck, all_layers, layers, base_layers, bodem_layers, extra_layers, df_hex_view
    if "sites_records" in locals(): del sites_records
    if "sites_costed_records" in locals(): del sites_costed_records
    gc.collect()

    # ========== Tabellen ==========
    with tables_container:
        render_tabs(df_filtered, threshold, ui["show_sites_layer"], st.session_state.get("sites_costed"))
    st.session_state["_map_dirty"] = False

    # --- expliciet opruimen om pieken kort te houden ---
    for _var in ["df_view_source", "df_indicative", "df_filtered", "df_hex_view"]:
        if _var in locals():
            try: del locals()[_var]
            except: pass
    gc.collect()

else:
    with map_container:
        # - Eerste keer openen → initiële instructie
        # - Daarna, als filters gewijzigd zijn → update-instructie
        # - Anders (nog niets gedaan) → neutrale instructie
        if st.session_state.get("_map_dirty"):
            st.info("Filters gewijzigd. Klik op 'Maak kaart' om de kaart bij te werken.")
        elif not st.session_state.get("first_hint_shown", False):
            st.info("Selecteer de gewenste filters. Klik op 'Maak kaart' om de kaart weer te geven.")
            st.session_state["first_hint_shown"] = True
        else:
            st.info("Klik op 'Maak kaart' om de kaart weer te geven.")
