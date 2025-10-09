# app.py
from __future__ import annotations

# ========== Imports ==========
import json
import gzip
from pathlib import Path

import h3
import orjson
import pandas as pd
import pydeck as pdk
import streamlit as st
import threading, time

# ---- interne modules (uit deze repo) ----
from core.config import (
    LAYER_CFG,
    BASEMAP_CFG,
    ENERGIEARMOEDE_PATH,
    KOOPWONINGEN_PATH,
    WOONCORPORATIE_PATH,
    SPOORDEEL_PATH,
    WATERDEEL_PATH,      # laat WATERDEEL_URL uit je config aan staan als je die wilt gebruiken
    DATA_CSV_PATH,       
    DATA_CSV_URL,
)

from core.utils import (
    format_dutch_number,
    get_dynamic_line_width,
    get_dynamic_resolution,
    colorize_geojson_cached,
    # nieuw centraal:
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
from core.io import load_geojson, load_data  # centrale loaders
from ui.sidebar import build_sidebar
from ui.kpis_and_tables import render_kpis, render_tabs

# RAM fix
def _periodic_cache_clear(interval_min: int = 30):
    """Leeg de Streamlit cache elke `interval_min` minuten in een achtergrondthread."""
    def _loop():
        while True:
            time.sleep(interval_min * 60)
            try:
                st.cache_data.clear()
                # optioneel: noteer het tijdstip in session_state voor UI-feedback
                st.session_state["last_cache_clear_ts"] = time.time()
                print(f"[AUTO-CLEAR] Cache leeggemaakt (interval={interval_min} min)")
            except Exception as e:
                print(f"[AUTO-CLEAR ERROR] {e}")
    t = threading.Thread(target=_loop, daemon=True)
    t.start()

# Start de cleaner bij app-opstart
if "auto_cache_cleaner_started" not in st.session_state:
    _periodic_cache_clear(30)  # elke 30 min
    st.session_state["auto_cache_cleaner_started"] = True

# ========== Streamlit pagina setup ==========
st.set_page_config(page_title="Friese Warmtevraagkaart", layout="wide")
st.markdown('<h1 style="font-size: 35px;">Friese Warmtevraagkaart (Heat Demand)</h1>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 16px; margin-top: -10px;">De kaart laat het gemiddelde jaarverbruik in 2024 zien van gas in m3, omgerekend naar kWh en MWh.</p>', unsafe_allow_html=True)
main_notice = st.empty()

# --- containers om de gewenste volgorde te forceren ---
kpi_container = st.container()
map_container = st.container()
tables_container = st.container()

# ========== GeoJSON / CSV laden ==========
# --- Laad geojsons (met extra buurt/gemeentenaam voor tooltips)
def _src(url_or_none, path: Path) -> str:
    # pydeck/geoloaders werken prima met een URL (string) of lokaal pad (string)
    return url_or_none or str(path)

_gj_common_props = ["buurtnaam", "gemeentenaam"]
# --- GeoJSONs
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

# --- Dataframe (CSV)
df_raw = load_data()

# ========== Sidebar / UI ==========
df_filtered_input, ui = build_sidebar(df_raw)

# ========== State init ==========
if "show_map" not in st.session_state:
    st.session_state.show_map = True
if "sites" not in st.session_state:
    st.session_state.sites = pd.DataFrame()
if "sites_costed" not in st.session_state:
    st.session_state.sites_costed = None

def _as_sorted_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return sorted(list(x))
    return [x]

# Bewaak filter-wijzigingen
# ===== Helpers voor stabiele vergelijkingen =====
def _as_sorted_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return sorted(list(x))
    return [x]

def _as_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def _as_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def _as_tuple_2(x, default=(0, 0)):
    try:
        a, b = x
        return (_as_int(a), _as_int(b))
    except Exception:
        return default

# ===== Bouw een volledig filters-profiel uit alle UI/state =====
def _build_filters_snapshot(ui: dict) -> dict:
    # Lagen-config keys (uit config)
    L = st.session_state.LAYER_CFG

    snap = {
        # Kaart & resolutie
        "zoom_level":          _as_int(ui.get("zoom_level")),
        "resolution":          _as_int(ui.get("resolution")),
        "extruded":            bool(ui.get("extruded")),
        "map_style":           ui.get("map_style", "light"),
        "hide_basemap":        bool(ui.get("hide_basemap", False)),

        # Hoofdlaag + indicatief
        "show_main_layer":     bool(ui.get("show_main_layer", True)),
        "show_indicative":     bool(ui.get("show_indicative_layer", True)),
        "threshold":           _as_float(ui.get("threshold", 50.0)),

        # Woonplaats / filters
        "woonplaats":          _as_sorted_list(ui.get("woonplaats_selectie")),
        "Energieklasse":       _as_sorted_list([str(x) for x in ui.get("energieklasse_selectie", [])]),
        "bouwjaar_range":      _as_tuple_2(ui.get("bouwjaar_range", (0, 3000))),
        "type_pand":           str(ui.get("pand_selectie", "")),

        # Woonlagen (extra layers) + opacity
        L["energiearmoede"]["toggle_key"]:  bool(st.session_state.get(L["energiearmoede"]["toggle_key"], False)),
        L["koopwoningen"]["toggle_key"]:    bool(st.session_state.get(L["koopwoningen"]["toggle_key"], False)),
        L["wooncorporatie"]["toggle_key"]:  bool(st.session_state.get(L["wooncorporatie"]["toggle_key"], False)),
        "extra_opacity":       _as_float(ui.get("extra_opacity", 0.55)),

        # Bodemlagen (spoor/water) + opacity
        L["spoordeel"]["toggle_key"]:       bool(st.session_state.get(L["spoordeel"]["toggle_key"], False)),
        L["waterdeel"]["toggle_key"]:       bool(st.session_state.get(L["waterdeel"]["toggle_key"], False)),
        "spoor_opacity":       _as_float(ui.get("spoor_opacity", 0.5)),
        "water_opacity":       _as_float(ui.get("water_opacity", 0.6)),

        # Participatie (voor KPI’s)
        "participatie":        _as_int(ui.get("participatie", st.session_state.get("participatie", 80))),

        # Sites/Collectieve warmtevoorziening (analyse)
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
    return snap

# ===== Init vorige filters =====
if "prev_filters" not in st.session_state:
    st.session_state.prev_filters = _build_filters_snapshot(ui)

# ===== Vergelijk huidige met vorige =====
current_filters = _build_filters_snapshot(ui)
prev = st.session_state.prev_filters

filters_changed = current_filters != prev  # diepe dict-vergelijking is nu genoeg

# ===== Reageer op filter-wijzigingen =====
if filters_changed:
    # (optioneel) debounce om niet meerdere renders per seconde te doen
    now = time.time()
    last = st.session_state.get("_last_auto_update", 0)
    if (now - last) > 0.6:  # 600 ms
        st.session_state.prev_filters = current_filters
        st.session_state.show_map = True   # auto “Maak Kaart”
        st.toast("De filters zijn gewijzigd. Kaart wordt automatisch bijgewerkt.")
        st.session_state["_last_auto_update"] = now

# ========== H3 indexering en aggregaties ==========
BASE_H3_RES = 12

def _build_res12(df_src: pd.DataFrame, ttl=1200-1800):
    lat_np = df_src["latitude"].astype("float32").to_numpy()
    lon_np = df_src["longitude"].astype("float32").to_numpy()
    h3_res12 = [h3.latlng_to_cell(float(la), float(lo), BASE_H3_RES) for la, lo in zip(lat_np, lon_np)]
    return df_src.assign(h3_r12=h3_res12)

# Rebuild bij afwijkende index/length
need_rebuild = (
    "df_with_h3_res12" not in st.session_state
    or len(st.session_state.df_with_h3_res12) != len(df_filtered_input)
    or not st.session_state.df_with_h3_res12.index.equals(df_filtered_input.index)
)

if need_rebuild:
    st.session_state.df_with_h3_res12 = _build_res12(df_filtered_input)
    # Ook de parent-cache leegmaken, anders blijven oude indices hangen
    st.session_state.h3_parent_cache = {}


if "h3_parent_cache" not in st.session_state:
    st.session_state.h3_parent_cache = {}  # {res: pd.Series}

def _ensure_parent_series_for(res: int) -> pd.Series:
    if res == BASE_H3_RES:
        return st.session_state.df_with_h3_res12["h3_r12"]
    if res in st.session_state.h3_parent_cache:
        return st.session_state.h3_parent_cache[res]
    parents = [h3.cell_to_parent(h, res) for h in st.session_state.df_with_h3_res12["h3_r12"]]
    ser = pd.Series(parents, index=st.session_state.df_with_h3_res12.index, name=f"h3_r{res}")
    st.session_state.h3_parent_cache[res] = ser
    return ser

# Snelle aggregatie op res12 + roll-up
@st.cache_data(show_spinner=False, max_entries=5, ttl=300)
def build_res12_agg(df_points_res12: pd.DataFrame):
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
               sum_vbos=("aantal_VBOs", "sum")
           )
           .reset_index()
    )
    return res12

@st.cache_data(show_spinner=False, max_entries=5, ttl=300)
def rollup_to_resolution(res12_agg: pd.DataFrame, target_res: int, _cache_key: int = 0, ttl=1200-1800):
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

# ========== Hoofdscherm ==========
if st.session_state.show_map:
    # Zorg dat we dezelfde rijen/volgorde houden als gefilterde input
    idx = df_filtered_input.index
    df_base = st.session_state.df_with_h3_res12
    df = df_base.reindex(idx)

    # vangnet: ontbrekende h3_r12 alsnog genereren
    if "h3_r12" in df.columns:
        missing = df["h3_r12"].isna()
        if missing.any():
            df.loc[missing, "h3_r12"] = [
                h3.latlng_to_cell(float(la), float(lo), BASE_H3_RES)
                for la, lo in zip(df.loc[missing, "latitude"], df.loc[missing, "longitude"])
            ]

    # Sampling (ongebruikt)
    zoom_level = int(ui["zoom_level"])
    if zoom_level <= 3:
        df_points = df.sample(frac=0.05, random_state=42)
    elif zoom_level <= 7:
        df_points = df.sample(frac=0.20, random_state=42)
    else:
        df_points = df

    # Parent-H3 per resolutie
    res = int(ui["resolution"])
    if res == BASE_H3_RES:
        df_points["h3_index"] = df_points["h3_r12"]
    else:
        parent_series = _ensure_parent_series_for(res).rename(f"h3_r{res}")
        if f"h3_r{res}" not in df_points.columns:
            df_points = df_points.join(parent_series, how="left")
        df_points["h3_index"] = df_points[f"h3_r{res}"]

    # Extra info voor tooltips
    df_extra_info = df_points[["h3_index", "woonplaats", "Energieklasse", "pandstatus"]].drop_duplicates(subset=["h3_index"])

    # Numerieke kolommen
    df_points["gemiddeld_jaarverbruik_mWh"] = pd.to_numeric(df_points["gemiddeld_jaarverbruik_mWh"], errors="coerce").fillna(0).astype("float32")
    df_points["kWh_per_m2"]                 = pd.to_numeric(df_points["kWh_per_m2"], errors="coerce").astype("float32")
    df_points["totale_oppervlakte"]         = pd.to_numeric(df_points["totale_oppervlakte"], errors="coerce").fillna(0).astype("float32")
    df_points["bouwjaar"]                   = pd.to_numeric(df_points["bouwjaar"], errors="coerce").astype("float32")
    df_points["aantal_VBOs"]                = pd.to_numeric(df_points["aantal_VBOs"], errors="coerce").fillna(0).astype("float32")

    # Per zoom: res12 agg + roll-up
    res12_agg = build_res12_agg(df_points[["h3_r12","kWh_per_m2","gemiddeld_jaarverbruik_mWh",
                                           "totale_oppervlakte","bouwjaar","aantal_VBOs"]])
    df_filtered = rollup_to_resolution(res12_agg, res, _cache_key=res)

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
        "gemiddeld_jaarverbruik_mWh_r","Energieklasse","bouwjaar",
        "MWh_per_ha","MWh_per_ha_r","area_ha","area_ha_r"
    ]]

    # NL-format kolommen voor tooltip (hexlaag)
    def _fmt0(x): return format_dutch_number(int(x), 0)
    def _fmt2(x): return format_dutch_number(x, 2)

    df_filtered["aantal_huizen_fmt"]                = df_filtered["aantal_huizen"].apply(_fmt0)
    df_filtered["aantal_VBOs_fmt"]                  = df_filtered["aantal_VBOs"].apply(_fmt0)
    df_filtered["MWh_per_ha_r_fmt"]                 = df_filtered["MWh_per_ha_r"].apply(_fmt2)
    df_filtered["gemiddeld_jaarverbruik_mWh_r_fmt"] = df_filtered["gemiddeld_jaarverbruik_mWh_r"].apply(_fmt0)
    df_filtered["area_ha_r_fmt"]                    = df_filtered["area_ha_r"].apply(_fmt2)
    df_filtered["kWh_per_m2_fmt"]                   = df_filtered["kWh_per_m2"].apply(_fmt0)
    df_filtered["totale_oppervlakte_fmt"]           = df_filtered["totale_oppervlakte"].apply(_fmt0)
    df_filtered["bouwjaar_fmt"]                     = df_filtered["bouwjaar"].astype(int)

    df_filtered["hex_section_display"]  = "block"
    df_filtered["site_section_display"] = "none"
    df_filtered["geo_section_display"]  = "none"

    # Indicatieve aandachtslaag
    df_filtered_area = df_filtered.copy()
    df_filtered_area["indicatief_aandachtsgebied"] = df_filtered_area["kWh_per_m2"] > threshold

    # --------- Warmtevoorziening (alleen als toggle aan) ---------
    if ui["show_sites_layer"]:
        shortlist_top_frac = 0.6
        threshold_kwh_m2 = float(ui["threshold"])
        k_val = int(st.session_state.kring_radius)

        centers_keep = shortlist_centers(df_filtered, threshold_kwh_m2=threshold_kwh_m2, top_frac=shortlist_top_frac)
        df_for_clusters = df_filtered.merge(centers_keep, on="h3_index", how="inner") if not centers_keep.empty else df_filtered

        cluster_params = {"k": k_val, "threshold": threshold_kwh_m2, "shortlist_frac": shortlist_top_frac}
        cache_key = filters_fingerprint(cluster_params, df_for_clusters["h3_index"].astype(str).unique())

        clusters = compute_clusters_cached(
            cache_key,
            df_for_clusters.loc[:, ["h3_index", "gemiddeld_jaarverbruik_mWh", "aantal_huizen"]],
            k_val
        )

        clusters = clusters.merge(
            df_filtered[["h3_index", "woonplaats", "kWh_per_m2", "aantal_VBOs", "gemiddeld_jaarverbruik_mWh"]],
            on="h3_index", how="left"
        )

        st.session_state.sites = select_sites_from_clusters(
            clusters,
            min_sep_cells=st.session_state.min_sep,
            topk=st.session_state.n_sites,
            cap_mwh=float(st.session_state.cap_mwh),
            cap_buildings=int(st.session_state.cap_buildings),
            ttl=1800,
        )

        # Voeg woonplaats toe + gebied_label
        sites = st.session_state.sites.merge(
            df_filtered[["h3_index", "woonplaats"]].drop_duplicates(),
            on="h3_index", how="left"
        )
        sites["gebied_label"] = sites["woonplaats"].fillna("Onbekend")

        # Kosten & formatting (NL)
        def _fmt0s(x): return format_dutch_number(int(x), 0)
        sites["cluster_buildings_fmt"]   = sites["cluster_buildings"].apply(_fmt0s)
        sites["cap_buildings_fmt"]       = sites["cap_buildings"].apply(_fmt0s)
        sites["connected_buildings_fmt"] = sites["connected_buildings"].apply(_fmt0s)
        sites["cluster_MWh_fmt"]         = sites["cluster_MWh"].apply(_fmt0s)
        sites["cap_MWh_fmt"]             = sites["cap_MWh"].apply(_fmt0s)
        sites["connected_MWh_fmt"]       = sites["connected_MWh"].apply(_fmt0s)
        sites["utilization_pct_fmt"]     = sites["utilization_pct"].apply(lambda x: f"{int(x)}")
        sites["vaste_kosten"]            = float(st.session_state.fixed_cost)
        sites["opex"]                    = float(st.session_state.opex_pct) / 100.0 * sites["vaste_kosten"]
        sites["variabele_kosten"]        = sites["connected_MWh"] * float(st.session_state.var_cost)
        sites["indicatieve_kosten_site"] = (sites["vaste_kosten"] + sites["opex"] + sites["variabele_kosten"]).round(0)

        # Tooltip secties
        sites["hex_section_display"]  = "none"
        sites["site_section_display"] = "block"
        sites["geo_section_display"]  = "none"

        st.session_state.sites = sites
        st.session_state.sites_costed = sites
    else:
        st.session_state.sites = pd.DataFrame()
        st.session_state.sites_costed = None

    # ========== Kaartlagen ==========
    # Woonlagen (GeoJSONs gefilterd op woonplaats bij zoom 11–12)
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

    # Bodemlagen (blijven altijd aan/uit volgens hun eigen toggles niet gekoppeld aan 'Geen achtergrondkaart')
    bodem_layers = create_bodem_layers(geojson_dict)

    # H3 hoofdlaag(en) per zoom
    layers = create_layers_by_zoom(df_filtered, ui["show_main_layer"], ui["extruded"], ui["zoom_level"])

    # --- Zorg dat sites altijd lat/lon hebben voor de ScatterplotLayer ---
    def _ensure_site_coords(df_sites: pd.DataFrame) -> pd.DataFrame:
        if df_sites is None or df_sites.empty:
            return df_sites
        need_lat = "lat" not in df_sites.columns
        need_lon = "lon" not in df_sites.columns
        if not (need_lat or need_lon):
            return df_sites
        # haal het celcentrum uit H3
        latlon = df_sites["h3_index"].apply(lambda h: pd.Series(h3.cell_to_latlng(h), index=["lat", "lon"]))
        df_out = df_sites.copy()
        for col in ["lat", "lon"]:
            if col not in df_out.columns:
                df_out[col] = latlon[col]
        return df_out

    if ui.get("show_sites_layer"):
        if st.session_state.sites is not None and not st.session_state.sites.empty:
            st.session_state.sites = _ensure_site_coords(st.session_state.sites)
        if st.session_state.sites_costed is not None and not st.session_state.sites_costed.empty:
            st.session_state.sites_costed = _ensure_site_coords(st.session_state.sites_costed)

    # Indicatieve aandachtslaag
    # Sites (bovenop de H3/indicatief)
    if ui["show_indicative_layer"]:
        layers.append(create_indicative_area_layer(df_filtered_area, threshold, ui["extruded"], ui["zoom_level"]))
    if ui["show_sites_layer"] and st.session_state.sites is not None and not st.session_state.sites.empty:
        layers.extend(create_site_layers(st.session_state.sites, st.session_state.sites_costed))

    # ===== Basemap toggle (alleen CARTO/OSM uitzetten) =====
    hide_bg = bool(ui.get("hide_basemap"))
    base_layers = build_base_layers(ui.get("map_style"), hide_bg)
    bodem_layers = create_bodem_layers(geojson_dict)

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

    lat, lon, zoom = _view_for_selection(df, ui["woonplaats_selectie"])
    st.session_state.view_state = pdk.ViewState(
        longitude=lon, latitude=lat, zoom=zoom,
        min_zoom=7.5, max_zoom=14, pitch=0, bearing=0,
    )

    # ========== KPI (boven de kaart) ==========
    with kpi_container:
        render_kpis(df_filtered, st.session_state.participatie)

    # ========== Kaart ==========
    # Deck-config: 'blank' canvas als 'Geen achtergrondkaart' aan staat
    deck_kwargs = {
        "map_style": "blank" if hide_bg else None
    }
    
    with map_container:
        st.pydeck_chart(
            pdk.Deck(
                layers=all_layers,
                initial_view_state=st.session_state.view_state,
                tooltip=build_deck_tooltip(),
                **deck_kwargs,
            ),
            width=None
        )

    # ========== Tabellen (onder de kaart) ==========
    with tables_container:
        render_tabs(df_filtered, threshold, ui["show_sites_layer"], st.session_state.get("sites_costed"))
