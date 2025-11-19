# app.py
from __future__ import annotations

# ========== Imports ==========
import gc
import math
from typing import Any

import h3
import pandas as pd
import pydeck as pdk
import streamlit as st
from pyproj import Transformer

# ---- interne modules ----
from core.config import (
    BASE_H3_RES,
    AVG_HA_BY_RES,
    LAYER_CFG,
    BASEMAP_CFG,
    ENERGIEARMOEDE_PATH,
    KOOPWONINGEN_PATH,
    WOONCORPORATIE_PATH,
    WATER_POTENTIE_PATH,
    BUURT_POTENTIE_PATH,
    WARMTENET_PATH,
)
from core.utils import (
    format_dutch_number,
    get_dynamic_line_width,
    get_dynamic_resolution,
    get_color,
    build_deck_tooltip,
    get_color_palette,
    extract_numeric_values,
    compute_quantile_breaks,
    format_numeric_range_labels,
)
from core.layers import (
    build_base_layers,
    create_layers_by_zoom,
    create_indicative_area_layer,
    create_site_layers,
    create_extra_layers,
    create_warmtenet_layers,
)
from core.h3sites import (
    shortlist_centers,
    filters_fingerprint,
    compute_clusters_cached,
    select_sites_from_clusters,
    aggregate_clusters,
)
from core.io import load_geojson, load_data
from ui.sidebar import build_sidebar
from ui.kpis_and_tables import render_kpis, render_tabs


def _extract_selected_hex_from_payload(state_obj) -> str | None:
    """Parse Streamlit/pydeck selection payload en geef h3_index terug."""
    if state_obj is None:
        return None

    objects = None
    if isinstance(state_obj, dict):
        if "objects" in state_obj:
            objects = state_obj.get("objects")
        elif "selection" in state_obj:
            sel = state_obj.get("selection", {})
            if isinstance(sel, dict):
                objects = sel.get("objects")
    else:
        sel_attr = getattr(state_obj, "selection", None)
        if isinstance(sel_attr, dict):
            objects = sel_attr.get("objects")
        elif hasattr(state_obj, "get"):
            maybe = state_obj.get("selection")
            if isinstance(maybe, dict):
                objects = maybe.get("objects")

    if not objects:
        return None

    first_obj = None
    if isinstance(objects, dict):
        first_obj = next(iter(objects.values()), None)
    elif isinstance(objects, list) and objects:
        first_obj = objects[0]

    if isinstance(first_obj, list) and first_obj:
        first_obj = first_obj[0]

    if isinstance(first_obj, dict) and "object" in first_obj and isinstance(first_obj["object"], dict):
        first_obj = first_obj["object"]

    if isinstance(first_obj, dict) and "properties" in first_obj and isinstance(first_obj["properties"], dict):
        candidate = first_obj["properties"].get("h3_index")
        if candidate:
            return str(candidate)

    if isinstance(first_obj, dict):
        for key in ("h3_index", "hex_id", "hexagon", "cell_id"):
            candidate = first_obj.get(key)
            if candidate:
                return str(candidate)

    return None


_RD_TO_WGS84 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)


def _first_coordinate(coords):
    if isinstance(coords, (list, tuple)):
        if coords and isinstance(coords[0], (int, float)):
            if len(coords) >= 2:
                return coords[0], coords[1]
            return None
        for sub in coords:
            sample = _first_coordinate(sub)
            if sample:
                return sample
    return None


def _needs_rd_to_wgs_conversion(gjson: dict) -> bool:
    if not gjson or gjson.get("type") != "FeatureCollection":
        return False
    for feat in gjson.get("features", []):
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates")
        sample = _first_coordinate(coords)
        if sample:
            x, y = sample
            if abs(x) <= 180 and abs(y) <= 90:
                return False
            if abs(x) > 200 or abs(y) > 200:
                return True
    return False


def _transform_coords_rd_to_wgs(coords):
    if isinstance(coords, (list, tuple)):
        if coords and isinstance(coords[0], (int, float)):
            if len(coords) >= 2:
                lon, lat = _RD_TO_WGS84.transform(coords[0], coords[1])
                return [float(lon), float(lat)]
            return coords
        return [_transform_coords_rd_to_wgs(c) for c in coords]
    return coords


def _convert_geojson_to_wgs84_if_needed(gjson: dict) -> dict:
    if not gjson or gjson.get("type") != "FeatureCollection":
        return gjson
    if not _needs_rd_to_wgs_conversion(gjson):
        return gjson
    feats_new = []
    for feat in gjson.get("features", []):
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates")
        new_geom = {
            "type": geom.get("type"),
            "coordinates": _transform_coords_rd_to_wgs(coords),
        }
        feats_new.append({
            "type": "Feature",
            "properties": feat.get("properties"),
            "geometry": new_geom,
        })
    return {"type": "FeatureCollection", "features": feats_new}


def _format_kwh_value(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{format_dutch_number(value, 0)} kWh"


def _format_percent_value(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{format_dutch_number(value * 100, 1)}%"


def _buurt_extra_rows(props: dict) -> str:
    demand = props.get("Demand_ontvangen_kWh")
    heat = props.get("heatDemand_kWh")
    rows = []
    if demand not in (None, ""):
        try:
            demand_val = float(demand)
        except (TypeError, ValueError):
            demand_val = None
        rows.append(f"<div class='tooltip-row'>Ontvangen warmtevraag: {_format_kwh_value(demand_val)}</div>")
    if heat not in (None, ""):
        try:
            heat_val = float(heat)
        except (TypeError, ValueError):
            heat_val = None
        rows.append(f"<div class='tooltip-row'>Totale heat demand: {_format_kwh_value(heat_val)}</div>")
    return "".join(rows)


def _build_water_potential_meta(gjson: dict) -> dict:
    cfg = LAYER_CFG["water_potentie"]
    n_colors = cfg.get("n_colors", 5)
    colors = get_color_palette(cfg.get("palette", "BuGn"), n_colors, cfg.get("alpha", 210))
    values = extract_numeric_values(gjson, cfg["prop_name"])
    breaks = compute_quantile_breaks(values, n_colors)
    if breaks and len(breaks) > n_colors - 1:
        breaks = breaks[: n_colors - 1]
    legend_labels = format_numeric_range_labels(breaks, suffix=cfg.get("tooltip_unit", "kWh"), decimals=0)
    return {
        "breaks": breaks,
        "colors": colors,
        "labels": legend_labels,
        "value_formatter": lambda v: _format_kwh_value(v),
        "extra_rows_fn": None,
        "default_opacity": 0.7,
        "location_row_display": "none",
    }


def _build_buurt_potential_meta(gjson: dict) -> dict:
    cfg = LAYER_CFG["buurt_potentie"]
    n_colors = cfg.get("n_colors", 5)
    colors = get_color_palette(cfg.get("palette", "YlOrRd"), n_colors, cfg.get("alpha", 210))
    breaks = [i / n_colors for i in range(1, n_colors)]
    legend_labels = format_numeric_range_labels([b * 100 for b in breaks], suffix="%", decimals=0)
    return {
        "breaks": breaks,
        "colors": colors,
        "labels": legend_labels,
        "value_formatter": lambda v: _format_percent_value(v),
        "extra_rows_fn": _buurt_extra_rows,
        "default_opacity": 0.7,
        "location_row_display": "block",
    }

def _build_warmtenet_meta(gjson: dict | None) -> dict:
    """Maak kleur- en legenda-info voor warmtenet model."""
    base_meta = {
        "color_map": {},
        "labels": {},
        "default_opacity": 0.85,
        "woonplaatsen": [],
        "types": [],
        "type_by_key": {},
        "wp_by_key": {},
    }
    if not gjson or not isinstance(gjson, dict):
        return base_meta

    def _distinct_colors(n: int, alpha: int) -> list[list[int]]:
        # verdeel tinten via golden ratio en varieer lichtheid om vergelijkbare tinten te voorkomen
        import colorsys
        colors = []
        if n <= 0:
            return colors
        for i in range(n):
            hue = (i * 0.618033988749895) % 1.0
            sat = 0.65
            val = 0.90 - (0.12 * ((i % 3) / 3))  # kleine variatie in helderheid
            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
            colors.append([int(r * 255), int(g * 255), int(b * 255), alpha])
        return colors

    labels: dict[str, dict[str, str]] = {}
    all_keys: list[str] = []
    woonplaatsen_all: set[str] = set()
    types_all: set[str] = set()
    type_by_key: dict[str, str] = {}
    wp_by_key: dict[str, str] = {}
    for feat in gjson.get("features", []):
        props = feat.get("properties") or {}
        key = str(props.get("bron_key") or "").strip()
        if not key:
            continue
        woonplaats_raw = str(props.get("woonplaats") or "").strip()
        bron_label = str(props.get("bron_id") or "").strip()
        type_bron = str(props.get("type_bron") or props.get("gegevensbron") or "").strip()
        if not bron_label and "__" in key:
            bron_label = key.split("__", 1)[1]
        pretty_label = bron_label or key
        labels.setdefault(key, {
            "label": pretty_label,
            "woonplaats_norm": woonplaats_raw.lower(),
        })
        all_keys.append(key)
        wp_by_key[key] = woonplaats_raw
        type_by_key[key] = type_bron
        if woonplaats_raw:
            woonplaatsen_all.add(woonplaats_raw)
        if type_bron:
            types_all.add(type_bron)

    unique_keys = sorted(set(all_keys))
    if not unique_keys:
        return base_meta

    cfg = LAYER_CFG.get("warmtenet_model", {})
    alpha = cfg.get("alpha", 230)
    colors = _distinct_colors(len(unique_keys), alpha)
    color_map = {key: colors[idx % len(colors)] for idx, key in enumerate(unique_keys)}

    return {
        "color_map": color_map,
        "labels": labels,
        "default_opacity": 0.85,
        "woonplaatsen": sorted(woonplaatsen_all),
        "types": sorted(types_all),
        "type_by_key": type_by_key,
        "wp_by_key": wp_by_key,
    }

# (optioneel) live RAM-meting in sidebar
#try:
#    import psutil, os
#    mem = psutil.Process(os.getpid()).memory_info().rss / 1e6
#    st.sidebar.write(f"RAM-gebruik: {mem:.1f} MB")
#except Exception:
#    pass

# TODO_RAMDEBUG: verwijder deze helper zodra RAM-diagnose is afgerond.
def _log_ram(label: str) -> None:
    """Log het huidige RAM-gebruik voor snelle diagnosestappen."""
    try:
        import os
        import psutil
    except Exception:
        return
    try:
        mem_mb = psutil.Process(os.getpid()).memory_info().rss / 1e6
        print(f"[RAM_DEBUG] {label}: {mem_mb:.1f} MB")
    except Exception:
        pass

# ========== Eerste init (NIET cache leegmaken) ==========
# Initialiseer een Streamlit-sessie éénmalig per gebruiker
if "app_initialized" not in st.session_state:
    st.session_state["app_initialized"] = True

# ========== Streamlit pagina setup ==========
st.set_page_config(page_title="Friese Warmtevraagkaart", layout="wide")
st.markdown('<h1 style="font-size: 35px;">Friese Warmtevraagkaart (Heat Demand)</h1>', unsafe_allow_html=True)
st.markdown(
    """
    <p style="font-size: 16px; margin-top: -10px;">
        De kaart laat het gemiddelde jaarverbruik in 2024 zien van gas in m³,
        omgerekend naar kWh en MWh.
    </p>
    """,
    unsafe_allow_html=True,
)

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
gjson_water_potentie = load_geojson(
    WATER_POTENTIE_PATH,
    keep_props=["Potentie_kWh", "id"],
    coord_precision=5,
)
gjson_water_potentie = _convert_geojson_to_wgs84_if_needed(gjson_water_potentie)
gjson_buurt_potentie = load_geojson(
    BUURT_POTENTIE_PATH,
    keep_props=["Perc_covered", "Demand_ontvangen_kWh", "heatDemand_kWh", "buurtnaam", "gemeentenaam"],
    coord_precision=5,
)
gjson_buurt_potentie = _convert_geojson_to_wgs84_if_needed(gjson_buurt_potentie)
gjson_warmtenet = load_geojson(
    WARMTENET_PATH,
    keep_props=[
        "layer",
        "woonplaats",
        "bron_key",
        "bron_id",
        "vraag_id",
        "gegevensbron",
        "type_bron",
        "bron_mwh_per_jaar",
        "vraag_mwh_per_jaar",
        "padlengte_m",
        "geometrie_lengte_m",
    ],
    coord_precision=5,
)
gjson_warmtenet = _convert_geojson_to_wgs84_if_needed(gjson_warmtenet)

potential_meta: dict[str, dict] = {}
if gjson_water_potentie:
    potential_meta["water_potentie"] = _build_water_potential_meta(gjson_water_potentie)
if gjson_buurt_potentie:
    potential_meta["buurt_potentie"] = _build_buurt_potential_meta(gjson_buurt_potentie)
warmtenet_meta = _build_warmtenet_meta(gjson_warmtenet)

df_raw = load_data()
_log_ram("after_load_data")

# ========== Sidebar / UI ==========
sidebar_out = build_sidebar(df_raw, potential_meta, warmtenet_meta)
map_button_clicked_sidebar = False
if isinstance(sidebar_out, tuple):
    if len(sidebar_out) == 3:
        df_filtered_input, ui, map_button_clicked_sidebar = sidebar_out
    else:
        df_filtered_input, ui = sidebar_out
else:
    df_filtered_input, ui = sidebar_out

map_button_clicked_main = st.button("Maak kaart")
map_button_clicked = bool(map_button_clicked_sidebar or map_button_clicked_main)
_log_ram("after_sidebar")

# ========== State init ==========
st.session_state.setdefault("show_map", False)
st.session_state.setdefault("sites", [])
st.session_state.setdefault("sites_costed", [])
st.session_state.setdefault("sites_ready", False)
st.session_state.setdefault("manual_site_h3", None)

st.session_state.setdefault("first_hint_shown", False)

# ===== Helpers voor stabiele vergelijkingen =====
def _as_sorted_list(x):
    """Converteer invoer naar een gesorteerde lijst voor Jaccard-vergelijkingen."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return sorted(list(x))
    return [x]

def _as_int(x, default=0):
    """Robuuste int-cast met fallbackwaarde."""
    try:
        return int(x)
    except Exception:
        return default

def _as_float(x, default=0.0):
    """Robuuste float-cast met fallbackwaarde."""
    try:
        return float(x)
    except Exception:
        return default

def _as_tuple_2(x, default=(0, 0)):
    """Converteer een iterabele naar tuple[int, int] voor bouwjaar slider."""
    try:
        a, b = x
        return (_as_int(a), _as_int(b))
    except Exception:
        return default

# ===== Filters-snapshot =====
def _build_filters_snapshot(ui: dict) -> dict:
    """Maak een hashbare snapshot van alle filters voor change-detectie."""
    L = st.session_state.get("LAYER_CFG", LAYER_CFG)
    return {
        "zoom_level":          _as_int(ui.get("zoom_level")),
        "resolution":          _as_int(ui.get("resolution")),
        "extruded":            bool(ui.get("extruded")),
        "map_style":           ui.get("map_style", "light"),
        "hide_basemap":        bool(ui.get("hide_basemap", False)),
        "show_main_layer":     bool(ui.get("show_main_layer", True)),
        "show_indicative_layer": bool(ui.get("show_indicative_layer", True)),
        "warmte_hex_opacity":  _as_float(ui.get("warmte_hex_opacity", st.session_state.get("warmte_hex_opacity", 0.6))),
        "threshold":           _as_float(ui.get("threshold", 50.0)),
        "gemeente":            _as_sorted_list(ui.get("gemeente_selectie")),
        "woonplaats":          _as_sorted_list(ui.get("woonplaats_selectie")),
        "Energieklasse":       _as_sorted_list([str(x) for x in ui.get("energieklasse_selectie", [])]),
        "bouwjaar_range":      _as_tuple_2(ui.get("bouwjaar_range", (0, 3000))),
        "type_pand":           str(ui.get("pand_selectie", "")),
        L["energiearmoede"]["toggle_key"]:  bool(st.session_state.get(L["energiearmoede"]["toggle_key"], False)),
        L["koopwoningen"]["toggle_key"]:    bool(st.session_state.get(L["koopwoningen"]["toggle_key"], False)),
        L["wooncorporatie"]["toggle_key"]:  bool(st.session_state.get(L["wooncorporatie"]["toggle_key"], False)),
        "extra_opacity":       _as_float(ui.get("extra_opacity", 0.55)),
        L["water_potentie"]["toggle_key"]:  bool(st.session_state.get(L["water_potentie"]["toggle_key"], False)),
        "water_potentie_opacity": _as_float(ui.get("water_potentie_opacity", st.session_state.get("water_potentie_opacity", 0.7))),
        L["buurt_potentie"]["toggle_key"]:  bool(st.session_state.get(L["buurt_potentie"]["toggle_key"], False)),
        "buurt_potentie_opacity": _as_float(ui.get("buurt_potentie_opacity", st.session_state.get("buurt_potentie_opacity", 0.7))),
        "participatie":        _as_int(ui.get("participatie", st.session_state.get("participatie", 80))),
        "show_sites_layer":    bool(ui.get("show_sites_layer", False)),
        "sites_hex_opacity":   _as_float(ui.get("sites_hex_opacity", st.session_state.get("sites_hex_opacity", 0.85))),
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
    """Snapshot zonder zoomvelden, handig voor UI-vergelijkingen."""
    snap = _build_filters_snapshot(ui).copy()
    snap.pop("zoom_level", None)
    snap.pop("resolution", None)
    return snap

def _changed_filter_keys(prev: dict, curr: dict) -> set[str]:
    """Bepaal welke filtervelden gewijzigd zijn tussen twee snapshots."""
    keys = set(prev) | set(curr)
    return {k for k in keys if prev.get(k) != curr.get(k)}

H3_RES13_COL = f"h3_r{BASE_H3_RES}"

if "prev_filters" not in st.session_state:
    st.session_state.prev_filters = _build_filters_snapshot(ui)

current_filters = _build_filters_snapshot(ui)
filters_changed = current_filters != st.session_state.prev_filters

if filters_changed:
    changed_keys = _changed_filter_keys(st.session_state.prev_filters, current_filters)
    st.session_state.prev_filters = current_filters
    woonplaats_only_change = bool(changed_keys) and changed_keys.issubset({"woonplaats"})
    visual_only_change = bool(changed_keys) and changed_keys.issubset({
        "warmte_hex_opacity",
        "sites_hex_opacity",
        "water_potentie_opacity",
        "buurt_potentie_opacity",
    })
    if woonplaats_only_change and st.session_state.get("show_map"):
        st.session_state["_map_changed"] = False
        st.session_state["sites_ready"] = False
    elif visual_only_change:
        st.session_state["_map_changed"] = False
    else:
        st.session_state.show_map = False
        st.session_state["_map_changed"] = True
        st.session_state["sites_ready"] = False
else:
    st.session_state["_map_changed"] = False

if map_button_clicked:
    st.session_state.show_map = True
    st.session_state["_map_changed"] = False

# ========== H3 indexering en aggregaties ==========
def _build_res13(df_src: pd.DataFrame):
    """Bereken de h3-index op basisresolutie (BASE_H3_RES) voor alle rijen."""
    lat_np = df_src["latitude"].astype("float32").to_numpy()
    lon_np = df_src["longitude"].astype("float32").to_numpy()
    base_cells = [h3.latlng_to_cell(float(la), float(lo), BASE_H3_RES) for la, lo in zip(lat_np, lon_np)]
    return df_src.assign(**{H3_RES13_COL: base_cells})

@st.cache_data(show_spinner=False, max_entries=2, ttl=1800)
def _build_res13_cached(df_src: pd.DataFrame):
    """Cache-wrapper rond _build_res13 om herhaald werk te voorkomen."""
    return _build_res13(df_src)

@st.cache_data(show_spinner=False, max_entries=6, ttl=1800)
def _ensure_parent_series_for_cached(df_with_res13: pd.DataFrame, res: int) -> pd.Series:
    """Geef h3-index op gewenste resolutie, bereken ouders indien nodig."""
    if res == BASE_H3_RES:
        return df_with_res13[H3_RES13_COL]
    parents = [h3.cell_to_parent(h, res) for h in df_with_res13[H3_RES13_COL]]
    return pd.Series(parents, index=df_with_res13.index, name=f"h3_r{res}")

# Kleiner cachevenster/TTL om RAM-opbouw te voorkomen
@st.cache_data(show_spinner=False, max_entries=2, ttl=240)
def build_res13_agg(df_points_res13: pd.DataFrame):
    """Aggregeer alle puntdata naar de basisresolutie."""
    tmp = df_points_res13.loc[
        :,
        [
            H3_RES13_COL,
            "kWh_per_m2",
            "gemiddeld_jaarverbruik_mWh",
            "totale_oppervlakte",
            "bouwjaar",
            "aantal_VBOs",
        ],
    ]
    kwh_sum = pd.to_numeric(tmp["kWh_per_m2"], errors="coerce").fillna(0).astype("float32")
    cnt = pd.Series(1, index=tmp.index, dtype="int32")
    res13 = (
        tmp.assign(kwh_sum=kwh_sum, cnt=cnt)
           .groupby(H3_RES13_COL, sort=False, observed=True)
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
    return res13

@st.cache_data(show_spinner=False, max_entries=2, ttl=240)
def rollup_to_resolution(res13_agg: pd.DataFrame, target_res: int, _cache_key: int = 0):
    """Rol de basisresolutie samen naar een doelresolutie en bereken afgeleide metrics."""
    if target_res == BASE_H3_RES:
        out = res13_agg.copy().rename(columns={H3_RES13_COL: "h3_index"})
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
    return out[
        [
            "h3_index",
            "kWh_per_m2",
            "gemiddeld_jaarverbruik_mWh",
            "totale_oppervlakte",
            "aantal_huizen",
            "bouwjaar",
            "aantal_VBOs",
        ]
    ]

def area_ha_for_res(res: int) -> float:
    """Gemiddelde hectare-oppervlakte per resolutie (fallback op basisres)."""
    return AVG_HA_BY_RES.get(res, AVG_HA_BY_RES[BASE_H3_RES])


def _build_map_dataframe(df_input: pd.DataFrame, res: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bereid dataframes voor de kaart en tooltips."""
    df_map = _build_res13_cached(df_input).reindex(df_input.index)
    _log_ram("after_h3_base_raw")

    if H3_RES13_COL in df_map.columns:
        missing = df_map[H3_RES13_COL].isna()
        if missing.any():
            df_map.loc[missing, H3_RES13_COL] = [
                h3.latlng_to_cell(float(la), float(lo), BASE_H3_RES)
                for la, lo in zip(df_map.loc[missing, "latitude"], df_map.loc[missing, "longitude"])
            ]

    if res == BASE_H3_RES:
        df_map["h3_index"] = df_map[H3_RES13_COL]
    else:
        parent_col = f"h3_r{res}"
        parent_series = _ensure_parent_series_for_cached(df_map, res).rename(parent_col)
        if parent_col not in df_map.columns:
            df_map = df_map.join(parent_series, how="left")
        df_map["h3_index"] = df_map[parent_col]

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

    res13_agg = build_res13_agg(
        df_map[[H3_RES13_COL, "kWh_per_m2", "gemiddeld_jaarverbruik_mWh",
                "totale_oppervlakte", "bouwjaar", "aantal_VBOs"]]
    )
    df_filtered = rollup_to_resolution(res13_agg, res, _cache_key=res)
    _log_ram("after_rollup_raw")
    del res13_agg

    return df_filtered, df_extra_info, df_map


def _build_site_records(
    sites_df: pd.DataFrame,
    df_filtered: pd.DataFrame,
    k_val: int,
) -> list[dict]:
    """
    Zet een sites-DataFrame om naar records met polygonen, hexagonen en geformatteerde metrics.
    """
    if sites_df is None or sites_df.empty:
        return []

    sites = sites_df.merge(
        df_filtered[["h3_index", "woonplaats"]].drop_duplicates(),
        on="h3_index",
        how="left",
    )
    woonplaats_series = sites["woonplaats"]
    if hasattr(woonplaats_series, "cat"):
        if "Onbekend" not in woonplaats_series.cat.categories:
            woonplaats_series = woonplaats_series.cat.add_categories(["Onbekend"])
    sites["woonplaats"] = woonplaats_series
    sites["gebied_label"] = sites["woonplaats"].fillna("Onbekend")

    def _safe_int(val, default=None):
        try:
            if val is None:
                return default
            if isinstance(val, float) and math.isnan(val):
                return default
        except Exception:
            pass
        try:
            return int(round(float(val)))
        except Exception:
            return default

    def _safe_float(val, default=None):
        try:
            if val is None:
                return default
            if isinstance(val, float) and math.isnan(val):
                return default
        except Exception:
            pass
        try:
            return float(val)
        except Exception:
            return default

    def _fmt0s(x):
        val = _safe_int(x)
        if val is None:
            return "-"
        return format_dutch_number(val, 0)

    def _fmt2s(x):
        val = _safe_float(x)
        if val is None:
            return "-"
        return format_dutch_number(val, 2)

    def _fmt4s(x):
        val = _safe_float(x)
        if val is None:
            return "-"
        return format_dutch_number(val, 4)

    def _fmt_year(x):
        val = _safe_int(x)
        if val is None:
            return "-"
        return str(val)

    records: list[dict] = []

    for idx, rec in enumerate(sites.itertuples(index=False), start=1):
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

        record["indicatieve_kosten_site"] = int(
            round(record["vaste_kosten"] + record["opex"] + record["variabele_kosten"])
        )
        record["hex_section_display"] = "block"
        record["site_section_display"] = "block"
        record["geo_section_display"] = "none"
        record["geo_extra_rows"] = ""
        record["gemeente_row_display"] = "block"
        record["buurt_row_display"] = "block"
        record["site_rank"] = idx

        cluster_buildings_val = max(record["cluster_buildings"], 0)
        if cluster_buildings_val > 0:
            cluster_mwh_val = float(record["cluster_MWh"])
            avg_mwh_per_pand = cluster_mwh_val / cluster_buildings_val
        else:
            avg_mwh_per_pand = 0.0
        record["MWh_per_pand"] = avg_mwh_per_pand
        record["MWh_per_pand_fmt"] = _fmt2s(avg_mwh_per_pand)

        hex_ids = list(h3.grid_disk(rec.h3_index, int(k_val)))
        df_site_hex = df_filtered[df_filtered["h3_index"].isin(hex_ids)].copy()

        coverage_polygons = []
        coverage_summary: dict[str, Any] = {}
        coverage_buildings: list[dict[str, Any]] = []
        coverage_hexes: list[dict[str, Any]] = []

        if hex_ids:
            try:
                multi_polys = h3.h3_set_to_multi_polygon(hex_ids, geo_json=True)
            except Exception:
                multi_polys = []
            for poly in multi_polys:
                for loop in poly:
                    coords = [[float(pt[1]), float(pt[0])] for pt in loop]
                    if coords and coords[0] != coords[-1]:
                        coords.append(coords[0])
                    coverage_polygons.append(coords)

        if not df_site_hex.empty:

            def _series_sum(df_local: pd.DataFrame, column_name: str, want_int: bool = False):
                if column_name not in df_local.columns:
                    return 0
                vals = pd.to_numeric(df_local[column_name], errors="coerce").fillna(0)
                total = float(vals.sum())
                return _safe_int(total, 0) if want_int else _safe_float(total, 0.0)

            def _series_mean(df_local: pd.DataFrame, column_name: str, want_int: bool = False):
                if column_name not in df_local.columns:
                    return 0 if want_int else 0.0
                vals = pd.to_numeric(df_local[column_name], errors="coerce").dropna()
                if vals.empty:
                    return 0 if want_int else 0.0
                avg = float(vals.mean())
                return _safe_int(avg, 0) if want_int else _safe_float(avg, 0.0)

            total_vbos = _series_sum(df_site_hex, "aantal_VBOs", True)
            total_huizen = _series_sum(df_site_hex, "aantal_huizen", True)
            total_mwh = _series_sum(df_site_hex, "gemiddeld_jaarverbruik_mWh")
            total_oppervlakte = _series_sum(df_site_hex, "totale_oppervlakte", True)
            total_area_ha = _series_sum(df_site_hex, "area_ha")
            total_area_m2 = _series_sum(df_site_hex, "area_m2")
            avg_kwh_m2 = _series_mean(df_site_hex, "kWh_per_m2")
            avg_bouwjaar = _series_mean(df_site_hex, "bouwjaar", True)
            avg_density = _series_mean(df_site_hex, "MWh_per_ha")
            avg_mwh_per_pand_summary = total_mwh / total_huizen if total_huizen else 0.0
            avg_area_ha = _series_mean(df_site_hex, "area_ha")
            avg_area_m2 = _series_mean(df_site_hex, "area_m2")

            coverage_summary = {
                "hex_count": len(df_site_hex),
                "hex_count_fmt": _fmt0s(len(df_site_hex)),
                "aantal_VBOs": total_vbos,
                "aantal_huizen": total_huizen,
                "aantal_VBOs_fmt": _fmt0s(total_vbos),
                "aantal_huizen_fmt": _fmt0s(total_huizen),
                "kWh_per_m2": avg_kwh_m2,
                "kWh_per_m2_fmt": _fmt0s(avg_kwh_m2),
                "gemiddeld_jaarverbruik_mWh": total_mwh,
                "gemiddeld_jaarverbruik_mWh_fmt": _fmt0s(total_mwh),
                "totale_oppervlakte": total_oppervlakte,
                "totale_oppervlakte_fmt": _fmt0s(total_oppervlakte),
                "area_ha_r": avg_area_ha,
                "area_ha_r_fmt": _fmt4s(avg_area_ha),
                "area_m2": avg_area_m2,
                "area_m2_fmt": _fmt0s(avg_area_m2),
                "area_ha_total": total_area_ha,
                "area_ha_total_fmt": _fmt2s(total_area_ha),
                "area_m2_total": total_area_m2,
                "area_m2_total_fmt": _fmt0s(total_area_m2),
                "MWh_per_ha": avg_density,
                "MWh_per_ha_fmt": _fmt2s(avg_density),
                "bouwjaar": avg_bouwjaar,
                "bouwjaar_fmt": _fmt_year(avg_bouwjaar),
                "MWh_per_pand": avg_mwh_per_pand_summary,
                "MWh_per_pand_fmt": _fmt2s(avg_mwh_per_pand_summary),
                "site_rank_label": record["site_rank"],
            }

            for cov_idx, cov in enumerate(df_site_hex.itertuples(index=False), start=1):
                cov_dict = {
                    "site_rank": idx,
                    "sub_rank": cov_idx,
                    "h3_index": getattr(cov, "h3_index", ""),
                    "geo_extra_rows": "",
                    "gemeente_row_display": "block",
                    "buurt_row_display": "block",
                    "woonplaats": getattr(cov, "woonplaats", "") or record.get("woonplaats", ""),
                    "aantal_huizen": _safe_int(getattr(cov, "aantal_huizen", 0), 0) or 0,
                    "aantal_VBOs": _safe_int(getattr(cov, "aantal_VBOs", 0), 0) or 0,
                    "MWh_per_ha_r": _safe_float(getattr(cov, "MWh_per_ha_r", 0.0), 0.0) or 0.0,
                    "gemiddeld_jaarverbruik_mWh_r": _safe_float(getattr(cov, "gemiddeld_jaarverbruik_mWh_r", 0.0), 0.0) or 0.0,
                    "area_ha_r": _safe_float(getattr(cov, "area_ha_r", 0.0), 0.0) or 0.0,
                    "area_m2": _safe_float(getattr(cov, "area_m2", 0.0), 0.0) or 0.0,
                    "kWh_per_m2": _safe_float(getattr(cov, "kWh_per_m2", 0.0), 0.0) or 0.0,
                    "totale_oppervlakte": _safe_int(getattr(cov, "totale_oppervlakte", 0), 0) or 0,
                    "bouwjaar": _safe_int(getattr(cov, "bouwjaar", 0), 0) or 0,
                    "aantal_huizen_fmt": _fmt0s(getattr(cov, "aantal_huizen", 0)),
                    "aantal_VBOs_fmt": _fmt0s(getattr(cov, "aantal_VBOs", 0)),
                    "MWh_per_ha_r_fmt": _fmt2s(getattr(cov, "MWh_per_ha_r", 0.0)),
                    "gemiddeld_jaarverbruik_mWh_r_fmt": _fmt0s(getattr(cov, "gemiddeld_jaarverbruik_mWh_r", 0)),
                    "area_ha_r_fmt": _fmt4s(getattr(cov, "area_ha_r", 0.0)),
                    "area_m2_fmt": _fmt0s(getattr(cov, "area_m2", 0.0)),
                    "kWh_per_m2_fmt": _fmt0s(getattr(cov, "kWh_per_m2", 0)),
                    "totale_oppervlakte_fmt": _fmt0s(getattr(cov, "totale_oppervlakte", 0)),
                    "bouwjaar_fmt": _fmt_year(getattr(cov, "bouwjaar", 0)),
                    "MWh_per_pand": _safe_float(getattr(cov, "MWh_per_pand", 0.0), 0.0) or 0.0,
                    "MWh_per_pand_fmt": _fmt2s(getattr(cov, "MWh_per_pand", 0.0)),
                    "hex_section_display": "block",
                    "site_section_display": "block",
                    "geo_section_display": "none",
                    "cluster_buildings": record["cluster_buildings"],
                    "cap_buildings": record["cap_buildings"],
                    "connected_buildings": record["connected_buildings"],
                    "cluster_MWh": record["cluster_MWh"],
                    "cap_MWh": record["cap_MWh"],
                    "connected_MWh": record["connected_MWh"],
                    "utilization_pct": record["utilization_pct"],
                    "cluster_buildings_fmt": record["cluster_buildings_fmt"],
                    "cap_buildings_fmt": record["cap_buildings_fmt"],
                    "connected_buildings_fmt": record["connected_buildings_fmt"],
                    "cluster_MWh_fmt": record["cluster_MWh_fmt"],
                    "cap_MWh_fmt": record["cap_MWh_fmt"],
                    "connected_MWh_fmt": record["connected_MWh_fmt"],
                    "utilization_pct_fmt": record["utilization_pct_fmt"],
                    "site_rank": idx,
                    "site_rank_label": idx,
                }
                coverage_hexes.append(cov_dict)

            if "polygon_shape" in df_site_hex.columns:
                coverage_buildings = df_site_hex.loc[:, ["h3_index", "polygon_shape"]].copy()
                coverage_buildings["polygon_shape"] = coverage_buildings["polygon_shape"].astype(str)
                coverage_buildings["site_rank"] = idx
                coverage_buildings["gebouw_id"] = coverage_buildings.index.astype(int) + 1
                coverage_buildings = coverage_buildings.to_dict("records")

        density_value = coverage_summary.get("MWh_per_ha", 0.0) if coverage_summary else 0.0
        record["MWh_per_ha"] = float(density_value or 0.0)
        record["MWh_per_ha_fmt"] = _fmt2s(record["MWh_per_ha"])

        record["coverage_polygons"] = coverage_polygons
        record["coverage_summary"] = coverage_summary
        record["coverage_buildings"] = coverage_buildings
        record["coverage_hexes"] = coverage_hexes
        record["coverage_hex_ids"] = hex_ids
        record["lat"], record["lon"] = h3.cell_to_latlng(rec.h3_index)
        records.append(record)

    return records


# ========== Hoofdscherm ==========
if st.session_state.show_map:
    res = int(ui["resolution"])
    zoom_level = int(ui.get("zoom_level", 0))
    threshold = float(ui["threshold"])

    df_filtered, df_extra_info, df_view_source = _build_map_dataframe(df_filtered_input, res)

    # afronden
    df_filtered["kWh_per_m2"] = df_filtered["kWh_per_m2"].round(0)
    df_filtered["gemiddeld_jaarverbruik_mWh"] = df_filtered["gemiddeld_jaarverbruik_mWh"].round(0)
    df_filtered["totale_oppervlakte"] = df_filtered["totale_oppervlakte"].round(0)
    df_filtered["bouwjaar"] = df_filtered["bouwjaar"].round(0)

    # Oppervlakte en dichtheid
    area_km2_lookup = {
        idx: float(h3.cell_area(idx, unit="km^2")) for idx in df_filtered["h3_index"].dropna().unique()
    }
    df_filtered["area_km2"] = df_filtered["h3_index"].map(area_km2_lookup).astype("float32")
    df_filtered["area_ha"] = (df_filtered["area_km2"] * 100.0).astype("float32")
    df_filtered["area_m2"] = (df_filtered["area_km2"] * 1_000_000.0).astype("float32")

    area_for_density = df_filtered["area_ha"].replace(0, pd.NA)
    df_filtered["MWh_per_ha"] = (
        df_filtered["gemiddeld_jaarverbruik_mWh"].div(area_for_density)
    ).fillna(0.0)
    df_filtered["area_ha_r"] = df_filtered["area_ha"]
    df_filtered["MWh_per_ha_r"] = df_filtered["MWh_per_ha"].round(2)
    df_filtered.drop(columns=["area_km2"], inplace=True)
    df_filtered["gemiddeld_jaarverbruik_mWh_r"] = df_filtered["gemiddeld_jaarverbruik_mWh"].round(0).astype(int)
    huizen_nonzero = df_filtered["aantal_huizen"].replace(0, pd.NA)
    df_filtered["MWh_per_pand"] = (
        df_filtered["gemiddeld_jaarverbruik_mWh"].div(huizen_nonzero)
    ).fillna(0.0)
    df_filtered["MWh_per_pand_r"] = df_filtered["MWh_per_pand"].round(2)

    # Compacte dtypes vasthouden om RAM onder controle te houden
    for col, dtype in [
        ("kWh_per_m2", "float32"),
        ("gemiddeld_jaarverbruik_mWh", "float32"),
        ("totale_oppervlakte", "float32"),
        ("MWh_per_ha", "float32"),
        ("MWh_per_ha_r", "float32"),
        ("MWh_per_pand", "float32"),
        ("MWh_per_pand_r", "float32"),
        ("area_ha", "float32"),
        ("area_m2", "float32"),
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
    df_filtered["scaled_elevation"] = (
        (df_filtered["kWh_per_m2"] - 10)
        / max((MAX_HEIGHT - 10), 1)
        * MAX_HEIGHT
    )
    df_filtered["scaled_elevation"] = df_filtered["scaled_elevation"].clip(lower=0, upper=threshold)

    # merge extra tooltip info
    df_filtered = df_filtered.merge(df_extra_info, on="h3_index", how="left")
    df_filtered = df_filtered[
        [
            "h3_index",
            "kWh_per_m2",
            "color",
            "woonplaats",
            "aantal_huizen",
            "aantal_VBOs",
            "scaled_elevation",
            "totale_oppervlakte",
            "gemiddeld_jaarverbruik_mWh",
            "gemiddeld_jaarverbruik_mWh_r",
            "bouwjaar",
            "MWh_per_ha",
            "MWh_per_ha_r",
            "MWh_per_pand",
            "MWh_per_pand_r",
            "area_ha",
            "area_ha_r",
            "area_m2",
        ]
    ]

    # --------- Warmtevoorziening (alleen als toggle aan én woonplaats geselecteerd) ---------
    woonplaatsen_selected = [wp for wp in ui.get("woonplaats_selectie", []) if wp]
    show_sites_layer = ui.get("show_sites_layer")
    current_sites_mode = ui.get("sites_mode", st.session_state.get("sites_mode", "auto"))
    manual_mode = current_sites_mode == "manual"

    if show_sites_layer and manual_mode:
        for payload in [
            st.session_state.get("main_map_deck_chart_selected_data"),
            st.session_state.get("main_map_deck_chart"),
        ]:
            selected_from_state = _extract_selected_hex_from_payload(payload)
            if selected_from_state:
                st.session_state["manual_site_h3"] = selected_from_state
                break

    allow_sites_auto = show_sites_layer and zoom_level >= 11 and woonplaatsen_selected
    allow_sites_manual = show_sites_layer and manual_mode
    allow_sites = allow_sites_auto or allow_sites_manual
    sites_records = []
    prev_sites_mode = st.session_state.get("_prev_sites_mode")
    if prev_sites_mode != current_sites_mode:
        st.session_state["_prev_sites_mode"] = current_sites_mode
        st.session_state.sites = []
        st.session_state.sites_costed = []
        st.session_state.sites_ready = False
        if current_sites_mode != "manual":
            st.session_state.pop("manual_site_h3", None)

    if allow_sites:
        if ui.get("reset_manual_site"):
            st.session_state.pop("manual_site_h3", None)
            st.session_state.sites = []
            st.session_state.sites_costed = []
            st.session_state.sites_ready = False

        sites_mode = current_sites_mode or "auto"
        k_val = int(st.session_state.kring_radius)

        if sites_mode == "auto":
            compute_requested = ui.get("compute_sites", False)
            if compute_requested:
                shortlist_top_frac = 0.85
                threshold_kwh_m2 = float(ui["threshold"])

                centers_keep = shortlist_centers(df_filtered, threshold_kwh_m2=threshold_kwh_m2, top_frac=shortlist_top_frac)
                df_for_clusters = df_filtered.merge(centers_keep, on="h3_index", how="inner") if not centers_keep.empty else df_filtered

                cluster_params = {"k": k_val, "threshold": threshold_kwh_m2, "shortlist_frac": shortlist_top_frac}
                cache_key = filters_fingerprint(cluster_params, df_for_clusters["h3_index"].astype(str).unique())

                cluster_input = df_for_clusters.loc[:, ["h3_index", "gemiddeld_jaarverbruik_mWh", "aantal_huizen"]]
                clusters = compute_clusters_cached(cache_key, cluster_input, k_val)
                _log_ram("after_clusters")

                clusters = clusters.merge(
                    df_filtered[["h3_index", "woonplaats", "kWh_per_m2", "aantal_VBOs", "gemiddeld_jaarverbruik_mWh"]],
                    on="h3_index",
                    how="left",
                )

                sites_df = select_sites_from_clusters(
                    clusters,
                    min_sep_cells=st.session_state.min_sep,
                    topk=st.session_state.n_sites,
                    cap_mwh=float(st.session_state.cap_mwh),
                    cap_buildings=int(st.session_state.cap_buildings),
                    ttl=1800,
                )

                records = _build_site_records(sites_df, df_filtered, k_val)
                st.session_state.sites = records
                st.session_state.sites_costed = records
                st.session_state.sites_ready = bool(records)
                del cluster_input, clusters, sites_df
            elif not st.session_state.get("sites_ready"):
                st.session_state.sites = []
                st.session_state.sites_costed = []
        else:
            manual_hex = st.session_state.get("manual_site_h3")
            if manual_hex:
                cluster_input_manual = df_filtered.loc[:, ["h3_index", "gemiddeld_jaarverbruik_mWh", "aantal_huizen"]]
                manual_cache_key = filters_fingerprint(
                    {"mode": "manual", "k": k_val},
                    cluster_input_manual["h3_index"].astype(str).unique(),
                )
                clusters_all = compute_clusters_cached(manual_cache_key, cluster_input_manual, k_val)
                manual_cluster = clusters_all[clusters_all["h3_index"] == manual_hex].copy()
                if manual_cluster.empty:
                    manual_cluster = pd.DataFrame(
                        [{
                            "h3_index": manual_hex,
                            "cluster_MWh": 0,
                            "cluster_buildings": 0,
                        }]
                    )

                cap_mwh_val = float(st.session_state.cap_mwh)
                cap_buildings_val = int(st.session_state.cap_buildings)

                manual_cluster["cluster_MWh"] = pd.to_numeric(manual_cluster["cluster_MWh"], errors="coerce").fillna(0).astype("int32")
                manual_cluster["cluster_buildings"] = pd.to_numeric(manual_cluster["cluster_buildings"], errors="coerce").fillna(0).astype("int32")

                manual_sites_df = select_sites_from_clusters(
                    manual_cluster,
                    min_sep_cells=0,
                    topk=1,
                    cap_mwh=cap_mwh_val,
                    cap_buildings=cap_buildings_val,
                    ttl=1800,
                )

                records = _build_site_records(manual_sites_df, df_filtered, k_val)
                st.session_state.sites = records
                st.session_state.sites_costed = records
                st.session_state.sites_ready = bool(records)
            else:
                st.session_state.sites = []
                st.session_state.sites_costed = []
                st.session_state.sites_ready = False

        if st.session_state.get("sites_ready"):
            sites_records = st.session_state.sites
    else:
        st.session_state.sites = []
        st.session_state.sites_costed = []
        st.session_state.sites_ready = False
        st.session_state.pop("manual_site_h3", None)

    if not st.session_state.get("sites_ready"):
        sites_records = []

    # ========== Kaartlagen ==========
    geojson_dict = {
        "energiearmoede": gjson_energiearmoede,
        "koopwoningen": gjson_koopwoningen,
        "corporatie": gjson_corporatie,
        "water_potentie": gjson_water_potentie,
        "buurt_potentie": gjson_buurt_potentie,
        "warmtenet": gjson_warmtenet,
    }

    extra_layers = []
    if any([
        st.session_state.get(LAYER_CFG["energiearmoede"]["toggle_key"]),
        st.session_state.get(LAYER_CFG["koopwoningen"]["toggle_key"]),
        st.session_state.get(LAYER_CFG["wooncorporatie"]["toggle_key"]),
        st.session_state.get(LAYER_CFG["water_potentie"]["toggle_key"]),
        st.session_state.get(LAYER_CFG["buurt_potentie"]["toggle_key"]),
    ]):
        extra_layers = create_extra_layers(
            geojson_dict,
            ui["woonplaats_selectie"],
            ui["zoom_level"],
            ui["extra_opacity"],
            potential_meta,
        )

    warmtenet_layers: list[pdk.Layer] = []
    if ui.get("show_warmtenet_model") and gjson_warmtenet:
        warmtenet_layers = create_warmtenet_layers(
            gjson_warmtenet,
            ui.get("warmtenet_wp_selectie", ui.get("woonplaats_selectie", [])),
            (warmtenet_meta or {}).get("color_map", {}),
            ui.get("warmtenet_selected_keys", []),
            (warmtenet_meta or {}).get("type_by_key", {}),
            ui.get("warmtenet_type_selectie", []),
            opacity=float(ui.get("warmtenet_opacity", (warmtenet_meta or {}).get("default_opacity", 0.85))),
        )

    # H3 hoofdlaag(en) per zoom
    base_hex_cols = [
        "h3_index",
        "color",
        "scaled_elevation",
        "woonplaats",
        "aantal_huizen",
        "aantal_VBOs",
        "MWh_per_ha_r",
        "gemiddeld_jaarverbruik_mWh_r",
        "MWh_per_pand_r",
        "area_ha_r",
        "area_m2",
        "kWh_per_m2",
        "totale_oppervlakte",
        "bouwjaar",
    ]
    df_hex_view = df_filtered.loc[:, base_hex_cols].copy()
    df_hex_view["geo_extra_rows"] = ""
    df_hex_view["gemeente_row_display"] = "block"
    df_hex_view["buurt_row_display"] = "block"

    def _fmt0_val(series):
        return series.astype("int64").map(lambda v: format_dutch_number(int(v), 0))

    def _fmt2_val(series):
        return series.astype("float32").map(lambda v: format_dutch_number(float(v), 2))

    def _fmt4_val(series):
        return series.astype("float32").map(lambda v: format_dutch_number(float(v), 4))

    df_hex_view["aantal_huizen_fmt"]                = _fmt0_val(df_hex_view["aantal_huizen"])
    df_hex_view["aantal_VBOs_fmt"]                  = _fmt0_val(df_hex_view["aantal_VBOs"])
    df_hex_view["MWh_per_ha_r_fmt"]                 = _fmt2_val(df_hex_view["MWh_per_ha_r"])
    df_hex_view["gemiddeld_jaarverbruik_mWh_r_fmt"] = _fmt0_val(df_hex_view["gemiddeld_jaarverbruik_mWh_r"])
    df_hex_view["MWh_per_pand_fmt"]                 = _fmt2_val(df_hex_view["MWh_per_pand_r"])
    df_hex_view["area_ha_r_fmt"]                    = _fmt4_val(df_hex_view["area_ha_r"])
    df_hex_view["area_m2_fmt"]                      = _fmt0_val(df_hex_view["area_m2"].round().astype("int64"))
    df_hex_view["kWh_per_m2_fmt"]                   = _fmt0_val(df_hex_view["kWh_per_m2"])
    df_hex_view["totale_oppervlakte_fmt"]           = _fmt0_val(df_hex_view["totale_oppervlakte"])
    df_hex_view["bouwjaar_fmt"]                     = df_hex_view["bouwjaar"].astype("int64").map(lambda v: str(int(v)))

    df_hex_view["hex_section_display"]  = "block"
    df_hex_view["site_section_display"] = "none"
    df_hex_view["geo_section_display"]  = "none"

    cols_for_hex = [
        "h3_index",
        "color",
        "scaled_elevation",
        "woonplaats",
        "aantal_huizen",
        "aantal_VBOs",
        "MWh_per_ha_r",
        "MWh_per_pand_r",
        "gemiddeld_jaarverbruik_mWh_r",
        "area_ha_r",
        "kWh_per_m2",
        "totale_oppervlakte",
        "bouwjaar",
        "aantal_huizen_fmt",
        "aantal_VBOs_fmt",
        "MWh_per_ha_r_fmt",
        "MWh_per_pand_fmt",
        "gemiddeld_jaarverbruik_mWh_r_fmt",
        "area_ha_r_fmt",
        "area_m2_fmt",
        "kWh_per_m2_fmt",
        "totale_oppervlakte_fmt",
        "bouwjaar_fmt",
        "hex_section_display",
        "site_section_display",
        "geo_section_display",
        "geo_extra_rows",
        "gemeente_row_display",
        "buurt_row_display",
    ]
    df_hex_view = df_hex_view.loc[:, cols_for_hex]
    indic_mask = df_filtered["kWh_per_m2"] > threshold
    df_indicative = df_hex_view.loc[indic_mask, :]
    _log_ram("before_pydeck_layers")
    warmte_opacity = float(ui.get("warmte_hex_opacity", st.session_state.get("warmte_hex_opacity", 0.6)))
    layers = create_layers_by_zoom(
        df_hex_view,
        ui["show_main_layer"],
        ui["extruded"],
        ui["zoom_level"],
        warmte_opacity,
    )

    site_layers = []
    if allow_sites and sites_records:
        sites_costed_records = st.session_state.sites_costed if st.session_state.get("sites_ready") else None
        site_layers = create_site_layers(
            sites_records,
            sites_costed_records,
            site_hex_opacity=float(ui.get("sites_hex_opacity", st.session_state.get("sites_hex_opacity", 0.85))),
        )

    # Indicatieve aandachtslaag
    if ui["show_indicative_layer"] and not df_indicative.empty:
        layers.append(create_indicative_area_layer(df_indicative, ui["extruded"], ui["zoom_level"], warmte_opacity))

    # Basemap
    hide_bg = bool(ui.get("hide_basemap"))
    basemap_style = ui.get("basemap_style", ui.get("map_style"))
    base_layers = build_base_layers(basemap_style, hide_bg)

    # Volgorde: basemap -> woonlagen -> H3/indicatief/sites
    all_layers = base_layers + extra_layers + layers + warmtenet_layers + site_layers

    # ========== ViewState ==========
    def _view_for_selection(df_full, woonplaatsen_geselecteerd):
        """Bepaal kaartcentrum en zoom op basis van de huidige selectie."""
        manual_hex_current = st.session_state.get("manual_site_h3")
        if current_sites_mode == "manual" and manual_hex_current:
            lat_manual, lon_manual = h3.cell_to_latlng(manual_hex_current)
            return lat_manual, lon_manual, 13.0
        friesland_center = (53.125, 5.75)
        friesland_zoom = 8
        min_zoom, max_zoom = 8, 13.0
        if not woonplaatsen_geselecteerd:
            return friesland_center[0], friesland_center[1], friesland_zoom
        df_sel = df_full[df_full["woonplaats"].isin(woonplaatsen_geselecteerd)]
        if df_sel.empty:
            return friesland_center[0], friesland_center[1], friesland_zoom
        lat_center = float(df_sel["latitude"].mean())
        lon_center = float(df_sel["longitude"].mean())
        if len(woonplaatsen_geselecteerd) == 1:
            return lat_center, lon_center, 13.0
        lat_min = float(df_sel["latitude"].min())
        lat_max = float(df_sel["latitude"].max())
        lon_min = float(df_sel["longitude"].min())
        lon_max = float(df_sel["longitude"].max())
        lat_span = max(0.0001, lat_max - lat_min)
        lon_span = max(0.0001, lon_max - lon_min)
        span = max(lat_span, lon_span)
        if span > 2.0:
            zoom = 8.0
        elif span > 1.0:
            zoom = 8.0
        elif span > 0.5:
            zoom = 9.0
        elif span > 0.25:
            zoom = 9.5
        elif span > 0.12:
            zoom = 10.0
        elif span > 0.06:
            zoom = 11.0
        elif span > 0.03:
            zoom = 12.0
        else:
            zoom = 13.0
        zoom = max(min_zoom, min(max_zoom, zoom))
        return lat_center, lon_center, zoom

    lat, lon, zoom = _view_for_selection(df_view_source, ui["woonplaats_selectie"])
    st.session_state.view_state = pdk.ViewState(
        longitude=lon, latitude=lat, zoom=zoom,
        min_zoom=7.5, max_zoom=16, pitch=0, bearing=0,
    )

    # ========== KPI ==========
    with kpi_container:
        render_kpis(df_filtered, st.session_state.participatie)

    # ========== Kaart render + cleanup ==========
    deck_kwargs = {"map_style": ui.get("map_style")}

    with map_container:
        deck = pdk.Deck(
            layers=all_layers,
            initial_view_state=st.session_state.view_state,
            tooltip=build_deck_tooltip(),
            **deck_kwargs,
        )

        manual_selection_active = show_sites_layer and manual_mode
        chart_key = "main_map_deck_chart"
        chart_kwargs = {
            "use_container_width": True,
            "key": chart_key,
        }
        if manual_selection_active:
            chart_kwargs.update(
                selection_mode="single-object",
                on_select="rerun",
            )
        chart_state = st.pydeck_chart(deck, **chart_kwargs)

        if manual_selection_active:
            selected_hex = None
            payload_candidates = [
                st.session_state.get(f"{chart_key}_selected_data"),
                st.session_state.get(chart_key),
                getattr(chart_state, "selection", None),
                chart_state,
            ]
            for payload in payload_candidates:
                selected_hex = _extract_selected_hex_from_payload(payload)
                if selected_hex:
                    break
            if selected_hex and st.session_state.get("manual_site_h3") != selected_hex:
                st.session_state["manual_site_h3"] = selected_hex

    # Opruimen om RAM-pieken terug te geven
    del deck, all_layers, layers, base_layers, extra_layers, warmtenet_layers, df_hex_view
    sites_records = None
    sites_costed_records = None
    gc.collect()

    # ========== Tabellen ==========
    with tables_container:
        render_tabs(df_filtered, threshold, ui["show_sites_layer"], st.session_state.get("sites_costed"))
    st.session_state["_map_changed"] = False

else:
    with map_container:
        # - Eerste keer openen -> initiële instructie
        # - Daarna, als filters gewijzigd zijn -> update-instructie
        # - Anders (nog niets gedaan) -> neutrale instructie
        if st.session_state.get("_map_changed"):
            st.info("De filters zijn gewijzigd. Klik op 'Maak kaart' om de kaart bij te werken.")
        elif not st.session_state.get("first_hint_shown", False):
            st.info("Selecteer de gewenste filters. Klik vervolgens op 'Maak kaart' om de kaart weer te geven.")
            st.session_state["first_hint_shown"] = True
        else:
            st.info("Klik op 'Maak kaart' om de kaart weer te geven.")
