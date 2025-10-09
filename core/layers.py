# core/layers.py
from __future__ import annotations

from typing import Iterable, List, Dict, Any, Union

import pydeck as pdk
import streamlit as st

from .config import LAYER_CFG, BASEMAP_CFG, WATERDEEL_PATH
from .utils import (
    get_layer_colors,
    _spoor_rgb_from_cfg,
    get_dynamic_line_width,
    colorize_geojson_cached,
    format_dutch_number,
)

JSONLike = Union[Dict[str, Any], List[Dict[str, Any]]]
Records = List[Dict[str, Any]]

# ------------------------------------------------------------
# Helpers: data normaliseren naar records (list[dict])
# ------------------------------------------------------------
def _to_records(data: Union[Records, "pd.DataFrame"]) -> Records:
    """Accepteer DataFrame of list[dict] en geef list[dict] terug."""
    if data is None:
        return []
    try:
        import pandas as pd  # lazy
        if isinstance(data, pd.DataFrame):
            return data.to_dict("records")
    except Exception:
        pass
    if isinstance(data, list):
        return [dict(x) for x in data]
    return []

def _fmt0(x):
    try:
        return format_dutch_number(int(x), 0)
    except Exception:
        return format_dutch_number(x, 0)

# ------------------------------------------------------------
# GeoJSON filteren op selectie (zoom 11–12)
# ------------------------------------------------------------
def filter_geojson_by_selection(gjson: dict, woonplaatsen: list[str] | None, zoom_level: int):
    if not gjson:
        return gjson
    if zoom_level < 11:
        return gjson
    if not woonplaatsen:
        return gjson
    wp = {str(w).strip().lower() for w in woonplaatsen}
    feats = []
    for f in gjson.get("features", []):
        pr = (f.get("properties") or {})
        gm = str(pr.get("gemeentenaam", "")).strip().lower()
        bn = str(pr.get("buurtnaam", "")).strip().lower()
        if gm in wp or bn in wp:
            feats.append(f)
    return {"type": "FeatureCollection", "features": feats}

# ------------------------------------------------------------
# Basemap
# ------------------------------------------------------------
def build_base_layers(style_key: str, hide_basemap: bool):
    """
    Basemap via TileLayer(s); 'Geen achtergrondkaart'.
    - hide_flag=True → geen achtergrondlagen
    """
    if hide_basemap:
        return []

    conf = BASEMAP_CFG.get(style_key, {})
    layers_local = []

    if conf.get("tile"):
        layers_local.append(pdk.Layer(
            "TileLayer",
            data=conf["tile"],
            minZoom=0,
            maxZoom=19,
            tileSize=256
        ))

    if conf.get("labels"):
        layers_local.append(pdk.Layer(
            "TileLayer",
            data=conf["labels"],
            minZoom=0,
            maxZoom=19,
            tileSize=256
        ))

    return layers_local

# ------------------------------------------------------------
# H3 hoofdlaag + indicatieve laag
# ------------------------------------------------------------
def create_main_layer(data_hex_df, show: bool, extruded: bool, zoom_level: int, elevation_scale: float):
    # verwacht een DataFrame (geen list[dict])
    return pdk.Layer(
        "H3HexagonLayer",
        data_hex_df,
        pickable=True, filled=True, extruded=extruded, coverage=1,
        auto_highlight=False,
        get_hexagon="h3_index",
        get_fill_color="color",
        get_elevation="scaled_elevation",
        elevation_scale=elevation_scale if extruded else 0,
        elevation_range=[0, 800.0],
        get_line_width=get_dynamic_line_width(zoom_level),
        visible=show,
    )

def create_indicative_area_layer(data, extruded: bool, zoom_level: int):
    """
    H3 laag voor indicatieve aandachtsgebieden. Verwacht een reeds gefilterde bron
    (DataFrame of list[dict]) met minimaal de kolom h3_index.
    """
    data_src = data
    return pdk.Layer(
        "H3HexagonLayer",
        data_src,
        pickable=True, filled=True, extruded=extruded,
        get_hexagon="h3_index",
        get_fill_color=[58, 27, 47, 200],
        get_line_color=[0, 0, 0, 0],
        get_line_width=get_dynamic_line_width(zoom_level),
        visible=True
    )

def create_layers_by_zoom(data_hex_df, show_main: bool, extruded: bool, zoom_level: int):
    # Verwacht hier een DataFrame (geen list[dict])
    layers = []
    if zoom_level <= 3:
        layers.append(create_main_layer(data_hex_df, show_main, extruded, zoom_level, 0.01))
    elif 4 <= zoom_level <= 7:
        layers.append(create_main_layer(data_hex_df, show_main, extruded, zoom_level, 0.05))
    elif 8 <= zoom_level <= 11:
        layers.append(create_main_layer(data_hex_df, show_main, extruded, zoom_level, 0.08))
    else:  # zoom_level >= 12
        layers.append(create_main_layer(data_hex_df, show_main, extruded, zoom_level, 0.10))
    return layers

# ------------------------------------------------------------
# Sites (H3 contour + scatter markers)
# ------------------------------------------------------------
def create_site_layers(
    sites_data: Union[Records, "pd.DataFrame"],
    sites_costed: Union[Records, "pd.DataFrame", None] = None
):
    """
    Maakt:
      - H3HexagonLayer contour voor gekozen sites
      - ScatterplotLayer markers met alle tooltip-velden (incl. *_fmt)
    """
    site_layers = []
    records = _to_records(sites_data)
    if not records:
        return site_layers

    # H3 contour (alleen id + display toggles)
    contour_records = [{
        "h3_index": r.get("h3_index"),
        "hex_section_display": r.get("hex_section_display", "none"),
        "site_section_display": r.get("site_section_display", "block"),
        "geo_section_display": r.get("geo_section_display", "none"),
    } for r in records if r.get("h3_index")]
    if contour_records:
        site_layers.append(pdk.Layer(
            "H3HexagonLayer",
            contour_records,
            pickable=False, filled=False, stroked=True,
            get_hexagon="h3_index",
            get_line_color=[26, 152, 80, 255],
            lineWidthMinPixels=2,
            visible=True
        ))

    # Scatter markers: gebruik 'costed' records indien aanwezig
    use_records = _to_records(sites_costed) if sites_costed is not None else records

    scatter_records = []
    for r in use_records:
        lon, lat = r.get("lon"), r.get("lat")
        if lon is None or lat is None:
            continue

        # ruwe waarden
        cluster_buildings = r.get("cluster_buildings")
        cap_buildings = r.get("cap_buildings")
        connected_buildings = r.get("connected_buildings")
        cluster_MWh = r.get("cluster_MWh")
        cap_MWh = r.get("cap_MWh")
        connected_MWh = r.get("connected_MWh")
        utilization_pct = r.get("utilization_pct")

        scatter_records.append({
            "lon": lon,
            "lat": lat,
            "woonplaats": r.get("woonplaats", ""),

            # raw
            "cluster_buildings": cluster_buildings,
            "cap_buildings": cap_buildings,
            "connected_buildings": connected_buildings,
            "cluster_MWh": cluster_MWh,
            "cap_MWh": cap_MWh,
            "connected_MWh": connected_MWh,
            "utilization_pct": utilization_pct,

            # formatted for tooltip (maak ze indien niet aanwezig)
            "cluster_buildings_fmt": r.get("cluster_buildings_fmt") or _fmt0(cluster_buildings),
            "cap_buildings_fmt": r.get("cap_buildings_fmt") or _fmt0(cap_buildings),
            "connected_buildings_fmt": r.get("connected_buildings_fmt") or _fmt0(connected_buildings),
            "cluster_MWh_fmt": r.get("cluster_MWh_fmt") or _fmt0(cluster_MWh),
            "cap_MWh_fmt": r.get("cap_MWh_fmt") or _fmt0(cap_MWh),
            "connected_MWh_fmt": r.get("connected_MWh_fmt") or _fmt0(connected_MWh),
            "utilization_pct_fmt": r.get("utilization_pct_fmt") or (str(int(utilization_pct)) if utilization_pct is not None else ""),

            # display-velden voor tooltip-secties
            "hex_section_display": r.get("hex_section_display", "none"),
            "site_section_display": r.get("site_section_display", "block"),
            "geo_section_display": r.get("geo_section_display", "none"),
        })

    if scatter_records:
        site_layers.append(pdk.Layer(
            "ScatterplotLayer",
            scatter_records,
            pickable=True,
            get_position=["lon", "lat"],
            get_radius=25,
            get_fill_color=[26, 152, 80, 255],
            radius_min_pixels=6,
            radius_max_pixels=10
        ))

    return site_layers

# ------------------------------------------------------------
# Woonlagen (energiearmoede/koop/corporatie)
# ------------------------------------------------------------
def _geojson_layer(data, name, fill_color, line_color, opacity=0.5):
    if data is None:
        return None
    return pdk.Layer(
        "GeoJsonLayer",
        data=data,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        get_fill_color=fill_color,
        get_line_color=line_color,
        get_line_width=1,
        lineWidthMinPixels=1,
        opacity=float(opacity),
    )

def create_extra_layers(geojson_dict: dict, woonplaats_selectie: list[str], zoom_level: int, extra_opacity: float = 0.4):
    """
    Woonlagen:
    - filteren op zoom+woonplaats
    - kleurtoekenning (cached)
    - labels/props voor tooltip meegeven
    """
    layers = []
    cfg = LAYER_CFG

    # Energiearmoede
    if st.session_state.get(cfg["energiearmoede"]["toggle_key"]):
        c = cfg["energiearmoede"]
        colors = get_layer_colors(c)
        gjson_src = filter_geojson_by_selection(geojson_dict.get("energiearmoede"), woonplaats_selectie, zoom_level)
        gjson_colored = colorize_geojson_cached(
            gjson_src, c["prop_name"], c["out_prop"], c["breaks"], colors,
            layer_label=c["legend_title"],
        )
        lyr = _geojson_layer(
            gjson_colored, "energiearmoede",
            fill_color=f"properties.{c['out_prop']}",
            line_color=c["line_color"],
            opacity=st.session_state.get("extra_opacity", extra_opacity),
        )
        if lyr: layers.append(lyr)

    # Koopwoningen
    if st.session_state.get(cfg["koopwoningen"]["toggle_key"]):
        c = cfg["koopwoningen"]
        colors = get_layer_colors(c)
        gjson_src = filter_geojson_by_selection(geojson_dict.get("koopwoningen"), woonplaats_selectie, zoom_level)
        gjson_colored = colorize_geojson_cached(
            gjson_src, c["prop_name"], c["out_prop"], c["breaks"], colors,
            layer_label=c["legend_title"],
        )
        lyr = _geojson_layer(
            gjson_colored, "koopwoningen",
            fill_color=f"properties.{c['out_prop']}",
            line_color=c["line_color"],
            opacity=st.session_state.get("extra_opacity", extra_opacity),
        )
        if lyr: layers.append(lyr)

    # Wooncorporatie
    if st.session_state.get(cfg["wooncorporatie"]["toggle_key"]):
        c = cfg["wooncorporatie"]
        colors = get_layer_colors(c)
        gjson_src = filter_geojson_by_selection(geojson_dict.get("corporatie"), woonplaats_selectie, zoom_level)
        gjson_colored = colorize_geojson_cached(
            gjson_src, c["prop_name"], c["out_prop"], c["breaks"], colors,
            layer_label=c["legend_title"],
        )
        lyr = _geojson_layer(
            gjson_colored, "wooncorporatie",
            fill_color=f"properties.{c['out_prop']}",
            line_color=c["line_color"],
            opacity=st.session_state.get("extra_opacity", extra_opacity),
        )
        if lyr: layers.append(lyr)

    return layers

# ------------------------------------------------------------
# Bodemlagen (spoor/water)
# ------------------------------------------------------------
def create_bodem_layers(geojson_dict: dict):
    """
    Bodemlagen onder de woonlagen:
    - Spoor: GeoJson lijnen
    - Water: MVT (uit URL) of niets als WATERDEEL_PATH None is.
    """
    layers = []
    cfg = LAYER_CFG

    # Spoor (GeoJSON -> lijn)
    wc_spoor = cfg["spoordeel"]
    if st.session_state.get(wc_spoor["toggle_key"]) and geojson_dict.get("spoordeel"):
        r, g, b = _spoor_rgb_from_cfg(cfg)
        layers.append(pdk.Layer(
            "GeoJsonLayer",
            data=geojson_dict["spoordeel"],
            pickable=False,
            stroked=True,
            filled=False,
            extruded=False,
            get_line_color=[r, g, b, 255],
            get_line_width=2,
            lineWidthMinPixels=2.5,
            opacity=float(st.session_state.get("spoor_opacity", 0.5)),
        ))

    # Water (MVT uit MBTiles via URL, alleen als path/URL gezet is)
    wc_water = cfg["waterdeel"]
    if st.session_state.get(wc_water["toggle_key"]) and WATERDEEL_PATH:
        r_w, g_w, b_w = int(wc_water["palette"][0]), int(wc_water["palette"][1]), int(wc_water["palette"][2])
        layers.append(pdk.Layer(
            "MVTLayer",
            data=WATERDEEL_PATH,
            pickable=False,
            filled=True,
            stroked=True,
            get_fill_color=[r_w, g_w, b_w, int(wc_water.get("alpha", 180))],
            get_line_color=[r_w, g_w, b_w, 255],
            lineWidthMinPixels=1,
            opacity=float(st.session_state.get("water_opacity", 0.6)),
            # minZoom=6, maxZoom=18,  # optioneel
        ))

    return layers
