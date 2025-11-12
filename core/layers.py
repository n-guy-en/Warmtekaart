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
    """Formatteer een getal als Nederlandse integer-string."""
    try:
        return format_dutch_number(int(x), 0)
    except Exception:
        return format_dutch_number(x, 0)

# ------------------------------------------------------------
# GeoJSON filteren op selectie (zoom 11–12)
# ------------------------------------------------------------
def filter_geojson_by_selection(gjson: dict, woonplaatsen: list[str] | None, zoom_level: int):
    """Beperk GeoJSON tot geselecteerde woonplaatsen bij hogere zoomniveaus."""
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
    - hide_flag=True -> geen achtergrondlagen
    """
    if hide_basemap:
        return []

    conf = BASEMAP_CFG.get(style_key, {})
    layers_local = []

    if conf.get("tile"):
        tile_kwargs = {
            "data": conf["tile"],
            "min_zoom": 0,
            "max_zoom": 19,
            "tile_size": 256,
        }
        attribution = conf.get("attribution")
        if attribution:
            tile_kwargs["attribution"] = f"'{attribution}'"
        layers_local.append(pdk.Layer("TileLayer", **tile_kwargs))

    if conf.get("labels"):
        label_kwargs = {
            "data": conf["labels"],
            "min_zoom": 0,
            "max_zoom": 19,
            "tile_size": 256,
        }
        label_attrib = conf.get("labels_attribution")
        if label_attrib:
            label_kwargs["attribution"] = f"'{label_attrib}'"
        layers_local.append(pdk.Layer("TileLayer", **label_kwargs))

    return layers_local

# ------------------------------------------------------------
# H3 hoofdlaag + indicatieve laag
# ------------------------------------------------------------
def create_main_layer(
    data_hex_df,
    show: bool,
    extruded: bool,
    zoom_level: int,
    elevation_scale: float,
    layer_opacity: float = 1.0,
):
    """Bouw de primaire H3-laag met kleur en hoogte op basis van energievraag."""
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
        opacity=float(layer_opacity),
    )

def create_indicative_area_layer(data, extruded: bool, zoom_level: int, layer_opacity: float = 1.0):
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
        visible=True,
        opacity=float(layer_opacity),
    )

def create_layers_by_zoom(
    data_hex_df,
    show_main: bool,
    extruded: bool,
    zoom_level: int,
    layer_opacity: float = 1.0,
):
    """Stel de hoofdlaag samen met een passende elevatieschaal per zoomniveau."""
    # Verwacht hier een DataFrame (geen list[dict])
    layers = []
    if zoom_level <= 3:
        layers.append(create_main_layer(data_hex_df, show_main, extruded, zoom_level, 0.01, layer_opacity))
    elif 4 <= zoom_level <= 7:
        layers.append(create_main_layer(data_hex_df, show_main, extruded, zoom_level, 0.05, layer_opacity))
    elif 8 <= zoom_level <= 11:
        layers.append(create_main_layer(data_hex_df, show_main, extruded, zoom_level, 0.08, layer_opacity))
    elif zoom_level == 12:
        layers.append(create_main_layer(data_hex_df, show_main, extruded, zoom_level, 0.10, layer_opacity))
    else:  # zoom_level >= 13
        layers.append(create_main_layer(data_hex_df, show_main, extruded, zoom_level, 0.12, layer_opacity))
    return layers

# ------------------------------------------------------------
# Sites (H3 contour + scatter markers)
# ------------------------------------------------------------
def create_site_layers(
    sites_data: Union[Records, "pd.DataFrame"],
    sites_costed: Union[Records, "pd.DataFrame", None] = None,
    site_hex_opacity: float = 1.0,
):
    """
    Maakt:
      - PolygonLayer (contour + semitransparant vlak) per warmtevoorziening
      - H3HexagonLayer gevuld met dezelfde groene kleur voor de individuele hexagonen
      - ScatterplotLayer markers met alle tooltip-velden (incl. *_fmt)
    """
    site_layers = []
    records = _to_records(sites_data)
    if not records:
        return site_layers

    base_fill = [26, 152, 80, 255]
    base_line = [0, 0, 0, 255]

    polygon_records = []
    hexagon_records = []

    for r in records:
        site_rank = int(r.get("site_rank") or 0) or 0
        coverage_summary = r.get("coverage_summary") or {}
        polygons = r.get("coverage_polygons") or []
        hexes = r.get("coverage_hexes") or []

        for poly in polygons:
            polygon_records.append({
                "polygon": poly,
                "site_rank": site_rank,
                "fill_color": base_fill,
                "line_color": base_line,
                "site_rank_label": coverage_summary.get("site_rank_label", site_rank),
                "woonplaats": r.get("woonplaats", ""),
                "cluster_buildings": r.get("cluster_buildings"),
                "cap_buildings": r.get("cap_buildings"),
                "connected_buildings": r.get("connected_buildings"),
                "cluster_MWh": r.get("cluster_MWh"),
                "cap_MWh": r.get("cap_MWh"),
                "connected_MWh": r.get("connected_MWh"),
                "utilization_pct": r.get("utilization_pct"),
                "cluster_buildings_fmt": r.get("cluster_buildings_fmt"),
                "cap_buildings_fmt": r.get("cap_buildings_fmt"),
                "connected_buildings_fmt": r.get("connected_buildings_fmt"),
                "cluster_MWh_fmt": r.get("cluster_MWh_fmt"),
                "cap_MWh_fmt": r.get("cap_MWh_fmt"),
                "connected_MWh_fmt": r.get("connected_MWh_fmt"),
                "utilization_pct_fmt": r.get("utilization_pct_fmt"),
                # aggregaties voor tooltip
                "aantal_huizen": coverage_summary.get("aantal_huizen"),
                "aantal_VBOs": coverage_summary.get("aantal_VBOs"),
                "gemiddeld_jaarverbruik_mWh_r": coverage_summary.get("gemiddeld_jaarverbruik_mWh_r"),
                "area_ha_r": coverage_summary.get("area_ha_r"),
                "area_m2": coverage_summary.get("area_m2"),
                "totale_oppervlakte": coverage_summary.get("totale_oppervlakte"),
                "area_ha_total": coverage_summary.get("area_ha_total"),
                "area_ha_total_fmt": coverage_summary.get("area_ha_total_fmt"),
                "area_m2_total": coverage_summary.get("area_m2_total"),
                "area_m2_total_fmt": coverage_summary.get("area_m2_total_fmt"),
                "kWh_per_m2": coverage_summary.get("kWh_per_m2"),
                "MWh_per_ha_r": coverage_summary.get("MWh_per_ha_r"),
                "bouwjaar": coverage_summary.get("bouwjaar"),
                "aantal_huizen_fmt": coverage_summary.get("aantal_huizen_fmt"),
                "aantal_VBOs_fmt": coverage_summary.get("aantal_VBOs_fmt"),
                "gemiddeld_jaarverbruik_mWh_r_fmt": coverage_summary.get("gemiddeld_jaarverbruik_mWh_r_fmt"),
                "area_ha_r_fmt": coverage_summary.get("area_ha_r_fmt"),
                "area_m2_fmt": coverage_summary.get("area_m2_fmt"),
                "totale_oppervlakte_fmt": coverage_summary.get("totale_oppervlakte_fmt"),
                "kWh_per_m2_fmt": coverage_summary.get("kWh_per_m2_fmt"),
                "MWh_per_ha_r_fmt": coverage_summary.get("MWh_per_ha_r_fmt"),
                "bouwjaar_fmt": coverage_summary.get("bouwjaar_fmt"),
                "hex_section_display": coverage_summary.get("hex_section_display", "block"),
                "site_section_display": coverage_summary.get("site_section_display", "block"),
                "geo_section_display": coverage_summary.get("geo_section_display", "none"),
            })

        for cov in hexes:
            cov_rec = dict(cov)
            cov_rec["site_rank"] = site_rank
            cov_rec["fill_color"] = cov_rec.get("fill_color", base_fill)
            cov_rec["line_color"] = cov_rec.get("line_color", base_line)
            hexagon_records.append(cov_rec)

    if polygon_records:
        site_layers.append(pdk.Layer(
            "PolygonLayer",
            polygon_records,
            pickable=True,
            stroked=True,
            filled=False,
            extruded=False,
            wireframe=False,
            get_polygon="polygon",
            get_fill_color=[0, 0, 0, 0],
            get_line_color=[0, 0, 0, 180],
            lineWidthMinPixels=2.5,
            lineWidthMaxPixels=10,
            opacity=1.0,
        ))

    if hexagon_records:
        site_layers.append(pdk.Layer(
            "H3HexagonLayer",
            hexagon_records,
            pickable=True,
            filled=True,
            stroked=False,
            extruded=False,
            get_hexagon="h3_index",
            get_fill_color="fill_color",
            opacity=float(site_hex_opacity),
        ))

    # Scatter markers: gebruik 'costed' records indien aanwezig
    use_records = _to_records(sites_costed) if sites_costed is not None else records

    scatter_records = []
    for r in use_records:
        lon, lat = r.get("lon"), r.get("lat")
        if lon is None or lat is None:
            continue
        site_rank = int(r.get("site_rank") or 0) or 0
        color = base_fill

        coverage_summary = r.get("coverage_summary") or {}

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
            "site_rank": site_rank,

            # raw
            "cluster_buildings": cluster_buildings,
            "cap_buildings": cap_buildings,
            "connected_buildings": connected_buildings,
            "cluster_MWh": cluster_MWh,
            "cap_MWh": cap_MWh,
            "connected_MWh": connected_MWh,
            "utilization_pct": utilization_pct,
            "area_ha": coverage_summary.get("area_ha_r"),
            "area_ha_fmt": coverage_summary.get("area_ha_r_fmt"),
            "area_m2": coverage_summary.get("area_m2"),
            "area_m2_fmt": coverage_summary.get("area_m2_fmt"),
            "area_ha_total": coverage_summary.get("area_ha_total"),
            "area_ha_total_fmt": coverage_summary.get("area_ha_total_fmt"),
            "area_m2_total": coverage_summary.get("area_m2_total"),
            "area_m2_total_fmt": coverage_summary.get("area_m2_total_fmt"),

            # formatted for tooltip (maak ze indien niet aanwezig)
            "cluster_buildings_fmt": r.get("cluster_buildings_fmt") or _fmt0(cluster_buildings),
            "cap_buildings_fmt": r.get("cap_buildings_fmt") or _fmt0(cap_buildings),
            "connected_buildings_fmt": r.get("connected_buildings_fmt") or _fmt0(connected_buildings),
            "cluster_MWh_fmt": r.get("cluster_MWh_fmt") or _fmt0(cluster_MWh),
            "cap_MWh_fmt": r.get("cap_MWh_fmt") or _fmt0(cap_MWh),
            "connected_MWh_fmt": r.get("connected_MWh_fmt") or _fmt0(connected_MWh),
            "utilization_pct_fmt": r.get("utilization_pct_fmt") or (str(int(utilization_pct)) if utilization_pct is not None else ""),

            # display-velden voor tooltip-secties
            "hex_section_display": coverage_summary.get("hex_section_display", r.get("hex_section_display", "none")),
            "site_section_display": r.get("site_section_display", "block"),
            "geo_section_display": r.get("geo_section_display", "none"),

            # aggregated hex data
            "aantal_huizen": coverage_summary.get("aantal_huizen"),
            "aantal_VBOs": coverage_summary.get("aantal_VBOs"),
            "gemiddeld_jaarverbruik_mWh_r": coverage_summary.get("gemiddeld_jaarverbruik_mWh_r"),
            "area_ha_r": coverage_summary.get("area_ha_r"),
            "totale_oppervlakte": coverage_summary.get("totale_oppervlakte"),
            "kWh_per_m2": coverage_summary.get("kWh_per_m2"),
            "MWh_per_ha_r": coverage_summary.get("MWh_per_ha_r"),
            "bouwjaar": coverage_summary.get("bouwjaar"),
            "aantal_huizen_fmt": coverage_summary.get("aantal_huizen_fmt"),
            "aantal_VBOs_fmt": coverage_summary.get("aantal_VBOs_fmt"),
            "gemiddeld_jaarverbruik_mWh_r_fmt": coverage_summary.get("gemiddeld_jaarverbruik_mWh_r_fmt"),
            "area_ha_r_fmt": coverage_summary.get("area_ha_r_fmt"),
            "totale_oppervlakte_fmt": coverage_summary.get("totale_oppervlakte_fmt"),
            "kWh_per_m2_fmt": coverage_summary.get("kWh_per_m2_fmt"),
            "MWh_per_ha_r_fmt": coverage_summary.get("MWh_per_ha_r_fmt"),
            "bouwjaar_fmt": coverage_summary.get("bouwjaar_fmt"),
            "site_rank_label": coverage_summary.get("site_rank_label", site_rank),

            "fill_color": color,
        })

    if scatter_records:
        site_layers.append(pdk.Layer(
            "ScatterplotLayer",
            scatter_records,
            pickable=True,
            get_position=["lon", "lat"],
            get_radius=25,
            get_fill_color="fill_color",
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
