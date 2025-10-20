# core/utils.py
from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import List, Dict, Any

import pandas as pd
import streamlit as st

# ============================================================
# Kleuren & legenda
# ============================================================

def _lazy_matplotlib():
    """
    Laadt matplotlib pas wanneer we het écht nodig hebben (scheelt init-RAM/starttijd).
    """
    import importlib
    mpl = importlib.import_module("matplotlib")
    return mpl


@lru_cache(maxsize=16)
def _cached_palette(palette_name: str, n: int, alpha: int) -> List[List[int]]:
    """
    Kleine cache rond colormap -> RGBA-lijst. Voorkomt herhaalde colormap-resolves.
    """
    mpl = _lazy_matplotlib()
    cmap = mpl.colormaps.get_cmap(palette_name).resampled(max(1, n))
    colors = []
    for i in range(max(1, n)):
        t = 0.0 if n == 1 else i / (n - 1)
        r, g, b, _ = cmap(t)
        colors.append([int(r * 255), int(g * 255), int(b * 255), int(alpha)])
    return colors


def get_color_palette(palette_name="OrRd", n=4, alpha=180):
    """
    Returns RGBA list of length n from a given colormap name (e.g. 'Greens', 'Purples', 'OrRd').
    Alpha is 0–255.
    """
    return _cached_palette(str(palette_name), int(n), int(alpha))


def get_layer_colors(layer_cfg: dict):
    """
    Maakt kleurenlijst layer-config (palette/breaks/n_colors/alpha).
    """
    return get_color_palette(
        palette_name=layer_cfg.get("palette", "OrRd"),
        n=layer_cfg.get("n_colors", max(2, len(layer_cfg.get("breaks", [])) + 1)),
        alpha=layer_cfg.get("alpha", 200),
    )


def _spoor_rgb_from_cfg(cfg: dict):
    """Returns (r,g,b) voor spoor uit LAYER_CFG['spoordeel'].""" 
    wc = cfg["spoordeel"]
    base = wc.get("palette", [0, 0, 0])
    if isinstance(base, (list, tuple)) and len(base) >= 3:
        r, g, b = int(base[0]), int(base[1]), int(base[2])
    else:
        r, g, b = 160, 189, 190
    return r, g, b


def legend_labels_from_breaks(breaks):
    pct = [int(b * 100) for b in breaks]
    return [f"< {pct[0]}%", f"{pct[0]}–{pct[1]}%", f"{pct[1]}–{pct[2]}%", f"≥ {pct[2]}%"]


def render_mini_legend(title, colors, labels):
    rows = "".join(
        f'<div class="ea-row"><span class="ea-swatch" style="background:rgba({c[0]},{c[1]},{c[2]},{c[3]/255});"></span> {lab}</div>'
        for c, lab in zip(colors, labels)
    )
    html = f"""
    <style>
      .ea-legend {{ background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:10px; font-family:Arial; font-size:12px; margin-bottom:20px;}}
      .ea-row {{ display:flex; align-items:center; margin:4px 0; }}
      .ea-swatch {{ width:16px; height:12px; border-radius:3px; margin-right:8px; border:1px solid #d1d5db; }}
    </style>
    <div class="ea-legend">
      <div style="font-weight:600; margin-bottom:6px;">{title}</div>
      {rows}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# Nummerformatting & parsing
# ============================================================

def format_dutch_number(num, decimals=2):
    """
    Engelse -> Nederlandse notatie.
    """
    if isinstance(num, int):
        return f"{num:,}".replace(",", ".")
    try:
        return f"{num:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return ""


def parse_dutch_int(text: str, fallback: int = 0) -> int:
    try:
        return int(text.replace(".", "").replace(",", ""))
    except Exception:
        return fallback


# ============================================================
# Zoom/line helpers
# ============================================================

def get_dynamic_resolution(zoom_level):
    """
    Pakt het zoom niveau van de slider om dynamisch de h3_index steeds opnieuw te berekenen en te weergeven op de kaart.
    """
    return zoom_level


def get_dynamic_line_width(zoom_level):
    """
    Wordt niet gebruikt maar wel handig voor later als je lijn diktes wilt aanpassen afhankelijk van het zoom niveau.
    """
    # Bewust 0 (geen stroke) om renderbuffers minimaal te houden.
    return 0


def get_hexagon_size(zoom_level: int) -> float:
    """
    Geeft een schatting van de zichtbare schaal (in kilometer)
    bij het opgegeven zoomniveau van de kaart.

    Deze waarden zijn bedoeld om aan te geven hoe groot het kaartgebied
    ongeveer is op elk zoomniveau. Het is niet de werkelijke H3-hexagon-grootte.
    """
    hexagon_sizes = {
        1: 5000, 2: 2500, 3: 1500, 4: 700, 5: 350, 6: 175, 7: 90,
        8: 35, 9: 17, 10: 8, 11: 4, 12: 2, 13: 1, 14: 0.5, 15: 0.2
    }
    return hexagon_sizes.get(zoom_level, 10)


# ============================================================
# Binning & kleuren hoofdlaag
# ============================================================

colorbrewer_colors = [
    [69, 117, 180, 150],   # <10 Donkerblauw
    [254, 224, 144, 150],  # 10-50 Lichtoranje
    [215, 48, 39, 150]     # >=50 Rood
]

def get_color(value):
    """
    Wanneer je dit aanpast de bins. Pas dan ook de legenda aan!
    - Donkerblauw is nu kleiner dan 10
    - Lichtoranje is tussen 10 en 50
    - Rood is boven 50
    """
    try:
        v = float(value)
    except Exception:
        return colorbrewer_colors[0]
    bins = [10, 50]
    for i, threshold in enumerate(bins):
        if v < threshold:
            return colorbrewer_colors[i]
    return colorbrewer_colors[-1]


# ============================================================
# View helper voor selectie
# ============================================================

def _view_for_selection(df_full, woonplaatsen_geselecteerd):
    """
    Bepaalt latitude, longitude en zoom:
    - Geen of veel woonplaatsen => Friesland overzicht
    - 1 woonplaats => zoom in op die gemeente
    - Meerdere woonplaatsen => center + zoom op de begrenzende box van de selectie
    """
    FRIESLAND_CENTER = (53.125, 5.75)
    FRIESLAND_ZOOM   = 8
    MIN_ZOOM, MAX_ZOOM = 8, 12.0

    if not woonplaatsen_geselecteerd:
        return FRIESLAND_CENTER[0], FRIESLAND_CENTER[1], FRIESLAND_ZOOM

    df_sel = df_full[df_full["woonplaats"].isin(woonplaatsen_geselecteerd)]
    if df_sel.empty:
        return FRIESLAND_CENTER[0], FRIESLAND_CENTER[1], FRIESLAND_ZOOM

    lat_center = float(df_sel["latitude"].mean())
    lon_center = float(df_sel["longitude"].mean())

    if len(woonplaatsen_geselecteerd) == 1:
        return lat_center, lon_center, 12.0

    lat_min, lat_max = float(df_sel["latitude"].min()), float(df_sel["latitude"].max())
    lon_min, lon_max = float(df_sel["longitude"].min()), float(df_sel["longitude"].max())
    lat_span = max(0.0001, lat_max - lat_min)
    lon_span = max(0.0001, lon_max - lon_min)
    span = max(lat_span, lon_span)

    if   span > 2.0:  zoom = 8.0
    elif span > 1.0:  zoom = 8.0
    elif span > 0.5:  zoom = 9.0
    elif span > 0.25: zoom = 9.0
    elif span > 0.12: zoom = 10.0
    else:             zoom = 11.0

    zoom = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
    return lat_center, lon_center, zoom


# ============================================================
# Procenthelpers + GeoJSON kleuring
# ============================================================

def _coerce_frac(v):
    """Accepteert 0–1 float, '0.15', '15', '15%', '15,2' etc. Geeft fractie 0–1 of None."""
    if v is None:
        return None
    try:
        if isinstance(v, str):
            s = v.strip().replace("%", "").replace(",", ".")
            v = float(s)
        v = float(v)
        if v > 1.0:
            v = v / 100.0
        if v < 0 or v > 1:
            return None
        return v
    except Exception:
        return None


def pct_color_from_breaks(v, breaks, colors):
    v = _coerce_frac(v)
    if v is None:
        return [200, 200, 200, 100]
    for t, c in zip(breaks, colors[:-1]):
        if v < t:
            return c
    return colors[-1]


@st.cache_data(show_spinner=False, max_entries=24)
def colorize_geojson_cached(gjson: dict, prop_name: str, out_prop: str, breaks: list, colors: list, layer_label: str = ""):
    """
    Schrijft per feature een RGBA in properties[out_prop] -> properties[prop_name].
    Cached, zodat de kleurtoewijzing niet steeds opnieuw doorlopen wordt.

    Optimalisatie:
    - Geen deepcopy van geometrieën.
    - Behoud minimale keys voor tooltip (buurt/gemeente + label + pct).
    """
    if not gjson or not isinstance(gjson, dict) or gjson.get("type") != "FeatureCollection":
        return gjson

    feats_new = []
    feats = gjson.get("features") or []
    for feat in feats:
        if not isinstance(feat, dict):
            continue
        props = dict((feat.get("properties") or {}))  # shallow copy properties
        v = props.get(prop_name)
        props[out_prop] = pct_color_from_breaks(v, breaks, colors)
        v_frac = _coerce_frac(v)
        props["_value_pct_fmt"] = int(round(v_frac * 100, 0)) if v_frac is not None else ""
        props["_layer_label"] = layer_label if layer_label else ""
        # --- Zorg dat buurt/gemeente altijd bestaan voor tooltip
        props["buurtnaam"] = props.get("buurtnaam", "")
        props["gemeentenaam"] = props.get("gemeentenaam", "")
        # --- Display-secties
        props["geo_section_display"]  = "block"
        props["hex_section_display"]  = "none"
        props["site_section_display"] = "none"

        geom = feat.get("geometry")  # geen diepe kopie
        feats_new.append({"type": "Feature", "properties": props, "geometry": geom})

    return {"type": "FeatureCollection", "features": feats_new}


# ============================================================
# Kleine utils die elders gebruikt worden
# ============================================================

def text_input_int(label: str, key: str, default: int) -> int:
    """
    Streamlit text input met Nederlandse integer notatie (1.234.567) -> int.
    """
    if key not in st.session_state:
        st.session_state[key] = default
    display_val = format_dutch_number(int(st.session_state[key]), 0)
    s = st.text_input(label, value=display_val, key=f"{key}_str")
    val = parse_dutch_int(s, fallback=default)
    st.session_state[key] = val
    return val


def build_deck_tooltip() -> dict:
    """
    PyDeck tooltip met de drie blokken:
    - geo_section_display (gemeente/buurt)
    - hex_section_display (H3 info)
    - site_section_display (collectieve voorziening)
    Houdt placeholders in sync met kolomnamen uit app.py.
    """
    html = """
    <style>
      .tooltip-wrapper {
        display:block;
        font-size:11px;
        line-height:1.35;
        max-width:360px;
      }
      .tooltip-section {
        padding:6px 8px;
        border-radius:6px;
        border:1px solid #e5e7eb;
        background:#f9fafb;
        margin-bottom:6px;
      }
      .tooltip-row {
        white-space:nowrap;
      }
      .tooltip-section h4 {
        margin:0 0 4px 0;
        font-size:12px;
        font-weight:700;
        letter-spacing:.15px;
      }
      .tooltip-highlight {
        background:#fef3c7;
        border-color:#f59e0b;
      }
    </style>
    <div class="tooltip-wrapper">
      <div class="tooltip-section" style="display:{geo_section_display};">
        <h4>Gebied</h4>
        <div class="tooltip-row"><strong>{_layer_label}</strong></div>
        <div class="tooltip-row">Waarde: {_value_pct_fmt}%</div>
        <div class="tooltip-row">Gemeente: {gemeentenaam}</div>
        <div class="tooltip-row">Buurt: {buurtnaam}</div>
      </div>
      <div class="tooltip-section" style="display:{hex_section_display};">
        <h4>Heat Demand</h4>
        <div class="tooltip-row">Woonplaats: {woonplaats}</div>
        <div class="tooltip-row">Aantal panden: {aantal_huizen_fmt}</div>
        <div class="tooltip-row">Aantal VBO's: {aantal_VBOs_fmt}</div>
        <div class="tooltip-row">Warmtevraag-dichtheid: {MWh_per_ha_r_fmt} MWh/ha</div>
        <div class="tooltip-row">Totale Heat Demand: {gemiddeld_jaarverbruik_mWh_r_fmt} MWh</div>
        <div class="tooltip-row">Oppervlakte cel: {area_ha_r_fmt} ha</div>
        <div class="tooltip-row">Gemiddelde Energiegebruik: {kWh_per_m2_fmt} kWh/m²</div>
        <div class="tooltip-row">Totale Oppervlakte: {totale_oppervlakte_fmt} m²</div>
        <div class="tooltip-row">Gemiddelde Bouwjaar: {bouwjaar_fmt}</div>
      </div>
      <div class="tooltip-section tooltip-highlight" style="display:{site_section_display};">
        <h4>Collectieve voorziening</h4>
        <div class="tooltip-row">Voorziening #: {site_rank_label}</div>
        <div class="tooltip-row">Gebouwen in radar: {cluster_buildings_fmt} | Capaciteit: {cap_buildings_fmt}</div>
        <div class="tooltip-row">Aangesloten gebouwen: {connected_buildings_fmt}</div>
        <div class="tooltip-row">Warmtevraag in radar (MWh): {cluster_MWh_fmt} | Capaciteit: {cap_MWh_fmt}</div>
        <div class="tooltip-row">Aangesloten MWh: {connected_MWh_fmt}</div>
                <div class="tooltip-row">Benutting: {utilization_pct_fmt}%</div>
      </div>
    </div>
    """
    return {
        "html": html,
        "style": {
            "backgroundColor": "white",
            "color": "black",
            "font-family": "Arial",
            "padding": "5px",
            "border-radius": "5px",
        },
    }
