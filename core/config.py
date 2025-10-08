# core/config.py
from __future__ import annotations
from pathlib import Path
import os


# probeer st.secrets als we in Streamlit draaien
try:
    import streamlit as st
    def _env_url(var: str) -> str | None:
        v = None
        try:
            v = st.secrets.get(var)  # werkt op Streamlit 1.18+
        except Exception:
            pass
        if not v:
            v = os.getenv(var, "").strip()
        return v or None
except Exception:
    # fallback buiten Streamlit (tests/CLI)
    def _env_url(var: str) -> str | None:
        v = os.getenv(var, "").strip()
        return v or None


# -----------------------------
# Project-relative base paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
LAYERS_DIR = DATA_DIR / "layers"

def _env_path(var: str, default: Path) -> Path:
    """Required file/dir; env var can override."""
    val = os.getenv(var)
    return Path(val).expanduser().resolve() if val else default

def _env_path_opt(var: str, default: Path | None) -> Path | None:
    """Optional file/dir; env var can override."""
    val = os.getenv(var)
    return Path(val).expanduser().resolve() if val else default

def _env_url(var: str) -> str | None:
    """Optional URL override."""
    val = os.getenv(var, "").strip()
    return val or None


# -----------------------------
# Data sources (Paths + optional URLs)
# -----------------------------
DATA_CSV_PATH        = _env_path("WARMTE_DATA_CSV", DATA_DIR / "data_kWh.parquet")
DATA_CSV_URL         = _env_url("WARMTE_URL_DATA_CSV")

ENERGIEARMOEDE_PATH  = _env_path("WARMTE_LYR_ENERGIEARMOEDE", LAYERS_DIR / "energiearmoede_frl.geojson.gz")

KOOPWONINGEN_PATH    = _env_path("WARMTE_LYR_KOOPWONINGEN",   LAYERS_DIR / "koopwoningen_frl.geojson.gz")

WOONCORPORATIE_PATH  = _env_path("WARMTE_LYR_WOONCORPORATIE", LAYERS_DIR / "wooncorporatie_frl.geojson.gz")

SPOORDEEL_PATH       = _env_path("WARMTE_LYR_SPOORDEEL",      LAYERS_DIR / "spoordeel.geojson.gz")

WATERDEEL_PATH       = _env_path_opt("WARMTE_LYR_WATERDEEL",  None)
#WATERDEEL_URL        = _env_url("WARMTE_URL_WATERDEEL")

# ================================
# Lagen-config
# ================================
LAYER_CFG = {
    "energiearmoede": {
        "toggle_key": "show_energiearmoede",
        "prop_name": "Percentage huishoudens met energiearmoede",
        "out_prop": "_ea_color",
        "breaks": [0.05, 0.10, 0.20],
        "palette": "OrRd",
        "n_colors": 4,
        "alpha": 200,
        "legend_title": "Energiearmoede (% huishoudens)",
        "line_color": [102, 51, 0, 120],
    },
    "koopwoningen": {
        "toggle_key": "show_koopwoningen",
        "prop_name": "Percentage koopwoningen",
        "out_prop": "_kw_color",
        "breaks": [0.40, 0.60, 0.80],
        "palette": "YlOrBr",
        "n_colors": 4,
        "alpha": 200,
        "legend_title": "Koopwoningen (% woningen)",
        "line_color": [0, 0, 0, 120],
    },
    "wooncorporatie": {
        "toggle_key": "show_corporatie",
        "prop_name": "Percentage wooncorporatie",
        "out_prop": "_wc_color",
        "breaks": [0.05, 0.10, 0.15],
        "palette": "YlGnBu",
        "n_colors": 4,
        "alpha": 200,
        "legend_title": "Wooncorporaties (% woningen)",
        "line_color": [60, 40, 120, 120],
    },
    "spoordeel": {
        "toggle_key": "show_spoordeel",
        "out_prop": "_wd_color",
        "palette": [0, 0, 0],
        "alpha": 200,
        "legend_title": "Spoordeel",
        "line_color": [0, 0, 0, 255],
    },
    "waterdeel": {
        "toggle_key": "show_waterdeel",
        "out_prop": "_water_color",
        "palette": [36, 122, 212],
        "alpha": 180,
        "legend_title": "Water (bodemlaag)",
        "line_color": [36, 122, 212, 255],
    },
}

# Gemeenschappelijke properties
GJ_COMMON_PROPS = ["buurtnaam", "gemeentenaam"]

# ================================
# Basemap-config
# ================================
BASEMAP_CFG = {
    "light": {
        "title": "Light",
        "tile": "https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        "legend_html": "Heldere achtergrond met subtiele labels. Bron: CARTO + OSM.",
    },
    "dark": {
        "title": "Dark",
        "tile": "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
        "legend_html": "Donkere achtergrond voor nadruk op thema-lagen. Bron: CARTO + OSM.",
    },
    "streets": {
        "title": "Streets",
        "tile": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "legend_html": "Standaard OSM-tegels met straten en labels. Bron: OSM.",
    },
    "outdoors": {
        "title": "Outdoors",
        "tile": "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        "legend_html": "Outdoor-achtig thema. Bron: CARTO + OSM.",
    },
    "satellite": {
        "title": "Satellite",
        "tile": "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/xyz/{z}/{x}/{y}.jpg",
        "legend_html": "Satellietbeeld zonder labels (2020 cloudless). Bron: EOX + Copernicus.",
    },
}