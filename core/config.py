"""Basisconfiguratie voor de Streamlit-app en scripts.

Deze module centraliseert alle paden, URL-overrides en laag instellingen zodat
het project vanuit één plaats kan worden geconfigureerd.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import streamlit as st
except Exception:  # Streamlit niet beschikbaar (CLI/tests)
    st = None  # type: ignore[assignment]

# -----------------------------
# Projectconstanten
# -----------------------------
BASE_H3_RES: int = 13
H3_AREA_RES0_KM2: float = 4_357_449.0
AVG_HA_BY_RES: dict[int, float] = {
    res: (H3_AREA_RES0_KM2 / (7 ** res)) * 100.0 #km2 -> ha
    for res in range(0, 16)
}


# -----------------------------
# Project-relative base paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
LAYERS_DIR = DATA_DIR / "layers"


def _get_secret(var: str) -> str | None:
    """Lees een configuratiewaarde uit Streamlit secrets indien beschikbaar."""
    if st is None:
        return None
    try:
        value: Any = st.secrets.get(var)
    except Exception:
        return None
    if value is None:
        return None
    return str(value).strip() or None


def _env_url(var: str) -> str | None:
    """Geef eerst een Streamlit-secret, anders een omgevingsvariabele terug."""
    secret_val = _get_secret(var)
    if secret_val:
        return secret_val
    val = os.getenv(var, "").strip()
    return val or None


def _env_path(var: str, default: Path) -> Path:
    """Verplicht pad. Laat een env-var het standaardpad overschrijven."""
    val = os.getenv(var)
    return Path(val).expanduser().resolve() if val else default


def _env_path_opt(var: str, default: Path | None) -> Path | None:
    """Optioneel pad: retourneer None wanneer er geen override is."""
    val = os.getenv(var)
    return Path(val).expanduser().resolve() if val else default


# -----------------------------
# Data sources (Paths + optional URLs)
# -----------------------------
DATA_CSV_PATH = _env_path("WARMTE_DATA_CSV", DATA_DIR / "data.parquet")
DATA_CSV_URL = _env_url("WARMTE_URL_DATA_CSV")

PRECOMPUTED_DIR = DATA_DIR / "precomputed"
H3_RES13_GROUPED_PATH = PRECOMPUTED_DIR / "h3_res13_grouped.parquet"

ENERGIEARMOEDE_PATH = _env_path(
    "WARMTE_LYR_ENERGIEARMOEDE", LAYERS_DIR / "energiearmoede_frl.geojson.gz"
)
KOOPWONINGEN_PATH = _env_path(
    "WARMTE_LYR_KOOPWONINGEN", LAYERS_DIR / "koopwoningen_frl.geojson.gz"
)
WOONCORPORATIE_PATH = _env_path(
    "WARMTE_LYR_WOONCORPORATIE", LAYERS_DIR / "wooncorporatie_frl.geojson.gz"
)
SPOORDEEL_PATH = _env_path("WARMTE_LYR_SPOORDEEL", LAYERS_DIR / "spoordeel.geojson.gz")
WATERDEEL_PATH = _env_path_opt("WARMTE_LYR_WATERDEEL", None)
WATER_POTENTIE_PATH = _env_path(
    "WARMTE_LYR_WATER_POTENTIE", LAYERS_DIR / "waterlichamen_potentie_extraqt.geojson.gz"
)
BUURT_POTENTIE_PATH = _env_path(
    "WARMTE_LYR_BUURT_POTENTIE", LAYERS_DIR / "buurt_potentie_extraqt.geojson.gz"
)


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
    "water_potentie": {
        "toggle_key": "show_water_potentie",
        "prop_name": "Potentie_kWh",
        "out_prop": "_water_pot_color",
        "palette": "YlGnBu",
        "n_colors": 5,
        "alpha": 210,
        "legend_title": "Waterlichamen potentie (kWh)",
        "tooltip_unit": "kWh",
        "line_color": [15, 70, 110, 140],
    },
    "buurt_potentie": {
        "toggle_key": "show_buurt_potentie",
        "prop_name": "Perc_covered",
        "out_prop": "_buurt_pot_color",
        "palette": "YlOrRd",
        "n_colors": 5,
        "alpha": 210,
        "legend_title": "Buurtpotentie (% gedekt)",
        "tooltip_unit": "%",
    },
}

# Gemeenschappelijke properties
GJ_COMMON_PROPS = ["buurtnaam", "gemeentenaam"]

# ================================
# Basemap-config
# ================================
BASEMAP_CFG = {
    "brt": {
        "title": "BRT Achtergrondkaart",
        "map_style": "https://api.pdok.nl/kadaster/brt-achtergrondkaart/ogc/v1/styles/standaard__webmercatorquad?f=mapbox",
        "attribution": "© BRT Achtergrondkaart, Kadaster (CC-BY 4.0)",
        "legend_html": "Bron: Kadaster via PDOK",
    },
}
