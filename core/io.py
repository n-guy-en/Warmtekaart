# core/io.py
import json
import gzip
from pathlib import Path
import re
import os

import orjson
import pandas as pd
import streamlit as st

from .config import (
    LAYER_CFG,
    DATA_CSV_PATH,
    DATA_CSV_URL,
    ENERGIEARMOEDE_PATH,
    KOOPWONINGEN_PATH,
    WOONCORPORATIE_PATH,
    SPOORDEEL_PATH,
    WATERDEEL_PATH,
    GJ_COMMON_PROPS,
)

# ============================================================
# GeoJSON loader (met property-filter & coördinaat-precisie)
# ============================================================

@st.cache_data(show_spinner=False)
def load_geojson(path: str | Path, keep_props=None, coord_precision: int = 3, ttl=3600):
    """
    Laadt een GeoJSON-bestand als dict.
    - keep_props: lijst met property-namen die je wilt behouden (alles daarbuiten wordt gestript)
    - coord_precision: aantal decimalen voor coördinaten (reductie van bestandsgrootte)
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None

    if p.suffix == ".gz":
        raw = gzip.open(p, "rb").read()
    else:
        raw = p.read_bytes()

    try:
        gj = orjson.loads(raw)
    except Exception:
        gj = json.loads(raw.decode("utf-8"))

    if not (gj and isinstance(gj, dict) and gj.get("type") == "FeatureCollection"):
        return gj

    feats = []
    kp = set(keep_props or [])
    factor = 10 ** coord_precision

    def _round_coords(obj):
        if isinstance(obj, list):
            return [_round_coords(x) for x in obj]
        if isinstance(obj, float):
            return int(obj * factor) / factor
        return obj

    for feat in gj.get("features", []):
        geom = feat.get("geometry")
        props = feat.get("properties", {}) or {}
        if kp:
            props = {k: props.get(k) for k in kp if k in props}
        if geom and geom.get("coordinates") is not None:
            geom = {"type": geom.get("type"), "coordinates": _round_coords(geom.get("coordinates"))}
        feats.append({"type": "Feature", "properties": props, "geometry": geom})

    return {"type": "FeatureCollection", "features": feats}

# ============================================================
# Google download cache-map voor externe downloads
# ============================================================
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _is_url(x: str | Path) -> bool:
    s = str(x)
    return s.startswith("http://") or s.startswith("https://")

def _is_gdrive(url: str) -> bool:
    u = url.lower()
    return ("drive.google.com" in u) or ("docs.google.com" in u)

def _extract_gdrive_file_id(url: str) -> str | None:
    # vangt zowel ?id=FILE_ID als /file/d/FILE_ID/ varianten
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]{10,})", url)
    if m: 
        return m.group(1)
    m = re.search(r"/file/d/([a-zA-Z0-9_-]{10,})", url)
    if m:
        return m.group(1)
    return None

def _gdrive_to_cache(url: str, filename_hint: str = "data_kWh.csv") -> Path:
    """Download 1x naar cache met gdown en retourneer lokaal pad."""
    try:
        import gdown  # lazy import
    except Exception as e:
        raise RuntimeError("gdown is vereist voor Google Drive-URL's (voeg gdown>=5.1 toe aan requirements).") from e

    file_id = _extract_gdrive_file_id(url)
    if not file_id:
        raise ValueError(f"Kon geen Google Drive file-id vinden in URL: {url}")

    # behoud extensie van hint (bv. .csv)
    suffix = Path(filename_hint).suffix or ".csv"
    cache_path = CACHE_DIR / f"gdrive_{file_id}{suffix}"

    if not cache_path.exists():
        gdown.download(f"https://drive.google.com/uc?id={file_id}", str(cache_path), quiet=False)

    return cache_path

# ===== Data loaders =====
@st.cache_data(show_spinner=False)
def load_data(src: str | Path | None = None, ttl=3600) -> pd.DataFrame:
    """
    Laadt CSV/Parquet uit:
      - Google Drive URL  -> download 1x naar cache (gdown), lees lokaal
      - Overige http(s)   -> pandas leest direct
      - Lokaal pad        -> pandas leest lokaal
    Als src None is: gebruik DATA_CSV_URL (secrets) of fallback naar DATA_CSV_PATH.
    Past daarna dezelfde kolom/dtype-opschoning toe als je monolith.
    """
    # 0) Bron bepalen en valideren (met fallback-logic)
    if src is None or (isinstance(src, (str, Path)) and str(src).strip() == ""):
        local_parquet = DATA_CSV_PATH.with_suffix(".parquet") if hasattr(DATA_CSV_PATH, "with_suffix") else None

        if local_parquet and Path(local_parquet).exists():
            src = local_parquet  # eerst lokaal Parquet
        elif DATA_CSV_PATH.exists():
            src = DATA_CSV_PATH  # dan lokaal CSV
        else:
            src = DATA_CSV_URL   # laatste optie: online CSV
            

    # 1) URL of lokaal pad bepalen
    if isinstance(src, (str, Path)) and _is_url(str(src)):
        s = str(src)
        if _is_gdrive(s):
            local_path = _gdrive_to_cache(s, filename_hint="data_kWh.csv")
            read_target = local_path
        else:
            read_target = s
    else:
        read_target = Path(src)

    # 2) Inlezen
    target_str = str(read_target).lower()
    if target_str.endswith(".parquet"):
        df = pd.read_parquet(read_target)
    else:
        try:
            df = pd.read_csv(read_target, engine="pyarrow")
        except Exception:
            df = pd.read_csv(read_target, low_memory=False)

    # 3) Dtypes/opschonen (ongewijzigd)
    cols_num_int = [
        "aantal_VBOs",
        "totale_oppervlakte",
        "bouwjaar",
        "gemiddeld_jaarverbruik",
        "gemiddeld_jaarverbruik_mWh",
    ]
    for c in cols_num_int:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["kWh_per_m2", "latitude", "longitude"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    return df


# ============================================================
# Convenience: alle themalagen vooraf laden (exacte keep_props)
# ============================================================

@st.cache_data(show_spinner=False)
def preload_geo_layers(ttl=3600):
    """
    Laadt alle geojson-lagen zoals in de monolith gedaan werd, met identieke keep_props.
    Retourneert dict met keys:
      - energiearmoede, koopwoningen, corporatie, spoordeel, water (optioneel)
    """
    gj_energiearmoede = load_geojson(
        ENERGIEARMOEDE_PATH,
        keep_props=[LAYER_CFG["energiearmoede"]["prop_name"], *GJ_COMMON_PROPS],
        coord_precision=3,
    )
    gj_koopwoningen = load_geojson(
        KOOPWONINGEN_PATH,
        keep_props=[LAYER_CFG["koopwoningen"]["prop_name"], *GJ_COMMON_PROPS],
        coord_precision=3,
    )
    gj_corporatie = load_geojson(
        WOONCORPORATIE_PATH,
        keep_props=[LAYER_CFG["wooncorporatie"]["prop_name"], *GJ_COMMON_PROPS],
        coord_precision=3,
    )
    gj_spoordeel = load_geojson(
        SPOORDEEL_PATH,
        keep_props=[],  # lijnen: geen extra props nodig
        coord_precision=3,
    )
    gj_water = None
    if WATERDEEL_PATH:
        gj_water = load_geojson(WATERDEEL_PATH, keep_props=[], coord_precision=3)

    return {
        "energiearmoede": gj_energiearmoede,
        "koopwoningen": gj_koopwoningen,
        "corporatie": gj_corporatie,
        "spoordeel": gj_spoordeel,
        "water": gj_water,
    }

