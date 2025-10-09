# core/io.py
import json
import gzip
from pathlib import Path
import re
import os

import orjson
import pandas as pd
import streamlit as st

# pandas copy-on-write voorkomt verborgen kopieën bij bewerkingen
pd.set_option("mode.copy_on_write", True)

from .config import (
    LAYER_CFG,
    DATA_CSV_PATH,
    DATA_CSV_URL,
    H3_RES12_GROUPED_PATH,
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
@st.cache_data(show_spinner=False, max_entries=8, ttl=86400)
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

    raw = gzip.open(p, "rb").read() if p.suffix == ".gz" else p.read_bytes()

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

    suffix = Path(filename_hint).suffix or ".csv"
    cache_path = CACHE_DIR / f"gdrive_{file_id}{suffix}"

    if not cache_path.exists():
        gdown.download(f"https://drive.google.com/uc?id={file_id}", str(cache_path), quiet=False)

    return cache_path


# ============================================================
# Data loader (RAM-geoptimaliseerd)
# ============================================================
@st.cache_data(show_spinner=False, max_entries=1)
def load_data(src: str | Path | None = None, ttl=3600) -> pd.DataFrame:
    """
    Laadt CSV/Parquet:
      - Parquet eerst (sneller + zuiniger)
      - CSV met pyarrow-engine (RAM vriendelijk)
      - Converteert kolommen naar compacte types (float32/int32/category)
    """
    # ---------- Bron bepalen ----------
    if src is None or (isinstance(src, (str, Path)) and str(src).strip() == ""):
        use_compact = os.getenv("WARMTE_USE_COMPACT", "").strip().lower() in {"1", "true", "yes"}
        compact_candidate = DATA_CSV_PATH.parent / "data_kWh_compact.parquet"
        if use_compact and compact_candidate.exists():
            src = compact_candidate
        else:
            local_parquet = DATA_CSV_PATH.with_suffix(".parquet")
            if local_parquet.exists():
                src = local_parquet
            elif DATA_CSV_PATH.exists():
                src = DATA_CSV_PATH
            else:
                src = DATA_CSV_URL

    # ---------- Download als URL ----------
    if isinstance(src, (str, Path)) and _is_url(str(src)):
        s = str(src)
        read_target = _gdrive_to_cache(s) if _is_gdrive(s) else s
    else:
        read_target = Path(src)

    # ---------- Kolommen & dtypes ----------
    usecols = [
        "aantal_VBOs", "totale_oppervlakte", "woonplaats", "Energieklasse",
        "latitude", "longitude", "bouwjaar", "pandstatus",
        "kWh_per_m2", "gemiddeld_jaarverbruik", "Dataset", "gemiddeld_jaarverbruik_mWh"
    ]

    csv_dtypes = {
        "aantal_VBOs": "Int32",
        "totale_oppervlakte": "Int32",
        "bouwjaar": "Int32",
        "gemiddeld_jaarverbruik": "float32",
        "gemiddeld_jaarverbruik_mWh": "float32",
        "kWh_per_m2": "float32",
        "latitude": "float32",
        "longitude": "float32",
    }

    # ---------- Inlezen ----------
    target_str = str(read_target).lower()
    if target_str.endswith(".parquet"):
        df = pd.read_parquet(read_target, columns=usecols, engine="pyarrow")
    else:
        try:
            df = pd.read_csv(
                read_target,
                engine="pyarrow",
                dtype=csv_dtypes,
                usecols=usecols,
            )
        except Exception:
            df = pd.read_csv(read_target, low_memory=False, usecols=usecols)

    # ---------- Numerieke types afdwingen ----------
    for c in ["latitude", "longitude", "kWh_per_m2", "gemiddeld_jaarverbruik_mWh"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    for c in ["aantal_VBOs", "totale_oppervlakte", "bouwjaar"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int32")

    # ---------- RAM reductie ----------
    # Strings -> categories (strenger: pas bij veel herhaling)
    for c in df.select_dtypes(include=["object"]).columns:
        try:
            nunique = df[c].nunique(dropna=True)
            if nunique and nunique <= 10000:
                if nunique <= 0.35 * len(df):  # strenger dan default
                    df[c] = df[c].astype("category")
        except Exception:
            pass

    if "pandstatus" in df.columns:
        valid_statuses = {"Pand in gebruik"}
        mask = df["pandstatus"].astype(str).isin(valid_statuses)
        df = df.loc[mask].reset_index(drop=True)
        df["pandstatus"] = df["pandstatus"].astype("category")

    # ---------- Extra safety ----------
    for c in ["lat", "lon"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    return df


# ============================================================
# Convenience: alle themalagen vooraf laden
# ============================================================
@st.cache_data(show_spinner=False, max_entries=2, ttl=86400)
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


@st.cache_data(show_spinner=False, max_entries=2)
def load_precomputed_h3_grouped(ttl=3600) -> pd.DataFrame | None:
    """
    Laadt het pre-geaggregeerde H3-bestand indien aanwezig.
    """
    if not H3_RES12_GROUPED_PATH.exists():
        return None
    df = pd.read_parquet(H3_RES12_GROUPED_PATH)

    for col in ["woonplaats", "Energieklasse", "Dataset"]:
        if col in df.columns:
            try:
                df[col] = df[col].astype("category")
            except Exception:
                pass

    for col in ["sum_mwh", "sum_area", "sum_kwh", "sum_vbos", "sum_lat", "sum_lon", "sum_bouwjaar"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    if "cnt" in df.columns:
        df["cnt"] = pd.to_numeric(df["cnt"], errors="coerce").astype("int32")

    return df
