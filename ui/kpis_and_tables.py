# ui/kpis_and_tables.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.utils import format_dutch_number

# =========================
# KPI CARDS
# =========================

def _nl_int(x) -> str:
    try:
        return f"{int(x):,}".replace(",", ".")
    except Exception:
        # fallback voor floats/NaN
        try:
            return format_dutch_number(x, 0)
        except Exception:
            return "0"

def _kpi_css():
    """Add CSS die de KPI-kaarten vormgeeft."""
    st.markdown("""
    <style>
    .kpi-card { background:#f6f8fb; border:1px solid #e5e7eb; border-radius:18px; padding:16px 18px; margin-bottom: 25px;}
    .kpi-title { margin:0 0 6px 0; font-size:14px; color:#6b7280; font-weight:600; letter-spacing:.2px }
    .kpi-value { font-size:32px; font-weight:800; color:#0b1324; letter-spacing:.3px }
    .kpi-sub { margin-top:6px; color:#6b7280; font-size:12px }
    </style>
    """, unsafe_allow_html=True)

def _kpi_card(title: str, value: str, sub: str):
    """Render één KPI-kaart met titel, hoofdwaarde en subtitel."""
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-title'>{title}</div>"
        f"<div class='kpi-value'>{value}</div>"
        f"<div class='kpi-sub'>{sub}</div></div>",
        unsafe_allow_html=True
    )

def render_kpis(df_filtered: pd.DataFrame, participatie_pct: int):
    """
    Toont 4 KPI-kaarten:
    - Totaal aantal panden
    - Deelnamegraad (panden)
    - Totale Heat Demand (MWh)
    - Deelnamegraad (MWh)
    """
    _kpi_css()

    # Gebruik .get met default Series om KeyError te vermijden (RAM-zuinig)
    s_panden = pd.to_numeric(df_filtered.get("aantal_huizen", pd.Series([], dtype="int32")), errors="coerce").fillna(0)
    s_mwh = pd.to_numeric(df_filtered.get("gemiddeld_jaarverbruik_mWh", pd.Series([], dtype="float32")), errors="coerce").fillna(0)

    totaal_panden = int(s_panden.sum()) if len(s_panden) else 0
    totaal_mwh = int(round(float(s_mwh.sum()))) if len(s_mwh) else 0

    pct = int(participatie_pct)
    panden_part = round(totaal_panden * pct / 100)
    mwh_part = round(totaal_mwh * pct / 100)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        _kpi_card("Totaal aantal panden", _nl_int(totaal_panden), "Aantal panden")
    with c2:
        _kpi_card(f"Deelnamegraad: {pct}%", _nl_int(panden_part), "Aantal panden")
    with c3:
        _kpi_card("Totale Heat Demand", _nl_int(totaal_mwh), "MWh")
    with c4:
        _kpi_card(f"Deelnamegraad: {pct}%", _nl_int(mwh_part), "MWh")


# =========================
# TABELLEN / TABS
# =========================

def _fmt0(x):
    try:
        return format_dutch_number(int(x), 0)
    except Exception:
        return format_dutch_number(x, 0)

def _fmt2(x):
    return format_dutch_number(x, 2)

def render_tabs(df_filtered: pd.DataFrame, threshold: float, show_sites_layer: bool, sites_costed: pd.DataFrame | None):
    """
    Tabs:
      - Top woonplaatsen (MWh)  [altijd]
      - Kandidaat-voorzieningen [alleen als show_sites_layer]
    RAM-zuinig: minimale kolomselecties, vectorized formatting.
    """
    if isinstance(sites_costed, list):
        sites_costed_df = pd.DataFrame(sites_costed)
    else:
        sites_costed_df = sites_costed

    if show_sites_layer:
        tab1, tab2 = st.tabs(["Top woonplaatsen (MWh)", "Kandidaat-voorzieningen"])
    else:
        (tab1,) = st.tabs(["Top woonplaatsen (MWh)"])

    # --- TAB 1: Top woonplaatsen (MWh) ---
    with tab1:
        # Beperk kolommen vóór groupby
        col_wp = "woonplaats"
        col_mwh = "gemiddeld_jaarverbruik_mWh"
        col_density = "MWh_per_ha"
        col_area = "area_ha"
        available_cols = set(df_filtered.columns)

        use_area = col_area in available_cols
        use_density_col = col_density in available_cols

        base_cols = [col_wp, col_mwh]
        if use_area:
            base_cols.append(col_area)
        if use_density_col and not use_area:
            # alleen meenemen als area ontbreekt; anders berekenen we het zelf
            base_cols.append(col_density)

        if set(base_cols) <= available_cols:
            df_wp = df_filtered.loc[:, base_cols]
        else:
            missing_cols = set(base_cols) - available_cols
            if missing_cols:
                # kan gebeuren bij lege datasets; maak lege df
                df_wp = pd.DataFrame(columns=base_cols)
            else:
                df_wp = df_filtered.loc[:, base_cols]

        if not df_wp.empty:
            s = pd.to_numeric(df_wp[col_mwh], errors="coerce").fillna(0)
            df_wp = df_wp.assign(**{col_mwh: s})

            agg_map = {col_mwh: "sum"}
            if use_area:
                area_series = pd.to_numeric(df_wp[col_area], errors="coerce").fillna(0)
                df_wp[col_area] = area_series
                agg_map[col_area] = "sum"
                density_source = "area"
            elif use_density_col:
                density_series = pd.to_numeric(df_wp[col_density], errors="coerce")
                df_wp[col_density] = density_series
                agg_map[col_density] = "mean"
                density_source = "col"
            else:
                density_source = None

            top_wp = (
                df_wp.groupby(col_wp, as_index=False, sort=False, observed=True)
                     .agg(agg_map)
                     .rename(columns={col_mwh: "MWh"})
                      .sort_values("MWh", ascending=False)
                      .head(15)
            )

            area_display_col = "Gebiedsoppervlakte voor warmtevraag (ha)"
            density_display_col = "Warmtevraag per ha (MWh)"

            if use_area and col_area in top_wp.columns:
                top_wp.rename(columns={col_area: area_display_col}, inplace=True)
                area_vals = top_wp[area_display_col].replace({0: pd.NA})
                top_wp[density_display_col] = top_wp["MWh"].div(area_vals)
            elif density_source == "col" and col_density in top_wp.columns:
                top_wp.rename(columns={col_density: density_display_col}, inplace=True)
            else:
                # geen bron beschikbaar; maak lege kolom
                top_wp[density_display_col] = pd.NA

            # Vectorized formatting (vermijd apply lambda per rij waar mogelijk)
            top_wp_fmt = top_wp.copy()
            top_wp_fmt["MWh"] = top_wp_fmt["MWh"].round(0).astype("int64").map(lambda v: f"{v:,}".replace(",", "."))
            if area_display_col in top_wp_fmt.columns:
                top_wp_fmt[area_display_col] = top_wp_fmt[area_display_col].map(
                    lambda v: "" if pd.isna(v) else _fmt2(float(v))
                )
            if density_display_col in top_wp_fmt.columns:
                top_wp_fmt[density_display_col] = top_wp_fmt[density_display_col].map(
                    lambda v: "" if pd.isna(v) else _fmt2(float(v))
                )
            st.dataframe(top_wp_fmt, width="stretch", height=420, hide_index=True)
        else:
            st.info("Geen gegevens om te tonen.")

    # --- TAB 3: Kandidaat-voorzieningen ---
    if show_sites_layer:
        with tab2:
            if sites_costed_df is not None and not sites_costed_df.empty:
                cols_keep = [
                    "site_rank", "gebied_label", "cluster_buildings", "cap_buildings", "connected_buildings",
                    "cluster_MWh", "cap_MWh", "connected_MWh", "MWh_per_ha", "utilization_pct", "indicatieve_kosten_site"
                ]
                have = [c for c in cols_keep if c in sites_costed_df.columns]
                out = sites_costed_df.loc[:, have].copy()
                if "site_rank" in out.columns:
                    out["site_rank"] = pd.to_numeric(out["site_rank"], errors="coerce").astype("Int32")
                rename_map = {
                    "site_rank": "Voorziening #",
                    "gebied_label": "Gebied",
                    "cluster_buildings": "Gebouwen\nin radar",
                    "cap_buildings": "Capaciteit\ngebouwen",
                    "connected_buildings": "Aangesloten\ngebouwen",
                    "cluster_MWh": "MWh\nin radar",
                    "cap_MWh": "Capaciteit\nMWh",
                    "connected_MWh": "Aangesloten\nMWh",
                    "MWh_per_ha": "Warmtevraag\n per ha (MWh)",
                    "utilization_pct": "Benutting\n(%)",
                    "indicatieve_kosten_site": "Indicatieve\njaarlast (€)"
                }
                out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns}, inplace=True)

                # Totaalrij (alleen over kolommen die bestaan)
                out_full = out.copy()
                totals_cols = [
                    "Gebouwen\nin radar", "Capaciteit\ngebouwen", "Aangesloten\ngebouwen",
                    "MWh\nin radar", "Capaciteit\nMWh", "Aangesloten\nMWh", "Indicatieve\njaarlast (€)"
                ]
                available_totals = {col: pd.to_numeric(out[col], errors="coerce").fillna(0).sum()
                                    for col in totals_cols if col in out.columns}
                if available_totals:
                    totals_values = []
                    for col_name in out.columns:
                        if col_name in available_totals:
                            totals_values.append(available_totals[col_name])
                        elif col_name == "Gebied":
                            totals_values.append("Totaal")
                        elif col_name == "Voorziening #":
                            totals_values.append("")
                        else:
                            series_col = out[col_name]
                            totals_values.append(np.nan if pd.api.types.is_numeric_dtype(series_col) else "")
                    totals_df = pd.DataFrame([totals_values], columns=out.columns)
                    out_full = pd.concat([out_full, totals_df], ignore_index=True)

                # Formatteringen (kolomsgewijs)
                out_fmt = out_full.copy()
                if "Voorziening #" in out_fmt.columns:
                    col = pd.to_numeric(out_fmt["Voorziening #"], errors="coerce")
                    out_fmt["Voorziening #"] = ""
                    mask = col.notna()
                    if mask.any():
                        out_fmt.loc[mask, "Voorziening #"] = col.loc[mask].astype("int64").astype(str)
                for col in [
                    "Gebouwen\nin radar", "Capaciteit\ngebouwen", "Aangesloten\ngebouwen",
                    "MWh\nin radar", "Capaciteit\nMWh", "Aangesloten\nMWh", "Indicatieve\njaarlast (€)"
                ]:
                    if col in out_fmt.columns:
                        s = pd.to_numeric(out_fmt[col], errors="coerce").fillna(0).round(0).astype("int64")
                        out_fmt[col] = s.map(lambda v: f"{v:,}".replace(",", "."))
                if "Warmtevraag\n per pand (MWh)" in out_fmt.columns:
                    s = pd.to_numeric(out_fmt["Warmtevraag\n per pand (MWh)"], errors="coerce")
                    out_fmt["Warmtevraag\n per pand (MWh)"] = s.map(
                        lambda v: "" if pd.isna(v) else f"{float(v):,.2f}".replace(",", "#").replace(".", ",").replace("#", ".")
                    )
                if "Warmtevraag\n per ha (MWh)" in out_fmt.columns:
                    s = pd.to_numeric(out_fmt["Warmtevraag\n per ha (MWh)"], errors="coerce")
                    out_fmt["Warmtevraag\n per ha (MWh)"] = s.map(
                        lambda v: "" if pd.isna(v) else f"{float(v):,.2f}".replace(",", "#").replace(".", ",").replace("#", ".")
                    )

                st.dataframe(out_fmt, width="stretch", height=440, hide_index=True)

            else:
                st.info("Geen locaties berekend. Pas instellingen aan en klik op ‘Maak Kaart’.")
