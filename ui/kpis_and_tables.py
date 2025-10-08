# ui/kpis_and_tables.py
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.utils import format_dutch_number


# =========================
# KPI CARDS
# =========================

def _nl_int(x) -> str:
    return f"{int(x):,}".replace(",", ".")


def _kpi_css():
    st.markdown("""
    <style>
    .kpi-card { background:#f6f8fb; border:1px solid #e5e7eb; border-radius:18px; padding:16px 18px; margin-bottom: 25px;}
    .kpi-title { margin:0 0 6px 0; font-size:14px; color:#6b7280; font-weight:600; letter-spacing:.2px }
    .kpi-value { font-size:32px; font-weight:800; color:#0b1324; letter-spacing:.3px }
    .kpi-sub { margin-top:6px; color:#6b7280; font-size:12px }
    </style>
    """, unsafe_allow_html=True)


def _kpi_card(title: str, value: str, sub: str):
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

    totaal_panden = int(df_filtered["aantal_huizen"].sum())
    totaal_mwh = int(df_filtered["gemiddeld_jaarverbruik_mWh"].sum())
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
    Bouwt tabs:
      - Top woonplaatsen (MWh)  [altijd]
      - Top aandachtsgebieden   [alleen als show_sites_layer]
      - Kandidaat-voorzieningen [alleen als show_sites_layer]
    Met exact dezelfde kolomnamen/formatting als monolith.
    """
    if show_sites_layer:
        tab1, tab2, tab3 = st.tabs(["Top woonplaatsen (MWh)", "Top aandachtsgebieden", "Kandidaat-voorzieningen"])
    else:
        (tab1,) = st.tabs(["Top woonplaatsen (MWh)"])

    # --- TAB 1: altijd tonen ---
    with tab1:
        top_wp = (
            df_filtered.groupby("woonplaats", as_index=False)["gemiddeld_jaarverbruik_mWh"]
            .sum().rename(columns={"gemiddeld_jaarverbruik_mWh": "MWh"})
            .sort_values("MWh", ascending=False).head(15)
        )
        top_wp_fmt = top_wp.copy()
        top_wp_fmt["MWh"] = top_wp_fmt["MWh"].apply(lambda x: format_dutch_number(x, 0))
        st.dataframe(top_wp_fmt, width="stretch", height=420, hide_index=True)

    # --- TAB 2: Top aandachtsgebieden ---
    if show_sites_layer:
        with tab2:
            hot_hex = (
                df_filtered[df_filtered["kWh_per_m2"] > threshold]
                .sort_values("kWh_per_m2", ascending=False)
                .loc[:, ["woonplaats", "kWh_per_m2", "MWh_per_ha", "gemiddeld_jaarverbruik_mWh_r"]]
                .head(30)
                .rename(columns={
                    "woonplaats": "Woonplaats",
                    "kWh_per_m2": "Gemiddeld energieverbruik (kWh/m²)",
                    "MWh_per_ha": "Warmtevraag-dichtheid (MWh/ha)",
                    "gemiddeld_jaarverbruik_mWh_r": "Totale Heat Demand"
                })
            )
            hot_hex_fmt = hot_hex.copy()
            hot_hex_fmt["Gemiddeld energieverbruik (kWh/m²)"] = hot_hex_fmt["Gemiddeld energieverbruik (kWh/m²)"].apply(lambda x: format_dutch_number(x, 0))
            hot_hex_fmt["Warmtevraag-dichtheid (MWh/ha)"] = hot_hex_fmt["Warmtevraag-dichtheid (MWh/ha)"].apply(lambda x: format_dutch_number(x, 2))
            hot_hex_fmt["Totale Heat Demand"] = hot_hex_fmt["Totale Heat Demand"].apply(lambda x: format_dutch_number(x, 0) + " MWh")
            st.dataframe(hot_hex_fmt, width="stretch", height=420, hide_index=True)

    # --- TAB 3: Kandidaat-voorzieningen ---
        with tab3:
            if sites_costed is not None and not sites_costed.empty:
                out = sites_costed[[
                    "gebied_label", "cluster_buildings", "cap_buildings", "connected_buildings",
                    "cluster_MWh", "cap_MWh", "connected_MWh", "utilization_pct", "indicatieve_kosten_site"
                ]].rename(columns={
                    "gebied_label": "Gebied",
                    "cluster_buildings": "Gebouwen\nin radar",
                    "cap_buildings": "Capaciteit\ngebouwen",
                    "connected_buildings": "Aangesloten\ngebouwen",
                    "cluster_MWh": "MWh\nin radar",
                    "cap_MWh": "Capaciteit\nMWh",
                    "connected_MWh": "Aangesloten\nMWh",
                    "utilization_pct": "Benutting\n(%)",
                    "indicatieve_kosten_site": "Indicatieve\njaarlast (€)"
                })

                total_row = pd.DataFrame([{
                    "Gebied": "Totaal",
                    "Gebouwen\nin radar": out["Gebouwen\nin radar"].sum(),
                    "Capaciteit\ngebouwen": out["Capaciteit\ngebouwen"].sum(),
                    "Aangesloten\ngebouwen": out["Aangesloten\ngebouwen"].sum(),
                    "MWh\nin radar": out["MWh\nin radar"].sum(),
                    "Capaciteit\nMWh": out["Capaciteit\nMWh"].sum(),
                    "Aangesloten\nMWh": out["Aangesloten\nMWh"].sum(),
                    "Benutting\n(%)": None,
                    "Indicatieve\njaarlast (€)": out["Indicatieve\njaarlast (€)"].sum()
                }])

                out = out.loc[:, out.notna().any()]
                total_row = total_row.loc[:, total_row.notna().any()]
                out_full = pd.concat([out, total_row], ignore_index=True)

                cols_to_format = [
                    "Gebouwen\nin radar", "Capaciteit\ngebouwen", "Aangesloten\ngebouwen",
                    "MWh\nin radar", "Capaciteit\nMWh", "Aangesloten\nMWh", "Indicatieve\njaarlast (€)"
                ]
                out_fmt = out_full.copy()
                for col in cols_to_format:
                    out_fmt[col] = out_fmt[col].apply(lambda x: format_dutch_number(x, decimals=0))

                st.dataframe(out_fmt, width="stretch", height=440, hide_index=True)
            else:
                st.info("Geen locaties berekend. Pas instellingen aan en klik op ‘Maak Kaart’.")
