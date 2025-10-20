# ui/sidebar.py
from __future__ import annotations

from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

from core.config import LAYER_CFG, BASEMAP_CFG
from core.utils import (
    format_dutch_number,
    get_dynamic_resolution,
    get_hexagon_size,
    get_layer_colors,
    legend_labels_from_breaks,
    render_mini_legend,
    text_input_int,
)

# ---------------------------
# Kleine helpers (RAM-zuinig)
# ---------------------------
def _fillna_categorical(df_in: pd.DataFrame, col: str, value: str = "Onbekend") -> pd.DataFrame:
    """Veilige NA -> 'Onbekend' voor categoricals zonder onnodige kopieën."""
    if col not in df_in.columns:
        return df_in
    s = df_in[col]
    try:
        from pandas.api.types import CategoricalDtype
        is_cat = isinstance(s.dtype, CategoricalDtype)
    except Exception:
        is_cat = False

    if is_cat:
        if value not in s.cat.categories:
            s = s.cat.add_categories([value])
        s = s.fillna(value)
    else:
        # cast naar category pas ná fill (voorkomt dubbele alloc)
        s = s.fillna(value).astype("category")
    df_in[col] = s
    return df_in


def _render_big_legend(current_threshold: int):
    """Render de hoofdlegenda voor de warmtevraaglaag."""
    legend_html = f"""
        <style>
            .legend {{
                background: white; padding: 10px; border-radius: 8px;
                font-family: Arial, sans-serif; font-size: 12px; color: black;
                border: 1px solid #e5e5e5; margin-bottom: 15px;
            }}
            .legend-title {{ font-weight: bold; margin-bottom: 10px; display: block; }}
            .color-box {{ width: 15px; height: 15px; display: inline-block; margin-right: 5px; }}
        </style>
        <div class="legend">
            <div class="legend-title">
                Gemiddelde Energieverbruik<br>(woon en utiliteit oppervlakte)
            </div>
            <div><span class="color-box" style="background-color: #4575b4;"></span> &lt; 10,0 kWh/m²</div>
            <div><span class="color-box" style="background-color: #fee090;"></span> 10,0 - 50,0 kWh/m²</div>
            <div><span class="color-box" style="background-color: #d73027;"></span> 50,0 - {current_threshold} (grenswaarde) kWh/m²</div>
            <div><span class="color-box" style="background-color: #3A1B2f;"></span> Potentie grenswaarde: {current_threshold} kWh/m²</div>
        </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)


def _render_bodem_legend(show_spoor: bool, show_water: bool):
    """Toon de legenda voor spoor- en waterlagen wanneer deze actief zijn."""
    from core.utils import _spoor_rgb_from_cfg
    r_s, g_s, b_s = _spoor_rgb_from_cfg(st.session_state.LAYER_CFG)
    wc = st.session_state.LAYER_CFG["waterdeel"]["palette"]
    r_w, g_w, b_w = int(wc[0]), int(wc[1]), int(wc[2])

    big_class = "ea-legend-wide" if show_water else "ea-legend"
    rows_html = ""
    if show_spoor:
        rows_html += f'<div class="ea-row"><span class="ea-line" style="background:rgba({r_s},{g_s},{b_s},1);"></span> Spoor</div>'
    if show_water:
        rows_html += f'<div class="ea-row"><span class="ea-box" style="background:rgba({r_w},{g_w},{b_w},0.75); border:1px solid rgba({r_w},{g_w},{b_w},1);"></span> Water (polygoon)</div>'

    if not rows_html:
        return

    html = f"""
    <style>
    .ea-legend, .ea-legend-wide {{
        background:#fff; border:1px solid #e5e7eb; border-radius:12px;
        padding:10px; font-family:Arial; font-size:12px; margin-bottom:20px;
        box-shadow: 0 1px 0 rgba(0,0,0,0.03);
    }}
    .ea-legend-wide {{ padding:14px 16px; font-size:13px; border-width: 1.5px; }}
    .ea-row {{ display:flex; align-items:center; margin:6px 0; }}
    .ea-line {{ width:30px; height:3px; display:inline-block; margin-right:8px; border-radius:2px; }}
    .ea-box  {{ width:16px; height:12px; display:inline-block; margin-right:8px; border-radius:3px; }}
    .ea-title {{ font-weight:700; margin-bottom:6px; }}
    </style>
    <div class="{big_class}">
      <div class="ea-title">Bodemlagen</div>
      {rows_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------
# Hoofdfunctie
# ---------------------------
def build_sidebar(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Bouwt de volledige sidebar en retourneert:
      - df (gefilterd)
      - ui (dict met alle gekozen waarden)
    """
    st.session_state.setdefault("grenswaarde_input", 100)
    st.session_state.setdefault("participatie", 80)
    st.session_state.setdefault("LAYER_CFG", LAYER_CFG)
    st.session_state.setdefault("BASEMAP_CFG", BASEMAP_CFG)

    ui: Dict[str, Any] = {}

    with st.sidebar:
        st.header("Opties")

        # ---------------- Kaart ----------------
        with st.expander("Kaart", expanded=True):
            ui["zoom_level"] = st.slider("Selecteer zoomniveau", min_value=9, max_value=12, value=10)
            ui["resolution"] = get_dynamic_resolution(ui["zoom_level"])
            hexagon_size = get_hexagon_size(ui["zoom_level"])
            st.markdown(
                f"<span style='font-size: 12px;'>Bij <b>zoomniveau {ui['zoom_level']}</b> is de kaart <b>ongeveer {hexagon_size} km breed</b>.</span>",
                unsafe_allow_html=True
            )
            with st.expander("Uitleg over zoomniveau"):
                st.write(
                    "Het zoomniveau bepaalt de mate van detail op de kaart:\n"
                    "- **9 en 10**: Specifieke buurten en industriegebieden zijn herkenbaar voor heel Friesland. \n"
                    "- **11 en 12**: Straatniveau. Vanaf dit niveau kan er worden gefilterd op gemeente en woonplaats. \n\n"
                    "De zoomniveaus zijn afgestemd op de vaste resoluties van het H3-grid en op de gewenste weergave op de kaart."
                )
            ui["extruded"] = st.toggle("3D Weergave", value=False, key="extruded")

        # ---------------- Lagen ----------------
        with st.expander("Lagen", expanded=True):
            st.subheader("Warmtevraaglaag")
            ui["show_main_layer"] = st.toggle("Energieverbruik", value=True, key="show_main_layer")

            current_threshold = st.session_state["grenswaarde_input"]
            _render_big_legend(current_threshold)

            ui["show_indicative_layer"] = st.toggle("Aandachtsgebieden tonen", value=True, key="show_indicative_layer")
            ui["threshold"] = st.number_input(
                "Stel de minimale grenswaarde (threshold) in per kWh/m²:",
                min_value=0,
                step=1,
                key="grenswaarde_input"
            )
            with st.expander("Wat doet de grenswaarde?"):
                st.write(
                    "*Pas de grenswaarde bovenin aan om te bepalen welk verbruik jij als grens van ‘extra aandacht’ ziet.* \n\n"
                    "**Kleuren in de kaart laten zien:**\n"
                    "- **< 10,0 kWh/m²** – lage warmtevraag (blauw)\n"
                    "- **10,0 – 50,0 kWh/m²** – gemiddelde warmtevraag (geel)\n"
                    "- **50,0 – grenswaarde** – hoge warmtevraag (rood)\n"
                    "- **> grenswaarde** – Aandachtsgebied (donkerpaars)\n\n"
                    "Alles boven de grenswaarde wordt als groen weergegeven op de kaart."
                )

            # Woonlagen + mini-legenda's
            st.subheader("Woonlagen")
            show_energiearmoede = st.toggle("Energiearmoede", value=False, key=LAYER_CFG["energiearmoede"]["toggle_key"])
            if show_energiearmoede:
                c = LAYER_CFG["energiearmoede"]
                colors = get_layer_colors(c)
                labels = legend_labels_from_breaks(c["breaks"])
                render_mini_legend(c["legend_title"], colors, labels)

            show_koopwoningen = st.toggle("Koopwoningen", value=False, key=LAYER_CFG["koopwoningen"]["toggle_key"])
            if show_koopwoningen:
                c = LAYER_CFG["koopwoningen"]
                colors = get_layer_colors(c)
                labels_kw = legend_labels_from_breaks(c["breaks"])
                render_mini_legend(c["legend_title"], colors, labels_kw)

            show_corporatie = st.toggle("Wooncorporatie", value=False, key=LAYER_CFG["wooncorporatie"]["toggle_key"])
            if show_corporatie:
                c = LAYER_CFG["wooncorporatie"]
                colors = get_layer_colors(c)
                labels_wc = legend_labels_from_breaks(c["breaks"])
                render_mini_legend(c["legend_title"], colors, labels_wc)

            if show_energiearmoede or show_koopwoningen or show_corporatie:
                ui["extra_opacity"] = st.slider(
                    "Transparantie woonlagen",
                    min_value=0.1,
                    max_value=1.0,
                    value=st.session_state.get("extra_opacity", 0.55),
                    key="extra_opacity"
                )
            else:
                ui["extra_opacity"] = st.session_state.setdefault("extra_opacity", 0.55)

            # Bodemlagen
            st.subheader("Bodemlagen")
            ui["show_spoorlaag"] = st.toggle("Spoorlaag", value=False, key=LAYER_CFG["spoordeel"]["toggle_key"])
            if ui["show_spoorlaag"]:
                ui["spoor_opacity"] = st.slider("Transparantie spoorlaag", 0.1, 1.0, 0.5, key="spoor_opacity")

            ui["show_waterlaag"] = st.toggle("Waterlaag", value=False, key=LAYER_CFG["waterdeel"]["toggle_key"])
            if ui["show_waterlaag"]:
                ui["water_opacity"] = st.slider("Transparantie waterlaag", 0.1, 1.0, 0.6, key="water_opacity")

            _render_bodem_legend(ui["show_spoorlaag"], ui["show_waterlaag"])

        # ---------------- Filters ----------------
        with st.expander("Filters", expanded=False):
            # Werk met één boolean mask i.p.v. herhaaldelijk df=df[...]
            df = df_in  # Copy-on-write staat aan in io.py -> masken zijn zuinig

            # Gemeente
            st.subheader("Gemeente")
            gemeente_selectie: List[str] = []
            gemeente_changed = False
            if "gemeentenaam" in df.columns:
                gemeenten_series = df["gemeentenaam"]
                if hasattr(gemeenten_series, "cat"):
                    gemeente_opties = [str(x) for x in gemeenten_series.cat.categories]
                else:
                    gemeente_opties = sorted({str(x) for x in gemeenten_series.dropna().unique()})
                prev_gemeente_selectie = st.session_state.get("_prev_gemeente_selectie", [])
                gemeente_default = [g for g in prev_gemeente_selectie if g in gemeente_opties]
                if not gemeente_default:
                    if "Leeuwarden" in gemeente_opties:
                        gemeente_default = ["Leeuwarden"]
                    else:
                        gemeente_default = gemeente_opties
                gemeente_selectie = st.multiselect(
                    "Filter op gemeente:",
                    options=gemeente_opties,
                    default=gemeente_default,
                )
                if not gemeente_selectie:
                    st.warning("Selecteer minimaal één gemeente.")
                    gemeente_selectie = gemeente_default or gemeente_opties
                prev_gemeente_set = set(prev_gemeente_selectie or [])
                current_gemeente_set = set(gemeente_selectie)
                gemeente_changed = current_gemeente_set != prev_gemeente_set
                st.session_state["_prev_gemeente_selectie"] = gemeente_selectie
                mask_gemeente = gemeenten_series.astype(str).isin(gemeente_selectie)
                df = df.loc[mask_gemeente]
            ui["gemeente_selectie"] = gemeente_selectie

            # Woonplaats
            st.subheader("Woonplaats")
            woonplaatsen = df["woonplaats"].dropna().unique().tolist()
            woonplaatsen_sorted = sorted(woonplaatsen)

            if 1 <= ui["zoom_level"] <= 10:
                # op lager zoomniveau: Friesland geheel, geen multiselect nodig
                friesland_woonplaatsen = woonplaatsen_sorted
                woonplaats_selectie = friesland_woonplaatsen
                mask_wp = df["woonplaats"].isin(woonplaats_selectie)
            else:
                prev_wp = st.session_state.get("woonplaats_selectie", [])
                prev_wp_filtered = [wp for wp in prev_wp if wp in woonplaatsen_sorted]
                if gemeente_changed and woonplaatsen_sorted:
                    default_wp = woonplaatsen_sorted
                else:
                    default_wp = prev_wp_filtered
                if not default_wp:
                    default_wp = woonplaatsen_sorted or ["Leeuwarden"]
                woonplaats_selectie = st.multiselect(
                    "Filter op woonplaats:",
                    options=woonplaatsen_sorted,
                    default=default_wp
                )
                if not woonplaats_selectie:
                    st.warning("Selecteer minimaal één woonplaats.")
                    woonplaats_selectie = woonplaatsen_sorted or ["Leeuwarden"]
                mask_wp = df["woonplaats"].isin(woonplaats_selectie)

            ui["woonplaats_selectie"] = woonplaats_selectie
            st.session_state["woonplaats_selectie"] = woonplaats_selectie
            df = df[mask_wp]

            # Energieklasse
            st.subheader("Energieklasse")
            df = _fillna_categorical(df, "Energieklasse", "Onbekend")
            energieklassen = df["Energieklasse"].cat.categories if hasattr(df["Energieklasse"], "cat") else sorted(df["Energieklasse"].dropna().unique())
            energieklasse_selectie = st.multiselect(
                "Filter op energieklasse:",
                options=[str(x) for x in energieklassen],
                default=[str(x) for x in energieklassen]
            )
            if not energieklasse_selectie:
                energieklasse_selectie = [str(x) for x in energieklassen]
            mask_en = df["Energieklasse"].astype(str).isin(energieklasse_selectie)
            df = df[mask_en]
            ui["energieklasse_selectie"] = energieklasse_selectie

            # Bouwjaar
            st.subheader("Bouwjaar")
            bouwjaar_num = pd.to_numeric(df["bouwjaar"], errors="coerce")
            min_year = int(np.nanmin(bouwjaar_num.to_numpy()))
            max_year = int(np.nanmax(bouwjaar_num.to_numpy()))
            by_lo, by_hi = st.slider("Filter op bouwjaar:", min_year, max_year, (min_year, max_year))
            mask_by = (bouwjaar_num >= by_lo) & (bouwjaar_num <= by_hi)
            df = df[mask_by]
            ui["bouwjaar_range"] = (by_lo, by_hi)

            # Type pand
            st.subheader("Type pand")
            typepand = [str(x) for x in (df["Dataset"].cat.categories if hasattr(df["Dataset"], "cat") else df["Dataset"].dropna().unique())]
            typepand_opties = ["Alle types"] + sorted(typepand)
            pand_selectie = st.selectbox("Selecteer type pand:", options=typepand_opties)
            if pand_selectie != "Alle types":
                df = df[df["Dataset"].astype(str) == pand_selectie]
            ui["pand_selectie"] = pand_selectie

            with st.expander("Uitleg over type pand"):
                st.write(
                    "Hier kun je een specifiek type pand (bron van de data) selecteren om alleen die gegevens op de kaart te tonen. "
                    "Standaard staan alle types aan.\n\n"
                    "**Beschikbare datalagen:**\n"
                    "- **Liander / Stedin** – Kleinverbruik (woningen)\n"
                    "- **Verrijkte BAG (TNO)** – Middel- tot grootverbruik \n"
                    "- **Alliander** – Middel- tot grootverbruik\n"
                )

        # ---------------- Participatie ----------------
        with st.expander("Participatie", expanded=False):
            st.session_state.participatie = st.slider(
                "Deelnamegraad (0% = niemand sluit aan, 100% = iedereen sluit aan)",
                min_value=0, max_value=100, value=st.session_state.participatie, step=1, key="participatie_slider"
            )
            ui["participatie"] = st.session_state.participatie

        # ---------------- Collectieve warmtevoorziening (analyse) ----------------
        selected_places_prior = ui.get("woonplaats_selectie") or st.session_state.get("woonplaats_selectie", [])
        can_analyse = (ui["zoom_level"] >= 11) and bool(selected_places_prior)
        info_html = "<p style='font-size:12px; color:#6b7280; margin-bottom:8px;'>Analyse alleen beschikbaar bij zoomniveau 11 en 12.</p>"

        with st.expander("Collectieve warmtevoorziening (analyse)", expanded=False):
            compute_sites = False
            if not can_analyse:
                st.markdown(info_html, unsafe_allow_html=True)
                ui["show_sites_layer"] = False
                st.session_state["show_sites_layer"] = False
            else:
                st.markdown(info_html, unsafe_allow_html=True)
                ui["show_sites_layer"] = st.toggle(
                    "Warmtevoorzieningen",
                    value=False,
                    key="show_sites_layer"
                )

                if ui["show_sites_layer"]:
                    if not st.session_state.get("sites_ready"):
                        st.info("Stel eerst de filters in en zorg dat de kaart zichtbaar is om warmtevoorzieningen te tonen.")
                    compute_sites = st.button("Bereken warmtevoorzieningen", key="compute_sites_button")

                    prev_kring = int(st.session_state.get("kring_radius", 3))
                    if prev_kring > 5:
                        prev_kring = 5
                        st.session_state.kring_radius = prev_kring
                    ui["kring_radius"] = st.slider("Bereik van de warmtevoorziening", 1, 5, prev_kring, 1, key="kring_radius")

                    prev_min_sep = int(st.session_state.get("min_sep", 3))
                    if prev_min_sep > 5:
                        prev_min_sep = 5
                        st.session_state.min_sep = prev_min_sep
                    ui["min_sep"] = st.slider("Minimale afstand tussen warmtevoorzieningen", 1, 5, prev_min_sep, 1, key="min_sep")
                    # Bewaar bestaande keuze maar klem deze binnen nieuwe grenzen
                    prev_n_sites = int(st.session_state.get("n_sites", 3))
                    if prev_n_sites > 20:
                        prev_n_sites = 20
                        st.session_state.n_sites = prev_n_sites
                    ui["n_sites"] = st.number_input("Aantal collectieve warmtevoorzieningen", min_value=1, max_value=20, value=prev_n_sites, step=1, key="n_sites")

                    ui["cap_mwh"] = text_input_int("Capaciteit per voorziening (MWh)", key="cap_mwh", default=100_000)
                    ui["cap_buildings"] = text_input_int("Max gebouwen per voorziening", key="cap_buildings", default=1_000)
                    ui["fixed_cost"] = text_input_int("Vaste kosten per locatie (€)", key="fixed_cost", default=25_000)
                    ui["var_cost"] = text_input_int("Variabele kosten (€ per MWh)", key="var_cost", default=35)

                    ui["opex_pct"] = st.number_input("Extra operationele kosten (% van vaste kosten)", min_value=0, max_value=100, value=10, step=1, key="opex_pct")

            ui["compute_sites"] = compute_sites

        # ---------------- Uitleg-blokken ----------------
        st.header("Uitleg")
        with st.expander("Uitleg H3", expanded=False):
            st.write("H3 is een hexagonaal raster dat gebieden verdeelt in zeshoeken van verschillende resoluties. "
                     "Elke hexagoon krijgt een unieke ID en bevat gegevens over de warmtebehoefte.")

        with st.expander("Uitleg analyse", expanded=False):
            st.markdown("""\
**Doel van de analyse**  
De analyse laat zien waar een **collectieve warmtevoorziening** (zoals een buurtbron of warmtenet) kansrijk kan zijn. Dit gebeurt door te kijken hoeveel energie en gebouwen er binnen de directe omgeving van een mogelijke locatie liggen en of deze plek past binnen de capaciteit van een voorziening.

**Werkwijze in hoofdlijnen**  
1. Rondom een centrale plek wordt gekeken naar omliggende buurten in de vorm van zeshoekige vakjes (hexagonen). De straal bepaalt hoe groot de omgeving is die wordt meegenomen.  
2. Per locatie wordt berekend hoeveel energie en hoeveel gebouwen in die omgeving aanwezig zijn. Daarbij geldt een maximum: een voorziening kan maar een bepaalde hoeveelheid warmte leveren en een maximum aantal gebouwen aansluiten.  
3. Alle locaties worden gerangschikt op de hoeveelheid warmte die daadwerkelijk kan worden aangesloten.  
4. Vervolgens worden de beste locaties geselecteerd, met een minimale onderlinge afstand zodat voorzieningen niet te dicht bij elkaar liggen.  

**Begrippen**  
- **k-ring:** de directe omgeving van een plek, gemeten in stappen van hexagonen. Hoe hoger de waarde, hoe verder de omgeving reikt.  
- **Minimale afstand tussen voorzieningen:** de onderlinge ruimte die wordt aangehouden, zodat meerdere voorzieningen niet op (bijna) dezelfde plek terechtkomen.  

**Verschil tussen deze twee**  
- De *k-ring* bepaalt hoe ver er wordt gekeken om te berekenen hoeveel gebouwen en energie bij één locatie horen (de invloedsstraal van een plek).  
- De *minimale afstand* zorgt ervoor dat twee geselecteerde voorzieningen niet te dicht naast elkaar komen te liggen (de spreiding tussen verschillende plekken).  

**K-ring in de praktijk (k = 1 t/m 5)**  
- **k = 1** – directe buren, circa 7 hexagonen. Denk aan een cluster van enkele panden binnen ~200 meter.  
- **k = 3** – kleine buurt, ±37 hexagonen. Bestrijkt ongeveer een paar straten.  
- **k = 5** – grotere buurt, ±91 hexagonen. Omvat een deelwijk of bedrijventerrein van enkele hectares.  
""")

    return df, ui
