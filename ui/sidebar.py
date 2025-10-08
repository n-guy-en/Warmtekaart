# ui/sidebar.py
from __future__ import annotations

from typing import Dict, Any, Tuple, List

import pandas as pd
import streamlit as st

from core.config import LAYER_CFG, BASEMAP_CFG
from core.utils import (
    format_dutch_number,
    get_dynamic_resolution,
    get_hexagon_size,
    legend_labels_from_breaks,
    render_mini_legend,
    text_input_int,
)


def _fillna_categorical(df_in: pd.DataFrame, col: str, value: str = "Onbekend") -> pd.DataFrame:
    """Veilige NA -> 'Onbekend' voor categoricals (exact gedrag monolith)."""
    if col not in df_in.columns:
        return df_in
    s = df_in[col]
    try:
        import pandas as pd  # local ref
        from pandas.api.types import CategoricalDtype
        is_cat = isinstance(s.dtype, CategoricalDtype)
    except Exception:
        is_cat = False

    if is_cat:
        if value not in s.cat.categories:
            s = s.cat.add_categories([value])
        s = s.fillna(value)
    else:
        s = s.fillna(value).astype("category")
    df_in = df_in.copy()
    df_in[col] = s
    return df_in


def _render_big_legend(current_threshold: int):
    legend_html = f"""
        <style>
            .legend {{
                background: white; padding: 10px; border-radius: 8px;
                font-family: Arial, sans-serif; font-size: 12px; color: black;
                box-shadow: 0px 0px 0px rgba(0,0,0,0.3); border: 1px solid #e5e5e5;
                margin-bottom: 15px;
            }}
            .legend-title {{
                font-weight: bold;
                margin-bottom: 10px;
                display: block;
            }}
            .color-box {{
                width: 15px; height: 15px; display: inline-block; margin-right: 5px;
            }}
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
        transition: all .15s ease;
    }}
    .ea-legend-wide {{
        padding:14px 16px; font-size:13px;
        border-width: 1.5px;
    }}
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


def build_sidebar(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Bouwt de volledige sidebar en retourneert:
      - df (gefilterd zoals in de monolith)
      - ui (dict met alle gekozen waarden)
    """
    st.session_state.setdefault("grenswaarde_input", 100)
    if "participatie" not in st.session_state:
        st.session_state["participatie"] = 80
    if "LAYER_CFG" not in st.session_state:
        st.session_state.LAYER_CFG = LAYER_CFG
    if "BASEMAP_CFG" not in st.session_state:
        st.session_state.BASEMAP_CFG = BASEMAP_CFG

    ui: Dict[str, Any] = {}

    with st.sidebar:
        st.header("Opties")

        # ---------------- Kaart ----------------
        with st.expander("Kaart", expanded=True):
            # ui["hide_basemap"] = st.toggle("Geen achtergrondkaart", value=False, key="hide_basemap")

            #basemap_keys = list(st.session_state.BASEMAP_CFG.keys())
            #default_key = "light" if "light" in st.session_state.BASEMAP_CFG else basemap_keys[0]
            #ui["map_style"] = st.selectbox(
            #    "Kies een kaartstijl:",
            #    options=basemap_keys,
            #    index=basemap_keys.index(default_key),
            #    format_func=lambda k: st.session_state.BASEMAP_CFG[k]["title"],
            #    key="map_style"
            #)

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
                    "- **11 en 12**: Straatniveau. Vanaf dit niveau kan de volledige dataset worden gefilterd op woonplaats. \n\n"
                    "Deze zoomniveaus zijn gebaseerd op de documentatie van [Mapbox](https://docs.mapbox.com/help/glossary/zoom-level/)."
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
                c = LAYER_CFG["energiearmoede"]; colors = legend_colors = None
                colors = legend_colors = None
                colors = legend_colors = None  # safeguard
                colors = colors or legend_colors  # noop
                colors = legend_colors = None  # reset
                colors = legend_colors = None
                colors = legend_colors = None
                # echte call:
                colors = legend_colors = None
                # kort en goed:
                c = LAYER_CFG["energiearmoede"]; colors = None
                from core.utils import get_layer_colors
                colors = get_layer_colors(c)
                labels = legend_labels_from_breaks(c["breaks"])
                render_mini_legend(c["legend_title"], colors, labels)

            show_koopwoningen = st.toggle("Koopwoningen", value=False, key=LAYER_CFG["koopwoningen"]["toggle_key"])
            if show_koopwoningen:
                c = LAYER_CFG["koopwoningen"]; from core.utils import get_layer_colors
                colors = get_layer_colors(c)
                labels_kw = legend_labels_from_breaks(c["breaks"])
                render_mini_legend(c["legend_title"], colors, labels_kw)

            show_corporatie = st.toggle("Wooncorporatie", value=False, key=LAYER_CFG["wooncorporatie"]["toggle_key"])
            if show_corporatie:
                c = LAYER_CFG["wooncorporatie"]; from core.utils import get_layer_colors
                colors = get_layer_colors(c)
                labels_wc = legend_labels_from_breaks(c["breaks"])
                render_mini_legend(c["legend_title"], colors, labels_wc)

            ui["extra_opacity"] = st.slider("Transparantie woonlagen", min_value=0.1, max_value=1.0, value=0.55, key="extra_opacity")

            # Bodemlagen
            st.subheader("Bodemlagen")
            ui["show_spoorlaag"] = st.toggle("Spoorlaag", value=False, key=LAYER_CFG["spoordeel"]["toggle_key"])
            ui["show_waterlaag"] = st.toggle("Waterlaag", value=False, key=LAYER_CFG["waterdeel"]["toggle_key"])

            ui["spoor_opacity"] = st.slider("Transparantie spoorlaag", 0.1, 1.0, 0.5, key="spoor_opacity")
            ui["water_opacity"] = st.slider("Transparantie waterlaag", 0.1, 1.0, 0.6, key="water_opacity")

            _render_bodem_legend(ui["show_spoorlaag"], ui["show_waterlaag"])

        # ---------------- Filters ----------------
        with st.expander("Filters", expanded=False):
            # Woonplaats
            st.subheader("Woonplaats")
            df = df_in.copy()
            woonplaatsen = df["woonplaats"].dropna().unique()
            if 1 <= ui["zoom_level"] <= 10:
                friesland_woonplaatsen = df["woonplaats"].unique()
                df = df[df["woonplaats"].isin(friesland_woonplaatsen)]
                woonplaats_selectie = friesland_woonplaatsen.tolist()
            else:
                woonplaats_selectie = st.multiselect(
                    "Filter op woonplaats:",
                    options=sorted(woonplaatsen),
                    default=["Leeuwarden"]
                )
                if not woonplaats_selectie:
                    st.warning("Selecteer minimaal één woonplaats.")
                    woonplaats_selectie = ["Leeuwarden"]
                df = df[df["woonplaats"].isin(woonplaats_selectie)]
            ui["woonplaats_selectie"] = woonplaats_selectie

            # Energieklasse
            st.subheader("Energieklasse")
            df = _fillna_categorical(df, "Energieklasse", "Onbekend")
            energieklassen = df["Energieklasse"].unique()
            energieklasse_selectie = st.multiselect(
                "Filter op energieklasse:",
                options=sorted([str(x) for x in energieklassen]),
                default=[str(x) for x in energieklassen]
            )
            if not energieklasse_selectie:
                energieklasse_selectie = [str(x) for x in energieklassen]
            df = df[df["Energieklasse"].astype(str).isin(energieklasse_selectie)]
            ui["energieklasse_selectie"] = energieklasse_selectie

            # Bouwjaar
            st.subheader("Bouwjaar")
            min_year = int(pd.to_numeric(df["bouwjaar"], errors="coerce").min())
            max_year = int(pd.to_numeric(df["bouwjaar"], errors="coerce").max())
            bouwjaar_range = st.slider("Filter op bouwjaar:", min_year, max_year, (min_year, max_year))
            df = df[(pd.to_numeric(df["bouwjaar"], errors="coerce") >= bouwjaar_range[0]) &
                    (pd.to_numeric(df["bouwjaar"], errors="coerce") <= bouwjaar_range[1])]
            ui["bouwjaar_range"] = bouwjaar_range

            # Type pand
            st.subheader("Type pand")
            df = _fillna_categorical(df, "Dataset", "Onbekend")
            typepand = sorted([str(x) for x in df["Dataset"].unique()])
            typepand_opties = ["Alle types"] + typepand
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
        with st.sidebar.expander("Collectieve warmtevoorziening (analyse)", expanded=False):
            ui["show_sites_layer"] = st.toggle("Warmtevoorzieningen", value=False, key="show_sites_layer")
            ui["kring_radius"] = st.slider("Bereik van de warmtevoorziening", 1, 20, 8, 1, key="kring_radius")
            ui["min_sep"] = st.slider("Minimale afstand tussen warmtevoorzieningen", 1, 20, 8, 1, key="min_sep")
            ui["n_sites"] = st.number_input("Aantal collectieve warmtevoorzieningen", min_value=1, max_value=25, value=8, step=1, key="n_sites")

            ui["cap_mwh"] = text_input_int("Capaciteit per voorziening (MWh)", key="cap_mwh", default=100_000)
            ui["cap_buildings"] = text_input_int("Max gebouwen per voorziening", key="cap_buildings", default=1_000)
            ui["fixed_cost"] = text_input_int("Vaste kosten per locatie (€)", key="fixed_cost", default=25_000)
            ui["var_cost"] = text_input_int("Variabele kosten (€ per MWh)", key="var_cost", default=35)

            ui["opex_pct"] = st.number_input("Extra operationele kosten (% van vaste kosten)", min_value=0, max_value=100, value=10, step=1, key="opex_pct")

        # ---------------- Uitleg-blokken ----------------
        st.header("Uitleg")
        with st.expander("Uitleg H3", expanded=False):
            st.write("H3 is een hexagonaal raster dat gebieden verdeelt in zeshoeken van verschillende resoluties. "
                     "Elke hexagoon krijgt een unieke ID en bevat gegevens over de warmtebehoefte.")

        with st.expander("Uitleg analyse", expanded=False):
            st.markdown("""
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
            """)

    return df, ui
