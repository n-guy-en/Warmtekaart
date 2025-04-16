import gdown
import streamlit as st
import pydeck as pdk
import pandas as pd
import h3
# # Download the file (e.g., 'data_KJ.csv') to the current directory
# *** Data laden met caching ***
@st.cache_data
def load_data():
    # file_path = "data_KJ - kopie.csv"
    # file_id = "1VDo2SGNtrtZR32XjAWQI4uKBa8zm9nxA"
    file_id = "1HAXE90xiqEShkCRhG92q_BC5Ie-wBfTh"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    output = "data_kWh.csv"
    gdown.download(url, output, quiet=False)

    df = pd.read_csv(output)
    # df = pd.read_csv(file_path)
    return df

df = load_data()

# Manual formatter
def format_dutch_number(num, decimals=2):
    if isinstance(num, int):
        return f"{num:,}".replace(",", ".")
    return f"{num:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")

def show_dutch_metric(value):
    formatted = format_dutch_number(value)
    st.metric(formatted)

# Alleen relevante kolommen inladen
df = df[["kWh_per_m2", "oppervlakte", "woonplaats", "Energieklasse", "huisnummer", "huisletter", "postcode", 
         "openbare_ruimte", "latitude", "longitude", "bouwjaar", "Energiebehoefte", "AandeelHernieuwbareEnergie",
         "Warmtebehoefte", "BerekendeCO2Emissie", "BerekendeEnergieverbruik", "PrimaireFossieleEnergie", "pandstatus"]]

df = df[df["pandstatus"] == "Pand in gebruik"]

#%%
# *** Dynamische H3-resolutie bepalen op basis van zoomniveau ***
def get_dynamic_resolution(zoom_level):
    return zoom_level  # Directe koppeling van zoomniveau aan resolutie

def get_dynamic_line_width(zoom_level):
    if zoom_level >= 15:
        return 0  # Verwijder de lijn bij hogere zoomniveaus
    elif 12 <= zoom_level <= 14:
        return 0.5
    elif 8 <= zoom_level < 12:
        return 8.5
    elif 7 <= zoom_level < 8:
        return 20
    else:
        return 50

# *** Kleurmapping ***
colorbrewer_colors = [
    [69, 117, 180, 255], # Donkerblauw (Geen potentie)
    [254, 224, 144, 255], # Lichtoranje (Goede potentie)
    [215, 48, 39, 255] # Rood (Zeer hoge potentie)
]

def get_color(value):
    bins = [10, 50]
    for i, threshold in enumerate(bins):
        if value < threshold:
            return colorbrewer_colors[i]
    return colorbrewer_colors[-1]  # For values >= 50

df["color"] = df["kWh_per_m2"].apply(get_color)

# *** Berekening H3 hexagoon in km ***
# Functie om de hexagon grootte te bepalen
def get_hexagon_size(zoom_level):
    hexagon_sizes = {
        1: 5000, 2: 2500, 3: 1500, 4: 700, 5: 350, 6: 175, 7: 90,
        8: 35, 9: 17, 10: 8, 11: 4, 12: 2, 13: 1, 14: 0.5, 15: 0.2
    }
    return hexagon_sizes.get(zoom_level, 8)  # Default naar zoomniveau 10

# *** Streamlit UI ***
st.markdown('<h1 style="font-size: 35px;">Friese Warmtevraagkaart (Heat Demand)</h1>', unsafe_allow_html=True)

with st.expander("ℹ Wat is een H3 hexagoon?"):
    st.write("H3 is een hexagonaal raster dat gebieden verdeelt in zeshoeken van verschillende resoluties. "
             "Elke hexagoon krijgt een unieke ID en bevat gegevens over de warmtebehoefte.")

# *** Sidebar opties ***
st.sidebar.header("Opties")

# *** Kies een kaartstijl ***
map_style = st.sidebar.selectbox(
    "Kies een kaartstijl:",
    options=["light", "dark", "streets", "outdoors", "satellite", "satellite-streets"],
    format_func=lambda x: x.capitalize()
)

map_style_url = f"mapbox://styles/mapbox/{map_style}-v9"

# *** Zoom slider ***
zoom_level = st.sidebar.slider("Selecteer zoomniveau", 9, 12, 10)
resolution = get_dynamic_resolution(zoom_level)
hexagon_size = get_hexagon_size(zoom_level)

# Directe feedback onder de slider
st.sidebar.markdown(f'<span style="font-size: smaller;">📍 Bij <b>zoomniveau {zoom_level}</b> is de kaart <b>ongeveer {hexagon_size} km breed</b>.</span>', unsafe_allow_html=True)

with st.sidebar.expander("ℹ Uitleg over zoomniveau"):
    st.write(
        "Het zoomniveau bepaalt de mate van detail op de kaart:\n"
        "- **9 en 10**: Specifieke buurten en industriegebieden zijn herkenbaar voor heel Friesland. \n"
        "- **11 en 12**: Straatniveau. Vanaf dit niveau kan de volledige dataset worden gefilterd op woonplaats. \n\n"
        "Deze zoomniveaus zijn gebaseerd op de documentatie van [Mapbox](https://docs.mapbox.com/help/glossary/zoom-level/)."
    )

# *** Woonplaatsen ***
# Unieke woonplaatsen ophalen
woonplaatsen = df["woonplaats"].dropna().unique()

# Bij zoomniveau 1-10: automatisch alle woonplaatsen in Friesland selecteren
if 1 <= zoom_level <= 10:
    # Alle woonplaatsen in Friesland
    friesland_woonplaatsen = df["woonplaats"].unique()
    df = df[df["woonplaats"].isin(friesland_woonplaatsen)]
    # Stel de woonplaats_selectie in als de lijst van woonplaatsen in Friesland
    woonplaats_selectie = friesland_woonplaatsen.tolist()
else:
    woonplaats_selectie = st.sidebar.multiselect(
        "Filter op woonplaats:",
        options=sorted(woonplaatsen),
        default=["Leeuwarden"]
    )

    # Zorg ervoor dat er altijd minimaal één woonplaats is geselecteerd
    if not woonplaats_selectie:
        st.warning("Selecteer minimaal één woonplaats.")
        # Terugvallen op standaardwaarde
        woonplaats_selectie = ["Leeuwarden"] 

df = df[df["woonplaats"].isin(woonplaats_selectie)]

# *** Grenswaarde voor kWh_per_m2 (indicatieve aandachtsgebieden) ***
grenswaarde = st.sidebar.number_input(
    "Stel de minimale grenswaarde (threshold) in per kWh/m²:",
    min_value=0,
    value=100,  # Standaard waarde
    step=1
)

# *** Energieklasse ***
df["Energieklasse"] = df["Energieklasse"].fillna("Onbekend")  # Vervang NaN door 'Onbekend'
energieklassen = df["Energieklasse"].unique()

energieklasse_selectie = st.sidebar.multiselect(
    "Filter op energieklasse:",
    options=sorted(energieklassen),
    default=energieklassen.tolist()  # Standaard selectie van **alle** energieklassen
)

# Als er geen energieklasse is geselecteerd (lege lijst), tonen we alle energieklassen
if not energieklasse_selectie:
    energieklasse_selectie = energieklassen.tolist()

# Data filteren op geselecteerde energieklassen
df = df[df["Energieklasse"].isin(energieklasse_selectie)]

# *** Bouwjaar ***
# Dynamically get the min and max year from the dataset
min_year = int(df["bouwjaar"].min())
max_year = int(df["bouwjaar"].max())

# Create a slider in the sidebar for users to filter by bouwjaar
bouwjaar_range = st.sidebar.slider(
    "Filter op bouwjaar:", 
    min_year, max_year, (min_year, max_year)
)

# Apply filter based on bouwjaar range
df = df[(df["bouwjaar"] >= bouwjaar_range[0]) & (df["bouwjaar"] <= bouwjaar_range[1])]

# *** 2D/3D-weergave ***
extruded = st.sidebar.toggle("3D Weergave", value=False)

# Uitleg over 3D-weergave
with st.sidebar.expander("ℹ Uitleg over 3D-weergave"):
    st.write(
        "Wanneer 3D-weergave is ingeschakeld, wordt de hoogte van de hexagonen aangepast op basis van het warmtegebruik.\n"
        "Dit helpt bij het visualiseren van gebieden met hogere of lagere energiebehoefte."
    )

# *** Maak alleen de kaart wanneer 'Maak Kaart' knop wordt gedrukt ***
# Initieel instellen van session_state voor kaartstatus
if "show_map" not in st.session_state:
    st.session_state.show_map = False

if "prev_filters" not in st.session_state:
    st.session_state.prev_filters = {
        "zoom_level": zoom_level,
        "woonplaats": woonplaats_selectie,
        "Energieklasse": energieklasse_selectie,
        "threshold": grenswaarde
    }

# **Waarschuwingslogica en uitzetten van de kaart direct bij wijzigingen in de filters**
filters_changed = (
    zoom_level != st.session_state.prev_filters["zoom_level"] or
    woonplaats_selectie != st.session_state.prev_filters["woonplaats"] or
    energieklasse_selectie != st.session_state.prev_filters["Energieklasse"]
)

if filters_changed:
    # Als filters zijn veranderd, zet de kaart uit
    st.session_state.show_map = False
    # Sla de huidige filters op
    st.session_state.prev_filters = {
        "zoom_level": zoom_level,
        "woonplaats": woonplaats_selectie,
        "Energieklasse": energieklasse_selectie
    }
    # Waarschuwing tonen
    st.warning("De filters zijn gewijzigd. Klik op 'Maak Kaart' om de kaart bij te werken.")

# **Knop voor het daadwerkelijk maken van de kaart**
if st.sidebar.button("Maak Kaart"):
    # Bij drukken van de knop wordt de kaart ingeschakeld
    st.session_state.show_map = True

# **Als de kaart moet worden weergegeven**
if st.session_state.show_map:
    # Data filtering per zoomniveau
    if zoom_level <= 3:
        df_filtered = df.sample(frac=0.05)
    elif zoom_level <= 7:
        df_filtered = df.sample(frac=0.20)
    elif zoom_level <= 11:
        df_filtered = df
    else:
        df_filtered = df

    # *H3 Index berekenen na filtering*
    df_filtered["h3_index"] = df_filtered.apply(lambda row: h3.latlng_to_cell(row["latitude"], row["longitude"], resolution), axis=1)

    # Opslaan van extra info voordat je groepeert
    df_extra_info = df_filtered[["h3_index", "woonplaats", "postcode", "openbare_ruimte", "huisnummer", "huisletter", 
                        "Energieklasse", "Energiebehoefte", "AandeelHernieuwbareEnergie",
                        "Warmtebehoefte", "BerekendeCO2Emissie", "BerekendeEnergieverbruik", "PrimaireFossieleEnergie"]].drop_duplicates(subset=["h3_index"])

    # Groeperen per H3-index: gemiddeld energieverbruik en oppervlakte berekenen
    df_filtered = df_filtered.groupby("h3_index").agg({
        "kWh_per_m2": "mean",  # Gemiddelde energieverbruik per hexagon
        "oppervlakte": "mean",  # Totale oppervlakte per hexagon
        "h3_index": "count", # Aantal panden in de hexagoon
        "bouwjaar": "mean" # Gemiddelde bouwjaar
    }).rename(columns={"h3_index": "aantal_huizen"}).reset_index()

    # Rond de gemiddelde waarde af op 1 decimaal
    df_filtered["kWh_per_m2"] = df_filtered["kWh_per_m2"].round(0)
    df_filtered["oppervlakte"] = df_filtered["oppervlakte"].round(0)
    df_filtered["bouwjaar"] = df_filtered["bouwjaar"].round(0)

    # Nieuwe kleur bepalen op basis van het gemiddelde energieverbruik
    df_filtered['color'] = df_filtered["kWh_per_m2"].apply(get_color)

    # Schaalvergroting aanpassen
    MAX_HEIGHT = 3500
    df_filtered["scaled_elevation"] = (df_filtered["kWh_per_m2"] - 10) / (50 - 10) * MAX_HEIGHT
    df_filtered["scaled_elevation"] = df_filtered["scaled_elevation"].clip(lower=0, upper=MAX_HEIGHT)

    # Merge extra info terug
    df_filtered = df_filtered.merge(df_extra_info, on="h3_index", how="left")

    # *Gebruik alleen de nodige kolommen*
    df_filtered = df_filtered[["h3_index", 
                    "kWh_per_m2", 
                    "color", 
                    "woonplaats", 
                    "aantal_huizen",
                    "huisnummer", 
                    "scaled_elevation", 
                    "oppervlakte", 
                    "postcode", 
                    "Energieklasse", 
                    "openbare_ruimte", 
                    "huisletter",
                    "bouwjaar",
                    "Energiebehoefte",
                    "AandeelHernieuwbareEnergie",
                    "Warmtebehoefte",
                    "BerekendeCO2Emissie",
                    "BerekendeEnergieverbruik",
                    "PrimaireFossieleEnergie"
                    ]]
    
    # *** Filter op basis van grenswaarde ***
    # Voeg een nieuwe kolom toe die aanduidt of een hexagoon boven de grenswaarde ligt
    df_filtered_area = df_filtered.copy()
    df_filtered_area["indicatief_aandachtsgebied"] = df_filtered_area["kWh_per_m2"] > grenswaarde

    # Laag voor indicatieve aandachtsgebieden
    def create_indicative_area_layer():
        return pdk.Layer(
            "H3HexagonLayer",
            df_filtered_area[df_filtered_area["indicatief_aandachtsgebied"] == True],
            pickable=True,
            filled=True,
            extruded=extruded,
            get_hexagon="h3_index",
            get_fill_color=[26, 152, 80],
            get_line_color=[0, 0, 0, 0],
            get_line_width=get_dynamic_line_width(zoom_level),
            visible=True
        )

    # Functie om de H3 laag aan te maken
    def create_layer(visible, elevation_scale):
        return pdk.Layer(
            "H3HexagonLayer",
            df_filtered,
            pickable=True,
            filled=True,
            extruded=extruded,
            coverage=1,
            get_hexagon="h3_index",
            get_fill_color="color",
            get_elevation="scaled_elevation",
            elevation_scale=elevation_scale if extruded else 0,
            elevation_range=[0, 800.0],
            get_line_width=get_dynamic_line_width(zoom_level), 
            visible=visible,
        )

    # Functie om de lagen dynamisch afhankelijk van het zoomniveau te maken
    def create_layers(df_filtered, zoom_level, extruded):
        layers = []
        if zoom_level <= 3:
            layers.append(create_layer(True, 0.01))
        if 4 <= zoom_level <= 7:
            layers.append(create_layer(True, 0.05))
        if 8 <= zoom_level <= 11:
            layers.append(create_layer(True, 0.08))
        if zoom_level >= 12:
            layers.append(create_layer(True, 0.10))
        return layers
    
    # Maak de lagen dynamisch afhankelijk van het zoomniveau
    layers = create_layers(df_filtered, zoom_level, extruded)
    layers.append(create_indicative_area_layer())

    # *ViewState correct bijwerken*
    # Pas de zoomniveau aan zoals aangegeven
    # if 14 <= zoom_level <= 15:
    #     adjusted_zoom_level = 18  # Gebruik altijd zoomniveau 18 voor zoomniveaus tussen 12 en 15
    # else:
    #     adjusted_zoom_level = zoom_level  # Gebruik het werkelijke zoomniveau voor zoomniveaus boven 15

    st.session_state.view_state = pdk.ViewState(
        longitude=df["longitude"].mean(),
        latitude=df["latitude"].mean(),
        zoom=zoom_level,  
        min_zoom=1,  
        max_zoom=18,  
        pitch=40.5,
        bearing=0,
    )

    # *Tooltip op de kaart*
    if zoom_level >= 11:
        tooltip_html = """
            <b>Postcode:</b> {postcode}<br>
            <b>Woonplaats:</b> {woonplaats}<br>
            <b>Aantal panden:</b> {aantal_huizen}<br>
            <b>Gemiddelde Energiegebruik:</b> {kWh_per_m2} kWh/m²<br>
            <b>Gemiddelde Oppervlakte:</b> {oppervlakte} m²<br>
            <b>Gemiddelde Bouwjaar:</b> {bouwjaar} <br>
            # Toevoegen totale heatdemand van de gegroepeerde panden in de h3-index
        """
    else:
        tooltip_html = """
            <b>Postcode:</b> {postcode}<br>
            <b>Woonplaats:</b> {woonplaats}<br>
            <b>Aantal panden:</b> {aantal_huizen}<br>
            <b>Gemiddelde Energiegebruik:</b> {kWh_per_m2} kWh/m²<br>
            <b>Gemiddelde Oppervlakte:</b> {oppervlakte} m²<br>
            <b>Gemiddelde Bouwjaar:</b> {bouwjaar} <br>
            # Toevoegen totale heatdemand van de gegroepeerde panden in de h3-index
        """

    tooltip = {
        "html": tooltip_html,
        "style": {
            "backgroundColor": "white",
            "color": "black",
            "font-family": "Arial",
            "padding": "5px",
            "border-radius": "5px"
        }
    }

    # *Tegels*
    # Aantal huizen berekenen
    totaal_aantal_huizen = df_filtered["aantal_huizen"].sum()
    totaal_aantal_huizen = format_dutch_number(totaal_aantal_huizen, decimals=0)

    # Totale heat demand berekenen
    totaal_heat_demand = df_filtered["kWh_per_m2"].mean()
    totaal_heat_demand = format_dutch_number(totaal_heat_demand, decimals=0)

    # CSS voor de tegel
    st.markdown("""
        <style>
        .tile-container {
            display: flex;
            gap: 10px; /* Gap between the tiles */
            margin-top: 0px;  /* Adjust gap between title and tiles */
        }

        .tile {
            background-color: #f0f4f8; 
            padding: 10px 30px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            width: 100%;
            text-align: center;
            font-family: 'Arial', sans-serif;
            border: 1px solid #dcdcdc; /* Neutral grey border */
            margin-bottom: 20px;
        }

        .tile h2 {
            margin: 0;
            font-size: 32px;
            color: #333333; /* Dark text for contrast */
        }

        .tile p {
            margin: 5px 0 0;
            font-size: 16px;
            color: #7f8c8d; /* Light grey text for paragraph */
        }

        </style>
    """, unsafe_allow_html=True)


    # HTML layout voor twee tegels
    st.markdown(f"""
        <div class="tile-container">
            <div class="tile">
                <h2>{totaal_aantal_huizen}</h2>
                <p>Aantal panden</p>
            </div>
            <div class="tile">
                <h2>{totaal_heat_demand} kWh/m²</h2>
                <p>Totale Head Demand</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=st.session_state.view_state,
            map_style=map_style_url,
            tooltip=tooltip 
        )
    )

    # *Legenda*
    legend_html = f"""
        <style>
            .legend {{
                position: absolute;
                bottom: 10px;
                left: 10px;
                width: 220px;
                background: white;
                padding: 10px;
                border-radius: 5px;
                font-family: Arial, sans-serif;
                font-size: 12px;
                color: black;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
            }}
            .legend-title {{
                font-weight: bold;
                margin-bottom: 5px;
            }}
            .color-box {{
                width: 15px;
                height: 15px;
                display: inline-block;
                margin-right: 5px;
            }}
        </style>
        <div class="legend">
            <div class="legend-title">Gemiddelde Warmtevraag in kWh/m²</div>
            <div><span class="color-box" style="background-color: #4575b4;"></span> &lt; 10,0 </div>
            <div><span class="color-box" style="background-color: #fee090;"></span> 10,0 - 50,0 </div>
            <div><span class="color-box" style="background-color: #d73027;"></span> &gt; 50,0 </div>
            <div><span class="color-box" style="background-color: #1a9850;"></span> Potentie grenswaarde: {grenswaarde} </div>
        </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)