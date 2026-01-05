import streamlit as st
import pandas as pd
import requests
import zipfile
import io
import pydeck as pdk
from datetime import datetime, timedelta

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Geopolitical Conflict Heat Map",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 Geopolitical Global Conflict Heat Map (By Country)")
st.markdown("""
**Data Source:** Direct CSV Downloads from `data.gdeltproject.org`.
**Logic:** Aggregating total negative impact ("Heat") by **Actor 2's Country Code**.
**Metric:** Sum of Scores for all negative events in the country.
""")

# 2. HELPER: GENERATE URLS
def generate_gdelt_urls(hours_back=1):
    base_url = "http://data.gdeltproject.org/gdeltv2/"
    urls = []
    
    # Round down to nearest 15 minutes
    now = datetime.utcnow()
    current_time = now - timedelta(minutes=15) 
    current_time = current_time.replace(second=0, microsecond=0)
    discard = current_time.minute % 15
    current_time -= timedelta(minutes=discard)
    
    steps = int(hours_back * 4) 
    for _ in range(steps):
        timestamp = current_time.strftime("%Y%m%d%H%M%S")
        url = f"{base_url}{timestamp}.export.CSV.zip"
        urls.append(url)
        current_time -= timedelta(minutes=15)
        
    return urls

# 3. CORE: DOWNLOAD & PROCESS
@st.cache_data(ttl=900) 
def load_gdelt_data(hours):
    urls = generate_gdelt_urls(hours)
    
    # --- MAPPING DICTIONARY ---
    # We still need Lat/Lon (48/49) momentarily to calculate the country's center point
    raw_mapping = {
        0:  'GlobalEventID',
        1:  'Day',
        6:  'Actor1Name',
        37: 'Actor1GeoCountry',  
        16: 'Actor2Name',
        45: 'Actor2GeoCountry',
        48: 'Lat',               # Used only to calculate country centroid
        49: 'Lon',               # Used only to calculate country centroid
        26: 'EventCode',       
        30: 'Goldstein',
        31: 'NumMentions',
        33: 'NumArticles',
        34: 'AvgTone',
        60: 'SourceURL'
    }
    
    sorted_mapping = dict(sorted(raw_mapping.items()))
    
    use_cols = list(sorted_mapping.keys())
    col_names_ordered = list(sorted_mapping.values())
    
    master_df = pd.DataFrame()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, url in enumerate(urls):
        status_text.text(f"Downloading batch {i+1}/{len(urls)}: {url.split('/')[-1]}...")
        progress_bar.progress((i + 1) / len(urls))
        
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    filename = z.namelist()[0]
                    with z.open(filename) as f:
                        df_chunk = pd.read_csv(
                            f, 
                            sep='\t', 
                            header=None, 
                            usecols=use_cols,     
                            names=col_names_ordered,
                            dtype=str
                        )
                        master_df = pd.concat([master_df, df_chunk], ignore_index=True)
        except Exception as e:
            continue
            
    progress_bar.empty()
    status_text.empty()
    
    if master_df.empty:
        return master_df
        
    # --- DATA TYPE CONVERSION ---
    numeric_cols = ['Goldstein', 'NumArticles', 'AvgTone', 'Lat', 'Lon']
    for col in numeric_cols:
        master_df[col] = pd.to_numeric(master_df[col], errors='coerce').fillna(0)

    master_df['EventDate'] = pd.to_datetime(master_df['Day'], format='%Y%m%d', errors='coerce')

    # --- FILTERING ---
    target_codes = [
        '1012','1014','102','1032','1033','1034','1041','1042','1043','1044','1052','1054','1055','1056','106','107','108','111','1121','1122','1123','1124','1125','113','114','115','116','121','1211','1212','122','1221','1222','1223','1224','123','1231','1232','1233','1234','124','1241','1242','1243','1244','1245','1246','125','126','127','128','129','130','131','1311','1312','1313','132','1321','1322','1323','1324','133','134','135','136','137','138','1381','1382','1383','1384','1385','139','140','141','1411','1412','1413','1414','142','1421','1422','1423','1424','143','1431','1432','1433','1434','144','1441','1442','1443','1444','145','1451','1452','1453','1454','150','151','152','153','154','155','160','161','162','1621','1622','1623','163','164','165','166','1661','1662','1663','1712','1721','1722','1723','1724','174','175','180','181','182','1821','1822','1823','183','1831','1832','1833','1834','184','185','186','190','191','192','193','194','195','1951','1952','196','200','201','202','203','204','2041','2042'
    ]
    master_df = master_df[master_df['EventCode'].isin(target_codes)]
    
    master_df['Score'] = (
        master_df['AvgTone'] * master_df['Goldstein'] * master_df['NumArticles']
    )
    
    # Ensure valid coordinates for aggregation
    master_df = master_df.dropna(subset=['Lat', 'Lon'])
    master_df = master_df[(master_df['Lat'] != 0) & (master_df['Lon'] != 0)]
    
    # Ensure Country Code is valid
    master_df = master_df[master_df['Actor2GeoCountry'].notna()]
    
    return master_df

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Data Settings")
data_range = st.sidebar.slider("Time Window (Hours)", min_value=1, max_value=72, value=1)

if st.sidebar.button("🔄 Refresh Data Now"):
    st.cache_data.clear()
    st.rerun()

# --- LOAD DATA ---
with st.spinner("Processing GDELT Event Stream..."):
    df = load_gdelt_data(data_range)

if df.empty:
    st.error("No data retrieved.")
    st.stop()

# --- PRE-PROCESSING ---
# 1. Filter Negative
filtered_df = df[
    (df['Goldstein'] < 0) & 
    (df['AvgTone'] < 0)
].copy()

# 2. Deduplicate URLs
filtered_df = filtered_df.drop_duplicates(subset=['SourceURL'])

# 3. AGGREGATION: Report Heat by Country
# We group by Actor2GeoCountry to get one row per country.
# We average Lat/Lon to find the "center" of activity to plot on the map.
country_df = filtered_df.groupby('Actor2GeoCountry').agg({
    'Score': 'sum',          # Total Heat (Sum of all negative scores)
    'NumArticles': 'sum',    # Total Articles
    'Lat': 'mean',           # Centroid Latitude
    'Lon': 'mean',           # Centroid Longitude
    'Actor2Name': 'count'    # Count of Events
}).rename(columns={'Actor2Name': 'EventCount'}).reset_index()

# Make Score positive for visualization sizing (Magnitude of Heat)
country_df['HeatIntensity'] = country_df['Score'].abs()
country_df = country_df.sort_values('HeatIntensity', ascending=False)

# --- VISUALIZATION ---
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"🗺️ Global Heat Map ({len(country_df)} Countries Active)")
    
    if not country_df.empty:
        # Scale radius based on intensity
        # We normalize to make sure circles are visible but not covering the whole map
        max_heat = country_df['HeatIntensity'].max()
        if max_heat == 0: max_heat = 1
        
        country_df['Radius'] = (country_df['HeatIntensity'] / max_heat) * 500000 + 50000
        
        map_data = country_df.to_dict(orient='records')
        
        # Layer 1: Heat Circles
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_data,
            get_position='[Lon, Lat]',
            get_radius='Radius',
            get_fill_color=[255, 50, 50, 140], # Red transparent
            get_line_color=[255, 255, 255],
            pickable=True,
            filled=True,
            stroked=True,
            line_width_min_pixels=1
        )
        
        # Layer 2: Country Labels
        text_layer = pdk.Layer(
            "TextLayer",
            data=map_data,
            get_position='[Lon, Lat]',
            get_text='Actor2GeoCountry',
            get_color=[255, 255, 255],
            get_size=12,
            get_alignment_baseline="'center'"
        )

        tooltip = {
           "html": "<b>Country: {Actor2GeoCountry}</b><br/>Total Heat: {Score}<br/>Events: {EventCount}<br/>Articles: {NumArticles}",
           "style": {"backgroundColor": "steelblue", "color": "white", "fontSize": "12px"}
        }

        st.pydeck_chart(pdk.Deck(
            layers=[scatter_layer, text_layer],
            initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=1),
            map_style=pdk.map_styles.CARTO_DARK,
            tooltip=tooltip
        ))
    else:
        st.warning("No negative events found in the selected timeframe.")

with col2:
    st.subheader("🔥 Hottest Countries")
    
    if not country_df.empty:
        # Display aggregated stats
        top_countries = country_df.head(15)
        for idx, row in top_countries.iterrows():
            st.metric(
                label=f"{row['Actor2GeoCountry']}",
                value=f"{row['Score']:.0f}",
                delta=f"{row['EventCount']} Events"
            )
            st.divider()

with st.expander("📂 Aggregated Country Data"):
    st.dataframe(country_df)