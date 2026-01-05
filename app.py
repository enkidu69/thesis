import streamlit as st
import pandas as pd
import requests
import zipfile
import io
import pydeck as pdk
from datetime import datetime, timedelta

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Geopolitical Conflict Map",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed" # Default to collapsed on mobile
)

# --- NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Global Conflict Map", "Keyword News Search"])

# ==============================================================================
# PAGE 1: GLOBAL CONFLICT MAP
# ==============================================================================
if page == "Global Conflict Map":

    st.title("🔥 Global Conflict Heat Map")
    
    # Hide details in an expander for mobile cleanliness
    with st.expander("ℹ️ About this Map"):
        st.markdown("""
        **Logic:** Aggregating negative impact by **Actor 2's Country Code**.
        **Location:** Coordinates are determined strictly by a **static lookup of the Country Code**.
        **Filters:** Event Codes, Negative Tone, Deduplicated URLs.
        """)

    # --- STATIC DATA: FIPS 10-4 Country Centroids ---
    COUNTRY_CENTROIDS = {
        'AF': [33.0, 65.0], 'AL': [41.0, 20.0], 'AG': [28.0, 3.0], 'AR': [-34.0, -64.0],
        'AS': [-25.0, 134.0], 'AU': [47.3, 13.3], 'AJ': [40.5, 47.5], 'BA': [26.0, 50.5],
        'BG': [24.0, 90.0], 'BE': [50.8, 4.3], 'BL': [-17.0, -65.0], 'BK': [44.0, 18.0],
        'BR': [-10.0, -55.0], 'BU': [43.0, 25.0], 'CA': [60.0, -95.0], 'CH': [35.0, 105.0],
        'CO': [4.0, -72.0], 'CS': [10.0, -84.0], 'HR': [45.1, 15.2], 'CU': [21.5, -80.0],
        'EZ': [50.0, 15.0], 'DA': [56.0, 10.0], 'EG': [27.0, 30.0], 'EN': [59.0, 26.0],
        'ET': [9.0, 40.0], 'FI': [64.0, 26.0], 'FR': [46.0, 2.0], 'GM': [51.0, 9.0],
        'GR': [39.0, 22.0], 'GT': [15.5, -90.25], 'HA': [19.0, -72.25], 'HO': [15.0, -86.5],
        'HU': [47.0, 20.0], 'IN': [20.0, 77.0], 'ID': [-5.0, 120.0], 'IR': [32.0, 53.0],
        'IZ': [33.0, 44.0], 'EI': [53.0, -8.0], 'IS': [31.5, 34.8], 'IT': [42.8, 12.8],
        'JA': [36.0, 138.0], 'JO': [31.0, 36.0], 'KZ': [48.0, 68.0], 'KE': [1.0, 38.0],
        'KN': [40.0, 127.0], 'KS': [37.0, 127.5], 'KU': [29.3, 47.6], 'LG': [57.0, 25.0],
        'LE': [33.8, 35.8], 'LY': [25.0, 17.0], 'LH': [56.0, 24.0], 'MY': [2.5, 112.5],
        'MX': [23.0, -102.0], 'MD': [47.0, 29.0], 'MG': [46.0, 105.0], 'MJ': [42.5, 19.3],
        'MO': [32.0, -5.0], 'BM': [22.0, 98.0], 'NP': [28.0, 84.0], 'NL': [52.5, 5.75],
        'NZ': [-41.0, 174.0], 'NU': [13.0, -85.0], 'NI': [10.0, 8.0], 'NO': [62.0, 10.0],
        'PK': [30.0, 70.0], 'PM': [9.0, -80.0], 'PE': [-10.0, -76.0], 'RP': [13.0, 122.0],
        'PL': [52.0, 20.0], 'PO': [39.5, -8.0], 'QA': [25.3, 51.25], 'RO': [46.0, 25.0],
        'RS': [60.0, 100.0], 'SA': [25.0, 45.0], 'RI': [44.0, 21.0], 'SN': [1.3, 103.8],
        'LO': [48.6, 19.7], 'SI': [46.0, 15.0], 'SF': [-29.0, 24.0], 'SP': [40.0, -4.0],
        'CE': [7.0, 81.0], 'SU': [15.0, 30.0], 'SW': [62.0, 15.0], 'SZ': [47.0, 8.0],
        'SY': [35.0, 38.0], 'TW': [23.5, 121.0], 'TH': [15.0, 100.0], 'TU': [39.0, 35.0],
        'TX': [40.0, 60.0], 'UP': [49.0, 32.0], 'AE': [24.0, 54.0], 'UK': [54.0, -2.0],
        'US': [38.0, -97.0], 'UY': [-33.0, -56.0], 'UZ': [41.0, 64.0], 'VE': [8.0, -66.0],
        'VM': [16.0, 106.0], 'YM': [15.0, 48.0], 'ZI': [-20.0, 30.0],
    }

    # HELPER: GENERATE URLS
    # UPDATED DEFAULT: 12 Hours
    def generate_gdelt_urls(hours_back=12):
        base_url = "http://data.gdeltproject.org/gdeltv2/"
        urls = []
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

    # CORE: DOWNLOAD & PROCESS
    @st.cache_data(ttl=900) 
    def load_gdelt_data(hours):
        urls = generate_gdelt_urls(hours)
        
        # Keys = CSV Indices, Values = Column Names
        raw_mapping = {
            0:  'GlobalEventID', 1:  'Day', 6:  'Actor1Name', 37: 'Actor1GeoCountry',  
            16: 'Actor2Name', 45: 'Actor2GeoCountry', 26: 'EventCode',       
            30: 'Goldstein', 31: 'NumMentions', 33: 'NumArticles', 34: 'AvgTone', 60: 'SourceURL'
        }
        
        sorted_mapping = dict(sorted(raw_mapping.items()))
        use_cols = list(sorted_mapping.keys())
        col_names_ordered = list(sorted_mapping.values())
        
        master_df = pd.DataFrame()
        progress_bar = st.progress(0)
        
        for i, url in enumerate(urls):
            progress_bar.progress((i + 1) / len(urls))
            try:
                r = requests.get(url, timeout=15)
                if r.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                        filename = z.namelist()[0]
                        with z.open(filename) as f:
                            df_chunk = pd.read_csv(f, sep='\t', header=None, usecols=use_cols, names=col_names_ordered, dtype=str)
                            master_df = pd.concat([master_df, df_chunk], ignore_index=True)
            except Exception:
                continue
                
        progress_bar.empty()
        
        if master_df.empty:
            return master_df
            
        numeric_cols = ['Goldstein', 'NumArticles', 'AvgTone']
        for col in numeric_cols:
            master_df[col] = pd.to_numeric(master_df[col], errors='coerce').fillna(0)
            
        master_df['EventDate'] = pd.to_datetime(master_df['Day'], format='%Y%m%d', errors='coerce')
        
        target_codes = [
            '1012','1014','102','1032','1033','1034','1041','1042','1043','1044','1052','1054','1055','1056','106','107','108','111','1121','1122','1123','1124','1125','113','114','115','116','121','1211','1212','122','1221','1222','1223','1224','123','1231','1232','1233','1234','124','1241','1242','1243','1244','1245','1246','125','126','127','128','129','130','131','1311','1312','1313','132','1321','1322','1323','1324','133','134','135','136','137','138','1381','1382','1383','1384','1385','139','140','141','1411','1412','1413','1414','142','1421','1422','1423','1424','143','1431','1432','1433','1434','144','1441','1442','1443','1444','145','1451','1452','1453','1454','150','151','152','153','154','155','160','161','162','1621','1622','1623','163','164','165','166','1661','1662','1663','1712','1721','1722','1723','1724','174','175','180','181','182','1821','1822','1823','183','1831','1832','1833','1834','184','185','186','190','191','192','193','194','195','1951','1952','196','200','201','202','203','204','2041','2042'
        ]
        master_df = master_df[master_df['EventCode'].isin(target_codes)]
        
        master_df['Score'] = (master_df['AvgTone'] * master_df['Goldstein'] * master_df['NumArticles'])
        
        return master_df

    st.sidebar.header("⚙️ Map Settings")
    # UPDATED DEFAULT: Value=12
    data_range = st.sidebar.slider("Time Window (Hours)", 1, 72, 12)

    with st.spinner("Processing GDELT Event Stream..."):
        df = load_gdelt_data(data_range)

    if df.empty:
        st.error("No data retrieved.")
    else:
        # --- FILTERS (Inside Expander for Mobile) ---
        with st.expander("🔍 Filter Actors (Click to Expand)", expanded=True):
            col_f1, col_f2 = st.columns(2)
            
            valid_countries = sorted([str(x) for x in df['Actor1GeoCountry'].unique() if len(str(x)) >= 2])
            valid_countries.insert(0, "All")
            with col_f1:
                a1_select = st.selectbox("Actor 1 (Origin)", valid_countries, index=0)
            
            filtered_df = df.copy()
            if a1_select != "All":
                filtered_df = filtered_df[filtered_df['Actor1GeoCountry'] == a1_select]
                
            valid_a2 = sorted([str(x) for x in filtered_df['Actor2GeoCountry'].unique() if len(str(x)) >= 2])
            valid_a2.insert(0, "All")
            with col_f2:
                a2_select = st.selectbox("Actor 2 (Target)", valid_a2, index=0)

            if a2_select != "All":
                filtered_df = filtered_df[filtered_df['Actor2GeoCountry'] == a2_select]

        # --- PRE-PROCESSING ---
        filtered_df = filtered_df[(filtered_df['Goldstein'] < 0) & (filtered_df['AvgTone'] < 0)]
        filtered_df = filtered_df.drop_duplicates(subset=['SourceURL'])

        def get_lat(code): return COUNTRY_CENTROIDS.get(code, [None, None])[0]
        def get_lon(code): return COUNTRY_CENTROIDS.get(code, [None, None])[1]

        filtered_df['MapLat'] = filtered_df['Actor2GeoCountry'].apply(get_lat)
        filtered_df['MapLon'] = filtered_df['Actor2GeoCountry'].apply(get_lon)
        filtered_df = filtered_df.dropna(subset=['MapLat', 'MapLon'])

        # --- AGGREGATION ---
        country_df = filtered_df.groupby('Actor2GeoCountry').agg({
            'Score': 'sum', 'NumArticles': 'sum', 'MapLat': 'first', 'MapLon': 'first', 'Actor2Name': 'count'
        }).rename(columns={'Actor2Name': 'EventCount'}).reset_index()

        country_df['HeatIntensity'] = country_df['Score'].abs()
        country_df = country_df.sort_values('HeatIntensity', ascending=False)

        # --- VISUALIZATION ---
        
        # 1. THE MAP
        st.subheader("🗺️ Conflict Map")
        if not country_df.empty:
            max_heat = country_df['HeatIntensity'].max()
            if max_heat == 0: max_heat = 1
            # Adjust radius calculation for better mobile visibility
            country_df['Radius'] = (country_df['HeatIntensity'] / max_heat) * 500000 + 80000
            
            map_data = country_df.to_dict(orient='records')
            
            st.pydeck_chart(pdk.Deck(
                layers=[
                    pdk.Layer("ScatterplotLayer", data=map_data, get_position='[MapLon, MapLat]', get_radius='Radius', get_fill_color=[255, 50, 50, 140], pickable=True, filled=True, stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=1),
                    pdk.Layer("TextLayer", data=map_data, get_position='[MapLon, MapLat]', get_text='Actor2GeoCountry', get_color=[255, 255, 255], get_size=12, get_alignment_baseline="'center'")
                ],
                initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=0.5), # Zoom out slightly for mobile context
                map_style=pdk.map_styles.CARTO_DARK,
                # Simplified Tooltip for mobile width
                tooltip={"html": "<div style='width: 150px;'><b>{Actor2GeoCountry}</b><br/>Heat: {Score}<br/>Events: {EventCount}</div>"}
            ), use_container_width=True) # Critical for mobile
        else:
            st.warning("No data matches current filters.")

        st.divider()

        # 2. DATA TABLES (Mobile optimized)
        tab1, tab2 = st.tabs(["🔥 Top Targets", "📂 Raw Events"])
        
        with tab1:
            if not country_df.empty:
                st.dataframe(
                    country_df[['Actor2GeoCountry', 'Score', 'EventCount']].head(20), 
                    hide_index=True, 
                    use_container_width=True # Fills mobile screen width
                )
            else:
                st.write("No data.")
                
        with tab2:
            st.dataframe(
                filtered_df[['EventDate', 'Actor1GeoCountry', 'Actor2GeoCountry', 'Score', 'SourceURL']].head(100),
                use_container_width=True
            )

# ==============================================================================
# PAGE 2: KEYWORD NEWS SEARCH (GDELT DOC API)
# ==============================================================================
elif page == "Keyword News Search":
    st.title("📰 News Search")
    st.caption("Search GDELT GKG (Last 3 months)")

    col_search, col_btn = st.columns([3, 1])
    with col_search:
        keyword = st.text_input("Keyword", "jetstereo", label_visibility="collapsed", placeholder="Enter keyword...")
    with col_btn:
        search_btn = st.button("Search", use_container_width=True)

    if search_btn and keyword:
        api_url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={keyword}&mode=artlist&maxrecords=250&timespan=3months&format=json"
        
        with st.spinner("Searching..."):
            try:
                r = requests.get(api_url)
                if r.status_code == 200:
                    data = r.json()
                    
                    if "articles" in data:
                        articles = data["articles"]
                        st.success(f"Found {len(articles)} articles.")
                        
                        news_df = pd.DataFrame(articles)
                        
                        # Mobile friendly card view
                        for idx, row in news_df.iterrows():
                            # Clean up the domain name
                            domain = row.get('domain', 'Unknown')
                            title = row.get('title', 'No Title')
                            
                            with st.container():
                                st.subheader(title)
                                st.caption(f"{domain} • {row.get('seendate', '')}")
                                if 'socialimage' in row and row['socialimage']:
                                    st.image(row['socialimage'], use_container_width=True)
                                st.link_button("Read Article", row.get('url', '#'))
                                st.divider()
                    else:
                        st.warning("No results.")
                else:
                    st.error(f"API Error: {r.status_code}")
            except Exception as e:
                st.error(f"Connection Error: {e}")