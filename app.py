import streamlit as st
import pandas as pd
import pydeck as pdk
import altair as alt
from datetime import datetime, timedelta
from urllib.parse import urlparse
from newspaper import Article
import nltk
import math
import re
from collections import Counter
from sqlalchemy import create_engine, text

# --- NLTK SETUP ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- DATABASE SETUP ---
DB_URI = 'postgresql://gdelt_admin:SuperSecurePassword123!@localhost:5432/gdelt_db'
engine = create_engine(DB_URI)

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Geopolitical Conflict Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    h1 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; font-size: 2.2rem; }
    
    div[data-testid="stDataFrame"] div[role="grid"] div[role="row"] div[role="gridcell"] {
        white-space: normal !important;
        line-height: 1.5 !important;
        height: auto !important;
        align-items: start !important;
        overflow-wrap: break-word !important;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    
    div.stButton > button {
        height: 2.6rem; 
        width: 100%;
        border-radius: 6px;
    }
    
    footer {display: none !important;}
    header {visibility: hidden;}
    .block-container { padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'time_window_days' not in st.session_state: st.session_state.time_window_days = 1 
if 'origin_country' not in st.session_state: st.session_state.origin_country = "All"
if 'target_country' not in st.session_state: st.session_state.target_country = "All"
if 'selected_country' not in st.session_state: st.session_state.selected_country = None
if 'deep_scan_data' not in st.session_state: st.session_state.deep_scan_data = None
if 'gkg_org_filter' not in st.session_state: st.session_state.gkg_org_filter = ""
if 'selected_gkg_event' not in st.session_state: st.session_state.selected_gkg_event = None

# --- CONSTANTS ---
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

# --- FUNCTIONS ---
def format_url_to_title(url):
    if not isinstance(url, str): return "Unknown Event"
    try:
        parsed = urlparse(url)
        path = parsed.path
        segments = path.split('/')
        slug = max(segments, key=len)
        if len(slug) < 4: return parsed.netloc
        title = slug.replace('-', ' ').replace('_', ' ').replace('.html', '').title()
        return title[:100]
    except: return "News Article"

def text_to_vector(text):
    words = re.compile(r'\w+').findall(text.lower())
    return Counter(words)

def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator: return 0.0
    return numerator / denominator

def verify_and_justify(url):
    import requests # imported here for safety
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
        }
        response = requests.get(url, headers=headers, timeout=4)
        if response.status_code != 200: return False, "⚠️ Link inaccessible or broken."
        
        article = Article(url)
        article.set_html(response.content)
        article.parse()
        
        text_content = f"{article.title} {article.text[:1500]}"
        if len(text_content) < 50: return False, "⚠️ Content too short for analysis."

        geopolitical_context = "cyber attack hack ransomware ddos data breach malware infosec phishing apt state sponsored cyber warfare espionage"
        vector_article = text_to_vector(text_content)
        vector_context = text_to_vector(geopolitical_context)
        score = get_cosine(vector_article, vector_context)
        
        try:
            article.nlp()
            summary_text = article.summary.replace('\n', ' ')[:300] + "..."
        except:
            summary_text = text_content[:300] + "..."

        if score > 0:
            return True, f"✅ Verified ({int(score*100)}%): {summary_text}"
        else:
            return False, f"⚠️ Low Relevance (0%): No cyber vocabulary match."
    except Exception as e: return False, f"⚠️ Analysis Error: {str(e)}"


# --- DATA LOADERS (POSTGRESQL) ---
@st.cache_data(ttl=3600, show_spinner=False) 
def load_gdelt_events(days):
    """Loads events directly from PostgreSQL."""
    query = text("""
            SELECT "GlobalEventID", "Day", "Actor1Name", "Actor1GeoCountry", 
                "Actor2Name", "Actor2GeoCountry", "EventCode", "Goldstein", 
                "NumMentions", "NumArticles", "AvgTone", "SourceURL", "EventDate"
            FROM gdelt_events
            WHERE "EventDate" >= CURRENT_DATE - INTERVAL '1 day' * :days
        """)
    
    try:
        with engine.connect() as conn:
            master_df = pd.read_sql_query(query, conn, params={'days': days})
            
        if master_df.empty: 
            return master_df
            
        # Ensure correct types
        for col in ['Goldstein', 'NumArticles', 'AvgTone']:
            master_df[col] = pd.to_numeric(master_df[col], errors='coerce').fillna(0)
            
        master_df['Score'] = (master_df['AvgTone'] * master_df['Goldstein'] * master_df['NumArticles'])
        return master_df
    except Exception as e:
        st.error(f"Error loading events from database: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_gkg_v2_data(days):
    """Loads GKG data directly from PostgreSQL."""
    query = text("""
            SELECT "Date", "SourceURL", "Themes", "Locations", "Organizations", "AvgTone"
            FROM gdelt_gkg
            WHERE "Date" >= CURRENT_TIMESTAMP - INTERVAL '1 day' * :days
        """)
    
    try:
        with engine.connect() as conn:
            gkg_df = pd.read_sql_query(query, conn, params={'days': days})
            
        if gkg_df.empty: 
            return pd.DataFrame()

        # Parse locations (we still need to do this step in python)
        def parse_location(loc_str):
            if not isinstance(loc_str, str): return None, None, None
            first = loc_str.split(';')[0]
            parts = first.split('#')
            if len(parts) > 5:
                try:
                    return parts[2], float(parts[4]), float(parts[5])
                except: return None, None, None
            return None, None, None

        parsed = gkg_df['Locations'].apply(parse_location)
        gkg_df['Country'] = [x[0] for x in parsed]
        gkg_df['Lat'] = [x[1] for x in parsed]
        gkg_df['Lon'] = [x[2] for x in parsed]
        
        final_df = gkg_df.dropna(subset=['Lat', 'Lon']).copy()
        
        if not final_df.empty:
            # Recreate NumArts since we dropped it in the DB insert script
            final_df['NumArts'] = 1 
            final_df['Weight'] = final_df['NumArts'] * final_df['AvgTone'].abs() 
            
        return final_df
    except Exception as e:
        st.error(f"Error loading GKG data from database: {e}")
        return pd.DataFrame()

# ... fetch_historical_trend ...
def fetch_historical_trend(origin, custom_query):
    import requests
    if origin != "All":
        query_parts = [f"sourcecountry:{origin}"]
        label = f"Media in {origin}"
    else:
        query_parts = []
        label = "Global Media"
    if custom_query.strip():
        keywords = [k.strip() for k in custom_query.split(',') if k.strip()]
        if len(keywords) > 1:
            processed_query = " AND ".join(keywords)
            query_parts.append(f"({processed_query})")
            label += f" reporting on '{processed_query}'"
        else:
            query_parts.append(keywords[0])
            label += f" reporting on '{keywords[0]}'"
    else:
        query_parts.append("cyber") 
        label += " (General Cyber)"
    
    final_query = " ".join(query_parts)
    api_base = "https://api.gdeltproject.org/api/v2/doc/doc"
    base_params = {'query': final_query, 'format': 'json', 'timespan': '3years', 'timelinesmooth': 5, 'timezoom': 'no'}
    try:
        vol_params = base_params.copy(); vol_params['mode'] = 'TimelineVolRaw'
        r_vol = requests.get(api_base, params=vol_params, timeout=15)
        tone_params = base_params.copy(); tone_params['mode'] = 'TimelineTone'
        r_tone = requests.get(api_base, params=tone_params, timeout=15)
        if r_vol.status_code == 200 and r_tone.status_code == 200:
            vol_json = r_vol.json(); tone_json = r_tone.json()
            if 'timeline' in vol_json:
                vol_data = vol_json['timeline'][0]['data']
                tone_data = tone_json['timeline'][0]['data']
                df_vol = pd.DataFrame(vol_data).rename(columns={'value': 'Volume'})
                df_tone = pd.DataFrame(tone_data).rename(columns={'value': 'AvgTone'})
                df = pd.merge(df_vol, df_tone, on='date')
                df['date'] = pd.to_datetime(df['date'])
                return df, label
    except Exception: return None, label
    return None, label

# ==============================================================================
# MAIN APP LAYOUT
# ==============================================================================
st.title("🔥 Geopolitical Conflict Monitor")

# Restored clean loading UI
with st.status("📡 Updating Intelligence Feeds...", expanded=True) as status:
    st.write(f"Fetching Events from Database (Last {st.session_state.time_window_days} Days)...")
    event_df = load_gdelt_events(st.session_state.time_window_days)
    st.write(f"Fetching GKG Data from Database (Last {st.session_state.time_window_days} Days)...")
    gkg_df = load_gkg_v2_data(st.session_state.time_window_days)
    status.update(label="Feeds Active", state="complete", expanded=False)

if not event_df.empty:
    all_actors = set(event_df['Actor1GeoCountry'].unique()) | set(event_df['Actor2GeoCountry'].unique())
    valid_countries = sorted([str(x) for x in all_actors if len(str(x)) >= 2])
    valid_countries.insert(0, "All")
    
    filtered_df = event_df.copy()
    if st.session_state.origin_country != "All":
        filtered_df = filtered_df[filtered_df['Actor1GeoCountry'] == st.session_state.origin_country]
    if st.session_state.target_country != "All":
        filtered_df = filtered_df[filtered_df['Actor2GeoCountry'] == st.session_state.target_country]

    filtered_df = filtered_df[(filtered_df['Goldstein'] < 0) & (filtered_df['AvgTone'] < 0)]
    filtered_df = filtered_df.drop_duplicates(subset=['SourceURL'])
    filtered_df['Score'] = filtered_df['Score'].astype(int)
    filtered_df['AbsScore'] = filtered_df['Score'].abs()
    filtered_df = filtered_df.sort_values('AbsScore', ascending=False)
    filtered_df['Title'] = filtered_df['SourceURL'].apply(format_url_to_title)
    filtered_df['Summary'] = filtered_df['Title']

    if st.session_state.deep_scan_data is not None:
        ds_df = st.session_state.deep_scan_data
        filtered_df = filtered_df.set_index('SourceURL')
        filtered_df.update(ds_df.set_index('SourceURL'))
        filtered_df = filtered_df.reset_index()

    def get_lat(code): return COUNTRY_CENTROIDS.get(code, [None, None])[0]
    def get_lon(code): return COUNTRY_CENTROIDS.get(code, [None, None])[1]
    filtered_df['MapLat'] = filtered_df['Actor2GeoCountry'].apply(get_lat)
    filtered_df['MapLon'] = filtered_df['Actor2GeoCountry'].apply(get_lon)
    map_df = filtered_df.dropna(subset=['MapLat', 'MapLon'])

    country_df = map_df.groupby('Actor2GeoCountry').agg({
        'Score': 'sum', 'NumArticles': 'sum', 'MapLat': 'first', 'MapLon': 'first', 'Actor1Name': 'count', 'Title': 'first'
    }).rename(columns={'Actor1Name': 'EventCount', 'Title': 'TopTitle'}).reset_index()
    country_df['HeatIntensity'] = country_df['Score'].abs()
    country_df = country_df.sort_values('HeatIntensity', ascending=False)

    left_panel, right_panel = st.columns([1, 1.2], gap="medium")
    with left_panel:
        st.subheader(f"📋 Event Feed (Last {st.session_state.time_window_days} Days)")
        st.dataframe(filtered_df[['EventDate', 'EventCode', 'Summary', 'SourceURL', 'Score']],
            column_config={"EventDate": st.column_config.DateColumn("Date", format="YYYY-MM-DD", width="small"),
                           "SourceURL": st.column_config.LinkColumn("Link", width="small")},
            use_container_width=True, height=700, hide_index=True)

    with right_panel:
        tc1, tc2 = st.columns([1, 1], vertical_alignment="bottom")
        with tc1:
            new_origin = st.selectbox("Origin Country", valid_countries, index=valid_countries.index(st.session_state.origin_country) if st.session_state.origin_country in valid_countries else 0)
            if new_origin != st.session_state.origin_country: st.session_state.origin_country = new_origin; st.rerun()
        with tc2:
            if st.button("🚀 Run Deep Scan", use_container_width=True):
                with st.status("🕵️ AI Analyst Working...", expanded=True):
                    verified_rows = []
                    candidates = filtered_df.head(200)
                    progress_bar = st.progress(0)
                    for i, (index, row) in enumerate(candidates.iterrows()):
                        progress_bar.progress((i + 1) / len(candidates))
                        is_rel, just = verify_and_justify(row['SourceURL'])
                        verified_rows.append({'SourceURL': row['SourceURL'], 'Summary': just})
                    progress_bar.empty()
                    if verified_rows:
                        st.session_state.deep_scan_data = pd.DataFrame(verified_rows)
                        st.rerun()

        if not country_df.empty:
            max_heat = country_df['HeatIntensity'].max() if country_df['HeatIntensity'].max() > 0 else 1
            country_df['Radius'] = (country_df['HeatIntensity'] / max_heat) * 500000 + 80000
            deck = pdk.Deck(layers=[
                pdk.Layer("ScatterplotLayer", data=country_df, get_position='[MapLon, MapLat]', get_radius='Radius', get_fill_color=[255, 50, 50, 140], pickable=True, auto_highlight=True, stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=2),
                pdk.Layer("TextLayer", data=country_df, get_position='[MapLon, MapLat]', get_text='Actor2GeoCountry', get_color=[255, 255, 255], get_size=12, get_alignment_baseline="'center'")
            ], initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=0.5), map_style=pdk.map_styles.CARTO_DARK, tooltip={"html": "<b>{Actor2GeoCountry}</b><br/>Heat: {Score}"})
            st.pydeck_chart(deck, use_container_width=True)
        else: st.warning("No data for map.")
else:
    st.info("No events found in the selected timeframe. Make sure the backfill worker is running!")

st.markdown("---")
st.markdown("### 📈 Historical Evolution")
gc1, gc2 = st.columns([1, 2])
with gc1: st.info(f"Source: **{st.session_state.origin_country}**")
with gc2: timeline_query = st.text_input("Timeline Theme/Query", placeholder="e.g. 'Cyber', 'Ransomware', 'China'")
trend_df, label = fetch_historical_trend(st.session_state.origin_country, timeline_query)
if trend_df is not None and not trend_df.empty:
    base = alt.Chart(trend_df).encode(x=alt.X('date:T', axis=alt.Axis(title='Date', format='%Y-%m-%d')))
    line_vol = base.mark_area(opacity=0.4, line=True, color='#3498db').encode(y='Volume:Q', tooltip=['date', 'Volume'])
    line_tone = base.mark_line(color='#e74c3c').encode(y=alt.Y('AvgTone:Q', scale=alt.Scale(reverse=True)))
    st.altair_chart(alt.layer(line_vol, line_tone).resolve_scale(y='independent').properties(height=350), use_container_width=True)

st.markdown("---")
st.header(f"🌍 Narrative Heatmap (GKG V2 Themes)")
st.info(f"Filtering Themes in: **{st.session_state.origin_country}**")

g_view = gkg_df.copy()
if not g_view.empty and st.session_state.origin_country != "All": 
    g_view = g_view[g_view['Country'] == st.session_state.origin_country]

org_col1, org_col2 = st.columns([3,1])
with org_col1:
    org_search = st.text_input("🏢 Filter by Organization", value=st.session_state.gkg_org_filter)
if org_search:
    st.session_state.gkg_org_filter = org_search
    g_view = g_view[g_view['Organizations'].astype(str).str.contains(org_search, case=False, na=False)]

slider_col, _ = st.columns([1, 1])
with slider_col:
    # Changed slider max from 30 to 365 days to access the full backfill
    new_days = st.slider("Analysis Window (Days)", 1, 365, st.session_state.time_window_days)
    if new_days != st.session_state.time_window_days: 
        st.session_state.time_window_days = new_days
        st.rerun()

if not g_view.empty:
    deck2 = pdk.Deck(layers=[
        pdk.Layer("HeatmapLayer", data=g_view, get_position='[Lon, Lat]', radius_pixels=60, intensity=1, threshold=0.3),
        pdk.Layer("ScatterplotLayer", data=g_view, get_position='[Lon, Lat]', get_radius=50000, get_fill_color='[0, 100, 255, 100]', pickable=True, auto_highlight=True)
    ], map_style=pdk.map_styles.CARTO_DARK, initial_view_state=pdk.ViewState(latitude=20, longitude=10, zoom=0.8),
    tooltip={"html": "<b>Weight:</b> {Weight}<br/><a href='{SourceURL}' target='_blank' style='color:#FFFF00'>Read Article</a>"})
    
    event = st.pydeck_chart(deck2, use_container_width=True, on_select="rerun", selection_mode="single-object")
    if event.selection and len(event.selection['objects']) > 0:
        obj = event.selection['objects'][0]
        st.info(f"📌 **Selected Event:** [Click to Read Article]({obj['SourceURL']}) (Weight: {obj['Weight']:.2f})")

    st.markdown("### 🔗 Top Event Sources")
    table_view = g_view.sort_values('Weight', ascending=False)[['Date', 'SourceURL', 'Organizations', 'Weight', 'AvgTone']].head(5000).copy()
    st.dataframe(table_view, column_config={
        "SourceURL": st.column_config.LinkColumn("Source Link", width="medium"),
        "Weight": st.column_config.NumberColumn("Weight", format="%.2f", width="small")
    }, use_container_width=True, hide_index=True)
else:
    st.warning("No GKG narrative data found for this filter.")