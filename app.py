import streamlit as st
import pandas as pd
import requests
import zipfile
import io
import pydeck as pdk
import altair as alt
from datetime import datetime, timedelta
from urllib.parse import urlparse
from newspaper import Article
import nltk
import math
import re
from collections import Counter

# --- NLTK SETUP ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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
if 'time_window_days' not in st.session_state: st.session_state.time_window_days = 7 
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
    """
    Enhanced AI Analyst with mimicked Browser Session Headers and strict 0% threshold.
    """
    try:
        # FULL BROWSER HEADERS MIMIC
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        
        response = requests.get(url, headers=headers, timeout=4)
        if response.status_code != 200: return False, "⚠️ Link inaccessible or broken."
        
        article = Article(url)
        article.set_html(response.content)
        article.parse()
        
        text_content = f"{article.title} {article.text[:1500]}"
        if len(text_content) < 50: return False, "⚠️ Content too short for analysis."

        geopolitical_context = """
    International relations military conflict war army navy air force troops defense 
    sovereignty border dispute territorial integrity diplomacy foreign policy 
    united nations nato european union alliance strategic interests statecraft
    hybrid warfare gray zone proxy war mercenary pmc wagner militia separatist 
    insurgency rebellion coup d'etat regime change martial law civil unrest 
    terrorism guerrilla paramilitary asymmetric warfare cyber warfare espionage 
    disinformation propaganda information operations sabotage
    """
        vector_article = text_to_vector(text_content)
        vector_context = text_to_vector(geopolitical_context)
        score = get_cosine(vector_article, vector_context)
        
        try:
            article.nlp()
            summary_text = article.summary.replace('\n', ' ')[:300] + "..."
        except:
            summary_text = text_content[:300] + "..."

        # CHANGED: ONLY 0% IS LOW RELEVANCE. Any positive match is verified.
        if score > 0:
            return True, f"✅ Verified ({int(score*100)}%): {summary_text}"
        else:
            return False, f"⚠️ Low Relevance (0%): No geopolitical vocabulary match."

    except Exception as e: return False, f"⚠️ Analysis Error: {str(e)}"

# --- URL GENERATORS ---
def generate_gdelt_event_urls(days_back):
    """Generates URLs for V2 Events (15 min updates) - 72 HOURS"""
    base_url = "http://data.gdeltproject.org/gdeltv2/"
    urls = []
    now = datetime.utcnow()
    hours_to_fetch = 72 
    current_time = now - timedelta(minutes=15) 
    current_time = current_time.replace(second=0, microsecond=0)
    discard = current_time.minute % 15
    current_time -= timedelta(minutes=discard)
    
    steps = int(hours_to_fetch * 4) 
    for _ in range(steps):
        timestamp = current_time.strftime("%Y%m%d%H%M%S")
        url = f"{base_url}{timestamp}.export.CSV.zip"
        urls.append(url)
        current_time -= timedelta(minutes=15)
    return urls

def generate_gkg_v1_urls(days_back):
    """Generates URLs for GKG V1 (Daily Files)"""
    base_url = "http://data.gdeltproject.org/gkg/"
    urls = []
    now = datetime.utcnow()
    current_date = now - timedelta(days=1)
    for _ in range(days_back):
        date_str = current_date.strftime("%Y%m%d")
        url = f"{base_url}{date_str}.gkg.csv.zip"
        urls.append(url)
        current_date -= timedelta(days=1)
    return urls

# --- DATA LOADERS ---
@st.cache_data(ttl=3600, show_spinner=False) 
def load_gdelt_events(days):
    urls = generate_gdelt_event_urls(3)
    raw_mapping = {0:'GlobalEventID', 1:'Day', 6:'Actor1Name', 37:'Actor1GeoCountry', 16:'Actor2Name', 45:'Actor2GeoCountry', 28:'EventCode', 30:'Goldstein', 31:'NumMentions', 33:'NumArticles', 34:'AvgTone', 60:'SourceURL'}
    sorted_mapping = dict(sorted(raw_mapping.items()))
    use_cols = list(sorted_mapping.keys())
    col_names_ordered = list(sorted_mapping.values())
    master_df = pd.DataFrame()
    
    progress_text = "Downloading 72h Event Feed..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, url in enumerate(urls):
        if i % 10 == 0: my_bar.progress((i + 1) / len(urls), text=f"Processing Event File {i+1}/{len(urls)}")
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    with z.open(z.namelist()[0]) as f:
                        df_chunk = pd.read_csv(f, sep='\t', header=None, usecols=use_cols, names=col_names_ordered, dtype=str)
                        master_df = pd.concat([master_df, df_chunk], ignore_index=True)
        except Exception: continue
    
    my_bar.empty()
    if master_df.empty: return master_df
    
    for col in ['Goldstein', 'NumArticles', 'AvgTone']:
        master_df[col] = pd.to_numeric(master_df[col], errors='coerce').fillna(0)
    master_df['EventDate'] = pd.to_datetime(master_df['Day'], format='%Y%m%d', errors='coerce')
    
    TARGET_ROOT_CODES = ['13', '15', '16', '17', '18', '19', '20']
    master_df = master_df[master_df['EventCode'].apply(lambda x: str(x).startswith(tuple(TARGET_ROOT_CODES)))]
    master_df['Score'] = (master_df['AvgTone'] * master_df['Goldstein'] * master_df['NumArticles'])
    return master_df

@st.cache_data(ttl=3600, show_spinner=False)
def load_gkg_v1_data(days):
    urls = generate_gkg_v1_urls(days)
    use_cols = [0, 1, 3, 4, 6, 7, 10]
    col_names = ['Date', 'NumArts', 'Themes', 'Locations', 'Organizations', 'ToneRaw', 'SourceURL']
    STRICT_THEMES = ['ARMEDCONFLICT', 'CYBER_ATTACK', 'TERROR', 'MILITARY', 'SECURITY_SERVICES', 'STATE_OF_EMERGENCY', 'BORDER', 'SANCTIONS', 'ELECTION_FRAUD', 'POLITICAL_TURMOIL', 'MANMADE_DISASTER_IMPLIED']
    pattern = '|'.join(STRICT_THEMES)
    
    master_rows = []
    progress_text = "Downloading GKG Daily Files..."
    my_bar = st.progress(0, text=progress_text)
    
    for i, url in enumerate(urls):
        my_bar.progress((i + 1) / len(urls), text=f"Processing Day {i+1}/{len(urls)}")
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    with z.open(z.namelist()[0]) as f:
                        for chunk in pd.read_csv(f, sep='\t', header=None, usecols=use_cols, names=col_names, dtype=str, encoding='utf-8', on_bad_lines='skip', chunksize=10000):
                            filtered_chunk = chunk[chunk['Themes'].str.contains(pattern, na=False, case=False)].copy()
                            if not filtered_chunk.empty:
                                filtered_chunk['AvgTone'] = filtered_chunk['ToneRaw'].apply(lambda x: float(str(x).split(',')[0]) if pd.notnull(x) else 0)
                                filtered_chunk['NumArts'] = pd.to_numeric(filtered_chunk['NumArts'], errors='coerce').fillna(1)
                                filtered_chunk = filtered_chunk[filtered_chunk['AvgTone'] < 0]
                                if not filtered_chunk.empty:
                                    master_rows.append(filtered_chunk)
        except: continue
    my_bar.empty()

    if not master_rows: return pd.DataFrame()
    gkg_df = pd.concat(master_rows, ignore_index=True)
    gkg_df['Weight'] = gkg_df['NumArts'] * gkg_df['AvgTone']
    
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
    return gkg_df.dropna(subset=['Lat', 'Lon'])

def fetch_historical_trend(origin, custom_query):
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
        query_parts.append("conflict") 
        label += " (General Conflict)"
    
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

with st.status("📡 Updating Intelligence Feeds...", expanded=True) as status:
    st.write("Fetching Real-time Events (Last 72h)...")
    event_df = load_gdelt_events(3)
    st.write(f"Fetching Historical GKG Data (Last {st.session_state.time_window_days} Days)...")
    gkg_df = load_gkg_v1_data(st.session_state.time_window_days)
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
        'Score': 'sum', 'NumArticles': 'sum', 'MapLat': 'first', 'MapLon': 'first', 'Actor2Name': 'count', 'Title': 'first'
    }).rename(columns={'Actor2Name': 'EventCount', 'Title': 'TopTitle'}).reset_index()
    country_df['HeatIntensity'] = country_df['Score'].abs()
    country_df = country_df.sort_values('HeatIntensity', ascending=False)

    left_panel, right_panel = st.columns([1, 1.2], gap="medium")
    with left_panel:
        st.subheader("📋 Event Feed (Last 72h)")
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

st.markdown("---")
st.markdown("### 📈 Historical Evolution")
gc1, gc2 = st.columns([1, 2])
with gc1: st.info(f"Source: **{st.session_state.origin_country}**")
with gc2: timeline_query = st.text_input("Timeline Theme/Query", placeholder="e.g. 'Trade', 'Macron', 'China'")
trend_df, label = fetch_historical_trend(st.session_state.origin_country, timeline_query)
if trend_df is not None and not trend_df.empty:
    base = alt.Chart(trend_df).encode(x=alt.X('date:T', axis=alt.Axis(title='Date', format='%Y-%m-%d')))
    line_vol = base.mark_area(opacity=0.4, line=True, color='#3498db').encode(y='Volume:Q', tooltip=['date', 'Volume'])
    line_tone = base.mark_line(color='#e74c3c').encode(y=alt.Y('AvgTone:Q', scale=alt.Scale(reverse=True)))
    st.altair_chart(alt.layer(line_vol, line_tone).resolve_scale(y='independent').properties(height=350), use_container_width=True)

st.markdown("---")
st.header(f"🌍 Narrative Heatmap (GKG V1 Themes)")
st.info(f"Filtering NEGATIVE Tone Narratives in: **{st.session_state.origin_country}** (Last {st.session_state.time_window_days} Days)")

g_view = gkg_df.copy()
if st.session_state.origin_country != "All": g_view = g_view[g_view['Country'] == st.session_state.origin_country]
org_col1, org_col2 = st.columns([3,1])
with org_col1:
    org_search = st.text_input("🏢 Filter by Organization", value=st.session_state.gkg_org_filter)
if org_search:
    st.session_state.gkg_org_filter = org_search
    g_view = g_view[g_view['Organizations'].astype(str).str.contains(org_search, case=False, na=False)]

# MOVED SLIDER BELOW FILTER
slider_col, _ = st.columns([1, 1])
with slider_col:
    new_days = st.slider("GKG Analysis Window (Days)", 1, 90, st.session_state.time_window_days)
    if new_days != st.session_state.time_window_days: st.session_state.time_window_days = new_days; st.rerun()

if not g_view.empty:
    deck2 = pdk.Deck(layers=[
        pdk.Layer("HeatmapLayer", data=g_view, get_position='[Lon, Lat]', radius_pixels=60, intensity=1, threshold=0.3),
        pdk.Layer("ScatterplotLayer", data=g_view, get_position='[Lon, Lat]', get_radius=50000, get_fill_color='[0, 100, 255, 100]', pickable=True, auto_highlight=True)
    ], map_style=pdk.map_styles.CARTO_DARK, initial_view_state=pdk.ViewState(latitude=20, longitude=10, zoom=0.8),
    tooltip={"html": "<b>Weight:</b> {Weight}<br/><a href='{SourceURL}' target='_blank' style='color:#FFFF00'>Read Article</a>"})
    
    # Selection Handler
    event = st.pydeck_chart(deck2, use_container_width=True, on_select="rerun", selection_mode="single-object")
    if event.selection and len(event.selection['objects']) > 0:
        obj = event.selection['objects'][0]
        st.info(f"📌 **Selected Event:** [Click to Read Article]({obj['SourceURL']}) (Weight: {obj['Weight']:.2f})")

    st.markdown("### 🔗 Top Negative Impact Sources (Sorted by Lowest Weight)")
    table_view = g_view.sort_values('Weight', ascending=True)[['Date', 'SourceURL', 'Organizations', 'Weight', 'AvgTone']].head(50).copy()
    st.dataframe(table_view, column_config={
        "SourceURL": st.column_config.LinkColumn("Source Link", width="medium"),
        "Weight": st.column_config.NumberColumn("Weight (NumArts * Tone)", format="%.2f", width="small")
    }, use_container_width=True, hide_index=True)
else:
    st.warning("No GKG narrative data found for this filter.")