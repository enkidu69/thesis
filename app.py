import streamlit as st
import pandas as pd
import requests
import zipfile
import io
import pydeck as pdk
from datetime import datetime, timedelta
from urllib.parse import urlparse
from newspaper import Article, Config 
import nltk

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
    
    /* DATAFRAME TEXT WRAPPING FIX */
    div[data-testid="stDataFrame"] div[role="grid"] div[role="row"] div[role="gridcell"] {
        white-space: normal !important;
        line-height: 1.5 !important;
        height: auto !important;
        align-items: start !important;
        overflow-wrap: break-word !important;
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    
    /* Button Alignment */
    div.stButton > button {
        height: 2.6rem; 
        width: 100%;
        border-radius: 6px;
    }
    
    /* REMOVE FOOTER & WHITESPACE */
    footer {display: none !important;}
    header {visibility: hidden;}
    .block-container { padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'time_window' not in st.session_state: st.session_state.time_window = 12
if 'origin_country' not in st.session_state: st.session_state.origin_country = "All"
if 'target_country' not in st.session_state: st.session_state.target_country = "All"
if 'selected_country' not in st.session_state: st.session_state.selected_country = None
if 'deep_scan_data' not in st.session_state: st.session_state.deep_scan_data = None

# --- CONSTANTS ---
GEOPOLITICAL_KEYWORDS = [
    "military", "army", "navy", "air force", "troops", "soldiers",
    "government", "parliament", "senate", "congress", "ministry", "minister", "president", "premier",
    "diplomat", "ambassador", "treaty", "agreement", "sanction", "embargo",
    "war", "conflict", "attack", "bomb", "missile", "strike", "terror", "crisis",
    "border", "territory", "sovereignty", "election", "vote", "party",
    "un", "united nations", "nato", "eu", "european union", "asean", "imf",
    "protest", "riot", "coup", "rebellion", "insurgent", "police", "security"
]

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

def verify_and_justify(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=4)
        if response.status_code != 200: return False, "⚠️ Analysis: Link inaccessible or broken."
        
        article = Article(url)
        article.set_html(response.content)
        article.parse()
        
        if not any(keyword in article.text.lower() for keyword in GEOPOLITICAL_KEYWORDS):
            return False, "⚠️ Analysis: No useful information (Irrelevant content)."
            
        try:
            article.nlp()
            summary = article.summary.replace('\n', ' ')
            if len(summary) > 400: summary = summary[:400] + "..."
            if not summary or len(summary) < 20: 
                summary = "⚠️ Analysis: No useful information (Content too short)."
        except: summary = "✅ Analysis: Relevant keywords found (Auto-summary failed)."
        return True, summary
    except Exception as e: return False, f"⚠️ Analysis Error: {str(e)}"

def generate_gdelt_urls(hours_back):
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

@st.cache_data(ttl=900, show_spinner=False) 
def load_gdelt_data(hours):
    urls = generate_gdelt_urls(hours)
    raw_mapping = {0:'GlobalEventID', 1:'Day', 6:'Actor1Name', 37:'Actor1GeoCountry', 16:'Actor2Name', 45:'Actor2GeoCountry', 26:'EventCode', 30:'Goldstein', 31:'NumMentions', 33:'NumArticles', 34:'AvgTone', 60:'SourceURL'}
    sorted_mapping = dict(sorted(raw_mapping.items()))
    use_cols = list(sorted_mapping.keys())
    col_names_ordered = list(sorted_mapping.values())
    master_df = pd.DataFrame()
    for i, url in enumerate(urls):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    filename = z.namelist()[0]
                    with z.open(filename) as f:
                        df_chunk = pd.read_csv(f, sep='\t', header=None, usecols=use_cols, names=col_names_ordered, dtype=str)
                        master_df = pd.concat([master_df, df_chunk], ignore_index=True)
        except Exception: continue
    if master_df.empty: return master_df
    
    for col in ['Goldstein', 'NumArticles', 'AvgTone']:
        master_df[col] = pd.to_numeric(master_df[col], errors='coerce').fillna(0)
    master_df['EventDate'] = pd.to_datetime(master_df['Day'], format='%Y%m%d', errors='coerce')
    
    target_codes = ['130','131','1311','1312','1313','132','1321','1322','1323','1324','133','134','135','136','137','138','1381','1382','1383','1384','1385','139','150','151','152','153','154','155','16','160','161','162','1621','1622','1623','163','164','165','166','1661','1662','1663','1712','1721','1722','1723','1724','174','175','180','181','182','1821','1822','1823','183','1831','1832','1833','1834','184','185','186','190','191','192','193','194','195','1951','1952','196','200','201','202','203','204','2041','2042']
    master_df = master_df[master_df['EventCode'].isin(target_codes)]
    master_df['Score'] = (master_df['AvgTone'] * master_df['Goldstein'] * master_df['NumArticles'])
    return master_df

def fetch_historical_trend(origin, custom_query):
    """Fetches 12-month Volume * Tone data using specific GDELT params."""
    
    # 1. BUILD QUERY
    # Logic: If Origin is "All", we search globally (keyword only).
    # If Origin is selected, we filter by sourcecountry.
    if origin != "All":
        query_parts = [f"sourcecountry:{origin}"]
        label = f"Media in {origin}"
    else:
        query_parts = []
        label = "Global Media"

    # Append keyword if provided
    if custom_query.strip():
        query_parts.append(custom_query)
        label += f" reporting on '{custom_query}'"
    else:
        # Default fallback if no keyword provided
        # CHANGED: Removed "tone:<-2" as it is not supported by the API in this context.
        # Replaced with "conflict" keyword to maintain app theme.
        query_parts.append("conflict") 
        label += " (General Conflict)"
    
    final_query = " ".join(query_parts)

    # 2. PARAMETERS (timelinesmooth=0, timezoom=yes, timespan=1Y)
    api_base = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    # We need TWO calls to calculate Vol * Tone (TimelineVolRaw * TimelineTone)
    base_params = {
        'query': final_query,
        'format': 'json',
        'timespan': '1Y',          # 1 Year
        'timelinesmooth': 0,       # No smoothing
        'timezoom': 'yes'          # Per user request
    }
    
    debug_info = {}
    
    try:
        # A. Fetch Volume (TimelineVolRaw)
        vol_params = base_params.copy()
        vol_params['mode'] = 'TimelineVolRaw'
        debug_info['vol'] = requests.Request('GET', api_base, params=vol_params).prepare().url
        r_vol = requests.get(api_base, params=vol_params, timeout=10)
        
        # B. Fetch Tone (TimelineTone)
        tone_params = base_params.copy()
        tone_params['mode'] = 'TimelineTone'
        debug_info['tone'] = requests.Request('GET', api_base, params=tone_params).prepare().url
        r_tone = requests.get(api_base, params=tone_params, timeout=10)
        
        if r_vol.status_code == 200 and r_tone.status_code == 200:
            vol_json = r_vol.json()
            tone_json = r_tone.json()
            
            # Extract Data
            if 'timeline' not in vol_json or not vol_json['timeline']: return None, label, debug_info
            vol_data = vol_json['timeline'][0]['data']
            
            if 'timeline' not in tone_json or not tone_json['timeline']: return None, label, debug_info
            tone_data = tone_json['timeline'][0]['data']
            
            df_vol = pd.DataFrame(vol_data).rename(columns={'value': 'Volume'})
            df_tone = pd.DataFrame(tone_data).rename(columns={'value': 'AvgTone'})
            
            # Merge & Process
            df = pd.merge(df_vol, df_tone, on='date')
            df['date'] = pd.to_datetime(df['date'])
            
            # Metric: Volume * Tone
            df['Impact'] = df['Volume'] * df['AvgTone']
            
            return df, label, debug_info
            
    except Exception:
        return None, label, debug_info
    
    return None, label, debug_info
# ==============================================================================
# MAIN APP LAYOUT
# ==============================================================================
st.title("🔥 Geopolitical Conflict Monitor")

# --- DATA LOADING ---
with st.status("📡 Updating Data Feed...", expanded=True) as status:
    df = load_gdelt_data(st.session_state.time_window)
    status.update(label="Feed Active", state="complete", expanded=False)

if not df.empty:
    # --- UNIFIED COUNTRY LIST ---
    all_actors = set(df['Actor1GeoCountry'].unique()) | set(df['Actor2GeoCountry'].unique())
    valid_countries = sorted([str(x) for x in all_actors if len(str(x)) >= 2])
    valid_countries.insert(0, "All")
    
    # --- FILTERS ---
    filtered_df = df.copy()
    if st.session_state.origin_country != "All":
        filtered_df = filtered_df[filtered_df['Actor1GeoCountry'] == st.session_state.origin_country]
    if st.session_state.target_country != "All":
        filtered_df = filtered_df[filtered_df['Actor2GeoCountry'] == st.session_state.target_country]

    # --- PROCESS DATA ---
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

    # ======================================================================
    # SPLIT LAYOUT
    # ======================================================================
    left_panel, right_panel = st.columns([1, 1.2], gap="medium")

    # --- LEFT PANEL: EVENT FEED ---
    with left_panel:
        st.subheader("📋 Event Feed")
        st.dataframe(
            filtered_df[['EventDate', 'Summary', 'SourceURL', 'Score']],
            column_config={
                "EventDate": st.column_config.DateColumn("Date", format="YYYY-MM-DD", width="small"),
                "Summary": st.column_config.TextColumn("Summary / Title", width="large"),
                "SourceURL": st.column_config.LinkColumn("Link", width="small"),
                "Score": st.column_config.NumberColumn("Heat", format="%d", width="small")
            },
            use_container_width=True,
            height=700,
            hide_index=True
        )

    # --- RIGHT PANEL: MAP & CONTROLS ---
    with right_panel:
        # TOP CONTROLS
        tc1, tc2 = st.columns([1, 1], vertical_alignment="bottom")
        with tc1:
            new_origin = st.selectbox("Origin Country", valid_countries, index=valid_countries.index(st.session_state.origin_country) if st.session_state.origin_country in valid_countries else 0)
            if new_origin != st.session_state.origin_country:
                st.session_state.origin_country = new_origin
                st.rerun()
        with tc2:
            if st.button("🚀 Run Deep Scan", use_container_width=True, help="Analyze top 20 events"):
                with st.status("🕵️ AI Analyst Working...", expanded=True) as status:
                    verified_rows = []
                    candidates = filtered_df.head(20)
                    progress_bar = st.progress(0)
                    for i, (index, row) in enumerate(candidates.iterrows()):
                        progress_bar.progress((i + 1) / len(candidates))
                        is_rel, just = verify_and_justify(row['SourceURL'])
                        verified_rows.append({'SourceURL': row['SourceURL'], 'Summary': just})
                    progress_bar.empty()
                    
                    if verified_rows:
                        st.session_state.deep_scan_data = pd.DataFrame(verified_rows)
                        status.update(label="Scan Complete!", state="complete", expanded=False)
                        st.rerun()
                    else:
                        status.update(label="Scan failed or no data", state="error")

        # MAP
        if not country_df.empty:
            max_heat = country_df['HeatIntensity'].max()
            if max_heat == 0: max_heat = 1
            country_df['Radius'] = (country_df['HeatIntensity'] / max_heat) * 500000 + 80000
            
            deck = pdk.Deck(
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=country_df,
                        get_position='[MapLon, MapLat]',
                        get_radius='Radius',
                        get_fill_color=[255, 50, 50, 140],
                        pickable=True,
                        auto_highlight=True,
                        stroked=True,
                        get_line_color=[255, 255, 255],
                        line_width_min_pixels=2
                    ),
                    pdk.Layer(
                        "TextLayer",
                        data=country_df,
                        get_position='[MapLon, MapLat]',
                        get_text='Actor2GeoCountry',
                        get_color=[255, 255, 255],
                        get_size=12,
                        get_alignment_baseline="'center'"
                    )
                ],
                initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=0.5),
                map_style=pdk.map_styles.CARTO_DARK,
                tooltip={"html": "<b>{Actor2GeoCountry}</b><br/>Heat: {Score}<br/>Events: {EventCount}<br/><i>Top Event: {TopTitle}</i>"}
            )
            selection = st.pydeck_chart(deck, use_container_width=True, on_select="rerun", selection_mode="single-object")
            if selection.selection and len(selection.selection['objects']) > 0:
                for layer_idx, objects in selection.selection['objects'].items():
                    if objects: st.session_state.selected_country = objects[0]['Actor2GeoCountry']
        else:
            st.warning("No data for map.")

        # SELECTED DETAIL
        if st.session_state.selected_country:
            st.info(f"📍 **Target Analysis: {st.session_state.selected_country}**")
            details_df = filtered_df[filtered_df['Actor2GeoCountry'] == st.session_state.selected_country].head(10).copy()
            st.dataframe(details_df[['Summary', 'Score', 'SourceURL']], column_config={"SourceURL": st.column_config.LinkColumn("Link", width="small"), "Score": st.column_config.NumberColumn("Heat", format="%d"), "Summary": st.column_config.TextColumn("Summary", width="large")}, use_container_width=True, hide_index=True)
            if st.button("Close Details"): st.session_state.selected_country = None; st.rerun()

        # BOTTOM CONTROLS
        st.markdown("---")
        bc1, bc2, bc3 = st.columns([1, 1, 1], vertical_alignment="bottom")
        with bc1:
            new_time = st.slider("Time Window (Hours)", 1, 72, st.session_state.time_window)
            if new_time != st.session_state.time_window: st.session_state.time_window = new_time; st.rerun()
        with bc2:
            new_target = st.selectbox("Target Country", valid_countries, index=valid_countries.index(st.session_state.target_country) if st.session_state.target_country in valid_countries else 0)
            if new_target != st.session_state.target_country: st.session_state.target_country = new_target; st.rerun()
        with bc3:
            st.download_button("📥 Download Raw Feed", filtered_df.to_csv(index=False), "gdelt_raw.csv", "text/csv", use_container_width=True)

# ==============================================================================
# HISTORICAL TREND GRAPH
# ==============================================================================
st.markdown("---")
st.markdown("### 📈 Historical Evolution (Last 12 Months)")

gc1, gc2 = st.columns([1, 2])
with gc1: st.info(f"Source: **{st.session_state.origin_country}**")
with gc2: timeline_query = st.text_input("Timeline Theme/Query", placeholder="e.g. 'Trade', 'Macron', 'China'", help="See how the Source Country reports on this topic.", label_visibility="collapsed")

trend_df, label, debug_info = fetch_historical_trend(st.session_state.origin_country, timeline_query)

with st.expander("🔌 API Query Debugger"):
    if debug_info:
        st.code(debug_info.get('vol', 'N/A'))
        st.code(debug_info.get('tone', 'N/A'))
    else: st.write("No query generated.")

if trend_df is not None and not trend_df.empty:
    st.line_chart(trend_df.set_index('date')['Impact'], color="#ff4b4b")
    st.caption(f"Showing **Volume × Average Tone** for: {label}")
else:
    st.info("No sufficient historical data found. Try specifying a Query.")

# ==============================================================================
# KEYWORD SEARCH (Bottom)
# ==============================================================================
st.markdown("---")
st.header("📰 Keyword News Search")

kc1, kc2 = st.columns([4, 1], vertical_alignment="bottom")
with kc1: 
    keyword = st.text_input("Search GDELT (Past 12 Months)", placeholder="e.g. 'Cyberattack', 'Border'", label_visibility="collapsed")
with kc2: 
    search_btn = st.button("🔍 Search Keyword", use_container_width=True)

if search_btn and keyword: st.session_state['last_kw'] = keyword

if 'last_kw' in st.session_state:
    kw = st.session_state['last_kw']
    api_url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={kw}&mode=artlist&maxrecords=50&timespan=12months&format=json"
    
    with st.spinner("Searching..."):
        try:
            r = requests.get(api_url)
            if r.status_code == 200:
                data = r.json()
                if "articles" in data:
                    news_df = pd.DataFrame(data["articles"])
                    st.success(f"Found {len(news_df)} articles.")
                    if st.button("🚀 Deep Scan Results"):
                        progress = st.progress(0)
                        verified = []
                        for i, row in news_df.iterrows():
                            progress.progress((i+1)/len(news_df))
                            rel, just = verify_and_justify(row['url'])
                            row['Justification'] = just
                            verified.append(row)
                        progress.empty()
                        v_df = pd.DataFrame(verified)
                        st.dataframe(v_df[['title', 'Justification', 'url']], column_config={"url": st.column_config.LinkColumn("Link")}, use_container_width=True)
                    else:
                        st.dataframe(news_df[['title', 'seendate', 'url']], column_config={"url": st.column_config.LinkColumn("Link")}, use_container_width=True)
        except Exception as e: st.error(str(e))