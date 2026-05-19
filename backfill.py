import pandas as pd
import requests
import zipfile
import io
import time
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

# --- CONFIGURATION ---
# Database Connection String: postgresql://user:password@host:port/database
DB_URI = 'postgresql://gdelt_admin:SuperSecurePassword123!@localhost:5432/gdelt_db'
engine = create_engine(DB_URI)

DAYS_TO_BACKFILL = 365
BASE_URL = "http://data.gdeltproject.org/gdeltv2/"

# Cyber Theme Dictionary
STRICT_THEMES = [
    'CYBER_ATTACK', 'HACK', 'HACKER', 'HACKING', 'DATA_BREACH', 'CYBER_SECURITY',
    'MALWARE', 'RANSOMWARE', 'VIRUS', 'TROJAN', 'SPYWARE', 'BOTNET',
    'DDOS', 'PHISHING', 'INFOSEC', 'CRIME_CYBER_CRIME', 'STATE_SPONSORED_CYBER'
]
GKG_PATTERN = '|'.join([f"(?:^|;){theme}," for theme in STRICT_THEMES])

# Target Event Codes (e.g., Conflict, Coercion, Assault)
TARGET_ROOT_CODES = ('13', '15', '16', '17', '18', '19', '20')

# --- INITIALIZE TRACKING TABLE ---
# This ensures we never process the same file twice
with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS processed_files (
            timestamp VARCHAR(20) PRIMARY KEY,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """))
    conn.commit()

def get_processed_timestamps():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT timestamp FROM processed_files")).fetchall()
        return set([row[0] for row in result])

def mark_as_processed(timestamp):
    with engine.connect() as conn:
        conn.execute(text("INSERT INTO processed_files (timestamp) VALUES (:ts)"), {"ts": timestamp})
        conn.commit()

def process_events(url, timestamp):
    raw_mapping = {0:'GlobalEventID', 1:'Day', 6:'Actor1Name', 37:'Actor1GeoCountry', 
                   16:'Actor2Name', 45:'Actor2GeoCountry', 28:'EventCode', 30:'Goldstein', 
                   31:'NumMentions', 33:'NumArticles', 34:'AvgTone', 60:'SourceURL'}
    sorted_mapping = dict(sorted(raw_mapping.items()))
    use_cols = list(sorted_mapping.keys())
    col_names = list(sorted_mapping.values())

    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f, sep='\t', header=None, usecols=use_cols, names=col_names, dtype=str)
                    
                    # Filter logic
                    df = df[df['EventCode'].apply(lambda x: str(x).startswith(TARGET_ROOT_CODES))]
                    for col in ['Goldstein', 'NumArticles', 'AvgTone']:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    df = df[(df['Goldstein'] < 0) & (df['AvgTone'] < 0)]
                    
                    if not df.empty:
                        df['EventDate'] = pd.to_datetime(df['Day'], format='%Y%m%d', errors='coerce')
                        df.to_sql('gdelt_events', engine, if_exists='append', index=False)
                        print(f"  -> Inserted {len(df)} Event records.")
        return True
    except Exception as e:
        print(f"  -> Event Error: {e}")
        return False

def process_gkg(url, timestamp):
    use_cols = [1, 4, 8, 9, 13, 15]
    col_names = ['Date', 'SourceURL', 'Themes', 'Locations', 'Organizations', 'ToneRaw']
    
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f, sep='\t', header=None, usecols=use_cols, names=col_names, dtype=str, on_bad_lines='skip')
                    
                    filtered = df[df['Themes'].str.contains(GKG_PATTERN, na=False, case=False, regex=True)].copy()
                    
                    if not filtered.empty:
                        filtered['AvgTone'] = filtered['ToneRaw'].apply(lambda x: float(str(x).split(',')[0]) if pd.notnull(x) else 0)
                        filtered = filtered[filtered['AvgTone'] < 0] # Keep negative tone
                        
                        # Strip ToneRaw to save DB space
                        filtered = filtered.drop(columns=['ToneRaw'])
                        filtered['Date'] = pd.to_datetime(filtered['Date'], format='%Y%m%d%H%M%S', errors='coerce')
                        
                        filtered.to_sql('gdelt_gkg', engine, if_exists='append', index=False)
                        print(f"  -> Inserted {len(filtered)} GKG records.")
        return True
    except Exception as e:
        print(f"  -> GKG Error: {e}")
        return False

# --- MAIN LOOP ---
print("🚀 Starting GDELT V2 Backfill Worker...")
processed = get_processed_timestamps()

# Align start time to the nearest 15 minutes
now = datetime.utcnow()
start_time = now.replace(second=0, microsecond=0)
start_time -= timedelta(minutes=start_time.minute % 15)
end_time = start_time - timedelta(days=DAYS_TO_BACKFILL)

current_time = start_time

while current_time >= end_time:
    ts = current_time.strftime("%Y%m%d%H%M00")
    
    if ts in processed:
        print(f"⏭️ Skipping {ts} (Already Processed)")
        current_time -= timedelta(minutes=15)
        continue
        
    print(f"\n📡 Processing {ts}...")
    
    event_url = f"{BASE_URL}{ts}.export.CSV.zip"
    gkg_url = f"{BASE_URL}{ts}.gkg.csv.zip"
    
    # Process both. If GDELT throws a 404, we just continue (missing files happen)
    process_events(event_url, ts)
    process_gkg(gkg_url, ts)
    
    # Mark as done so we don't repeat it if the script restarts
    mark_as_processed(ts)
    
    current_time -= timedelta(minutes=15)
    time.sleep(1) # Be polite to GDELT servers