import pandas as pd
import requests, zipfile, io, csv, os, random, string
from pathlib import Path
import numpy as np

desk = os.getcwd()

# GDELT v2 column names (61 cols)
GDELT_COLUMNS = [
    "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate", "Actor1Code",
    "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode", "Actor1EthnicCode",
    "Actor1Religion1Code", "Actor1Religion2Code", "Actor1Type1Code", "Actor1Type2Code",
    "Actor1Type3Code", "Actor2Code", "Actor2Name", "Actor2CountryCode",
    "Actor2KnownGroupCode", "Actor2EthnicCode", "Actor2Religion1Code",
    "Actor2Religion2Code", "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass",
    "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone",
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode", "Actor1Geo_ADM1Code",
    "Actor1Geo_ADM2Code", "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code",
    "Actor2Geo_ADM2Code", "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode", "ActionGeo_ADM1Code",
    "ActionGeo_ADM2Code", "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID",
    "DATEADDED", "SOURCEURL"
]


# --- Load master list ---
urlmaster="http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
master = pd.read_csv(urlmaster, sep=" ", header=None, names=["a", "b", "urls"])

master["datetime"] = pd.to_datetime(
    master["urls"].str.extract(r"(\d{14})(?=\.export)")[0],
    format="%Y%m%d%H%M%S",
    errors="coerce"
)

master = master.dropna(subset=["datetime"]).drop(columns=["a", "b"])

# --- Date range filter ---
start_dt = pd.to_datetime("20250916000000", format="%Y%m%d%H%M%S")
end_dt   = pd.to_datetime("20250922004500", format="%Y%m%d%H%M%S")

master2 = master[master["datetime"].between(start_dt, end_dt)]

# --- Download & parse files ---
all_dfs = []
for url in master2.urls:
    try:
        print(f"Downloading {url}")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for member in z.namelist():
                if member.lower().endswith(".csv"):
                    with z.open(member) as f:
                        wrapper = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
                        df = pd.read_csv(
                            wrapper,
                            sep="\t",
                            header=None,
                            names=GDELT_COLUMNS,
                            low_memory=False,
                            quoting=csv.QUOTE_NONE,
                            on_bad_lines="skip"
                        )
                        all_dfs.append(df)
    except Exception as e:
        print(f"Error processing {url}: {e}")

Data = pd.concat(all_dfs, ignore_index=True)
Data=Data[(Data["IsRootEvent"] == 1)]
Data['GeoCountries']  = Data["Actor1Geo_CountryCode"]+Data["Actor2Geo_CountryCode"]
Data['Goldstein*diffusion'] = Data["GoldsteinScale"]*(Data["NumMentions"]+Data["NumSources"]+Data["NumArticles"])
# --- Filter by country ---
#Data = Data[(Data["GeoCountries"] == "UK") | (Data["GeoCountries"] == "ISLE")]
Data=Data[Data["GeoCountries"].str.contains("UK", na=False)]



# --- Fix DATEADDED ---
Data["DATEADDED"] = pd.to_datetime(Data["DATEADDED"], format="%Y%m%d%H%M%S", errors="coerce")
Data["DATEADDED_DISPLAY"] = Data["DATEADDED"].dt.strftime("%Y-%m-%d %H:%M:%S")

Data = Data.drop(columns=["SQLDATE", "MonthYear","Year","ActionGeo_FeatureID","FractionDate","Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat","Actor1Geo_Long","Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code", "Actor2Geo_Lat","Actor2Geo_Long"])

# Conversione in datetime
Data["datetime"] = pd.to_datetime(Data["DATEADDED"], format="%Y%m%d%H%M%S")

# Estrazione ora (arrotondata all’ora intera)
Data["hour"] = Data["datetime"].dt.to_period("H")

# Trasformazioni log(1+x) sulle metriche di diffusione
Data["M"] = np.log1p(Data["NumMentions"])
Data["S"] = np.log1p(Data["NumSources"])
Data["A"] = np.log1p(Data["NumArticles"])

# Fattore di diffusione
Data["D"] = Data["M"] + Data["S"] + Data["A"]

# Gravità conflittuale (solo eventi negativi)
Data["C"] = Data["GoldsteinScale"].apply(lambda x: max(0, -x))

# Contributo di rischio per evento
Data["EventRisk"] = Data["C"] * Data["D"]

# Aggregazione per Paese e ora
agg = (
    Data.groupby(["ActionGeo_CountryCode", "hour"])
    .agg(
        RawRisk=("EventRisk", "sum"),
        MediaVol=("NumArticles", "sum"),
        Events=("EventRisk", "count")
    )
    .reset_index()
)

# Normalizzazione rischio per volume mediale
agg["Index"] = agg["RawRisk"] / (agg["MediaVol"] + 1)

# Standardizzazione z-score (rispetto a tutto il dataset)
agg["Index_zscore"] = (agg["Index"] - agg["Index"].mean()) / agg["Index"].std()

# Risultati
print(agg.head(20))

# --- Save to Excel ---
random_str = "".join(random.choices(string.ascii_letters + string.digits, k=8))
filename = Path(desk) /f"gdelt_{random_str}.xlsx"
Data.to_excel(filename, index=False, engine="xlsxwriter")

print(f"Saved {len(Data)} rows -> {filename}")