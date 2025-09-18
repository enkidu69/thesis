import pandas as pd
import glob
from pathlib import Path
import zipfile
import io
import csv
from typing import Optional, Tuple, List
import os
import random
import string
import requests
from datetime import datetime

desk=os.getcwd()
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

master = pd.read_csv('masterfilelist.txt', sep=" ", header=None, names=["a", "b", "urls"])
master['datetime'] = master['urls'].str[37:14]
master=master.drop(columns=['a','b'])
master['datetime'] = pd.to_datetime(master["datetime"].strftime('%YYYY%MM%DD%HH%MM%SS'))
print(master)
#exit()

mask = (df['master'] > start_date) & (df['date'] <= end_date)

start_dt = "20230901000000"
end_dt = "20230902000000"
    

master2 = master[master['datetime'].between(start_dt, end_dt)]

print(master2)

exit()

for url in master.urls:
    df_list = []
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
                        df_list.append(df)
    except Exception as e:
        print(f"Error processing {url}: {e}")


def load_gdelt_period_from_local(masterfile: str, start: str, end: str) -> pd.DataFrame:
    """
    Load GDELT data for a given period using a local masterfilelist.txt.
    Args:
        masterfile (str): path to masterfilelist.txt
        start (str): start datetime in 'YYYYMMDDHHMMSS'
        end (str): end datetime in 'YYYYMMDDHHMMSS'
    """
    master = fetch_master_list_local(masterfile)
    start_dt = datetime.strptime(start, "%Y%m%d%H%M%S")
    end_dt = datetime.strptime(end, "%Y%m%d%H%M%S")
    print(start_dt,end_dt)
    subset = master[(master["datetime"] >= start_dt) & (master["datetime"] <= end_dt)]

    if subset.empty:
        print("No files match the requested period.")
        return pd.DataFrame(columns=GDELT_COLUMNS)

    urls = subset["url"].tolist()
    return download_and_parse(urls)


# Example usage:
df = load_gdelt_period_from_local("masterfilelist.txt", "20230901000000", "20230903000000")
# print(df.shape)
# print(df.head())

print(df)
# 2) custom glob:
#    df = load_gdelt_data("data/**/*.CSV.zip")
# 3) single file or directory:
#    df = load_gdelt_data("data/2019-01-01.CSV.zip")
random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
filename= desk+'\\gdelt'+random_str+'.xlsx'
writer = pd.ExcelWriter(filename,engine='xlsxwriter')
df.to_excel(writer, index = False)
writer.close()