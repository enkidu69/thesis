import pandas as pd
import requests, zipfile, io, csv, os, random, string
from pathlib import Path
import numpy as np
import mplfinance as mpf
import re
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 

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


# CAMEO event codes used for geopolitical risk (from the study)

# External risk events
roote=[]

#new root codes, only threat ones considered
rcodes = [10,11,12,13,14,15,16,17,18,19,20]
gov="GOV"
#new selection of codes
#scodes=[12,16,212,214,232,233,234,243,244,252,253,254,255,256,26,27,28,312,314,32,332,333,334,354,355,356,36,37,38,39,46,50,51,52,53,54,55,56,57,6,60,61,62,63,64,71,72,73,74,75,811,812,813,814,82,83,831,832,833,834,841,85,86,861,862,863,87,871,872,873,874,92,93,94,1012,1014,102,1032,1033,1034,1041,1042,1043,1044,1052,1054,1055,1056,106,107,108,111,1121,1122,1123,1124,1125,113,114,115,116,121,1211,1212,122,1221,1222,1223,1224,123,1231,1232,1233,1234,124,1241,1242,1243,1244,1245,1246,125,126,127,128,129,130,131,1311,1312,1313,132,1321,1322,1323,1324,133,134,135,136,137,138,1381,1382,1383,1384,1385,139,140,141,1411,1412,1413,1414,142,1421,1422,1423,1424,143,1431,1432,1433,1434,144,1441,1442,1443,1444,145,1451,1452,1453,1454,150,151,152,153,154,155,16,160,161,162,1621,1622,1623,163,164,165,166,1661,1662,1663,1712,1721,1722,1723,1724,174,175,176,180,181,182,1821,1822,1823,183,1831,1832,1833,1834,184,185,186,190,191,192,193,194,195,1951,1952,196,200,201,202,203,204,2041,2042]
scodes=[212,214,232,234,256,312,314,32,332,334,356,62,72,74,87,872,873,874,93,94,1012,1014,1032,1033,1034,1056,111,1122,1123,1124,1125,114,121,1211,1212,122,1221,1222,1223,1224,123,1231,1232,1233,1234,124,1241,1242,1243,1244,1245,1246,125,126,127,128,129,130,131,1311,1312,1313,132,1321,1322,1323,1324,133,134,135,136,137,138,1381,1382,1383,1384,1385,139,145,1451,1452,1453,1454,150,151,152,153,154,155,16,160,161,162,1621,1622,1623,164,165,166,1661,1662,1663,1712,1721,1722,1723,1724,174,175,176,180,181,182,1821,1822,1823,183,1831,1832,1833,1834,184,185,186,190,191,192,193,194,195,1951,1952,196,200,201,202,203,204,2041,2042]


#old codes
codes1 = [
    24,   # Call for political reform
    25,   # Call for compromise
    28,   # Call for mediation
    9,    # Investigate
    104,  # Demand political reform
    105,  # Demand target compromise
    11,   # Criticism
    13,   # Threat
    15,   # Show of military force
    16,   # Reduce relations
    17,   # Coerce
    18,   # Assault / Attack
    19,   # Military clash
    20]    # Non-conventional mass violence
# Internal risk events
sub = [    #120,190,#test
    24,   # Call for political reform
    25,   # Call for compromise
    104,  # Demand political reform
    105,  # Demand target compromise
    112,  # Condemn
    113,  # Express collective opposition
    123,  # Refuse political reform
    124,  # Refuse to yield / compromise
    125,  # Refuse to meet / mediate
    127,  # Refuse agreement
    128,  # Defy law / norms / rules
    13,   # Threat
    14,   # Protest / Demonstrate
    15,   # Show of military force
    17,   # Coerce
    18,   # Assault / Attack
    19,   # Military clash
    20    # Non-conventional mass violence
]


# --- Date range filter ---

print("importing dates")

#UK attack heathrow
start_dt = pd.to_datetime("20250801000000", format="%Y%m%d%H%M%S")
end_dt = pd.to_datetime("20250925010000", format="%Y%m%d%H%M%S")

#FR TV5 monde
#start_dt = pd.to_datetime("20150301000000", format="%Y%m%d%H%M%S")
#end_dt   = pd.to_datetime("20150414010000", format="%Y%m%d%H%M%S")

#test
#start_dt = pd.to_datetime("20250918000000", format="%Y%m%d%H%M%S")
#end_dt   = pd.to_datetime("20250925010000", format="%Y%m%d%H%M%S")



Sourcename=str(start_dt)+str(end_dt)+".xlsx"
Sourcename=Sourcename.replace(" ","")
Sourcename=Sourcename.replace(":","")
#Sourcename = re.sub(r"\s+", "", Sourcename, flags=re.UNICODE)
#Sourcename = re.sub(r"\s+", "", Sourcename, flags=re.UNICODE)




#check if table is already available
desk=str(os.getcwd())

# Specify path
path = desk+'/'+Sourcename

# Check whether the specified path exists or not
isExist = os.path.exists(path)
#print(path)
#print(isExist)



# --- Download & parse files ---
all_dfs = []
if isExist is False:

    print("importing from internet")
# --- Load master list ---
    urlmaster="http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"


    master = pd.read_csv(urlmaster, sep=" ", header=None, names=["a", "b", "urls"])

    master["datetime"] = pd.to_datetime(
        master["urls"].str.extract(r"(\d{14})(?=\.export)")[0],
        format="%Y%m%d%H%M%S",
        errors="coerce"
    )

    master = master.dropna(subset=["datetime"]).drop(columns=["a", "b"])
    master2 = master[master["datetime"].between(start_dt, end_dt)]
    
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
                            #added dtype=str
                            df = pd.read_csv(
                                wrapper,
                                sep="\t",
                                header=None, #dtype=str,
                                names=GDELT_COLUMNS,
                                low_memory=False,
                                quoting=csv.QUOTE_NONE,
                                on_bad_lines="skip"
                            )
                            #print(df)
                            #drop columns before processing to reduce processing need
                            #
                            #print(df)
                            
                            #exit()
                            #df = df.drop(columns=["SQLDATE", "MonthYear","Year","ActionGeo_FeatureID","FractionDate","Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat","Actor1Geo_Long","Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code", "Actor2Geo_Lat","Actor2Geo_Long"])
                            #allow only values less than 0 from goldstein
                            #df= df[(df['GoldsteinScale'] < 0)]
                            
                            
                            #filter only GOV types
                                                
                            #df=df[(df['Actor1Type1Code'] == "GOV")|(df['Actor2Type1Code'] == "GOV")]
                            
                            #filter for root events
                            
                            #df=df[(df["IsRootEvent"]== 1)]
                            #df['GeoCountries']  = df["Actor1Geo_CountryCode"]+df["Actor2Geo_CountryCode"]
                            #df=df[df["GeoCountries"].str.contains("UK", na=False)]
                            #filter based on rootcodes instead of codes
                            #df=df[df["EventRootCode"].astype(int).isin(rcodes)]
                            #df=df[df["EventCode"].astype(int).isin(codes)]
                            
                            #filter only for relevant CAMEO events

                            #df=df[df["EventCode"].astype(int).isin(codes)]
                            all_dfs.append(df)
                            
        except Exception as e:
            print(f"Error processing {url}: {e}")

    Data = pd.concat(all_dfs, ignore_index=True)
    #clean data before saving
    Data = Data.drop(columns=["SQLDATE", "MonthYear","Year","ActionGeo_FeatureID","FractionDate","Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat","Actor1Geo_Long","Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code", "Actor2Geo_Lat","Actor2Geo_Long"])
    #drop duplicate articles
    Data = Data.drop_duplicates(subset=["SOURCEURL"],keep='last')
    #filter by CAMEO cat
    Data=Data[Data["EventCode"].astype(int).isin(scodes)]
#save Dataframe as excel file
# --- Save to Excel ---
    #print(Data)
    count_row = Data.shape[0]
    if count_row>6000000:
        df1 = my_dataframe.iloc[:600000,:]
        df2 = my_dataframe.iloc[600000:1200000,:]
        df3 = my_dataframe.iloc[1200000:,:]
        filename1 = Path(desk) /f"{Sourcename}_1"
        filename1 = Path(desk) /f"{Sourcename}_2"
        filename1 = Path(desk) /f"{Sourcename}_3"
        Data.to_excel(filename1, index=False, engine="xlsxwriter", engine_kwargs={'options': {'strings_to_urls': False}})
        Data.to_excel(filename2, index=False, engine="xlsxwriter",engine_kwargs={'options': {'strings_to_urls': False}})
        Data.to_excel(filename3, index=False, engine="xlsxwriter",engine_kwargs={'options': {'strings_to_urls': False}})
    else:
        filename = Path(desk) /f"{Sourcename}"
        Data.to_excel(filename, index=False, engine="xlsxwriter", engine_kwargs={'options': {'strings_to_urls': False}})
else:
    print("importing from local file")
    Data = pd.read_excel(Sourcename)

print("Import complted. Cleaning data")
#exit()
#Data=Data[(Data['Actor1Type1Code'] == "GOV") | (Data['Actor2Type1Code'] == "GOV")]


Data['GeoCountries']  = Data["Actor1Geo_CountryCode"]+"+"+Data["Actor2Geo_CountryCode"]
Data['Goldstein*diffusion'] = Data["GoldsteinScale"]*(Data["NumMentions"]+Data["NumSources"]+Data["NumArticles"])


# --- Filter by country ---
#Data = Data[(Data["GeoCountries"] == "UK") | (Data["GeoCountries"] == "ISLE")]
Data=Data[Data["GeoCountries"].str.contains("FR", na=False)|Data["GeoCountries"].str.contains("UK", na=False)]


#drop duplicates
Data=Data.drop_duplicates(subset=['SOURCEURL', 'GeoCountries'], keep='last')



# --- Fix DATEADDED ---
Data["DATEADDED"] = pd.to_datetime(Data["DATEADDED"], format="%Y%m%d%H%M%S", errors="coerce")
Data["DATEADDED_DISPLAY"] = Data["DATEADDED"].dt.strftime("%Y-%m-%d %H:%M:%S")

# Conversione in datetime
Data["datetime"] = pd.to_datetime(Data["DATEADDED"], format="%Y%m%d%H%M%S")

# Estrazione ora (arrotondata all’ora intera)
Data["hour"] = Data["datetime"].dt.to_period("h")

Data["Goldstein*NumArticles"] = Data["GoldsteinScale"] * Data["NumArticles"]
Data["AvgTone*NumArticles"] = Data["AvgTone"] * Data["NumArticles"]

#sort by datetime
Data = Data.sort_values('datetime')

#apply moving averages
countries = ['UK']
#calculate counterparts based on the provided list. Remember that overlaps will take the value of the last entry in the countries list
for country in countries:
    Data['Counterpart'] = np.where(Data['Actor1Geo_CountryCode'] == country, Data['Actor2Geo_CountryCode'], Data['Actor1Geo_CountryCode'])
    Data['Counterpart'] = np.where(Data['Actor2Geo_CountryCode'] == country, Data['Actor1Geo_CountryCode'], Data['Actor2Geo_CountryCode'])

masks = {country: Data['GeoCountries'].str.contains(country, na=False) for country in countries}
axes = plt.gca()
#print(masks.items())

#3 parameters to measure
#bilateral for the lowest 5
#per category of countries
#overall

print("building averages and plots")


#mask data for selected countries
for country, mask in masks.items():
    Data.loc[mask, 'Country'] = country
    subset = Data.loc[mask].copy()
    
    if subset.empty:
        print(f"No data for {country}, skipping.")
        continue

    # Sort by datetime so rolling and plotting work correctly
    subset = subset.sort_values('datetime')
    #write the country of reference that was masked for future usage
    
    #extract unique counterparts and create a mask
    counterparts=subset["Counterpart"].unique()
    #masks = {country: Data['GeoCountries'].str.contains(country, na=False) for country in countries}
    MaskCounterparts = {counterpart: subset['Counterpart'].str.contains(counterpart, na=False) for counterpart in counterparts}
    #(df[(df['state'] == 'NY') | (df['state'] == 'TX')])
    
    
    for counterpart in counterparts:
        CounterpartDF=subset[(subset['Country'].str.contains(country, na=False) | subset['Counterpart'].str.contains(counterpart, na=False))]
        #subset2 = np.where((Data['Country'].str==country) & (Data['Counterpart'].str==counterpart))
        #subset2=Data.loc[MaskCounterparts].copy()
        
        if CounterpartDF.empty:
            print(f"No data for {counterpart}, skipping.")
            continue
        print(CounterpartDF)
        exit()
        # Compute 1-day rolling mean
        subset[f'{counterpart}_MovingAvg'] = (
            subset.set_index('datetime')['GoldsteinScale']
            .rolling('1D', min_periods=1)
            .mean()
            .values
        )
        # Compute 7-day rolling mean
        subset[f'{counterpart}_7DMovingAvg'] = (
            subset.set_index('datetime')['AvgTone*NumArticles']
            .rolling('7D', min_periods=10)
            .mean()
            .values
        )
        
            # Compute 1M rolling mean
        subset[f'{counterpart}_1MMovingAvg'] = (
            subset.set_index('datetime')['AvgTone*NumArticles']
            .rolling('30D', min_periods=50)
            .mean()
            .values
        )
            # Compute 1-day rolling mean on  tone only
        subset[f'{counterpart}_1DMovingAvgTone'] = (
            subset.set_index('datetime')['AvgTone']
            .rolling('1D', min_periods=1)
            .mean()
            .values
        )
        
        

        # Save the moving avg back into Data (optional, to keep consistency)
        Data.loc[MaskCounterparts, f'{country}_MovingAvg'] = subset[f'{country}_MovingAvg'].values
        #saving this mask's country as first reference point 
        Data.loc[MaskCounterparts, 'Country'] = country
        #Data.loc[mask, 'Counterpart'] = np.where(Data['Actor1Geo_CountryCode'] == country, df['col2'], df['col1'])
        
        
        #mask2 = Data['Actor1Geo_CountryCode'].str.contains(country, na=False)
        #subset2=Data.loc[mask2].copy()
        #Data.loc[mask2, 'Counterpart'] = mask2['Actor2Geo_CountryCode']
        #mask2 = Data['Actor2Geo_CountryCode'].str.contains(country, na=False)
        #subset2=Data.loc[mask2].copy()
        #Data.loc[mask2, 'Counterpart'] = mask2['Actor1Geo_CountryCode']
    #    print(subset2)
    #    exit()
        #column_name = 'my_channel'
        #df.loc[mask, column_name] = 0
        #mask2 = Data['GeoCountries'].str.contains(country, na=False) for country in countries}
        
        #mask = df.my_channel > 20000
        #column_name = 'my_channel'
        #df.loc[mask, column_name] = 0

        # --- Plot ---
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Primary axis (original)
        ax1.plot(
            subset['datetime'],
            subset[f'{counterpart}_1DMovingAvgTone'],
            label='_1DMovingAvgTone',
            color='tab:blue',
            alpha=0.7
        )
        ax1.set_xlabel("Date")
        ax1.set_ylabel("1DMovingAvgTone", color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Secondary axis (1D moving avg Tone*Num)
        ax2 = ax1.twinx()
        ax2.plot(
            subset['datetime'],
            subset[f'{counterpart}_MovingAvg'],
            label='1D Moving Avg',
            color='tab:red',
            linewidth=2
        )
        ax2.set_ylabel("1D Moving Avg", color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        
        
            # 3 axis (7D moving avg)
        ax3 = ax1.twinx()
        ax3.plot(
            subset['datetime'],
            subset[f'{counterpart}_7DMovingAvg'],
            label='7D Moving Avg',
            color='tab:green',
            linewidth=1
        )
        ax3.set_ylabel("7D Moving Avg", color='tab:green')
        ax3.tick_params(axis='y', labelcolor='tab:green')
        
        # 4 axis (1 M moving avg)
        
        ax4 = ax1.twinx()
        ax4.plot(
            subset['datetime'],
            subset[f'{counterpart}_1MMovingAvg'],
            label='1M Moving Avg',
            color='tab:pink',
            linewidth=1
        )
        ax4.set_ylabel("1M Moving Avg", color='tab:pink')
        ax4.tick_params(axis='y', labelcolor='tab:pink')
        
        

        # --- Ensure full date range and proper tick spacing ---
        ax1.set_xlim(subset['datetime'].min(), subset['datetime'].max())  # full X range
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())             # smart tick spacing
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))   # readable format
        plt.xticks(rotation=45, ha='right')

        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        lines_3, labels_3 = ax3.get_legend_handles_labels()
        lines_4, labels_4 = ax4.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2+lines_3+lines_4, labels_1 + labels_2+labels_3+labels_4, loc='best')

        # Title, grid, layout
        plt.title(f"{country} — AvgTone * NumArticles vs 7D Moving Avg")
        ax1.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()

# Gravità conflittuale (solo eventi negativi)
Data["C"] = Data["GoldsteinScale"].apply(lambda x: max(0, -x))


# Contributo di rischio per evento
Data["EventRiskNegative*Numarticles"] = Data["C"] * Data["NumArticles"]


# --- Save to Excel ---
random_str = "".join(random.choices(string.ascii_letters + string.digits, k=8))
filename = Path(desk) /f"gdelt_2_{random_str}.xlsx"
Data.to_excel(filename, index=False, engine="xlsxwriter")

print(f"Saved {len(Data)} rows -> {filename}")


plt.show()













    

# Trasformazioni log(1+x) sulle metriche di diffusione
#Data["M"] = np.log1p(Data["NumMentions"])
#Data["S"] = np.log1p(Data["NumSources"])
#Data["A"] = np.log1p(Data["NumArticles"])

# Fattore di diffusione
#Data["D"] = Data["M"] + Data["S"] + Data["A"]





#consider also positive events
#Data["EventRiskAll"] = Data["GoldsteinScale"] * Data["D"]

#Data["DATEADDED"] = pd.to_datetime(Data["DATEADDED"], errors="coerce")
#Data = Data.dropna(subset=["DATEADDED"])
#Data = Data[Data['C'] > 0]






#plotting as financial 
#need for a cumulated


#ohlc = Data.set_index("DATEADDED")["EventRiskAll"].resample("D").ohlc()
#ohlc.index.name = "DATEADDED"  # make sure index is DateTime
#print(ohlc)

#mpf.plot(ohlc, type="candle", style="charles", title="Goldstein Index Candlestick", ylabel="Goldstein Scale")

#second plottint
#ohlc1 = Data.set_index("DATEADDED")["GoldsteinScale"].resample("D").ohlc()
#ohlc1.index.name = "DATEADDED"  # make sure index is DateTime
#print(ohlc1)

#mpf.plot(ohlc1, type="candle", style="charles", title="Goldstein Index Candlestick", ylabel="Goldstein Scale")






# Aggregazione per Paese e ora
#agg = (
#    Data.groupby(["ActionGeo_CountryCode", "hour"])
#    .agg(
#        RawRisk=("EventRiskNegative", "sum"),
#        MediaVol=("NumArticles", "sum"),
#        Events=("EventRiskNegative", "count")
#    )
#    .reset_index()
#)

# Normalizzazione rischio per volume mediale
#agg["Index"] = agg["RawRisk"] / (agg["MediaVol"] + 1)
#
# Standardizzazione z-score (rispetto a tutto il dataset)
#agg["Index_zscore"] = (agg["Index"] - agg["Index"].mean()) / agg["Index"].std()

# Risultati
#print(agg.head(20))

