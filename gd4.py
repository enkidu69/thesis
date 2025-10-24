import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import requests
import zipfile
import io
import csv
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime, timedelta
import gc
import signal
import sys
import random
import string
import time
#measure exec time
start_time = time.time()
#remove warning
pd.options.mode.chained_assignment = None  # default='warn'



# FIPS to country name mapping
FIPS_TO_ISO2 = {'AF': 'AFGHANISTAN','AL': 'ALBANIA','AG': 'ALGERIA','AQ': 'AMERICAN SAMOA','AN': 'ANDORRA','AO': 'ANGOLA','AV': 'ANGUILLA','AY': 'ANTARCTICA','AC': 'ANTIGUA AND BARBUDA','AR': 'ARGENTINA','AM': 'ARMENIA','AA': 'ARUBA','AT': 'ASHMORE AND CARTIER ISLANDS','AS': 'AUSTRALIA','AU': 'AUSTRIA','AJ': 'AZERBAIJAN','BF': 'BAHAMAS, THE','BA': 'BAHRAIN','FQ': 'BAKER ISLAND','BG': 'BANGLADESH','BB': 'BARBADOS','BS': 'BASSAS DA INDIA','BO': 'BELARUS','BE': 'BELGIUM','BH': 'BELIZE','BN': 'BENIN','BD': 'BERMUDA','BT': 'BHUTAN','BL': 'BOLIVIA','BK': 'BOSNIA AND HERZEGOVINA','BC': 'BOTSWANA','BV': 'BOUVET ISLAND','BR': 'BRAZIL','IO': 'BRITISH INDIAN OCEAN TERRITORY','VI': 'BRITISH VIRGIN ISLANDS','BX': 'BRUNEI','BU': 'BULGARIA','UV': 'BURKINA','BM': 'BURMA','BY': 'BURUNDI','CB': 'CAMBODIA','CM': 'CAMEROON','CA': 'CANADA','CV': 'CAPE VERDE','CJ': 'CAYMAN ISLANDS','CT': 'CENTRAL AFRICAN REPUBLIC','CD': 'CHAD','CI': 'CHILE','CH': 'CHINA','KT': 'CHRISTMAS ISLAND','IP': 'CLIPPERTON ISLAND','CK': 'COCOS (KEELING) ISLANDS','CO': 'COLOMBIA','CN': 'COMOROS','CF': 'CONGO','CW': 'COOK ISLANDS','CR': 'CORAL SEA ISLANDS','CS': 'COSTA RICA','IV': 'COTE DIVOIRE','HR':'CROATIA','CU': 'CUBA','CY': 'CYPRUS','EZ': 'CZECH REPUBLIC','DA': 'DENMARK','DJ': 'DJIBOUTI','DO': 'DOMINICA','DR': 'DOMINICAN REPUBLIC','EC': 'ECUADOR','EG': 'EGYPT','ES': 'EL SALVADOR','EK': 'EQUATORIAL GUINEA','ER': 'ERITREA','EN': 'ESTONIA','ET': 'ETHIOPIA','EU': 'EUROPA ISLAND','FK': 'FALKLAND ISLANDS (ISLAS MALVINAS)','FO': 'FAROE ISLANDS','FM': 'FEDERATED STATES OF MICRONESIA','FJ': 'FIJI','FI': 'FINLAND','FR': 'FRANCE','FG': 'FRENCH GUIANA','FP': 'FRENCH POLYNESIA','FS': 'FRENCH SOUTHERN AND ANTARCTIC LANDS','GB': 'GABON','GA': 'GAMBIA, THE','GZ': 'GAZA STRIP','GG': 'GEORGIA','GM': 'GERMANY state/land','GH': 'GHANA','GI': 'GIBRALTAR','GO': 'GLORIOSO ISLANDS','GR': 'GREECE','GL': 'GREENLAND','GJ': 'GRENADA','GP': 'GUADELOUPE','GQ': 'GUAM','GT': 'GUATEMALA','GK': 'GUERNSEY','GV': 'GUINEA','PU': 'GUINEA-BISSAU','GY': 'GUYANA','HA': 'HAITI','HM': 'HEARD ISLAND AND MCDONALD ISLANDS','HO': 'HONDURAS','HK': 'HONG KONG','HQ': 'HOWLAND ISLAND','HU': 'HUNGARY','IC': 'ICELAND','IN': 'INDIA','ID': 'INDONESIA','IR': 'IRAN','IZ': 'IRAQ','EI': 'IRELAND','IS': 'ISRAEL','IT': 'ITALY','JM': 'JAMAICA','JN': 'JAN MAYEN','JA': 'JAPAN','DQ': 'JARVIS ISLAND','JE': 'JERSEY','JQ': 'JOHNSTON ATOLL','JO': 'JORDAN','JU': 'JUAN DE NOVA ISLAND','KZ': 'KAZAKHSTAN','KE': 'KENYA','KQ': 'KINGMAN REEF','KR': 'KIRIBATI','KN': 'KOREA, DEMOCRATIC PEOPLES REPUBLIC OF','KS': 'KOREA, REPUBLIC OF','KU': 'KUWAIT','KG': 'KYRGYZSTAN ','LA': 'LAOS','LG': 'LATVIA','LE': 'LEBANON','LT': 'LESOTHO','LI': 'LIBERIA','LY': 'LIBYA','LS': 'LIECHTENSTEIN','LH': 'LITHUANIA','LU': 'LUXEMBOURG','MC': 'MACAU','MK': 'MACEDONIA','MA': 'MADAGASCAR','MI': 'MALAWI','MY': 'MALAYSIA','MV': 'MALDIVES','ML': 'MALI','MT': 'MALTA','IM': 'MAN, ISLE OF','RM': 'MARSHALL ISLANDS','MB': 'MARTINIQUE','MR': 'MAURITANIA','MP': 'MAURITIUS','MF': 'MAYOTTE','MX': 'MEXICO','MQ': 'MIDWAY ISLANDS','MD': 'MOLDOVA','MN': 'MONACO','MG': 'MONGOLIA','MW': 'MONTENEGRO','MH': 'MONTSERRAT','MO': 'MOROCCO','MZ': 'MOZAMBIQUE','WA': 'NAMIBIA','NR': 'NAURU','BQ': 'NAVASSA ISLAND','NP': 'NEPAL','NL': 'NETHERLANDS','NT': 'NETHERLANDS ANTILLES','NC': 'NEW CALEDONIA','NZ': 'NEW ZEALAND','NU': 'NICARAGUA','NG': 'NIGER','NI': 'NIGERIA','NE': 'NIUE','NF': 'NORFOLK ISLAND','CQ': 'NORTHERN MARIANA ISLANDS','NO': 'NORWAY','MU': 'OMAN','PK': 'PAKISTAN','LQ': 'PALMYRA ATOLL','PM': 'PANAMA','PP': 'PAPUA NEW GUINEA','PF': 'PARACEL ISLANDS','PA': 'PARAGUAY','PE': 'PERU','RP': 'PHILIPPINES','PC': 'PITCAIRN ISLANDS','PL': 'POLAND','PO': 'PORTUGAL','RQ': 'PUERTO RICO','QA': 'QATAR','RE': 'REUNION','RO': 'ROMANIA','RS': 'RUSSIA','RW': 'RWANDA','SC': 'ST. KITTS AND NEVIS','SH': 'ST. HELENA','ST': 'ST. LUCIA','SB': 'ST. PIERRE AND MIQUELON','VC': 'ST. VINCENT AND THE GRENADINES','SM': 'SAN MARINO','TP': 'SAO TOME AND PRINCIPE','SA': 'SAUDI ARABIA','SG': 'SENEGAL','SR': 'SERBIA','SE': 'SEYCHELLES','SL': 'SIERRA LEONE','SN': 'SINGAPORE','LO': 'SLOVAKIA','SI': 'SLOVENIA','BP': 'SOLOMON ISLANDS','SO': 'SOMALIA','SF': 'SOUTH AFRICA','SX': 'SOUTH GEORGIA AND THE SOUTH SANDWICH ISLANDS','SP': 'SPAIN','PG': 'SPRATLY ISLANDS','CE': 'SRI LANKA','SU': 'SUDAN','NS': 'SURINAME','SV': 'SVALBARD','WZ': 'SWAZILAND','SW': 'SWEDEN','SZ': 'SWITZERLAND','SY': 'SYRIA','TI': 'TAJIKISTAN','TZ': 'TANZANIA','TH': 'THAILAND','TO': 'TOGO','TL': 'TOKELAU','TN': 'TONGA','TD': 'TRINIDAD AND TOBAGO','TE': 'TROMELIN ISLAND','PS': 'TRUST TERRITORY OF THE PACIFIC ISLANDS (PALAU)','TS': 'TUNISIA','TU': 'TURKEY','TX': 'TURKMENISTAN','TK': 'TURKS AND CAICOS ISLANDS','TV': 'TUVALU','UG': 'UGANDA','UP': 'UKRAINE','TC': 'UNITED ARAB EMIRATES','UK': 'UNITED KINGDOM','UK': 'UNITED KINGDOM','UK': 'UNITED KINGDOM','UK': 'UNITED KINGDOM','US': 'UNITED STATES','UY': 'URUGUAY','UZ': 'UZBEKISTAN','NH': 'VANUATU','VT': 'VATICAN CITY','VE': 'VENEZUELA','VM': 'VIETNAM','VQ': 'VIRGIN ISLANDS','WQ': 'WAKE ISLAND','WF': 'WALLIS AND FUTUNA','WE': 'WEST BANK','WI': 'WESTERN SAHARA','WS': 'WESTERN SAMOA','YM': 'YEMEN','CG': 'ZAIRE','ZA': 'ZAMBIA','ZI': 'ZIMBABWE','TW': 'TAIWAN'}

# Global variable to track interruption
interrupted = False

syear=2023
#start_month=1 
#start_day=1
#end_month=12
#end_day=31
##start_date = datetime(year, start_month, start_day)
#end_date = datetime(year, end_month, end_day)
#current_date = start_date

def signal_handler(sig, frame):
    """Handle Ctrl+C interruption gracefully"""
    global interrupted
    interrupted = True
    print(f"\n\n⚠️  Process interrupted by user! Cleaning up...")
    sys.exit(1)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

def download_gdelt_day(year, month, day, retries=3):
    """
    Download GDELT data for a specific day using direct HTTP download
    """
    date_str = f"{year}{month:02d}{day:02d}"
    url = f"http://data.gdeltproject.org/events/{date_str}.export.CSV.zip"
    
    for attempt in range(retries):
        try:
            print(f"  Downloading {year}-{month:02d}-{day:02d} (attempt {attempt + 1})...")
            
            # Download the zip file
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Extract CSV from zip
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    # Read CSV content
                    content = f.read().decode('utf-8')
                    
            # Parse CSV into DataFrame
            df = pd.read_csv(io.StringIO(content), sep='\t', header=None, low_memory=False)
            
            # GDELT 1.0 column names
            columns = [
                'GLOBALEVENTID', 'SQLDATE', 'MonthYear', 'Year', 'FractionDate',
                'Actor1Code', 'Actor1Name', 'Actor1CountryCode', 'Actor1KnownGroupCode',
                'Actor1EthnicCode', 'Actor1Religion1Code', 'Actor1Religion2Code',
                'Actor1Type1Code', 'Actor1Type2Code', 'Actor1Type3Code',
                'Actor2Code', 'Actor2Name', 'Actor2CountryCode', 'Actor2KnownGroupCode',
                'Actor2EthnicCode', 'Actor2Religion1Code', 'Actor2Religion2Code',
                'Actor2Type1Code', 'Actor2Type2Code', 'Actor2Type3Code',
                'IsRootEvent', 'EventCode', 'EventBaseCode', 'EventRootCode',
                'QuadClass', 'GoldsteinScale', 'NumMentions', 'NumSources',
                'NumArticles', 'AvgTone', 'Actor1Geo_Type', 'Actor1Geo_FullName',
                'Actor1Geo_CountryCode', 'Actor1Geo_ADM1Code', 'Actor1Geo_Lat',
                'Actor1Geo_Long', 'Actor1Geo_FeatureID', 'Actor2Geo_Type',
                'Actor2Geo_FullName', 'Actor2Geo_CountryCode', 'Actor2Geo_ADM1Code',
                'Actor2Geo_Lat', 'Actor2Geo_Long', 'Actor2Geo_FeatureID',
                'ActionGeo_Type', 'ActionGeo_FullName', 'ActionGeo_CountryCode',
                'ActionGeo_ADM1Code', 'ActionGeo_Lat', 'ActionGeo_Long',
                'ActionGeo_FeatureID', 'DATEADDED', 'SOURCEURL'
            ]
            
            # Assign column names (only for available columns)
            df.columns = columns[:len(df.columns)]
            #print(df) OK
##########################################
            print(f"  ✓ Downloaded {len(df):,} events for {year}-{month:02d}-{day:02d}")
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Network error for {year}-{month:02d}-{day:02d}: {e}")
            if attempt < retries - 1:
                print(f"  Retrying...")
                continue
            else:
                return None
        except Exception as e:
            print(f"  ✗ Error processing {year}-{month:02d}-{day:02d}: {e}")
            return None
    
    return None

def download_gdelt_data_direct():
    """
    Download GDELT data directly from their servers day by day
    """

    
    temp_dir = 'temp_data_direct'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    syear=2023
    start_month=1
    start_day=1
    eyear=2024
    end_month=10
    end_day=23
    start_date = datetime(syear, start_month, start_day)
    end_date = datetime(eyear, end_month, end_day)
    current_date = start_date
    
    daily_files = []
    
    print(f"Downloading GDELT data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print("This may take a while as we download day by day...")
    
    while current_date <= end_date and not interrupted:
        year = current_date.year
        month = current_date.month
        day = current_date.day
        
        daily_filename = f"daily_{year}{month:02d}{day:02d}.pkl"
        daily_path = os.path.join(temp_dir, daily_filename)
        
        # Check if daily file already exists
        if os.path.exists(daily_path):
            print(f"✓ Found existing file: {daily_filename}")
            daily_files.append(daily_path)
        else:
            # Download daily data
            df_daily = download_gdelt_day(year, month, day)
            #print(df_daily) OK
            
            if df_daily is not None and len(df_daily) > 0:
                # Filter to essential columns immediately to save memory
                essential_columns = ['GLOBALEVENTID',
                    'SQLDATE', 'SOURCEURL', 'Actor1Geo_CountryCode', 'Actor2Geo_CountryCode',
                    'EventRootCode', 'EventBaseCode', 'EventCode', 'GoldsteinScale', 
                    'AvgTone', 'NumMentions', 'NumArticles', 'NumSources'
                ]

                available_columns = [col for col in essential_columns if col in df_daily.columns]
                df_filtered = df_daily[available_columns].copy()
                #print(df_filtered) OK
                # Convert FIPS country codes to country names
                #def convert_country_code(code):
                #    if pd.isna(code) or code == '':
                #        return code
                #    return FIPS_TO_ISO2.get(code, code)
                
                #if 'Actor1Geo_CountryCode' in df_filtered.columns:
                #    df_filtered['Actor1Geo_CountryCode'] = df_filtered['Actor1Geo_CountryCode'].apply(convert_country_code)
                #if 'Actor2Geo_CountryCode' in df_filtered.columns:
                #    df_filtered['Actor2Geo_CountryCode'] = df_filtered['Actor2Geo_CountryCode'].apply(convert_country_code)
                
                # Save daily file
                df_filtered.to_pickle(daily_path)
                
                daily_files.append(daily_path)
                print(f"  ✓ Saved daily data: {len(df_filtered):,} events")
                
                # Clear memory
                del df_daily, df_filtered
                gc.collect()
            else:
                print(f"  ✗ No data available for {year}-{month:02d}-{day:02d}")
                # Create empty file to mark this day as processed
                pd.DataFrame().to_pickle(daily_path)
                daily_files.append(daily_path)
        
        current_date += timedelta(days=1)
    
    return daily_files

def main():
    """
    SEQUENTIAL EXECUTION - Step by step process
    """
    global interrupted
    
    print("="*70)
    print("SEQUENTIAL GDELT DATA ANALYSIS - DIRECT DOWNLOAD")
    print("="*70)
    print("Press Ctrl+C at any time to interrupt the process")
    
    # STEP 1: INITIALIZE PARAMETERS
    print("\nSTEP 1: Initializing parameters...")
    
    # Fixed parameters
    YEAR = syear  # Using 2024 for actual data
    FOCAL_COUNTRIES = ['RS','IZ','IS','US', 'FR', 'UK', 'IT','SP']  # Countries we're analyzing
    
    # Event codes to filter for (political/diplomatic events)
    EVENT_CODES = ['012','016','0212','0214','0232','0233','0234','0243','0244','0252','0253','0254','0255','0256','026','027','028','0312','0314','032','0332','0333','0334','0354','0355','0356','036','037','038','039','046','050','051','052','053','054','055','056','057','06','060','061','062','063','064','071','072','073','074','075','0811','0812','0813','0814','082','083','0831','0832','0833','0834','0841','085','086','0861','0862','0863','087','0871','0872','0873','0874','092','093','094','1012','1014','102','1032','1033','1034','1041','1042','1043','1044','1052','1054','1055','1056','106','107','108','111','1121','1122','1123','1124','1125','113','114','115','116','121','1211','1212','122','1221','1222','1223','1224','123','1231','1232','1233','1234','124','1241','1242','1243','1244','1245','1246','125','126','127','128','129','130','131','1311','1312','1313','132','1321','1322','1323','1324','133','134','135','136','137','138','1381','1382','1383','1384','1385','139','140','141','1411','1412','1413','1414','142','1421','1422','1423','1424','143','1431','1432','1433','1434','144','1441','1442','1443','1444','145','1451','1452','1453','1454','150','151','152','153','154','155','16','160','161','162','1621','1622','1623','163','164','165','166','1661','1662','1663','1712','1721','1722','1723','1724','174','175','180','181','182','1821','1822','1823','183','1831','1832','1833','1834','184','185','186','190','191','192','193','194','195','1951','1952','196','200','201','202','203','204','2041','2042']

    print(f"Year: {YEAR}")
    print(f"Focal countries: {', '.join(FOCAL_COUNTRIES)}")
    print(f"Event codes: {len(EVENT_CODES)} types")

    # STEP 2: GET USER INPUT FOR COUNTERPARTS
    print("\nSTEP 2: Selecting counterpart countries...")

    # Default selection
    default_counterparts = ['RS','IZ','IS','US', 'FR', 'UK', 'IT','SP']
    print(f"Default counterparts: {', '.join(default_counterparts)}")

    # Simple user input
    print("\nOptions:")
    print("1. Use default counterparts")
    print("2. Enter custom counterparts")

    try:
        choice = "1"#input("Choose option (1/2): ").strip()
    except KeyboardInterrupt:
        print("\nInterrupted during user input!")
        return

    if choice == "1":
        selected_counterparts = default_counterparts
    else:
        try:
            custom_input = input("Enter country names separated by commas (e.g., RUSSIA,CHINA,IRAN): ").strip()
        except KeyboardInterrupt:
            print("\nInterrupted during user input!")
            return
            
        if custom_input:
            selected_counterparts = [name.strip().upper() for name in custom_input.split(',')]
            selected_counterparts = selected_counterparts
        else:
            print("No input provided. Using defaults.")
            selected_counterparts = default_counterparts

    print(f"Selected counterparts: {', '.join(selected_counterparts)}")

    # STEP 3: DOWNLOAD DATA DIRECTLY FROM GDELT
    print("\nSTEP 3: Downloading GDELT data directly...")
    
    daily_files = download_gdelt_data_direct()
    
    if interrupted:
        print("Download interrupted!")
        return

    print(f"✓ Downloaded {len(daily_files)} daily files")

    # STEP 4: PROCESS AND FILTER DATA
    print("\nSTEP 4: Processing and filtering data...")

    # Verify all daily files are readable
    valid_daily_files = []
    for daily_file in daily_files:
        try:
            test_df = pd.read_pickle(daily_file)
            if len(test_df) > 0:
                valid_daily_files.append(daily_file)
            else:
                print(f"⚠️  Empty daily file: {os.path.basename(daily_file)}")
        except Exception as e:
            print(f"⚠️  Corrupted daily file: {os.path.basename(daily_file)} - {e}")

    daily_files = valid_daily_files
    print(f"✓ Valid daily files ready for processing: {len(daily_files)}")

    all_country_data = []
    all_counterparts_found = set()

    # Process all daily files
    for i, daily_file in enumerate(daily_files):
        if interrupted:
            break
            
        print(f"Processing daily file {i+1}/{len(daily_files)}: {os.path.basename(daily_file)}")
        
        try:
            # Load daily file checked OK
            df_daily = pd.read_pickle(daily_file)
            #########print(df_daily) OK
            if len(df_daily) > 0:
                print("df daily from pickle file: OK")
            #else:
            
            
            # Get date from filename for verification
            filename = os.path.basename(daily_file)
            date_str = filename.replace('daily_', '').replace('.pkl', '')
            file_date = datetime.strptime(date_str, '%Y%m%d')
            print(f"  Processing date: {file_date.strftime('%Y-%m-%d')}")
            
            # Process each focal country
            daily_country_data = []
            for focal_country in FOCAL_COUNTRIES:
                # Filter for events where focal country is involved
                #print(focal_country)
                #print(df_daily)
                #daily = f'daily_scores_{YEAR}_.csv'
                #Data.to_excel(filename, index=False, engine="xlsxwriter")
                #df_daily.to_csv(daily, index=False)
                #print(focal_country)
                #exit()
                focal_filter = df_daily[
                    (df_daily['Actor1Geo_CountryCode'] == focal_country) | 
                    (df_daily['Actor2Geo_CountryCode'] == focal_country)
                ]

                df_filtered = focal_filter
                
                # Filter by event codes
                if EVENT_CODES and len(df_filtered) > 0:
#######################print(EVENT_CODES) REMOVED TEMPORAROY
                    #df_filtered = df_filtered[df_filtered['EventBaseCode'].isin(EVENT_CODES)]
                    #print(df_filtered)
                    #Data=Data[Data["EventCode"].astype(int).isin(scodes)]
                    if len(df_filtered) > 0:
                        # Create relationship column
                        def get_relationship_pair(row):
                            if row['Actor1Geo_CountryCode'] == focal_country:
                                counterpart = row['Actor2Geo_CountryCode']
                            else:
                                counterpart = row['Actor1Geo_CountryCode']
                            
                            # Collect all counterparts found
                            if pd.notna(counterpart) and counterpart != '':
                                all_counterparts_found.add(counterpart)
                            
                            return f"{focal_country}-{counterpart}"
                    
                    df_filtered['RelationshipPair'] = df_filtered.apply(get_relationship_pair, axis=1)
                    df_filtered['FocalCountry'] = focal_country

                    
                    # Convert date to proper format
                    df_filtered['Date'] = pd.to_datetime(df_filtered['SQLDATE'].astype(str), format='%Y%m%d', errors='coerce')
                    
                    # Drop rows with invalid dates
                    df_filtered = df_filtered.dropna(subset=['Date'])
                    
                    daily_country_data.append(df_filtered)
                    print(f"  {focal_country}: {len(df_filtered)} events")
            
            all_country_data.extend(daily_country_data)
            
            # Clear memory
            del df_daily, daily_country_data
            gc.collect()
            
        except Exception as e:
            print(f"  ✗ Error processing daily file {daily_file}: {e}")
            continue

    if interrupted:
        print("Processing interrupted!")
        return

    # Combine all data
    if all_country_data:
        combined_df = pd.concat(all_country_data, ignore_index=True)
        
        # Verify date range in combined data
        if not combined_df.empty:
            min_combined_date = combined_df['Date'].min()
            max_combined_date = combined_df['Date'].max()
            print(f"✓ Combined data date range: {min_combined_date.strftime('%Y-%m-%d')} to {max_combined_date.strftime('%Y-%m-%d')}")
        
        print(f"✓ Combined {len(combined_df):,} total events")
        
        # Filter by selected counterparts
        print(f"Filtering for selected counterparts: {', '.join(selected_counterparts)}")
        counterpart_filter = False
        for focal_country in FOCAL_COUNTRIES:
            for counterpart in selected_counterparts:
                
                relationship = f"{focal_country}-{counterpart}"
                counterpart_filter |= (combined_df['RelationshipPair'] == relationship)


        before_filter = len(combined_df)
        combined_df = combined_df[counterpart_filter].copy()
        
        print(f"✓ Filtered to {len(combined_df):,} events (from {before_filter:,})")
        
        # Final date range check
        if not combined_df.empty:
            final_min_date = combined_df['Date'].min()
            final_max_date = combined_df['Date'].max()
            print(f"✓ Final data date range: {final_min_date.strftime('%Y-%m-%d')} to {final_max_date.strftime('%Y-%m-%d')}")
    else:
        print("✗ No data found after processing!")
        return

    # [Rest of the processing steps 5-8 remain similar to before...]
    # STEP 5: CALCULATE NEGATIVITY SCORES
    print("\nSTEP 5: Calculating negativity scores...")

    if len(combined_df) == 0:
        print("✗ No data to calculate scores!")
        return

    # Group data by date, country, and relationship
    daily_relationships = combined_df.groupby(['Date', 'FocalCountry', 'RelationshipPair']).agg({
        'AvgTone': ['mean', 'std', 'count'],
        'GoldsteinScale': ['mean', 'std'],
        'EventBaseCode': 'count',
        'NumMentions': 'sum',
        'NumArticles': 'sum',
        'NumSources': 'sum'
    }).round(4)

    # Flatten column names
    daily_relationships.columns = ['_'.join(col).strip() for col in daily_relationships.columns.values]
    daily_relationships = daily_relationships.rename(columns={
        'AvgTone_mean': 'Daily_AvgTone',
        'AvgTone_std': 'Daily_AvgTone_Std',
        'AvgTone_count': 'Daily_AvgTone_Count',
        'GoldsteinScale_mean': 'Daily_Goldstein',
        'GoldsteinScale_std': 'Daily_Goldstein_Std',
        'EventBaseCode_count': 'Daily_EventCount',
        'NumMentions_sum': 'Daily_NumMentions',
        'NumArticles_sum': 'Daily_NumArticles',
        'NumSources_sum': 'Daily_NumSources'
    }).reset_index()

    # Filter by year
    #daily_relationships = daily_relationships[daily_relationships['Date'].dt.year == YEAR]

    # Calculate scores for each relationship
    daily_scores = pd.DataFrame()

    for focal_country in FOCAL_COUNTRIES:
        if interrupted:
            break
            
        country_data = daily_relationships[daily_relationships['FocalCountry'] == focal_country].copy()
        
        for relationship in country_data['RelationshipPair'].unique():
            rel_data = country_data[country_data['RelationshipPair'] == relationship].copy()
            rel_data = rel_data.sort_values('Date')
            
            # Use 28-day rolling window
            window = min(28, len(rel_data))
            
            # Calculate rolling statistics
            rel_data['AvgTone_Rolling_Mean'] = rel_data['Daily_AvgTone'].rolling(window, min_periods=1).mean()
            rel_data['AvgTone_Rolling_Std'] = rel_data['Daily_AvgTone'].rolling(window, min_periods=1).std()
            
            # Calculate negativity scores
            rel_data['Negativity_Score'] = -rel_data['Daily_AvgTone']  # Negative tone = more negative
            rel_data['Negativity_ZScore'] = -rel_data['AvgTone_Rolling_Mean'] / rel_data['AvgTone_Rolling_Std']
            rel_data['Negativity_ZScore'] = rel_data['Negativity_ZScore'].replace([np.inf, -np.inf], np.nan).fillna(0)
            rel_data['Composite_Negativity_Score'] = rel_data['Negativity_Score'] + rel_data['Negativity_ZScore']
            
            # Calculate impact indices (tone × volume)
            rel_data['Tone_Article_Index'] = rel_data['Daily_AvgTone'] * rel_data['Daily_NumArticles']
            rel_data['Negativity_Article_Index'] = rel_data['Negativity_Score'] * rel_data['Daily_NumArticles']
            
            # Calculate z-score for Tone-Article Index
            rel_data['Tone_Article_Rolling_Mean'] = rel_data['Tone_Article_Index'].rolling(window, min_periods=1).mean()
            rel_data['Tone_Article_Rolling_Std'] = rel_data['Tone_Article_Index'].rolling(window, min_periods=1).std()
            rel_data['Tone_Article_ZScore'] = (rel_data['Tone_Article_Index'] - rel_data['Tone_Article_Rolling_Mean']) / rel_data['Tone_Article_Rolling_Std']
            rel_data['Tone_Article_ZScore'] = rel_data['Tone_Article_ZScore'].replace([np.inf, -np.inf], np.nan).fillna(0)
            
            daily_scores = pd.concat([daily_scores, rel_data])

    if interrupted:
        print("Score calculation interrupted!")
        return
    #remove first 3 chars from rel pair to retrieve counterpart
    daily_scores['Counterpart'] = daily_scores['RelationshipPair'].str[3:]
    #print(daily_scores['Counterpart'])
    #create a new rel pair that is unique per pair
    
    daily_scores['newrel']=np.where(daily_scores['Counterpart']<daily_scores['FocalCountry'],daily_scores['Counterpart']+daily_scores['FocalCountry'],daily_scores['FocalCountry']+daily_scores['Counterpart'] )

    print(f"✓ Calculated scores for {len(daily_scores)} daily relationships")

    # STEP 6: CREATE VISUALIZATION REPORT
    import random
    import string
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    country_str = '_'.join(FOCAL_COUNTRIES)
    pdf_path = os.path.join(f'{country_str}_Analysis_{YEAR}_{random_str}.pdf')

    print(f"PDF will be saved as: {pdf_path}")

    # Create color palette for focal countries (distinct colors)
    focal_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    focal_color_dict = dict(zip(FOCAL_COUNTRIES, focal_colors[:len(FOCAL_COUNTRIES)]))

    # Create line styles for counterparts
    counterpart_styles = ['-', '--', '-.', ':']
    counterpart_style_dict = {}
    for i, counterpart in enumerate(selected_counterparts):
        counterpart_style_dict[counterpart] = counterpart_styles[i % len(counterpart_styles)]

    try:
        with PdfPages(pdf_path) as pdf:
            plt.style.use('default')
            
            # PAGE 1: Overview - Focus on showing different colors for FR and UK clearly
            print("Creating overview page...")
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # Larger figure for better visibility
            fig.suptitle(f'Bilateral Analysis: {" vs ".join(FOCAL_COUNTRIES)} ({YEAR})\nTime Period', 
                        fontsize=16, fontweight='bold')
            
            # Get the full date range for setting x-axis limits
            if not daily_scores.empty:
                min_date = daily_scores['Date'].min()
                max_date = daily_scores['Date'].max()
            else:
                min_date = start_date
                max_date = end_date
            
            # Plot 1: Negativity Scores - Different colors for focal countries
            for counterpart in selected_counterparts:
                for focal_country in FOCAL_COUNTRIES:
                    rel_data = daily_scores[
                        (daily_scores['FocalCountry'] == focal_country) & 
                        (daily_scores['RelationshipPair'] == f"{focal_country}-{counterpart}")
                    ]
                    if not rel_data.empty:
                        color = focal_color_dict[focal_country]
                        linestyle = counterpart_style_dict[counterpart]
                        line = axes[0, 0].plot(rel_data['Date'], rel_data['Negativity_Score'], 
                                             label=f'{focal_country}-{counterpart}', 
                                             color=color, linestyle=linestyle, linewidth=2.5)
            
            axes[0, 0].set_title('Daily Negativity Scores\n(Higher = More Negative)', fontweight='bold', fontsize=12)
            axes[0, 0].set_ylabel('Negativity Score', fontsize=10)
            axes[0, 0].set_xlim([min_date, max_date])  # Ensure full time period is visible
            axes[0, 0].xaxis.set_major_locator(plt.MaxNLocator(8))  # Reasonable number of date labels
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Tone-Article Impact Z-Score - Different colors for focal countries
            for counterpart in selected_counterparts:
                for focal_country in FOCAL_COUNTRIES:
                    rel_data = daily_scores[
                        (daily_scores['FocalCountry'] == focal_country) & 
                        (daily_scores['RelationshipPair'] == f"{focal_country}-{counterpart}")
                    ]
                    if not rel_data.empty:
                        color = focal_color_dict[focal_country]
                        linestyle = counterpart_style_dict[counterpart]
                        axes[0, 1].plot(rel_data['Date'], rel_data['Tone_Article_ZScore'], 
                                       label=f'{focal_country}-{counterpart}', 
                                       color=color, linestyle=linestyle, linewidth=2.5)
            
            axes[0, 1].set_title('Tone-Article Impact Z-Score\n(Z-score of Tone × Articles)', fontweight='bold', fontsize=12)
            axes[0, 1].set_ylabel('Z-Score', fontsize=10)
            axes[0, 1].set_xlim([min_date, max_date])  # Ensure full time period is visible
            axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.7, label='Mean')
            axes[0, 1].axhline(y=2, color='red', linestyle=':', alpha=0.7, label='+2 Std Dev')
            axes[0, 1].axhline(y=-2, color='red', linestyle=':', alpha=0.7, label='-2 Std Dev')
            axes[0, 1].xaxis.set_major_locator(plt.MaxNLocator(8))
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Event Counts by Focal Country (separate bars for each focal country)
            event_counts_by_focal = {}
            for focal_country in FOCAL_COUNTRIES:
                event_counts_by_focal[focal_country] = []
                for counterpart in selected_counterparts:
                    rel_data = daily_scores[
                        (daily_scores['FocalCountry'] == focal_country) & 
                        (daily_scores['RelationshipPair'] == f"{focal_country}-{counterpart}")
                    ]
                    total_events = rel_data['Daily_EventCount'].sum() if not rel_data.empty else 0
                    event_counts_by_focal[focal_country].append(total_events)
            
            # Create grouped bar chart to clearly show focal country differences
            x_pos = np.arange(len(selected_counterparts))
            bar_width = 0.35
            
            for i, focal_country in enumerate(FOCAL_COUNTRIES):
                color = focal_color_dict[focal_country]
                offset = i * bar_width
                axes[1, 0].bar(x_pos + offset, event_counts_by_focal[focal_country], bar_width, 
                              label=focal_country, color=color, alpha=0.8)
            
            axes[1, 0].set_title('Total Event Count by Counterpart and Focal Country', fontweight='bold', fontsize=12)
            axes[1, 0].set_ylabel('Total Events', fontsize=10)
            axes[1, 0].set_xlabel('Counterparts', fontsize=10)
            axes[1, 0].set_xticks(x_pos + bar_width * (len(FOCAL_COUNTRIES) - 1) / 2)
            axes[1, 0].set_xticklabels(selected_counterparts)
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            # Plot 4: Average Tone Comparison - Clear focal country comparison
            avg_tone_by_focal = {}
            for focal_country in FOCAL_COUNTRIES:
                avg_tone_by_focal[focal_country] = []
                tone_labels = []
                for counterpart in selected_counterparts:
                    rel_data = daily_scores[
                        (daily_scores['FocalCountry'] == focal_country) & 
                        (daily_scores['RelationshipPair'] == f"{focal_country}-{counterpart}")
                    ]
                    if not rel_data.empty:
                        avg_tone = rel_data['Daily_AvgTone'].mean()
                        avg_tone_by_focal[focal_country].append(avg_tone)
                    else:
                        avg_tone_by_focal[focal_country].append(0)
            
            # Create grouped bar chart for average tone
            for i, focal_country in enumerate(FOCAL_COUNTRIES):
                color = focal_color_dict[focal_country]
                offset = i * bar_width
                axes[1, 1].bar(x_pos + offset, avg_tone_by_focal[focal_country], bar_width, 
                              label=focal_country, color=color, alpha=0.8)
            
            axes[1, 1].set_title('Average Tone by Counterpart and Focal Country', fontweight='bold', fontsize=12)
            axes[1, 1].set_ylabel('Average Tone', fontsize=10)
            axes[1, 1].set_xlabel('Counterparts', fontsize=10)
            axes[1, 1].set_xticks(x_pos + bar_width * (len(FOCAL_COUNTRIES) - 1) / 2)
            axes[1, 1].set_xticklabels(selected_counterparts)
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            # Add a text box with color explanation
            color_explanation = "Color Guide:\n" + "\n".join([f"{country}: {focal_color_dict[country]}" for country in FOCAL_COUNTRIES])
            fig.text(0.02, 0.02, color_explanation, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # PAGE 2: Time Series Comparison - Focus on clear focal country comparison over time
            print("Creating time series comparison page...")
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Time Series Analysis: Focal Country Comparison ({YEAR})', 
                        fontsize=16, fontweight='bold')
            
            # Plot 1: Negativity Score Comparison (aggregated by focal country)
            for focal_country in FOCAL_COUNTRIES:
                # Aggregate data for this focal country across all counterparts
                country_data = daily_scores[daily_scores['FocalCountry'] == focal_country]
                if not country_data.empty:
                    daily_agg = country_data.groupby('Date').agg({
                        'Negativity_Score': 'mean',
                        'Daily_EventCount': 'sum'
                    }).reset_index()
                    
                    color = focal_color_dict[focal_country]
                    axes[0, 0].plot(daily_agg['Date'], daily_agg['Negativity_Score'], 
                                   label=focal_country, color=color, linewidth=3)
            
            axes[0, 0].set_title('Average Daily Negativity Score\n(All Counterparts Combined)', fontweight='bold', fontsize=12)
            axes[0, 0].set_ylabel('Negativity Score', fontsize=10)
            axes[0, 0].set_xlim([min_date, max_date])
            axes[0, 0].xaxis.set_major_locator(plt.MaxNLocator(8))
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Tone-Article Z-Score Comparison (aggregated by focal country)
            for focal_country in FOCAL_COUNTRIES:
                country_data = daily_scores[daily_scores['FocalCountry'] == focal_country]
                if not country_data.empty:
                    daily_agg = country_data.groupby('Date').agg({
                        'Tone_Article_ZScore': 'mean'
                    }).reset_index()
                    
                    color = focal_color_dict[focal_country]
                    axes[0, 1].plot(daily_agg['Date'], daily_agg['Tone_Article_ZScore'], 
                                   label=focal_country, color=color, linewidth=3)
            
            axes[0, 1].set_title('Average Tone-Article Z-Score\n(All Counterparts Combined)', fontweight='bold', fontsize=12)
            axes[0, 1].set_ylabel('Z-Score', fontsize=10)
            axes[0, 1].set_xlim([min_date, max_date])
            axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.7)
            axes[0, 1].axhline(y=2, color='red', linestyle=':', alpha=0.7)
            axes[0, 1].axhline(y=-2, color='red', linestyle=':', alpha=0.7)
            axes[0, 1].xaxis.set_major_locator(plt.MaxNLocator(8))
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Event Volume Comparison
            for focal_country in FOCAL_COUNTRIES:
                country_data = daily_scores[daily_scores['FocalCountry'] == focal_country]
                if not country_data.empty:
                    daily_agg = country_data.groupby('Date').agg({
                        'Daily_EventCount': 'sum'
                    }).reset_index()
                    
                    color = focal_color_dict[focal_country]
                    axes[1, 0].plot(daily_agg['Date'], daily_agg['Daily_EventCount'], 
                                   label=focal_country, color=color, linewidth=3)
            
            axes[1, 0].set_title('Daily Event Volume\n(All Counterparts Combined)', fontweight='bold', fontsize=12)
            axes[1, 0].set_ylabel('Number of Events', fontsize=10)
            axes[1, 0].set_xlim([min_date, max_date])
            axes[1, 0].xaxis.set_major_locator(plt.MaxNLocator(8))
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Average Tone Comparison
            for focal_country in FOCAL_COUNTRIES:
                country_data = daily_scores[daily_scores['FocalCountry'] == focal_country]
                if not country_data.empty:
                    daily_agg = country_data.groupby('Date').agg({
                        'Daily_AvgTone': 'mean'
                    }).reset_index()
                    
                    color = focal_color_dict[focal_country]
                    axes[1, 1].plot(daily_agg['Date'], daily_agg['Daily_AvgTone'], 
                                   label=focal_country, color=color, linewidth=3)
            
            axes[1, 1].set_title('Average Daily Tone\n(All Counterparts Combined)', fontweight='bold', fontsize=12)
            axes[1, 1].set_ylabel('Average Tone', fontsize=10)
            axes[1, 1].set_xlim([min_date, max_date])
            axes[1, 1].xaxis.set_major_locator(plt.MaxNLocator(8))
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Additional pages for individual counterparts (existing code)
            print("Creating individual counterpart pages...")
            for counterpart in selected_counterparts:
                if interrupted:
                    break
                    
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                fig.suptitle(f'Detailed Analysis: Relations with {counterpart}', fontsize=14, fontweight='bold')
                
                metrics = ['Negativity_Score', 'Tone_Article_ZScore', 'Daily_EventCount', 'Daily_AvgTone']
                titles = ['Negativity Score', 'Tone-Article Z-Score', 'Daily Event Count', 'Average Tone']
                
                for idx, (metric, title) in enumerate(zip(metrics, titles)):
                    row = idx // 2
                    col = idx % 2
                    
                    for focal_country in FOCAL_COUNTRIES:
                        rel_data = daily_scores[
                            (daily_scores['FocalCountry'] == focal_country) & 
                            (daily_scores['RelationshipPair'] == f"{focal_country}-{counterpart}")
                        ]
                        if not rel_data.empty:
                            color = focal_color_dict[focal_country]
                            axes[row, col].plot(rel_data['Date'], rel_data[metric], 
                                              label=focal_country, linewidth=2.5, marker='o',
                                              color=color, markersize=4, markevery=7)
                    
                    axes[row, col].set_title(title, fontweight='bold')
                    axes[row, col].set_ylabel(title)
                    axes[row, col].set_xlim([min_date, max_date])
                    axes[row, col].xaxis.set_major_locator(plt.MaxNLocator(6))
                    axes[row, col].tick_params(axis='x', rotation=45)
                    axes[row, col].legend()
                    axes[row, col].grid(True, alpha=0.3)
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

        print(f"✓ PDF report saved to: {pdf_path}")

    except Exception as e:
        print(f"✗ Error creating PDF report: {e}")



        
    

    # STEP 7: SAVE RESULTS
    print("\nSTEP 7: Saving results...")

    # Save combined data with random string
    combined_filename = f"combined_data_{YEAR}_{random_str}.pkl"
    #combined_df.to_pickle(combined_filename)
    print(f"✓ Combined data saved to: {combined_filename}")

    # Save daily scores with random string
    scores_filename = f'daily_scores_{YEAR}_{random_str}.csv'
    daily_scores.to_csv(scores_filename, index=False)
    print(f"✓ Daily scores saved to: {scores_filename}")


    # STEP 8: SUMMARY
    print("\n" + "="*70)
    print("PROCESSING COMPLETE - SUMMARY")
    print("="*70)
    print(f"✓ Year analyzed: {YEAR}")
    print(f"✓ Focal countries: {', '.join(FOCAL_COUNTRIES)}")
    print(f"✓ Counterparts analyzed: {', '.join(selected_counterparts)}")
    print(f"✓ Total events processed: {len(combined_df):,}")
    print(f"✓ Daily scores calculated: {len(daily_scores):,}")

    # Show actual date ranges from data
    if not combined_df.empty:
        data_start = combined_df['Date'].min().strftime('%Y-%m-%d')
        data_end = combined_df['Date'].max().strftime('%Y-%m-%d')
        print(f"✓ Actual data date range: {data_start} to {data_end}")

    print(f"✓ Report generated: {pdf_path}")
    print(f"✓ Data files saved: {combined_filename}, {scores_filename}")
    print("--- %s seconds ---" % (time.time() - start_time))

# Run the sequential analysis
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user! Exiting...")
    except Exception as e:
        print(f"\n\nError during execution: {e}")
        import traceback
        traceback.print_exc()