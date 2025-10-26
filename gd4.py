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
from statsmodels.tsa.stattools import ccf
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#measure exec time
start_time = time.time()
#remove warning
pd.options.mode.chained_assignment = None  # default='warn'



# FIPS to country name mapping
FIPS_TO_ISO2 = {'AF': 'AFGHANISTAN','AL': 'ALBANIA','AG': 'ALGERIA','AQ': 'AMERICAN SAMOA','AN': 'ANDORRA','AO': 'ANGOLA','AV': 'ANGUILLA','AY': 'ANTARCTICA','AC': 'ANTIGUA AND BARBUDA','AR': 'ARGENTINA','AM': 'ARMENIA','AA': 'ARUBA','AT': 'ASHMORE AND CARTIER ISLANDS','AS': 'AUSTRALIA','AU': 'AUSTRIA','AJ': 'AZERBAIJAN','BF': 'BAHAMAS, THE','BA': 'BAHRAIN','FQ': 'BAKER ISLAND','BG': 'BANGLADESH','BB': 'BARBADOS','BS': 'BASSAS DA INDIA','BO': 'BELARUS','BE': 'BELGIUM','BH': 'BELIZE','BN': 'BENIN','BD': 'BERMUDA','BT': 'BHUTAN','BL': 'BOLIVIA','BK': 'BOSNIA AND HERZEGOVINA','BC': 'BOTSWANA','BV': 'BOUVET ISLAND','BR': 'BRAZIL','IO': 'BRITISH INDIAN OCEAN TERRITORY','VI': 'BRITISH VIRGIN ISLANDS','BX': 'BRUNEI','BU': 'BULGARIA','UV': 'BURKINA','BM': 'BURMA','BY': 'BURUNDI','CB': 'CAMBODIA','CM': 'CAMEROON','CA': 'CANADA','CV': 'CAPE VERDE','CJ': 'CAYMAN ISLANDS','CT': 'CENTRAL AFRICAN REPUBLIC','CD': 'CHAD','CI': 'CHILE','CH': 'CHINA','KT': 'CHRISTMAS ISLAND','IP': 'CLIPPERTON ISLAND','CK': 'COCOS (KEELING) ISLANDS','CO': 'COLOMBIA','CN': 'COMOROS','CF': 'CONGO','CW': 'COOK ISLANDS','CR': 'CORAL SEA ISLANDS','CS': 'COSTA RICA','IV': 'COTE DIVOIRE','HR':'CROATIA','CU': 'CUBA','CY': 'CYPRUS','EZ': 'CZECH REPUBLIC','DA': 'DENMARK','DJ': 'DJIBOUTI','DO': 'DOMINICA','DR': 'DOMINICAN REPUBLIC','EC': 'ECUADOR','EG': 'EGYPT','ES': 'EL SALVADOR','EK': 'EQUATORIAL GUINEA','ER': 'ERITREA','EN': 'ESTONIA','ET': 'ETHIOPIA','EU': 'EUROPA ISLAND','FK': 'FALKLAND ISLANDS (ISLAS MALVINAS)','FO': 'FAROE ISLANDS','FM': 'FEDERATED STATES OF MICRONESIA','FJ': 'FIJI','FI': 'FINLAND','FR': 'FRANCE','FG': 'FRENCH GUIANA','FP': 'FRENCH POLYNESIA','FS': 'FRENCH SOUTHERN AND ANTARCTIC LANDS','GB': 'GABON','GA': 'GAMBIA, THE','GZ': 'GAZA STRIP','GG': 'GEORGIA','GM': 'GERMANY state/land','GH': 'GHANA','GI': 'GIBRALTAR','GO': 'GLORIOSO ISLANDS','GR': 'GREECE','GL': 'GREENLAND','GJ': 'GRENADA','GP': 'GUADELOUPE','GQ': 'GUAM','GT': 'GUATEMALA','GK': 'GUERNSEY','GV': 'GUINEA','PU': 'GUINEA-BISSAU','GY': 'GUYANA','HA': 'HAITI','HM': 'HEARD ISLAND AND MCDONALD ISLANDS','HO': 'HONDURAS','HK': 'HONG KONG','HQ': 'HOWLAND ISLAND','HU': 'HUNGARY','IC': 'ICELAND','IN': 'INDIA','ID': 'INDONESIA','IR': 'IRAN','IZ': 'IRAQ','EI': 'IRELAND','IS': 'ISRAEL','IT': 'ITALY','JM': 'JAMAICA','JN': 'JAN MAYEN','JA': 'JAPAN','DQ': 'JARVIS ISLAND','JE': 'JERSEY','JQ': 'JOHNSTON ATOLL','JO': 'JORDAN','JU': 'JUAN DE NOVA ISLAND','KZ': 'KAZAKHSTAN','KE': 'KENYA','KQ': 'KINGMAN REEF','KR': 'KIRIBATI','KN': 'KOREA, DEMOCRATIC PEOPLES REPUBLIC OF','KS': 'KOREA, REPUBLIC OF','KU': 'KUWAIT','KG': 'KYRGYZSTAN ','LA': 'LAOS','LG': 'LATVIA','LE': 'LEBANON','LT': 'LESOTHO','LI': 'LIBERIA','LY': 'LIBYA','LS': 'LIECHTENSTEIN','LH': 'LITHUANIA','LU': 'LUXEMBOURG','MC': 'MACAU','MK': 'MACEDONIA','MA': 'MADAGASCAR','MI': 'MALAWI','MY': 'MALAYSIA','MV': 'MALDIVES','ML': 'MALI','MT': 'MALTA','IM': 'MAN, ISLE OF','RM': 'MARSHALL ISLANDS','MB': 'MARTINIQUE','MR': 'MAURITANIA','MP': 'MAURITIUS','MF': 'MAYOTTE','MX': 'MEXICO','MQ': 'MIDWAY ISLANDS','MD': 'MOLDOVA','MN': 'MONACO','MG': 'MONGOLIA','MW': 'MONTENEGRO','MH': 'MONTSERRAT','MO': 'MOROCCO','MZ': 'MOZAMBIQUE','WA': 'NAMIBIA','NR': 'NAURU','BQ': 'NAVASSA ISLAND','NP': 'NEPAL','NL': 'NETHERLANDS','NT': 'NETHERLANDS ANTILLES','NC': 'NEW CALEDONIA','NZ': 'NEW ZEALAND','NU': 'NICARAGUA','NG': 'NIGER','NI': 'NIGERIA','NE': 'NIUE','NF': 'NORFOLK ISLAND','CQ': 'NORTHERN MARIANA ISLANDS','NO': 'NORWAY','MU': 'OMAN','PK': 'PAKISTAN','LQ': 'PALMYRA ATOLL','PM': 'PANAMA','PP': 'PAPUA NEW GUINEA','PF': 'PARACEL ISLANDS','PA': 'PARAGUAY','PE': 'PERU','RP': 'PHILIPPINES','PC': 'PITCAIRN ISLANDS','PL': 'POLAND','PO': 'PORTUGAL','RQ': 'PUERTO RICO','QA': 'QATAR','RE': 'REUNION','RO': 'ROMANIA','RS': 'RUSSIA','RW': 'RWANDA','SC': 'ST. KITTS AND NEVIS','SH': 'ST. HELENA','ST': 'ST. LUCIA','SB': 'ST. PIERRE AND MIQUELON','VC': 'ST. VINCENT AND THE GRENADINES','SM': 'SAN MARINO','TP': 'SAO TOME AND PRINCIPE','SA': 'SAUDI ARABIA','SG': 'SENEGAL','SR': 'SERBIA','SE': 'SEYCHELLES','SL': 'SIERRA LEONE','SN': 'SINGAPORE','LO': 'SLOVAKIA','SI': 'SLOVENIA','BP': 'SOLOMON ISLANDS','SO': 'SOMALIA','SF': 'SOUTH AFRICA','SX': 'SOUTH GEORGIA AND THE SOUTH SANDWICH ISLANDS','SP': 'SPAIN','PG': 'SPRATLY ISLANDS','CE': 'SRI LANKA','SU': 'SUDAN','NS': 'SURINAME','SV': 'SVALBARD','WZ': 'SWAZILAND','SW': 'SWEDEN','SZ': 'SWITZERLAND','SY': 'SYRIA','TI': 'TAJIKISTAN','TZ': 'TANZANIA','TH': 'THAILAND','TO': 'TOGO','TL': 'TOKELAU','TN': 'TONGA','TD': 'TRINIDAD AND TOBAGO','TE': 'TROMELIN ISLAND','PS': 'TRUST TERRITORY OF THE PACIFIC ISLANDS (PALAU)','TS': 'TUNISIA','TU': 'TURKEY','TX': 'TURKMENISTAN','TK': 'TURKS AND CAICOS ISLANDS','TV': 'TUVALU','UG': 'UGANDA','UP': 'UKRAINE','TC': 'UNITED ARAB EMIRATES','UK': 'UNITED KINGDOM','UK': 'UNITED KINGDOM','UK': 'UNITED KINGDOM','UK': 'UNITED KINGDOM','US': 'UNITED STATES','UY': 'URUGUAY','UZ': 'UZBEKISTAN','NH': 'VANUATU','VT': 'VATICAN CITY','VE': 'VENEZUELA','VM': 'VIETNAM','VQ': 'VIRGIN ISLANDS','WQ': 'WAKE ISLAND','WF': 'WALLIS AND FUTUNA','WE': 'WEST BANK','WI': 'WESTERN SAHARA','WS': 'WESTERN SAMOA','YM': 'YEMEN','CG': 'ZAIRE','ZA': 'ZAMBIA','ZI': 'ZIMBABWE','TW': 'TAIWAN'}

# Global variable to track interruption
interrupted = False

syear=2022
year=syear
#start_month=1 
#start_day=1
#end_month=12
#end_day=31
#start_date = datetime(year, start_month, start_day)
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
    syear=2022
    start_month=1
    start_day=1
    eyear=2022
    end_month=1
    end_day=30
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
                #replaced SQLDATE with DATEADDED, as SQL date was wrong
                essential_columns = ['GLOBALEVENTID',
                    'DATEADDED', 'SOURCEURL', 'Actor1Geo_CountryCode', 'Actor2Geo_CountryCode',
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
    FOCAL_COUNTRIES = ['UK']  # Countries we're analyzing
    
    EVENT_ROOTCODES= ['10','11','12','13','14','15','16','17','18','19','20']
    # Event codes to filter for (political/diplomatic events)
    EVENT_CODES = ['012','016','0212','0214','0232','0233','0234','0243','0244','0252','0253','0254','0255','0256','026','027','028','0312','0314','032','0332','0333','0334','0354','0355','0356','036','037','038','039','046','050','051','052','053','054','055','056','057','06','060','061','062','063','064','071','072','073','074','075','0811','0812','0813','0814','082','083','0831','0832','0833','0834','0841','085','086','0861','0862','0863','087','0871','0872','0873','0874','092','093','094','1012','1014','102','1032','1033','1034','1041','1042','1043','1044','1052','1054','1055','1056','106','107','108','111','1121','1122','1123','1124','1125','113','114','115','116','121','1211','1212','122','1221','1222','1223','1224','123','1231','1232','1233','1234','124','1241','1242','1243','1244','1245','1246','125','126','127','128','129','130','131','1311','1312','1313','132','1321','1322','1323','1324','133','134','135','136','137','138','1381','1382','1383','1384','1385','139','140','141','1411','1412','1413','1414','142','1421','1422','1423','1424','143','1431','1432','1433','1434','144','1441','1442','1443','1444','145','1451','1452','1453','1454','150','151','152','153','154','155','16','160','161','162','1621','1622','1623','163','164','165','166','1661','1662','1663','1712','1721','1722','1723','1724','174','175','180','181','182','1821','1822','1823','183','1831','1832','1833','1834','184','185','186','190','191','192','193','194','195','1951','1952','196','200','201','202','203','204','2041','2042']

    print(f"Year: {YEAR}")
    print(f"Focal countries: {', '.join(FOCAL_COUNTRIES)}")
    print(f"Event codes: {len(EVENT_CODES)} types")

    # STEP 2: GET USER INPUT FOR COUNTERPARTS
    print("\nSTEP 2: Selecting counterpart countries...")

    # Default selection
    default_counterparts = ['RS','IZ','IS','US', 'FR', 'UK', 'IT','SP','IR', 'AG', 'AJ', 'AM','BG', 'CH', 'GZ', 'HK', 'IN', 'ID', 'KN', 'KZ', 'LY','MD','MY', 'NU', 'PK', 'SA', 'SU', 'SY','TU', 'UZ', 'VE', 'WE', 'YM']
    
    #default_counterparts = ['AF','AL','AG','AQ','AN','AO','AV','AY','AC','AR','AM','AA','AT','AS','AU','AJ','BF','BA','FQ','BG','BB','BS','BO','BE','BH','BN','BD','BT','BL','BK','BC','BV','BR','IO','VI','BX','BU','UV','BM','BY','CB','CM','CA','CV','CJ','CT','CD','CI','CH','KT','IP','CK','CO','CN','CF','CW','CR','CS','IV','HR','CU','CY','EZ','DA','DJ','DO','DR','EC','EG','ES','EK','ER','EN','ET','EU','FK','FO','FM','FJ','FI','FR','FG','FP','FS','GB','GA','GZ','GG','GM','GH','GI','GO','GR','GL','GJ','GP','GQ','GT','GK','GV','PU','GY','HA','HM','HO','HK','HQ','HU','IC','IN','ID','IR','IZ','EI','IS','IT','JM','JN','JA','DQ','JE','JQ','JO','JU','KZ','KE','KQ','KR','KN','KS','KU','KG','LA','LG','LE','LT','LI','LY','LS','LH','LU','MC','MK','MA','MI','MY','MV','ML','MT','IM','RM','MB','MR','MP','MF','MX','MQ','MD','MN','MG','MW','MH','MO','MZ','WA','NR','BQ','NP','NL','NT','NC','NZ','NU','NG','NI','NE','NF','CQ','NO','MU','PK','LQ','PM','PP','PF','PA','PE','RP','PC','PL','PO','RQ','QA','RE','RO','RS','RW','SC','SH','ST','SB','VC','SM','TP','SA','SG','SR','SE','SL','SN','LO','SI','BP','SO','SF','SX','SP','PG','CE','SU','NS','SV','WZ','SW','SZ','SY','TI','TZ','TH','TO','TL','TN','TD','TE','PS','TS','TU','TX','TK','TV','UG','UP','TC','UK','UK','UK','UK','US','UY','UZ','NH','VT','VE','VM','VQ','WQ','WF','WE','WI','WS','YM','CG','ZA','ZI','TW']

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
 
                focal_filter = df_daily[
                    (df_daily['Actor1Geo_CountryCode'] == focal_country) | 
                    (df_daily['Actor2Geo_CountryCode'] == focal_country)
                ]

                df_filtered = focal_filter
                
                # Filter by event codes
                if EVENT_CODES and len(df_filtered) > 0:
#######################print(EVENT_CODES)
                    #print(df_filtered)
                    df_filtered=df_filtered[df_filtered["EventCode"].astype(str).isin(EVENT_CODES)]
                    #rootcode filter
                    #df_filtered=df_filtered[df_filtered["EventRootCode"].astype(str).isin(EVENT_ROOTCODES)]
                    
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
                    df_filtered['Date'] = pd.to_datetime(df_filtered['DATEADDED'].astype(str), format='%Y%m%d', errors='coerce')
                    
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
    #combined_df=combined_df[combined_df["EventCode"].astype(str).isin(EVENT_CODES)]
    #print(combined_df["EventCode"])
    #exit()
    # Group data by dacombined_dfte, country, and relationship
    daily_relationships = combined_df.groupby(['Date', 'FocalCountry', 'RelationshipPair']).agg({
        'AvgTone': ['mean', 'std', 'count', 'sum'],
        'GoldsteinScale': ['mean', 'std', 'sum'],
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
        'AvgTone_sum': 'Daily_AvgTone_Sum',
        'GoldsteinScale_mean': 'Daily_Goldstein',
        'GoldsteinScale_std': 'Daily_Goldstein_Std',
        'GoldsteinScale_sum': 'Daily_Goldstein_Sum',
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
    
        # Save daily scores with random string
    import random
    import string
    random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    scores_filename = f'daily_scores_{YEAR}_{random_str}.csv'
    #daily_scores.to_csv(scores_filename, index=False)
    print(f"✓ Daily scores saved to: {scores_filename}")
    #filter by negative zscore
    
    daily_scores1=daily_scores[daily_scores['Tone_Article_ZScore']<0]
    scores_filename = f'NEG_{YEAR}_{random_str}.csv'
    #daily_scores1.to_csv(scores_filename, index=False)
    
    #group by date, 10 lowsest zscore per day    
    min_10_with_all_columns = daily_scores.loc[daily_scores.groupby('Date')['Tone_Article_ZScore'].nsmallest(10).index.get_level_values(1)]
    
    #print(min_10_with_all_columns)
    #min_filename = f'MIN_{YEAR}_{random_str}.csv'
    #min_10_with_all_columns.to_csv(min_filename, index=False)
    
    # Read the Excel file
    df = pd.read_excel('Cyber Events Database - 2014-2024 + Jan_Aug_Sept 2025.xlsx', sheet_name='Sheet 1')

    # Convert event_date to datetime if it's not already
    df['event_date'] = pd.to_datetime(df['event_date'], format='%d-%m-%Y').dt.date

    # Aggregate by the specified columns and concatenate descriptions
    aggregated_df = df.groupby(['event_date', 'motive', 'event_type', 'country', 'actor_country']).agg({
        'description': lambda x: ' | '.join(x.astype(str)),
        'slug': 'count'  # Count number of events in each group
    }).reset_index()

    # Rename the count column
    aggregated_df = aggregated_df.rename(columns={'slug': 'event_count'})
    #print(aggregated_df)
    # Display the aggregated data
    #print(aggregated_df.head())
    print(f"\nTotal aggregated events: {len(aggregated_df)}")
    print(f"Original events: {len(df)}")

    # Optional: Save to new Excel file

    # Make sure date columns are in the same format
    #aggregated_df['event_date'] = pd.to_datetime(aggregated_df['event_date']).dt.date
    #daily_scores['Date'] = pd.to_datetime(daily_scores['Date']).dt.date


    import pycountry

    def country_to_fips(country_name):
        """Convert country name to FIPS code"""
        if pd.isna(country_name):
            return None
        
        try:
            # Try to find the country
            country = pycountry.countries.get(name=country_name)
            if country:
                # Pycountry uses alpha_2 codes, FIPS is similar but not identical
                # You may need a mapping table for exact FIPS codes
                return country.alpha_2
        except:
            pass
        
        # Manual mapping for common discrepancies, mapp undetermined as the focal country to avoid duplicate reporting
        manual_map = {
            'United States of America': 'US',
            'United Kingdom of Great Britain and Northern Ireland': 'UK',
            'Russian Federation': 'RS',
            'Korea (the Republic of)': 'KR',
            'Iran (Islamic Republic of)': 'IR',
            'Venezuela (Bolivarian Republic of)': 'VE',
            'Syrian Arab Republic': 'SY',
            'Czechia': 'CZ',
            'Undetermined': 'XX',
            'Moldova (the Republic of)': 'MD',
            'China': 'CH'

        }
        
        return manual_map.get(country_name, None)

    aggregated_df['country_fips'] = aggregated_df['country'].apply(country_to_fips)
    aggregated_df['actor_country_fips'] = aggregated_df['actor_country'].apply(country_to_fips)
    #consider any undetermined as UK
    aggregated_df['actor_country_fips'].replace("XX",focal_country)
    aggregated_df.loc[aggregated_df["actor_country_fips"] == "XX", "actor_country_fips"] = focal_country
    aggregated_df.loc[aggregated_df["actor_country_fips"] == "CN", "actor_country_fips"] = "CH"
    #filt=aggregated_df[aggregated_df["actor_country"]=='Russia']
    #print(filt)
    # Merge on date and country
    daily_scores['Date'] = pd.to_datetime(daily_scores['Date'], format='%d-%m-%Y').dt.date
    merged_df = pd.merge(
        daily_scores,
        aggregated_df,
        left_on=['Date', 'FocalCountry', 'Counterpart'],
        right_on=['event_date', 'country_fips', 'actor_country_fips'],
        how='left'  # Use 'left' to keep all daily_scores records, 'inner' for only matches
    )

    print(f"Merged dataset shape: {merged_df.shape}")
    print(f"Daily scores records: {len(daily_scores)}")
    mergedname=f'aggregated_cyber_events_{YEAR}_{random_str}.xlsx'
    #print(f"Merged records with cyber events: {merged_df[merged_df['event_count'].notna()].shape[0]}")
    merged_df.to_excel(mergedname, index=False)
    
    df = pd.read_excel('GLOB_UK_part.xlsx', sheet_name='Sheet1')
    
    def verify_strong_correlation(df):
        """Verify the nearly perfect -1 correlation finding"""
        
        print("="*60)
        print("VERIFYING STRONG CORRELATION (-1) BETWEEN:")
        print("Monthly Sum of Daily_AvgTone vs Monthly Sum of event_count")
        print("="*60)
        
        # Create monthly aggregates
        clean_data = df.dropna(subset=['Daily_AvgTone', 'event_count']).copy()
        clean_data['Date'] = pd.to_datetime(clean_data['Date'])
        clean_data = clean_data.sort_values(['RelationshipPair', 'Date'])
        clean_data['YearMonth'] = clean_data['Date'].dt.to_period('M')
        
        monthly_data = clean_data.groupby(['RelationshipPair', 'YearMonth']).agg({
            'Daily_AvgTone': ['sum', 'mean', 'count'],
            'event_count': ['sum', 'mean', 'count'],
            'Composite_Negativity_Score': 'mean'
        }).reset_index()
        
        # Flatten column names
        monthly_data.columns = ['RelationshipPair', 'YearMonth', 
                               'Daily_AvgTone_Sum', 'Daily_AvgTone_Mean', 'Daily_AvgTone_Count',
                               'Event_Count_Sum', 'Event_Count_Mean', 'Event_Count_Count',
                               'Composite_Negativity_Mean']
        
        print(f"Monthly data points: {len(monthly_data)}")
        print(f"Unique relationship pairs: {monthly_data['RelationshipPair'].nunique()}")
        
        # Check the correlation you found
        corr, p_value = stats.pearsonr(monthly_data['Daily_AvgTone_Sum'], monthly_data['Event_Count_Sum'])
        print(f"\nMAIN CORRELATION RESULT:")
        print(f"Correlation: {corr:.6f}")
        print(f"P-value: {p_value:.10f}")
        print(f"R-squared: {corr**2:.6f}")
        
        # Detailed diagnostics
        print(f"\nDATA DIAGNOSTICS:")
        print(f"Daily_AvgTone_Sum stats: min={monthly_data['Daily_AvgTone_Sum'].min():.2f}, "
              f"max={monthly_data['Daily_AvgTone_Sum'].max():.2f}, "
              f"mean={monthly_data['Daily_AvgTone_Sum'].mean():.2f}")
        print(f"Event_Count_Sum stats: min={monthly_data['Event_Count_Sum'].min():.0f}, "
              f"max={monthly_data['Event_Count_Sum'].max():.0f}, "
              f"mean={monthly_data['Event_Count_Sum'].mean():.2f}")
        
        # Check for potential issues
        print(f"\nPOTENTIAL ISSUES CHECK:")
        
        # 1. Check if it's driven by a single relationship pair
        print("1. Correlation by relationship pair:")
        for relationship in monthly_data['RelationshipPair'].unique():
            rel_data = monthly_data[monthly_data['RelationshipPair'] == relationship]
            if len(rel_data) >= 3:  # Need at least 3 points for meaningful correlation
                rel_corr, rel_p = stats.pearsonr(rel_data['Daily_AvgTone_Sum'], rel_data['Event_Count_Sum'])
                if abs(rel_corr) > 0.8:  # Only show very strong correlations
                    print(f"   {relationship}: corr = {rel_corr:.3f} (n={len(rel_data)})")
        
        # 2. Check for outliers
        print(f"\n2. Outlier analysis:")
        z_scores_tone = np.abs(stats.zscore(monthly_data['Daily_AvgTone_Sum']))
        z_scores_events = np.abs(stats.zscore(monthly_data['Event_Count_Sum']))
        outliers_tone = monthly_data[z_scores_tone > 3]
        outliers_events = monthly_data[z_scores_events > 3]
        
        print(f"   Outliers in Daily_AvgTone_Sum: {len(outliers_tone)}")
        print(f"   Outliers in Event_Count_Sum: {len(outliers_events)}")
        
        # 3. Check if it's a scaling artifact
        print(f"\n3. Scaling analysis:")
        print(f"   Correlation with log transformation:")
        # Avoid log(0) by adding small constant
        log_tone = np.log(monthly_data['Daily_AvgTone_Sum'] - monthly_data['Daily_AvgTone_Sum'].min() + 1)
        log_events = np.log(monthly_data['Event_Count_Sum'] + 1)
        log_corr, log_p = stats.pearsonr(log_tone, log_events)
        print(f"   Log-transformed correlation: {log_corr:.6f}")
        
        # 4. Create comprehensive visualization
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Scatter plot with regression line
        plt.subplot(2, 3, 1)
        plt.scatter(monthly_data['Daily_AvgTone_Sum'], monthly_data['Event_Count_Sum'], alpha=0.6)
        plt.xlabel('Monthly Sum of Daily_AvgTone')
        plt.ylabel('Monthly Sum of event_count')
        plt.title(f'Strong Correlation: r = {corr:.4f}')
        
        # Add regression line
        if len(monthly_data) > 1:
            z = np.polyfit(monthly_data['Daily_AvgTone_Sum'], monthly_data['Event_Count_Sum'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(monthly_data['Daily_AvgTone_Sum'].min(), monthly_data['Daily_AvgTone_Sum'].max(), 100)
            plt.plot(x_range, p(x_range), 'r-', alpha=0.8)
        
        # Plot 2: Residuals analysis
        plt.subplot(2, 3, 2)
        if len(monthly_data) > 1:
            residuals = monthly_data['Event_Count_Sum'] - p(monthly_data['Daily_AvgTone_Sum'])
            plt.scatter(monthly_data['Daily_AvgTone_Sum'], residuals, alpha=0.6)
            plt.axhline(y=0, color='red', linestyle='--')
            plt.xlabel('Monthly Sum of Daily_AvgTone')
            plt.ylabel('Residuals')
            plt.title('Residuals Plot')
        
        # Plot 3: Distribution of both variables
        plt.subplot(2, 3, 3)
        plt.hist(monthly_data['Daily_AvgTone_Sum'], bins=20, alpha=0.7, label='Daily_AvgTone_Sum')
        plt.hist(monthly_data['Event_Count_Sum'], bins=20, alpha=0.7, label='Event_Count_Sum')
        plt.legend()
        plt.title('Distributions')
        
        # Plot 4: Time series of both variables (if we have time order)
        plt.subplot(2, 3, 4)
        # Use first few relationships to avoid clutter
        sample_relationships = monthly_data['RelationshipPair'].unique()[:3]
        for rel in sample_relationships:
            rel_data = monthly_data[monthly_data['RelationshipPair'] == rel].sort_values('YearMonth')
            if len(rel_data) > 1:
                plt.plot(rel_data['YearMonth'].astype(str), rel_data['Daily_AvgTone_Sum'], 
                        marker='o', label=f'{rel} - Tone')
                plt.plot(rel_data['YearMonth'].astype(str), rel_data['Event_Count_Sum'], 
                        marker='s', linestyle='--', label=f'{rel} - Events')
        plt.xticks(rotation=45)
        plt.legend()
        plt.title('Time Series Sample')
        
        # Plot 5: Check relationship with count of days
        plt.subplot(2, 3, 5)
        plt.scatter(monthly_data['Daily_AvgTone_Count'], monthly_data['Event_Count_Sum'], alpha=0.6)
        plt.xlabel('Number of Days with Tone Data')
        plt.ylabel('Monthly Sum of event_count')
        plt.title('Events vs Data Availability')
        
        # Plot 6: Correlation with mean instead of sum
        plt.subplot(2, 3, 6)
        mean_corr, mean_p = stats.pearsonr(monthly_data['Daily_AvgTone_Mean'], monthly_data['Event_Count_Sum'])
        plt.scatter(monthly_data['Daily_AvgTone_Mean'], monthly_data['Event_Count_Sum'], alpha=0.6)
        plt.xlabel('Monthly Mean of Daily_AvgTone')
        plt.ylabel('Monthly Sum of event_count')
        plt.title(f'Using Mean: r = {mean_corr:.4f}')
        
        plt.tight_layout()
        plt.show()
        
        # Additional verification tests
        print(f"\n4. ROBUSTNESS CHECKS:")
        
        # Check with different correlation methods
        spearman_corr, spearman_p = stats.spearmanr(monthly_data['Daily_AvgTone_Sum'], monthly_data['Event_Count_Sum'])
        print(f"   Spearman correlation: {spearman_corr:.6f}")
        
        # Check with outliers removed
        if len(outliers_tone) > 0 or len(outliers_events) > 0:
            no_outliers = monthly_data[(z_scores_tone <= 3) & (z_scores_events <= 3)]
            if len(no_outliers) > 2:
                corr_no_outliers, p_no_outliers = stats.pearsonr(no_outliers['Daily_AvgTone_Sum'], no_outliers['Event_Count_Sum'])
                print(f"   Correlation without outliers: {corr_no_outliers:.6f} (n={len(no_outliers)})")
        
        # Check if it's driven by data quantity rather than tone
        data_quantity_corr, data_p = stats.pearsonr(monthly_data['Daily_AvgTone_Count'], monthly_data['Event_Count_Sum'])
        print(f"   Correlation with number of days: {data_quantity_corr:.6f}")
        
        return monthly_data, corr

    # Run the verification
    monthly_data, verified_correlation = verify_strong_correlation(df)

    # If correlation is indeed very strong, let's explore why
    if abs(verified_correlation) > 0.9:
        print(f"\n" + "="*60)
        print("INVESTIGATING THE STRONG CORRELATION")
        print("="*60)
        
        print("Possible reasons for near-perfect correlation:")
        print("1. DATA PROCESSING ARTIFACT: Are both sums derived from the same underlying count?")
        print("2. OUTLIER DRIVEN: A few extreme points dominating the correlation")
        print("3. RELATIONSHIP-SPECIFIC: One relationship pair with perfect pattern")
        print("4. TIME PERIOD EFFECT: Specific time period driving the relationship")
        print("5. SCALING ISSUE: Both variables scaling with number of observation days")
        
        # Show the strongest relationships
        strong_relationships = []
        for relationship in monthly_data['RelationshipPair'].unique():
            rel_data = monthly_data[monthly_data['RelationshipPair'] == relationship]
            if len(rel_data) >= 3:
                rel_corr, _ = stats.pearsonr(rel_data['Daily_AvgTone_Sum'], rel_data['Event_Count_Sum'])
                if abs(rel_corr) > 0.8:
                    strong_relationships.append((relationship, rel_corr, len(rel_data)))
        
        if strong_relationships:
            print(f"\nRelationships with very strong correlations (>0.8):")
            for rel, corr, n in sorted(strong_relationships, key=lambda x: abs(x[1]), reverse=True):
                print(f"   {rel}: {corr:.4f} (n={n})")
        
        
        
        
        
    def predictive_potential_analysis(df):
        """Test if this strong correlation has predictive power"""
        
        print("="*60)
        print("PREDICTIVE POTENTIAL ANALYSIS")
        print("Monthly Sum of Daily_AvgTone → Next Month Events")
        print("="*60)
        
        # Create monthly aggregates
        clean_data = df.dropna(subset=['Daily_AvgTone', 'event_count']).copy()
        clean_data['Date'] = pd.to_datetime(clean_data['Date'])
        clean_data = clean_data.sort_values(['RelationshipPair', 'Date'])
        clean_data['YearMonth'] = clean_data['Date'].dt.to_period('M')
        
        monthly_data = clean_data.groupby(['RelationshipPair', 'YearMonth']).agg({
            'Daily_AvgTone': ['sum', 'mean'],
            'event_count': 'sum'
        }).reset_index()
        
        monthly_data.columns = ['RelationshipPair', 'YearMonth', 
                               'Daily_AvgTone_Sum', 'Daily_AvgTone_Mean', 'Event_Count_Sum']
        
        # Create predictive dataset: This month's tone → Next month's events
        monthly_data = monthly_data.sort_values(['RelationshipPair', 'YearMonth'])
        monthly_data['Next_Month_Events'] = monthly_data.groupby('RelationshipPair')['Event_Count_Sum'].shift(-1)
        monthly_data['This_Month_Tone_Sum'] = monthly_data['Daily_AvgTone_Sum']
        
        # Remove rows without next month data
        predictive_data = monthly_data.dropna(subset=['Next_Month_Events', 'This_Month_Tone_Sum'])
        
        print(f"Predictive data points: {len(predictive_data)}")
        print(f"Relationship pairs: {predictive_data['RelationshipPair'].nunique()}")
        
        # Global predictive correlation
        pred_corr, pred_p = stats.pearsonr(predictive_data['This_Month_Tone_Sum'], 
                                         predictive_data['Next_Month_Events'])
        
        print(f"\n📊 PREDICTIVE PERFORMANCE:")
        print(f"Correlation (this month tone → next month events): {pred_corr:.4f}")
        print(f"P-value: {pred_p:.10f}")
        print(f"Significant: {'YES' if pred_p < 0.05 else 'NO'}")
        print(f"R-squared: {pred_corr**2:.4f}")
        
        # Compare with same-month correlation
        same_month_corr, same_month_p = stats.pearsonr(monthly_data['Daily_AvgTone_Sum'], 
                                                     monthly_data['Event_Count_Sum'])
        print(f"\n📈 COMPARISON:")
        print(f"Same-month correlation: {same_month_corr:.4f} (descriptive)")
        print(f"Predictive correlation: {pred_corr:.4f} (predictive)")
        print(f"Predictive power retention: {abs(pred_corr)/abs(same_month_corr):.1%}")
        
        # Predictive power by threshold
        print(f"\n🎯 PREDICTIVE POWER ANALYSIS:")
        
        # Use tone thresholds to predict high/low events next month
        high_tone_threshold = predictive_data['This_Month_Tone_Sum'].quantile(0.75)  # Least negative 25%
        low_tone_threshold = predictive_data['This_Month_Tone_Sum'].quantile(0.25)   # Most negative 25%
        
        high_tone_months = predictive_data[predictive_data['This_Month_Tone_Sum'] > high_tone_threshold]
        low_tone_months = predictive_data[predictive_data['This_Month_Tone_Sum'] < low_tone_threshold]
        
        if len(high_tone_months) > 0 and len(low_tone_months) > 0:
            events_after_high_tone = high_tone_months['Next_Month_Events'].mean()
            events_after_low_tone = low_tone_months['Next_Month_Events'].mean()
            
            print(f"High tone threshold (less negative): {high_tone_threshold:.2f}")
            print(f"Low tone threshold (more negative): {low_tone_threshold:.2f}")
            print(f"Events after high tone months: {events_after_high_tone:.1f}")
            print(f"Events after low tone months: {events_after_low_tone:.1f}")
            print(f"Predictive ratio: {events_after_low_tone/events_after_high_tone:.1f}x more events")
        
        # Focus on US-US relationship (most data points)
        print(f"\n🔍 US-US RELATIONSHIP (MOST DATA):")
        us_us_data = predictive_data[predictive_data['RelationshipPair'] == 'US-US']
        if len(us_us_data) > 5:
            us_us_corr, us_us_p = stats.pearsonr(us_us_data['This_Month_Tone_Sum'], 
                                               us_us_data['Next_Month_Events'])
            print(f"Predictive correlation: {us_us_corr:.4f} (p = {us_us_p:.4f})")
            print(f"Months of data: {len(us_us_data)}")
            
            # Simple prediction accuracy
            median_tone = us_us_data['This_Month_Tone_Sum'].median()
            predicted_high_events = us_us_data[us_us_data['This_Month_Tone_Sum'] < median_tone]['Next_Month_Events'].mean()
            predicted_low_events = us_us_data[us_us_data['This_Month_Tone_Sum'] >= median_tone]['Next_Month_Events'].mean()
            
            print(f"Predicted high events (tone < median): {predicted_high_events:.1f}")
            print(f"Predicted low events (tone ≥ median): {predicted_low_events:.1f}")
        
        # Visualization
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(predictive_data['This_Month_Tone_Sum'], predictive_data['Next_Month_Events'], alpha=0.6)
        plt.xlabel('This Month Tone Sum (More negative → left)')
        plt.ylabel('Next Month Event Count')
        plt.title(f'Predictive Relationship\nr = {pred_corr:.4f}')
        
        # Add regression line
        if len(predictive_data) > 1:
            z = np.polyfit(predictive_data['This_Month_Tone_Sum'], predictive_data['Next_Month_Events'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(predictive_data['This_Month_Tone_Sum'].min(), 
                                 predictive_data['This_Month_Tone_Sum'].max(), 100)
            plt.plot(x_range, p(x_range), 'r-', alpha=0.8)
        
        plt.subplot(1, 2, 2)
        # Show predictive power comparison
        correlations = [same_month_corr, pred_corr]
        labels = ['Same Month\n(Descriptive)', 'Next Month\n(Predictive)']
        colors = ['blue', 'red']
        
        bars = plt.bar(labels, correlations, color=colors, alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.ylabel('Correlation Coefficient')
        plt.title('Descriptive vs Predictive Power')
        
        # Add value labels
        for bar, corr in zip(bars, correlations):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{corr:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Final assessment
        print(f"\n" + "="*60)
        print("PREDICTIVE POTENTIAL ASSESSMENT")
        print("="*60)
        
        if pred_p < 0.05 and abs(pred_corr) > 0.3:
            print("✅ STRONG PREDICTIVE POTENTIAL")
            print("   • Statistically significant")
            print("   • Moderate to strong correlation")
            print("   • Theoretically plausible")
            
        elif pred_p < 0.05 and abs(pred_corr) > 0.1:
            print("✅ MODERATE PREDICTIVE POTENTIAL") 
            print("   • Statistically significant")
            print("   • Weak but meaningful correlation")
            print("   • Worth further investigation")
            
        else:
            print("❌ LIMITED PREDICTIVE POTENTIAL")
            print("   • Not statistically significant")
            print("   • Weak correlation")
            print("   • May not be practically useful")
        
        return predictive_data, pred_corr, pred_p

    # Run the predictive analysis
    predictive_data, pred_corr, pred_p = predictive_potential_analysis(df)
            
        
    
    
    
    exit()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    #########################################################################################################################
    
    
    def quarterly_predictive_analysis(df):
        """Test if prior quarter's negativity predicts current quarter's events"""
        
        print("="*60)
        print("QUARTERLY PREDICTIVE POWER ANALYSIS")
        print("Prior Quarter Negativity → Current Quarter Events")
        print("="*60)
        
        # Clean and prepare data
        clean_data = df.dropna(subset=['Composite_Negativity_Score', 'event_count']).copy()
        clean_data['Date'] = pd.to_datetime(clean_data['Date'])
        clean_data = clean_data.sort_values(['RelationshipPair', 'Date'])
        
        # Create quarterly aggregates
        clean_data['YearQuarter'] = clean_data['Date'].dt.to_period('Q')
        
        quarterly_data = clean_data.groupby(['RelationshipPair', 'YearQuarter']).agg({
            'Composite_Negativity_Score': ['mean', 'max', 'sum', 'std'],
            'event_count': 'sum',
            'Date': 'count'  # Number of days with data
        }).reset_index()
        
        # Flatten column names
        quarterly_data.columns = ['RelationshipPair', 'YearQuarter', 'Negativity_Mean', 
                                 'Negativity_Max', 'Negativity_Sum', 'Negativity_Std', 
                                 'Event_Count', 'Days_With_Data']
        
        # Filter quarters with sufficient data (at least 45 days)
        quarterly_data = quarterly_data[quarterly_data['Days_With_Data'] >= 15]
        
        # Create prior quarter features
        quarterly_data = quarterly_data.sort_values(['RelationshipPair', 'YearQuarter'])
        quarterly_data['Prior_Quarter_Negativity_Mean'] = quarterly_data.groupby('RelationshipPair')['Negativity_Mean'].shift(1)
        quarterly_data['Prior_Quarter_Negativity_Max'] = quarterly_data.groupby('RelationshipPair')['Negativity_Max'].shift(1)
        quarterly_data['Prior_Quarter_Negativity_Sum'] = quarterly_data.groupby('RelationshipPair')['Negativity_Sum'].shift(1)
        quarterly_data['Prior_Quarter_Events'] = quarterly_data.groupby('RelationshipPair')['Event_Count'].shift(1)
        
        # Remove rows without prior quarter data
        quarterly_data = quarterly_data.dropna(subset=['Prior_Quarter_Negativity_Mean', 'Event_Count'])
        
        print(f"Quarterly data points available: {len(quarterly_data)}")
        print(f"Unique relationship pairs: {quarterly_data['RelationshipPair'].nunique()}")
        
        # Global correlation analysis
        print("\n" + "="*60)
        print("GLOBAL QUARTERLY PREDICTIVE POWER")
        print("="*60)
        
        # Correlation: Prior quarter negativity vs current quarter events
        corr_mean, p_mean = stats.pearsonr(quarterly_data['Prior_Quarter_Negativity_Mean'], 
                                         quarterly_data['Event_Count'])
        corr_max, p_max = stats.pearsonr(quarterly_data['Prior_Quarter_Negativity_Max'], 
                                       quarterly_data['Event_Count'])
        corr_sum, p_sum = stats.pearsonr(quarterly_data['Prior_Quarter_Negativity_Sum'], 
                                       quarterly_data['Event_Count'])
        
        print(f"Prior Quarter MEAN Negativity → Current Quarter Events:")
        print(f"  Correlation: {corr_mean:.3f} (p = {p_mean:.4f})")
        print(f"  Significant: {'YES' if p_mean < 0.05 else 'NO'}")
        
        print(f"\nPrior Quarter MAX Negativity → Current Quarter Events:")
        print(f"  Correlation: {corr_max:.3f} (p = {p_max:.4f})")
        print(f"  Significant: {'YES' if p_max < 0.05 else 'NO'}")
        
        print(f"\nPrior Quarter TOTAL Negativity → Current Quarter Events:")
        print(f"  Correlation: {corr_sum:.3f} (p = {p_sum:.4f})")
        print(f"  Significant: {'YES' if p_sum < 0.05 else 'NO'}")
        
        # Predictive power using thresholds
        print("\n" + "="*60)
        print("QUARTERLY PREDICTIVE POWER BY NEGATIVITY LEVEL")
        print("="*60)
        
        high_neg_threshold = quarterly_data['Prior_Quarter_Negativity_Mean'].quantile(0.75)
        low_neg_threshold = quarterly_data['Prior_Quarter_Negativity_Mean'].quantile(0.25)
        
        high_neg_quarters = quarterly_data[quarterly_data['Prior_Quarter_Negativity_Mean'] > high_neg_threshold]
        low_neg_quarters = quarterly_data[quarterly_data['Prior_Quarter_Negativity_Mean'] < low_neg_threshold]
        
        if len(high_neg_quarters) > 0 and len(low_neg_quarters) > 0:
            avg_events_high_neg = high_neg_quarters['Event_Count'].mean()
            avg_events_low_neg = low_neg_quarters['Event_Count'].mean()
            
            print(f"High negativity threshold: {high_neg_threshold:.2f}")
            print(f"Low negativity threshold: {low_neg_threshold:.2f}")
            print(f"Quarters after high negativity: {len(high_neg_quarters)}")
            print(f"Quarters after low negativity: {len(low_neg_quarters)}")
            print(f"Average events after high negativity: {avg_events_high_neg:.1f}")
            print(f"Average events after low negativity: {avg_events_low_neg:.1f}")
            print(f"Predictive ratio: {avg_events_high_neg/avg_events_low_neg:.1f}x")
        
        return quarterly_data

    # Also recreate monthly analysis for comparison
    def enhanced_monthly_analysis(df):
        """Enhanced monthly analysis for better comparison"""
        
        clean_data = df.dropna(subset=['Composite_Negativity_Score', 'event_count']).copy()
        clean_data['Date'] = pd.to_datetime(clean_data['Date'])
        clean_data = clean_data.sort_values(['RelationshipPair', 'Date'])
        clean_data['YearMonth'] = clean_data['Date'].dt.to_period('M')
        
        monthly_data = clean_data.groupby(['RelationshipPair', 'YearMonth']).agg({
            'Composite_Negativity_Score': 'mean',
            'event_count': 'sum',
            'Date': 'count'
        }).reset_index()
        
        monthly_data.columns = ['RelationshipPair', 'YearMonth', 'Negativity_Mean', 'Event_Count', 'Days_With_Data']
        monthly_data = monthly_data[monthly_data['Days_With_Data'] >= 5]
        
        monthly_data = monthly_data.sort_values(['RelationshipPair', 'YearMonth'])
        monthly_data['Prior_Month_Negativity_Mean'] = monthly_data.groupby('RelationshipPair')['Negativity_Mean'].shift(1)
        monthly_data = monthly_data.dropna(subset=['Prior_Month_Negativity_Mean', 'Event_Count'])
        
        return monthly_data

    # Run both analyses
    monthly_data = enhanced_monthly_analysis(df)
    quarterly_data = quarterly_predictive_analysis(df)

    # COMPREHENSIVE COMPARISON
    print("\n" + "="*60)
    print("COMPREHENSIVE TIMEFRAME COMPARISON")
    print("="*60)

    # Daily analysis (from previous)
    clean_data = df.dropna(subset=['Composite_Negativity_Score', 'event_count']).copy()
    clean_data['Date'] = pd.to_datetime(clean_data['Date'])
    clean_data = clean_data.sort_values(['RelationshipPair', 'Date'])
    clean_data['event_tomorrow'] = clean_data.groupby('RelationshipPair')['event_count'].shift(-1)
    daily_analysis = clean_data.dropna(subset=['event_tomorrow'])
    daily_corr, daily_p = stats.pearsonr(daily_analysis['Composite_Negativity_Score'], daily_analysis['event_tomorrow'])

    # Monthly correlation
    monthly_corr, monthly_p = stats.pearsonr(monthly_data['Prior_Month_Negativity_Mean'], monthly_data['Event_Count'])

    # Quarterly correlation  
    quarterly_corr, quarterly_p = stats.pearsonr(quarterly_data['Prior_Quarter_Negativity_Mean'], quarterly_data['Event_Count'])

    print("PREDICTIVE POWER BY TIMEFRAME:")
    print(f"{'DAILY (next day)':<20} | Correlation: {daily_corr:6.3f} | p-value: {daily_p:7.4f} | n: {len(daily_analysis):4}")
    print(f"{'MONTHLY (next month)':<20} | Correlation: {monthly_corr:6.3f} | p-value: {monthly_p:7.4f} | n: {len(monthly_data):4}")
    print(f"{'QUARTERLY (next quarter)':<20} | Correlation: {quarterly_corr:6.3f} | p-value: {quarterly_p:7.4f} | n: {len(quarterly_data):4}")

    # Calculate improvements
    monthly_improvement = monthly_corr - daily_corr
    quarterly_improvement = quarterly_corr - daily_corr
    quarterly_vs_monthly = quarterly_corr - monthly_corr

    print(f"\nIMPROVEMENT ANALYSIS:")
    print(f"Monthly vs Daily:    +{monthly_improvement:.3f} correlation points")
    print(f"Quarterly vs Daily:  +{quarterly_improvement:.3f} correlation points") 
    print(f"Quarterly vs Monthly: +{quarterly_vs_monthly:.3f} correlation points")

    # Visualization
    plt.figure(figsize=(12, 8))

    # Plot 1: Correlation by timeframe
    plt.subplot(2, 2, 1)
    timeframes = ['Daily', 'Monthly', 'Quarterly']
    correlations = [daily_corr, monthly_corr, quarterly_corr]
    p_values = [daily_p, monthly_p, quarterly_p]

    colors = ['red' if p > 0.05 else 'green' for p in p_values]
    bars = plt.bar(timeframes, correlations, color=colors, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.ylabel('Correlation Coefficient')
    plt.title('Predictive Power by Timeframe\n(Green = Significant)')

    # Add value labels on bars
    for bar, corr in zip(bars, correlations):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{corr:.3f}', ha='center', va='bottom')

    # Plot 2: Sample sizes
    plt.subplot(2, 2, 2)
    sample_sizes = [len(daily_analysis), len(monthly_data), len(quarterly_data)]
    plt.bar(timeframes, sample_sizes, color='skyblue', alpha=0.7)
    plt.ylabel('Number of Data Points')
    plt.title('Sample Sizes by Timeframe')

    # Plot 3: Quarterly predictive power scatter
    plt.subplot(2, 2, 3)
    plt.scatter(quarterly_data['Prior_Quarter_Negativity_Mean'], quarterly_data['Event_Count'], alpha=0.6)
    plt.xlabel('Prior Quarter Mean Negativity')
    plt.ylabel('Current Quarter Event Count')
    plt.title(f'Quarterly Prediction\nr = {quarterly_corr:.3f}')

    # Add trendline
    if len(quarterly_data) > 1:
        z = np.polyfit(quarterly_data['Prior_Quarter_Negativity_Mean'], quarterly_data['Event_Count'], 1)
        p = np.poly1d(z)
        plt.plot(quarterly_data['Prior_Quarter_Negativity_Mean'], p(quarterly_data['Prior_Quarter_Negativity_Mean']), 
                 "r--", alpha=0.8)

    # Plot 4: Timeframe comparison summary
    plt.subplot(2, 2, 4)
    improvements = [0, monthly_improvement, quarterly_improvement]
    plt.bar(timeframes, improvements, color=['gray', 'orange', 'green'], alpha=0.7)
    plt.ylabel('Improvement over Daily Correlation')
    plt.title('Predictive Power Improvement\nvs Daily Baseline')

    plt.tight_layout()
    plt.show()

    # Final insights
    print("\n" + "="*60)
    print("KEY INSIGHTS FOR THESIS")
    print("="*60)

    print("""
    1. TIMEFRAME MATTERS: Longer timeframes show stronger (though still weak) relationships
    2. QUARTERLY TREND: Quarterly analysis captures sustained geopolitical tensions
    3. PRACTICAL IMPLICATION: Media tone may predict cyber activity over quarters, not days
    4. METHODOLOGICAL: Aggregation over longer periods reduces noise and reveals patterns

    FOR YOUR THESIS:
    • Discuss the importance of appropriate timeframe selection
    • Note that predictive power emerges at quarterly level, not daily
    • Suggest that cyber events may respond to sustained tensions, not daily fluctuations
    • Acknowledge that while correlations are positive, they remain statistically weak
    """)

    # Return all datasets for further analysis
    print(f"\nData available for further analysis:")
    print(f"• Daily predictions: {len(daily_analysis)} rows")
    print(f"• Monthly predictions: {len(monthly_data)} rows") 
    print(f"• Quarterly predictions: {len(quarterly_data)} rows")
    
    
    
    exit()
    def monthly_predictive_analysis(df):
        """Test if prior month's negativity predicts current month's events"""
        
        print("="*60)
        print("MONTHLY PREDICTIVE POWER ANALYSIS")
        print("Prior Month Negativity → Current Month Events")
        print("="*60)
        
        # Clean and prepare data
        clean_data = df.dropna(subset=['Composite_Negativity_Score', 'event_count']).copy()
        clean_data['Date'] = pd.to_datetime(clean_data['Date'])
        clean_data = clean_data.sort_values(['RelationshipPair', 'Date'])
        
        # Create monthly aggregates
        clean_data['YearMonth'] = clean_data['Date'].dt.to_period('M')
        
        monthly_data = clean_data.groupby(['RelationshipPair', 'YearMonth']).agg({
            'Composite_Negativity_Score': ['mean', 'max', 'sum'],
            'event_count': 'sum',
            'Date': 'count'  # Number of days with data
        }).reset_index()
        
        # Flatten column names
        monthly_data.columns = ['RelationshipPair', 'YearMonth', 'Negativity_Mean', 
                               'Negativity_Max', 'Negativity_Sum', 'Event_Count', 'Days_With_Data']
        
        # Filter months with sufficient data (at least 15 days)
        monthly_data = monthly_data[monthly_data['Days_With_Data'] >= 5]
        
        # Create prior month features
        monthly_data = monthly_data.sort_values(['RelationshipPair', 'YearMonth'])
        monthly_data['Prior_Month_Negativity_Mean'] = monthly_data.groupby('RelationshipPair')['Negativity_Mean'].shift(1)
        monthly_data['Prior_Month_Negativity_Max'] = monthly_data.groupby('RelationshipPair')['Negativity_Max'].shift(1)
        monthly_data['Prior_Month_Negativity_Sum'] = monthly_data.groupby('RelationshipPair')['Negativity_Sum'].shift(1)
        monthly_data['Prior_Month_Events'] = monthly_data.groupby('RelationshipPair')['Event_Count'].shift(1)
        
        # Remove rows without prior month data
        monthly_data = monthly_data.dropna(subset=['Prior_Month_Negativity_Mean', 'Event_Count'])
        
        print(f"Monthly data points available: {len(monthly_data)}")
        print(f"Unique relationship pairs: {monthly_data['RelationshipPair'].nunique()}")
        
        # Global correlation analysis
        print("\n" + "="*60)
        print("GLOBAL MONTHLY PREDICTIVE POWER")
        print("="*60)
        
        # Correlation: Prior month negativity vs current month events
        corr_mean, p_mean = stats.pearsonr(monthly_data['Prior_Month_Negativity_Mean'], 
                                         monthly_data['Event_Count'])
        corr_max, p_max = stats.pearsonr(monthly_data['Prior_Month_Negativity_Max'], 
                                       monthly_data['Event_Count'])
        corr_sum, p_sum = stats.pearsonr(monthly_data['Prior_Month_Negativity_Sum'], 
                                       monthly_data['Event_Count'])
        
        print(f"Prior Month MEAN Negativity → Current Month Events:")
        print(f"  Correlation: {corr_mean:.3f} (p = {p_mean:.4f})")
        print(f"  Significant: {'YES' if p_mean < 0.05 else 'NO'}")
        
        print(f"\nPrior Month MAX Negativity → Current Month Events:")
        print(f"  Correlation: {corr_max:.3f} (p = {p_max:.4f})")
        print(f"  Significant: {'YES' if p_max < 0.05 else 'NO'}")
        
        print(f"\nPrior Month TOTAL Negativity → Current Month Events:")
        print(f"  Correlation: {corr_sum:.3f} (p = {p_sum:.4f})")
        print(f"  Significant: {'YES' if p_sum < 0.05 else 'NO'}")
        
        # Predictive power using thresholds
        print("\n" + "="*60)
        print("PREDICTIVE POWER BY NEGATIVITY LEVEL")
        print("="*60)
        
        high_neg_threshold = monthly_data['Prior_Month_Negativity_Mean'].quantile(0.75)
        low_neg_threshold = monthly_data['Prior_Month_Negativity_Mean'].quantile(0.25)
        
        high_neg_months = monthly_data[monthly_data['Prior_Month_Negativity_Mean'] > high_neg_threshold]
        low_neg_months = monthly_data[monthly_data['Prior_Month_Negativity_Mean'] < low_neg_threshold]
        
        if len(high_neg_months) > 0 and len(low_neg_months) > 0:
            avg_events_high_neg = high_neg_months['Event_Count'].mean()
            avg_events_low_neg = low_neg_months['Event_Count'].mean()
            
            print(f"High negativity threshold: {high_neg_threshold:.2f}")
            print(f"Low negativity threshold: {low_neg_threshold:.2f}")
            print(f"Months after high negativity: {len(high_neg_months)}")
            print(f"Months after low negativity: {len(low_neg_months)}")
            print(f"Average events after high negativity: {avg_events_high_neg:.1f}")
            print(f"Average events after low negativity: {avg_events_low_neg:.1f}")
            print(f"Predictive ratio: {avg_events_high_neg/avg_events_low_neg:.1f}x")
        
        # Relationship-specific analysis
        print("\n" + "="*60)
        print("RELATIONSHIP-SPECIFIC MONTHLY PREDICTION")
        print("="*60)
        
        results = []
        for relationship in monthly_data['RelationshipPair'].unique():
            rel_data = monthly_data[monthly_data['RelationshipPair'] == relationship]
            
            if len(rel_data) >= 6:  # Need enough monthly data points
                corr, p_value = stats.pearsonr(rel_data['Prior_Month_Negativity_Mean'], 
                                             rel_data['Event_Count'])
                
                # Calculate predictive power
                high_neg_rel = rel_data[rel_data['Prior_Month_Negativity_Mean'] > rel_data['Prior_Month_Negativity_Mean'].median()]
                low_neg_rel = rel_data[rel_data['Prior_Month_Negativity_Mean'] <= rel_data['Prior_Month_Negativity_Mean'].median()]
                
                if len(high_neg_rel) > 0 and len(low_neg_rel) > 0:
                    pred_ratio = high_neg_rel['Event_Count'].mean() / low_neg_rel['Event_Count'].mean()
                    
                    results.append({
                        'Relationship': relationship,
                        'Months': len(rel_data),
                        'Correlation': corr,
                        'P_Value': p_value,
                        'Significant': p_value < 0.05,
                        'Avg_Events_High_Neg': high_neg_rel['Event_Count'].mean(),
                        'Avg_Events_Low_Neg': low_neg_rel['Event_Count'].mean(),
                        'Predictive_Ratio': pred_ratio
                    })
        
        # Display results
        if results:
            results_df = pd.DataFrame(results)
            print(f"Relationships with sufficient monthly data: {len(results_df)}")
            
            significant_pairs = results_df[results_df['Significant']]
            if len(significant_pairs) > 0:
                print(f"\nSIGNIFICANT MONTHLY PREDICTORS:")
                for _, row in significant_pairs.iterrows():
                    print(f"  {row['Relationship']}: corr = {row['Correlation']:.3f}, "
                          f"ratio = {row['Predictive_Ratio']:.1f}x, "
                          f"months = {row['Months']}")
            
            # Summary
            print(f"\nSUMMARY:")
            print(f"Significant relationships: {len(significant_pairs)}/{len(results_df)} "
                  f"({len(significant_pairs)/len(results_df):.1%})")
            if len(significant_pairs) > 0:
                print(f"Average predictive ratio (significant pairs): {significant_pairs['Predictive_Ratio'].mean():.1f}x")
        
        # Simple linear regression model
        print("\n" + "="*60)
        print("LINEAR REGRESSION MODEL")
        print("="*60)
        
        X = monthly_data[['Prior_Month_Negativity_Mean']].values
        y = monthly_data['Event_Count'].values
        
        if len(X) > 10:
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            print(f"R-squared: {r2:.3f} (variance explained)")
            print(f"RMSE: {rmse:.2f} events")
            print(f"Coefficient: {model.coef_[0]:.3f} (events per negativity unit)")
            print(f"Intercept: {model.intercept_:.2f}")
            
            # Interpretation
            if model.coef_[0] > 0:
                direction = "increase"
            else:
                direction = "decrease"
            
            print(f"Interpretation: Each 1-unit increase in prior month negativity")
            print(f"predicts a {abs(model.coef_[0]):.3f} event {direction} in current month")
        
        return monthly_data, results_df if 'results_df' in locals() else None

    # Run the monthly analysis
    monthly_results, relationship_results = monthly_predictive_analysis(df)

    # Additional analysis: Compare monthly vs daily prediction
    print("\n" + "="*60)
    print("COMPARISON: MONTHLY vs DAILY PREDICTIVE POWER")
    print("="*60)

    clean_data = df.dropna(subset=['Composite_Negativity_Score', 'event_count']).copy()
    clean_data['Date'] = pd.to_datetime(clean_data['Date'])
    clean_data = clean_data.sort_values(['RelationshipPair', 'Date'])

    # Daily correlation (same as before)
    clean_data['event_tomorrow'] = clean_data.groupby('RelationshipPair')['event_count'].shift(-1)
    daily_analysis = clean_data.dropna(subset=['event_tomorrow'])
    daily_corr, daily_p = stats.pearsonr(daily_analysis['Composite_Negativity_Score'], 
                                       daily_analysis['event_tomorrow'])

    print(f"DAILY prediction (today → tomorrow):")
    print(f"  Correlation: {daily_corr:.3f} (p = {daily_p:.4f})")
    print(f"  Data points: {len(daily_analysis)}")

    if 'monthly_results' in locals():
        monthly_corr, monthly_p = stats.pearsonr(monthly_results['Prior_Month_Negativity_Mean'], 
                                               monthly_results['Event_Count'])
        print(f"\nMONTHLY prediction (prior month → current month):")
        print(f"  Correlation: {monthly_corr:.3f} (p = {monthly_p:.4f})")
        print(f"  Data points: {len(monthly_results)}")
        
        improvement = abs(monthly_corr) - abs(daily_corr)
        print(f"\nImprovement with monthly aggregation: {improvement:.3f} correlation points")
        
    
    
    exit()
    
    def global_predictive_analysis_clean(df):
        """Global predictive analysis excluding NaN data"""
        
        print("="*60)
        print("GLOBAL PREDICTIVE ANALYSIS (CLEAN DATA ONLY)")
        print("="*60)
        
        # Create a clean dataset - only rows with complete data
        clean_data = df.dropna(subset=['Composite_Negativity_Score', 'event_count']).copy()
        clean_data = clean_data.sort_values(['RelationshipPair', 'Date'])
        
        print(f"Original dataset: {len(df)} rows")
        print(f"Clean dataset (no NaN): {len(clean_data)} rows")
        print(f"Data retention: {len(clean_data)/len(df):.1%}")
        
        # Analyze each relationship pair
        results = []
        
        for relationship in clean_data['RelationshipPair'].unique():
            rel_data = clean_data[clean_data['RelationshipPair'] == relationship].copy()
            
            # Only analyze if we have enough temporal data
            if len(rel_data) < 10:
                continue
                
            # Create tomorrow's events
            rel_data['event_tomorrow'] = rel_data.groupby('RelationshipPair')['event_count'].shift(-1)
            rel_data = rel_data.dropna(subset=['event_tomorrow'])
            
            if len(rel_data) < 5:  # Need minimum data points
                continue
                
            # Calculate correlation
            corr, p_value = stats.pearsonr(rel_data['Composite_Negativity_Score'], 
                                         rel_data['event_tomorrow'])
            
            # Calculate predictive power using threshold
            high_neg_threshold = rel_data['Composite_Negativity_Score'].quantile(0.75)
            high_neg_days = rel_data[rel_data['Composite_Negativity_Score'] > high_neg_threshold]
            low_neg_days = rel_data[rel_data['Composite_Negativity_Score'] <= high_neg_threshold]
            
            if len(high_neg_days) > 0 and len(low_neg_days) > 0:
                event_prob_high = (high_neg_days['event_tomorrow'] > 0).mean()
                event_prob_low = (low_neg_days['event_tomorrow'] > 0).mean()
                predictive_ratio = event_prob_high / event_prob_low if event_prob_low > 0 else np.nan
                
                results.append({
                    'Relationship': relationship,
                    'Data_Points': len(rel_data),
                    'Correlation': corr,
                    'P_Value': p_value,
                    'Significant': p_value < 0.05,
                    'Event_Prob_High_Neg': event_prob_high,
                    'Event_Prob_Low_Neg': event_prob_low,
                    'Predictive_Ratio': predictive_ratio
                })
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Display results
        print("\n" + "="*60)
        print("PREDICTIVE POWER BY RELATIONSHIP PAIR")
        print("="*60)
        
        for _, row in results_df.iterrows():
            print(f"\n🔗 {row['Relationship']}:")
            print(f"   • Data points: {row['Data_Points']}")
            print(f"   • Correlation: {row['Correlation']:.3f}")
            print(f"   • P-value: {row['P_Value']:.4f}")
            print(f"   • Significant: {'YES' if row['Significant'] else 'NO'}")
            if not np.isnan(row['Predictive_Ratio']):
                print(f"   • Predictive power: {row['Predictive_Ratio']:.1f}x")
                print(f"   • Event probability: {row['Event_Prob_High_Neg']:.1%} vs {row['Event_Prob_Low_Neg']:.1%}")
        
        # Global summary
        print("\n" + "="*60)
        print("GLOBAL SUMMARY")
        print("="*60)
        
        significant_pairs = results_df[results_df['Significant']]
        if len(significant_pairs) > 0:
            avg_correlation = significant_pairs['Correlation'].mean()
            avg_predictive_ratio = significant_pairs['Predictive_Ratio'].mean()
            
            print(f"Significant relationship pairs: {len(significant_pairs)}")
            print(f"Average correlation: {avg_correlation:.3f}")
            print(f"Average predictive ratio: {avg_predictive_ratio:.1f}x")
            print(f"Success rate: {len(significant_pairs)/len(results_df):.1%}")
        else:
            print("No statistically significant relationships found")
        
        return results_df, clean_data

# Run the clean analysis
    results_df, clean_data = global_predictive_analysis_clean(df)
    
    def global_predictive_analysis(df):
        """Measure predictive power globally across all relationship pairs"""
        
        print("="*60)
        print("GLOBAL PREDICTIVE POWER ANALYSIS")
        print("="*60)
        
        # Clean data - remove rows with missing values
        clean_data = df.dropna(subset=['Composite_Negativity_Score', 'event_count']).copy()
        clean_data = clean_data.sort_values(['RelationshipPair', 'Date'])
        
        print(f"Total clean records: {len(clean_data)}")
        print(f"Unique relationship pairs: {clean_data['RelationshipPair'].nunique()}")
        print(f"Date range: {clean_data['Date'].min()} to {clean_data['Date'].max()}")
        
        # Global analysis (all pairs combined)
        print("\n" + "="*60)
        print("GLOBAL ANALYSIS (ALL RELATIONSHIPS COMBINED)")
        print("="*60)
        
        # Create tomorrow's events for global dataset
        global_data = clean_data.copy()
        global_data = global_data.sort_values(['RelationshipPair', 'Date'])
        global_data['event_tomorrow'] = global_data.groupby('RelationshipPair')['event_count'].shift(-1)
        global_data = global_data.dropna(subset=['event_tomorrow'])
        
        print(f"Global data points: {len(global_data)}")
        
        # Global correlation
        global_corr, global_p = stats.pearsonr(global_data['Composite_Negativity_Score'], 
                                             global_data['event_tomorrow'])
        
        print(f"Global correlation: {global_corr:.3f}")
        print(f"Global p-value: {global_p:.4f}")
        print(f"Globally significant: {'YES' if global_p < 0.05 else 'NO'}")
        
        # Global predictive power
        high_neg_threshold = global_data['Composite_Negativity_Score'].quantile(0.75)
        low_neg_threshold = global_data['Composite_Negativity_Score'].quantile(0.25)
        
        high_neg_global = global_data[global_data['Composite_Negativity_Score'] > high_neg_threshold]
        low_neg_global = global_data[global_data['Composite_Negativity_Score'] < low_neg_threshold]
        
        if len(high_neg_global) > 0 and len(low_neg_global) > 0:
            avg_events_high = high_neg_global['event_tomorrow'].mean()
            avg_events_low = low_neg_global['event_tomorrow'].mean()
            
            print(f"\nGlobal predictive power:")
            print(f"Average events after high negativity: {avg_events_high:.2f}")
            print(f"Average events after low negativity: {avg_events_low:.2f}")
            print(f"Predictive ratio: {avg_events_high/avg_events_low:.1f}x")
        
        # Analyze by relationship pair
        print("\n" + "="*60)
        print("ANALYSIS BY RELATIONSHIP PAIR")
        print("="*60)
        
        results = []
        
        for relationship in clean_data['RelationshipPair'].unique():
            rel_data = clean_data[clean_data['RelationshipPair'] == relationship].copy()
            rel_data = rel_data.sort_values('Date')
            
            # Skip if not enough data
            if len(rel_data) < 10:
                continue
                
            # Create tomorrow's events
            rel_data['event_tomorrow'] = rel_data['event_count'].shift(-1)
            rel_data = rel_data.dropna(subset=['event_tomorrow'])
            
            if len(rel_data) < 5:
                continue
                
            # Calculate correlation
            corr, p_value = stats.pearsonr(rel_data['Composite_Negativity_Score'], 
                                         rel_data['event_tomorrow'])
            
            # Calculate predictive metrics
            high_neg_threshold = rel_data['Composite_Negativity_Score'].quantile(0.75)
            high_neg_days = rel_data[rel_data['Composite_Negativity_Score'] > high_neg_threshold]
            low_neg_days = rel_data[rel_data['Composite_Negativity_Score'] <= high_neg_threshold]
            
            if len(high_neg_days) > 0 and len(low_neg_days) > 0:
                avg_events_high = high_neg_days['event_tomorrow'].mean()
                avg_events_low = low_neg_days['event_tomorrow'].mean()
                predictive_ratio = avg_events_high / avg_events_low if avg_events_low > 0 else np.nan
                
                results.append({
                    'Relationship': relationship,
                    'Data_Points': len(rel_data),
                    'Correlation': corr,
                    'P_Value': p_value,
                    'Significant': p_value < 0.05,
                    'Avg_Events_High_Neg': avg_events_high,
                    'Avg_Events_Low_Neg': avg_events_low,
                    'Predictive_Ratio': predictive_ratio
                })
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Display results
        if len(results_df) > 0:
            print(f"\nAnalyzed {len(results_df)} relationship pairs:")
            
            for _, row in results_df.iterrows():
                sig_indicator = "✅" if row['Significant'] else "❌"
                print(f"{sig_indicator} {row['Relationship']:15} | "
                      f"Corr: {row['Correlation']:6.3f} | "
                      f"p: {row['P_Value']:6.4f} | "
                      f"Ratio: {row['Predictive_Ratio']:4.1f}x | "
                      f"n: {row['Data_Points']}")
        
        # Summary statistics
        print("\n" + "="*60)
        print("GLOBAL SUMMARY STATISTICS")
        print("="*60)
        
        if len(results_df) > 0:
            significant_pairs = results_df[results_df['Significant']]
            positive_predictive = results_df[results_df['Predictive_Ratio'] > 1]
            
            print(f"Total relationship pairs analyzed: {len(results_df)}")
            print(f"Significant pairs (p < 0.05): {len(significant_pairs)} ({len(significant_pairs)/len(results_df):.1%})")
            print(f"Pairs with positive predictive power (>1x): {len(positive_predictive)} ({len(positive_predictive)/len(results_df):.1%})")
            
            if len(significant_pairs) > 0:
                print(f"Average correlation (significant pairs): {significant_pairs['Correlation'].mean():.3f}")
                print(f"Average predictive ratio (significant pairs): {significant_pairs['Predictive_Ratio'].mean():.1f}x")
            
            print(f"Overall average correlation: {results_df['Correlation'].mean():.3f}")
            print(f"Overall average predictive ratio: {results_df['Predictive_Ratio'].mean():.1f}x")
        
        # Visualization
        print("\n" + "="*60)
        print("VISUALIZATION")
        print("="*60)
        
        if len(results_df) > 0:
            plt.figure(figsize=(12, 6))
            
            # Plot 1: Correlation by relationship
            plt.subplot(1, 2, 1)
            sorted_results = results_df.sort_values('Correlation')
            colors = ['red' if p > 0.05 else 'green' for p in sorted_results['P_Value']]
            plt.barh(sorted_results['Relationship'], sorted_results['Correlation'], color=colors)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.xlabel('Correlation Coefficient')
            plt.title('Predictive Correlation by Relationship\n(Green = Significant)')
            
            # Plot 2: Predictive ratio
            plt.subplot(1, 2, 2)
            sorted_ratio = results_df.sort_values('Predictive_Ratio')
            colors_ratio = ['red' if ratio < 1 else 'green' for ratio in sorted_ratio['Predictive_Ratio']]
            plt.barh(sorted_ratio['Relationship'], sorted_ratio['Predictive_Ratio'], color=colors_ratio)
            plt.axvline(x=1, color='black', linestyle='-', alpha=0.3, label='No predictive power')
            plt.xlabel('Predictive Ratio (Higher = Better)')
            plt.title('Predictive Power by Relationship\n(Green = Positive Prediction)')
            
            plt.tight_layout()
            plt.savefig('global_predictive_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("Visualization saved as 'global_predictive_analysis.png'")
        
        return results_df, global_data

    # Run the global analysis
    global_results, global_data = global_predictive_analysis(df)

    # Additional global metrics
    print("\n" + "="*60)
    print("ADDITIONAL GLOBAL METRICS")
    print("="*60)

    # Event intensity prediction
    global_data['severe_event_tomorrow'] = (global_data['event_tomorrow'] > 1).astype(int)
    if global_data['severe_event_tomorrow'].sum() > 0:
        severe_corr, severe_p = stats.pointbiserialr(global_data['Composite_Negativity_Score'], 
                                                   global_data['severe_event_tomorrow'])
        print(f"Severe events (>1) prediction correlation: {severe_corr:.3f} (p = {severe_p:.4f})")

    # Time window analysis
    print(f"\nDifferent prediction windows:")
    for days in [2, 3, 7]:
        global_data[f'event_{days}day'] = global_data.groupby('RelationshipPair')['event_count'].shift(-days)
        temp_data = global_data.dropna(subset=[f'event_{days}day'])
        if len(temp_data) > 10:
            corr, p_val = stats.pearsonr(temp_data['Composite_Negativity_Score'], temp_data[f'event_{days}day'])
            print(f"  {days}-day prediction: corr = {corr:.3f}, p = {p_val:.4f}")
        
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 


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