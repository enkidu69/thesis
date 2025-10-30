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

syear=2020
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
    syear=2020
    start_month=1
    start_day=1
    eyear=2025
    end_month=9
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
        daily_path3 = os.path.join("E:", daily_filename)
        daily_path2 = os.path.join("D:", "\\", "Data", daily_filename)

        # Check if daily file already exists
        if os.path.exists(daily_path3):
            print(f"✓ Found existing file in E: {daily_filename}")
            daily_files.append(daily_path3)
            
        elif os.path.exists(daily_path):
            print(f"✓ Found existing file : {daily_filename}")
            daily_files.append(daily_path)
            
        elif os.path.exists(daily_path2):
            print(f"✓ Found existing file : {daily_filename}")
            daily_files.append(daily_path2)


        else:
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
#############################################################################################COUNTRY
    EVENT_ROOTCODES= ['10','11','12','13','14','15','16','17','18','19','20']
    # Event codes to filter for (political/diplomatic events)
    EVENT_CODES = ['012','016','0212','0214','0232','0233','0234','0243','0244','0252','0253','0254','0255','0256','026','027','028','0312','0314','032','0332','0333','0334','0354','0355','0356','036','037','038','039','046','050','051','052','053','054','055','056','057','06','060','061','062','063','064','071','072','073','074','075','0811','0812','0813','0814','082','083','0831','0832','0833','0834','0841','085','086','0861','0862','0863','087','0871','0872','0873','0874','092','093','094','1012','1014','102','1032','1033','1034','1041','1042','1043','1044','1052','1054','1055','1056','106','107','108','111','1121','1122','1123','1124','1125','113','114','115','116','121','1211','1212','122','1221','1222','1223','1224','123','1231','1232','1233','1234','124','1241','1242','1243','1244','1245','1246','125','126','127','128','129','130','131','1311','1312','1313','132','1321','1322','1323','1324','133','134','135','136','137','138','1381','1382','1383','1384','1385','139','140','141','1411','1412','1413','1414','142','1421','1422','1423','1424','143','1431','1432','1433','1434','144','1441','1442','1443','1444','145','1451','1452','1453','1454','150','151','152','153','154','155','16','160','161','162','1621','1622','1623','163','164','165','166','1661','1662','1663','1712','1721','1722','1723','1724','174','175','180','181','182','1821','1822','1823','183','1831','1832','1833','1834','184','185','186','190','191','192','193','194','195','1951','1952','196','200','201','202','203','204','2041','2042']

    print(f"Year: {YEAR}")
    print(f"Focal countries: {', '.join(FOCAL_COUNTRIES)}")
    print(f"Event codes: {len(EVENT_CODES)} types")

    # STEP 2: GET USER INPUT FOR COUNTERPARTS
    print("\nSTEP 2: Selecting counterpart countries...")

    # Default selection
    #default_counterparts = ['RS','IZ','IS','US', 'FR', 'UK', 'IT','SP','IR', 'AG', 'AJ', 'AM','BG', 'CH', 'GZ', 'HK', 'IN', 'ID', 'KN', 'KZ', 'LY','MD','MY', 'NU', 'PK', 'SA', 'SU', 'SY','TU', 'UZ', 'VE', 'WE', 'YM', 'XX']
    default_counterparts = ['RS']
    
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
            #print(df_daily)
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
                #print(df_filtered)
                # Filter by event codes
                if len(df_filtered) > 0:
#######################print(EVENT_CODES)####################################################################################EVENT CODE FILTERING#######################################################################################################
                    #print(df_filtered)
                    #df_filtered=df_filtered[df_filtered["EventCode"].astype(str).isin(EVENT_CODES)]
                    #print("EVENT CODES USED")
                    #rootcode filter
                    df_filtered=df_filtered[df_filtered["EventRootCode"].astype(str).isin(EVENT_ROOTCODES)]
                    print("ROOT CODES USED")
                    #df_filtered=df_filtered[df_filtered['Actor2Geo_CountryCode'].replace("", focal_country)]


                    #print(len(df_filtered))
                    if len(df_filtered) > 0:
                        # Create relationship column
                        cols = ['Actor2Geo_CountryCode', 'Actor1Geo_CountryCode']

                        df_filtered[cols] = (
                            df_filtered[cols]
                            .apply(lambda x: x.str.strip().replace('', np.nan))
                            .fillna(focal_country)
                        )

                        def get_relationship_pair(row):
                            if row['Actor1Geo_CountryCode'] == focal_country:
                                counterpart = row['Actor2Geo_CountryCode']
                            else:
                                counterpart = row['Actor1Geo_CountryCode']
                            
                            
                            # Collect all counterparts found
                            if pd.notna(counterpart) and counterpart != '':
                            #if counterpart != '':
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
#######COUNTERPARTFILTER#################################ignored counterpart filter###################################################################################################################################
        #combined_df = combined_df[counterpart_filter].copy()
        #print("COUNTERPART FILTER")
        
        print("NO COUNTERPART FILTER  -ALL COUNTRIES USED")
        
        print(f"✓ Filtered to {len(combined_df):,} events (from {before_filter:,})")
        
        # Final date range check
        if not combined_df.empty:
            final_min_date = combined_df['Date'].min()
            final_max_date = combined_df['Date'].max()
            print(f"✓ Final data date range: {final_min_date.strftime('%Y-%m-%d')} to {final_max_date.strftime('%Y-%m-%d')}")
    else:
        print("✗ No data found after processing!")
        return

    
    # STEP 5: CALCULATE NEGATIVITY SCORES
    print("\nSTEP 5: Calculating negativity scores...")
    #print the dataframe to excel to check data
    #combined_df


    #combinedname=f'coombinedDF_{YEAR}.xlsx'
    #combined_df.to_excel(combinedname, index=False)
    #exit()
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
            'China': 'CH',
            'Turkey': 'TU',
            'Taiwan (Province of China)': 'TW',
            'Holy See': 'VT',
            'Korea (the Democratic People\'s Republic of)': 'KN',
            'Bolivia (Plurinational State of)': 'BL',
            'European Union': 'EU',
            'Sint Maarten':'NL',
            'Republic of North Macedonia': 'MK',
            'Lebanon': 'LE',
            'Kosovo': 'KS',
            'Macau': 'CH',
            'Saint Thomas': 'US',
            'Vietnam': 'VM',
            'Kazakstan': 'KZ'
            
            
            
            
            

        }
       
        
        
        return manual_map.get(country_name, None)
    attacks=pd.DataFrame()
    aggregated_df['country_fips'] = aggregated_df['country'].apply(country_to_fips)
    aggregated_df['actor_country_fips'] = aggregated_df['actor_country'].apply(country_to_fips)
    aggregated_df.loc[aggregated_df["actor_country_fips"] == "CN", "actor_country_fips"] = "CH"
    aggregated_df.loc[aggregated_df["actor_country_fips"] == "RU", "actor_country_fips"] = "RS"
    aggregated_df.loc[aggregated_df["actor_country_fips"] == "NG", "actor_country_fips"] = "NI"
    #filter focal countries
    attacksfilter1=False
    attacksfilter2=False
    xx=False
    attacks=pd.DataFrame()
    #filter by focal country
    for focal_country in FOCAL_COUNTRIES:
        attacksfilter1 |= (aggregated_df['country_fips'] == focal_country)
        aggregated_df1 = aggregated_df[attacksfilter1].copy()
        #replace XX with same country to track all of them
        aggregated_df1.loc[aggregated_df1["actor_country_fips"] == "XX", "actor_country_fips"] = focal_country

        attacks=pd.concat([aggregated_df1,attacks], axis=0, ignore_index=True)
    
    
    for counterpart in selected_counterparts:
        attacksfilter2 |= (aggregated_df['actor_country_fips'] == counterpart)
    #filter counterparts to remove in case of all countries!!###########################################################
    
    #attacks = aggregated_df[attacksfilter2].copy()

    
    #consider any undetermined as UK
    #aggregated_df['actor_country_fips'].replace("XX",focal_country)
    #aggregated_df.loc[aggregated_df["actor_country_fips"] == "XX", "actor_country_fips"] = focal_country
    
    #filt=aggregated_df[aggregated_df["actor_country"]=='Russia']
    #print(filt)
    # Merge on date and country
    daily_scores['Date'] = pd.to_datetime(daily_scores['Date'], format='%d-%m-%Y').dt.date
    
    ###########################NOT MERGING ANYMORE
    
    
    merged_df = pd.merge(
        daily_scores,
        attacks,
        left_on=['Date', 'FocalCountry'],
        right_on=['event_date', 'country_fips'],
        how='left'  # Use 'left' to keep all daily_scores records, 'inner' for only matches
    )
    

    #clean attack data
    
    daily_attacks=merged_df[['event_date', 'motive', 'event_type', 'country', 'actor_country', 'description', 'event_count', 'country_fips','actor_country_fips']]  
    merged_df = merged_df.drop(columns=['event_date', 'motive', 'event_type', 'country', 'actor_country', 'description', 'event_count', 'country_fips','actor_country_fips'])
    daily_attacks=daily_attacks.drop_duplicates()

    merged = pd.merge(
        merged_df,
        daily_attacks,
        left_on=['Date', 'FocalCountry','Counterpart'],
        right_on=['event_date', 'country_fips','actor_country_fips'],
        how='left'  # Use 'left' to keep all daily_scores records, 'inner' for only matches
    )
    
    
    
    print(f"Merged dataset shape: {merged.shape}")
    print(f"Daily scores records: {len(daily_scores)}")
    mergedname=f'aggregated_cyber_events_{YEAR}_{random_str}.xlsx'
    #print(f"Merged records with cyber events: {merged_df[merged_df['event_count'].notna()].shape[0]}")
    daily_scores.to_excel(mergedname, index=False)
    #dailyscores OK
    attackname=f'cyberevents_{focal_country}.xlsx'
    attacks.to_excel(attackname, index=False)
   # df = pd.read_excel('GLOB_UK.xlsx', sheet_name='Sheet1')


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