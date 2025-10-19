import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from gdelt import gdelt
import os
import requests, zipfile, io, csv, os, random, string
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

# FIPS to ISO2 country code mapping
FIPS_TO_ISO2 = {
    'AF': 'AF', 'AX': 'AX', 'AL': 'AL', 'DZ': 'DZ', 'AS': 'AS', 'AD': 'AD', 'AO': 'AO', 'AI': 'AI',
    'AQ': 'AQ', 'AG': 'AG', 'AR': 'AR', 'AM': 'AM', 'AW': 'AW', 'AU': 'AU', 'AT': 'AT', 'AZ': 'AZ',
    'BS': 'BS', 'BH': 'BH', 'BD': 'BD', 'BB': 'BB', 'BY': 'BY', 'BE': 'BE', 'BZ': 'BZ', 'BJ': 'BJ',
    'BM': 'BM', 'BT': 'BT', 'BO': 'BO', 'BQ': 'BQ', 'BA': 'BA', 'BW': 'BW', 'BV': 'BV', 'BR': 'BR',
    'IO': 'IO', 'BN': 'BN', 'BG': 'BG', 'BF': 'BF', 'BI': 'BI', 'KH': 'KH', 'CM': 'CM', 'CA': 'CA',
    'CV': 'CV', 'KY': 'KY', 'CF': 'CF', 'TD': 'TD', 'CL': 'CL', 'CN': 'CN', 'CX': 'CX', 'CC': 'CC',
    'CO': 'CO', 'KM': 'KM', 'CG': 'CG', 'CD': 'CD', 'CK': 'CK', 'CR': 'CR', 'CI': 'CI', 'HR': 'HR',
    'CU': 'CU', 'CW': 'CW', 'CY': 'CY', 'CZ': 'CZ', 'DK': 'DK', 'DJ': 'DJ', 'DM': 'DM', 'DO': 'DO',
    'EC': 'EC', 'EG': 'EG', 'SV': 'SV', 'GQ': 'GQ', 'ER': 'ER', 'EE': 'EE', 'ET': 'ET', 'FK': 'FK',
    'FO': 'FO', 'FJ': 'FJ', 'FI': 'FI', 'FR': 'FR', 'GF': 'GF', 'PF': 'PF', 'TF': 'TF', 'GA': 'GA',
    'GM': 'GM', 'GE': 'GE', 'DE': 'DE', 'GH': 'GH', 'GI': 'GI', 'GR': 'GR', 'GL': 'GL', 'GD': 'GD',
    'GP': 'GP', 'GU': 'GU', 'GT': 'GT', 'GG': 'GG', 'GN': 'GN', 'GW': 'GW', 'GY': 'GY', 'HT': 'HT',
    'HM': 'HM', 'VA': 'VA', 'HN': 'HN', 'HK': 'HK', 'HU': 'HU', 'IS': 'IS', 'IN': 'IN', 'ID': 'ID',
    'IR': 'IR', 'IQ': 'IQ', 'IE': 'IE', 'IM': 'IM', 'IL': 'IL', 'IT': 'IT', 'JM': 'JM', 'JP': 'JP',
    'JE': 'JE', 'JO': 'JO', 'KZ': 'KZ', 'KE': 'KE', 'KI': 'KI', 'KP': 'KP', 'KR': 'KR', 'KW': 'KW',
    'KG': 'KG', 'LA': 'LA', 'LV': 'LV', 'LB': 'LB', 'LS': 'LS', 'LR': 'LR', 'LY': 'LY', 'LI': 'LI',
    'LT': 'LT', 'LU': 'LU', 'MO': 'MO', 'MK': 'MK', 'MG': 'MG', 'MW': 'MW', 'MY': 'MY', 'MV': 'MV',
    'ML': 'ML', 'MT': 'MT', 'MH': 'MH', 'MQ': 'MQ', 'MR': 'MR', 'MU': 'MU', 'YT': 'YT', 'MX': 'MX',
    'FM': 'FM', 'MD': 'MD', 'MC': 'MC', 'MN': 'MN', 'ME': 'ME', 'MS': 'MS', 'MA': 'MA', 'MZ': 'MZ',
    'MM': 'MM', 'NA': 'NA', 'NR': 'NR', 'NP': 'NP', 'NL': 'NL', 'NC': 'NC', 'NZ': 'NZ', 'NI': 'NI',
    'NE': 'NE', 'NG': 'NG', 'NU': 'NU', 'NF': 'NF', 'MP': 'MP', 'NO': 'NO', 'OM': 'OM', 'PK': 'PK',
    'PW': 'PW', 'PS': 'PS', 'PA': 'PA', 'PG': 'PG', 'PY': 'PY', 'PE': 'PE', 'PH': 'PH', 'PN': 'PN',
    'PL': 'PL', 'PT': 'PT', 'PR': 'PR', 'QA': 'QA', 'RE': 'RE', 'RO': 'RO', 'RU': 'RU', 'RW': 'RW',
    'BL': 'BL', 'SH': 'SH', 'KN': 'KN', 'LC': 'LC', 'MF': 'MF', 'PM': 'PM', 'VC': 'VC', 'WS': 'WS',
    'SM': 'SM', 'ST': 'ST', 'SA': 'SA', 'SN': 'SN', 'RS': 'RS', 'SC': 'SC', 'SL': 'SL', 'SG': 'SG',
    'SX': 'SX', 'SK': 'SK', 'SI': 'SI', 'SB': 'SB', 'SO': 'SO', 'ZA': 'ZA', 'GS': 'GS', 'SS': 'SS',
    'ES': 'ES', 'LK': 'LK', 'SD': 'SD', 'SR': 'SR', 'SJ': 'SJ', 'SZ': 'SZ', 'SE': 'SE', 'CH': 'CH',
    'SY': 'SY', 'TW': 'TW', 'TJ': 'TJ', 'TZ': 'TZ', 'TH': 'TH', 'TL': 'TL', 'TG': 'TG', 'TK': 'TK',
    'TO': 'TO', 'TT': 'TT', 'TN': 'TN', 'TR': 'TR', 'TM': 'TM', 'TC': 'TC', 'TV': 'TV', 'UG': 'UG',
    'UA': 'UA', 'AE': 'AE', 'GB': 'GB', 'US': 'US', 'UM': 'UM', 'UY': 'UY', 'UZ': 'UZ', 'VU': 'VU',
    'VE': 'VE', 'VN': 'VN', 'VG': 'VG', 'VI': 'VI', 'WF': 'WF', 'EH': 'EH', 'YE': 'YE', 'ZM': 'ZM',
    'ZW': 'ZW'
}

def filter_and_convert_columns(df, focal_countries):
    """
    Filter columns to keep only essential ones and convert FIPS to ISO2 country codes
    """
    # Define columns to keep
    essential_columns = [
        'SQLDATE', 'SOURCEURL',
        'Actor1Geo_CountryCode', 'Actor2Geo_CountryCode',
        'EventRootCode', 'EventBaseCode', 'EventCode',
        'GoldsteinScale', 'AvgTone', 'NumMentions', 'NumArticles', 'NumSources'
    ]
    
    # Filter columns - only keep those that exist in the dataframe
    available_columns = [col for col in essential_columns if col in df.columns]
    df_filtered = df[available_columns].copy()
    
    print(f"Filtered columns from {len(df.columns)} to {len(df_filtered.columns)}")
    print(f"Keeping columns: {list(df_filtered.columns)}")
    
    # Convert country codes from FIPS to ISO2
    def convert_country_code(code):
        if pd.isna(code) or code == '':
            return code
        # GDELT uses FIPS codes which are mostly the same as ISO2, but we'll use mapping for safety
        return FIPS_TO_ISO2.get(code, code)  # Return original if not in mapping
    
    # Convert both actor country codes
    df_filtered['Actor1Geo_CountryCode'] = df_filtered['Actor1Geo_CountryCode'].apply(convert_country_code)
    df_filtered['Actor2Geo_CountryCode'] = df_filtered['Actor2Geo_CountryCode'].apply(convert_country_code)
    
    # Verify our focal countries are in the correct format
    for country in focal_countries:
        if country not in FIPS_TO_ISO2.values():
            print(f"Warning: Focal country {country} may not be in ISO2 format")
    
    return df_filtered

def calculate_negativity_scores(df, focal_countries, year):
    """
    Calculate DAILY negativity scores based on AvgTone for multiple focal countries
    """
    
    # Group by Date, FocalCountry and Relationship Pair
    daily_relationships = df.groupby(['Date', 'FocalCountry', 'RelationshipPair']).agg({
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
    
    # ENSURE WE ONLY HAVE DATES FROM THE SPECIFIED YEAR
    daily_relationships = daily_relationships[daily_relationships['Date'].dt.year == year]
    
    # Calculate negativity scores and Z-scores
    daily_scores = pd.DataFrame()
    
    for focal_country in focal_countries:
        country_data = daily_relationships[daily_relationships['FocalCountry'] == focal_country].copy()
        
        for relationship in country_data['RelationshipPair'].unique():
            rel_data = country_data[country_data['RelationshipPair'] == relationship].copy()
            
            # Sort by date
            rel_data = rel_data.sort_values('Date')
            
            # Use smaller window for limited date range
            window = min(7, len(rel_data))
            
            # Calculate rolling statistics for AvgTone (focus on negativity)
            rel_data['AvgTone_Rolling_Mean'] = rel_data['Daily_AvgTone'].rolling(window, min_periods=1).mean()
            rel_data['AvgTone_Rolling_Std'] = rel_data['Daily_AvgTone'].rolling(window, min_periods=1).std()
            
            # Calculate Negativity Score (negative AvgTone = higher negativity)
            rel_data['Negativity_Score'] = -rel_data['Daily_AvgTone']  # Direct inversion
            
            # Calculate Negativity Z-Score (how negative compared to historical average)
            rel_data['Negativity_ZScore'] = -rel_data['AvgTone_Rolling_Mean'] / rel_data['AvgTone_Rolling_Std']
            
            # Handle division by zero
            rel_data['Negativity_ZScore'] = rel_data['Negativity_ZScore'].replace([np.inf, -np.inf], np.nan)
            rel_data['Negativity_ZScore'] = rel_data['Negativity_ZScore'].fillna(0)
            
            # Create composite risk score focusing on negativity
            rel_data['Composite_Negativity_Score'] = (
                rel_data['Negativity_Score'] +  # Direct negativity
                rel_data['Negativity_ZScore']   # Relative negativity
            )
            
            daily_scores = pd.concat([daily_scores, rel_data])
    
    return daily_scores

def find_common_counterparts(daily_scores, focal_countries, min_events=5):
    """
    Find common counterpart countries that both focal countries interact with
    """
    common_relationships = []
    
    for focal1 in focal_countries:
        for focal2 in focal_countries:
            if focal1 != focal2:
                # Get relationships for each focal country
                rel1 = set([rel.split('-')[1] for rel in 
                           daily_scores[daily_scores['FocalCountry'] == focal1]['RelationshipPair'].unique()])
                rel2 = set([rel.split('-')[1] for rel in 
                           daily_scores[daily_scores['FocalCountry'] == focal2]['RelationshipPair'].unique()])
                
                # Find common counterparts
                common_counterparts = rel1.intersection(rel2)
                
                for counterpart in common_counterparts:
                    # Check if both relationships have sufficient events
                    rel1_data = daily_scores[
                        (daily_scores['FocalCountry'] == focal1) & 
                        (daily_scores['RelationshipPair'] == f"{focal1}-{counterpart}")
                    ]
                    rel2_data = daily_scores[
                        (daily_scores['FocalCountry'] == focal2) & 
                        (daily_scores['RelationshipPair'] == f"{focal2}-{counterpart}")
                    ]
                    
                    if len(rel1_data) >= min_events and len(rel2_data) >= min_events:
                        common_relationships.append({
                            'counterpart': counterpart,
                            f'{focal1}_events': len(rel1_data),
                            f'{focal2}_events': len(rel2_data),
                            f'{focal1}_avg_negativity': rel1_data['Negativity_Score'].mean(),
                            f'{focal2}_avg_negativity': rel2_data['Negativity_Score'].mean()
                        })
    
    return pd.DataFrame(common_relationships)

def create_comparison_pdf_report(daily_scores, focal_countries, year, top_common=10, output_dir='reports'):
    """
    Create PDF report comparing two focal countries against common counterparts
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    country_str = '_'.join(focal_countries)
    pdf_path = os.path.join(output_dir, f'{country_str}_Comparison_Analysis_{year}.pdf')
    
    # Find common counterparts
    common_df = find_common_counterparts(daily_scores, focal_countries)
    
    if common_df.empty:
        print("No common counterparts found with sufficient data!")
        return None
    
    # Sort by total negativity (sum of both countries' negativity)
    common_df['total_negativity'] = common_df[f'{focal_countries[0]}_avg_negativity'] + common_df[f'{focal_countries[1]}_avg_negativity']
    common_df = common_df.sort_values('total_negativity', ascending=False)
    top_common = common_df.head(top_common)
    
    print(f"Found {len(common_df)} common counterparts")
    print(f"Top {len(top_common)} most negative common relationships:")
    print(top_common[['counterpart', 'total_negativity']].to_string(index=False))
    
    with PdfPages(pdf_path) as pdf:
        plt.style.use('default')
        sns.set_palette("tab10")
        
        # PAGE 1: Overview - Negativity comparison for top common counterparts
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Negativity Comparison: {" vs ".join(focal_countries)} with Common Counterparts ({year})', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Get top common counterparts
        top_counterparts = top_common['counterpart'].head(8).tolist()
        
        # Plot 1: Daily Negativity Score Comparison
        for i, counterpart in enumerate(top_counterparts):
            for focal_country in focal_countries:
                rel_data = daily_scores[
                    (daily_scores['FocalCountry'] == focal_country) & 
                    (daily_scores['RelationshipPair'] == f"{focal_country}-{counterpart}")
                ]
                if not rel_data.empty:
                    color = 'red' if focal_country == focal_countries[0] else 'blue'
                    linestyle = '-' if focal_country == focal_countries[0] else '--'
                    axes[0, 0].plot(rel_data['Date'], rel_data['Negativity_Score'], 
                                   label=f'{focal_country}-{counterpart}', 
                                   color=color, linestyle=linestyle, linewidth=2,
                                   marker='o' if focal_country == focal_countries[0] else 's',
                                   markersize=4)
        
        axes[0, 0].set_title('Daily Negativity Score Comparison\n(Higher = More Negative)', 
                            fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Negativity Score', fontsize=10)
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Plot 2: Negativity Z-Score Comparison
        for i, counterpart in enumerate(top_counterparts):
            for focal_country in focal_countries:
                rel_data = daily_scores[
                    (daily_scores['FocalCountry'] == focal_country) & 
                    (daily_scores['RelationshipPair'] == f"{focal_country}-{counterpart}")
                ]
                if not rel_data.empty:
                    color = 'red' if focal_country == focal_countries[0] else 'blue'
                    linestyle = '-' if focal_country == focal_countries[0] else '--'
                    axes[0, 1].plot(rel_data['Date'], rel_data['Negativity_ZScore'], 
                                   label=f'{focal_country}-{counterpart}', 
                                   color=color, linestyle=linestyle, linewidth=2,
                                   marker='o' if focal_country == focal_countries[0] else 's',
                                   markersize=4)
        
        axes[0, 1].set_title('Negativity Z-Score Comparison\n(Relative to Historical Average)', 
                            fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Z-Score', fontsize=10)
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Plot 3: Composite Negativity Score
        for i, counterpart in enumerate(top_counterparts):
            for focal_country in focal_countries:
                rel_data = daily_scores[
                    (daily_scores['FocalCountry'] == focal_country) & 
                    (daily_scores['RelationshipPair'] == f"{focal_country}-{counterpart}")
                ]
                if not rel_data.empty:
                    color = 'red' if focal_country == focal_countries[0] else 'blue'
                    linestyle = '-' if focal_country == focal_countries[0] else '--'
                    axes[1, 0].plot(rel_data['Date'], rel_data['Composite_Negativity_Score'], 
                                   label=f'{focal_country}-{counterpart}', 
                                   color=color, linestyle=linestyle, linewidth=2,
                                   marker='o' if focal_country == focal_countries[0] else 's',
                                   markersize=4)
        
        axes[1, 0].set_title('Composite Negativity Score\n(Combined Absolute and Relative Negativity)', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Composite Score', fontsize=10)
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Plot 4: Event Count Comparison
        for i, counterpart in enumerate(top_counterparts):
            for focal_country in focal_countries:
                rel_data = daily_scores[
                    (daily_scores['FocalCountry'] == focal_country) & 
                    (daily_scores['RelationshipPair'] == f"{focal_country}-{counterpart}")
                ]
                if not rel_data.empty:
                    color = 'red' if focal_country == focal_countries[0] else 'blue'
                    axes[1, 1].bar([f'{focal_country}-{counterpart}'], 
                                  [rel_data['Daily_EventCount'].sum()],
                                  color=color, alpha=0.7, label=focal_country)
        
        axes[1, 1].set_title('Total Event Count by Relationship', 
                            fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Total Events', fontsize=10)
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend(fontsize=8)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # PAGE 2: Side-by-side comparison for each common counterpart
        relationships_per_page = 4
        total_pages = (len(top_counterparts) + relationships_per_page - 1) // relationships_per_page
        
        for page in range(total_pages):
            start_idx = page * relationships_per_page
            end_idx = start_idx + relationships_per_page
            page_counterparts = top_counterparts[start_idx:end_idx]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Side-by-Side Comparison - Page {page+1}', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            for i, counterpart in enumerate(page_counterparts):
                row = i // 2
                col = i % 2
                
                if row < 2:  # Ensure we don't exceed subplot indices
                    # Get data for both countries
                    data1 = daily_scores[
                        (daily_scores['FocalCountry'] == focal_countries[0]) & 
                        (daily_scores['RelationshipPair'] == f"{focal_countries[0]}-{counterpart}")
                    ]
                    data2 = daily_scores[
                        (daily_scores['FocalCountry'] == focal_countries[1]) & 
                        (daily_scores['RelationshipPair'] == f"{focal_countries[1]}-{counterpart}")
                    ]
                    
                    # Plot both countries' negativity scores
                    if not data1.empty:
                        axes[row, col].plot(data1['Date'], data1['Negativity_Score'], 
                                          label=focal_countries[0], color='red', linewidth=2, marker='o')
                    if not data2.empty:
                        axes[row, col].plot(data2['Date'], data2['Negativity_Score'], 
                                          label=focal_countries[1], color='blue', linewidth=2, marker='s')
                    
                    axes[row, col].set_title(f'With {counterpart}\nNegativity Comparison', 
                                           fontsize=11, fontweight='bold')
                    axes[row, col].set_ylabel('Negativity Score', fontsize=9)
                    axes[row, col].tick_params(axis='x', rotation=45)
                    axes[row, col].legend(fontsize=8)
                    axes[row, col].grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(len(page_counterparts), 4):
                row = i // 2
                col = i % 2
                if row < 2:
                    axes[row, col].set_visible(False)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    
    print(f"PDF report saved to: {pdf_path}")
    return pdf_path

# MAIN EXECUTION
if __name__ == "__main__":
    # Initialize GDELT
    g1 = gdelt(version=1)
    
    # Parameters
    YEAR = 2025
    FOCAL_COUNTRIES = ['FR', 'UK']  # France vs UK comparison
    EVENT_CODES = ['012','016','0212','0214','0232','0233','0234','0243','0244','0252','0253','0254','0255','0256','026','027','028','0312','0314','032','0332','0333','0334','0354','0355','0356','036','037','038','039','046','050','051','052','053','054','055','056','057','06','060','061','062','063','064','071','072','073','074','075','0811','0812','0813','0814','082','083','0831','0832','0833','0834','0841','085','086','0861','0862','0863','087','0871','0872','0873','0874','092','093','094','1012','1014','102','1032','1033','1034','1041','1042','1043','1044','1052','1054','1055','1056','106','107','108','111','1121','1122','1123','1124','1125','113','114','115','116','121','1211','1212','122','1221','1222','1223','1224','123','1231','1232','1233','1234','124','1241','1242','1243','1244','1245','1246','125','126','127','128','129','130','131','1311','1312','1313','132','1321','1322','1323','1324','133','134','135','136','137','138','1381','1382','1383','1384','1385','139','140','141','1411','1412','1413','1414','142','1421','1422','1423','1424','143','1431','1432','1433','1434','144','1441','1442','1443','1444','145','1451','1452','1453','1454','150','151','152','153','154','155','16','160','161','162','1621','1622','1623','163','164','165','166','1661','1662','1663','1712','1721','1722','1723','1724','174','175','180','181','182','1821','1822','1823','183','1831','1832','1833','1834','184','185','186','190','191','192','193','194','195','1951','1952','196','200','201','202','203','204','2041','2042']
    
    try:
        print(f"Downloading GDELT data for {YEAR}...")
        print("Press Ctrl+C at any time to interrupt the download")
        
        # DOWNLOAD DATA DIRECTLY IN MAIN (INTERRUPTIBLE)
        start_date = f"{YEAR}-10-15"
        end_date = f"{YEAR}-10-18"
        
        df = g1.Search([start_date, end_date], table='events', coverage=False)
        print(f"Downloaded {len(df):,} total events")
        
        # Remove duplicate URLs
        df = df.drop_duplicates(subset=["SOURCEURL"], keep='last')
        print(f"After removing URL duplicates: {len(df):,} events")
        
        # FILTER COLUMNS AND CONVERT COUNTRY CODES
        df = filter_and_convert_columns(df, FOCAL_COUNTRIES)
        
        all_country_data = []
        
        for focal_country in FOCAL_COUNTRIES:
            print(f"\nAnalyzing {focal_country}...")
            
            # Filter for events where focal country is either Actor1 OR Actor2
            focal_filter = (
                (df['Actor1Geo_CountryCode'] == focal_country) | 
                (df['Actor2Geo_CountryCode'] == focal_country)
            )
            
            df_filtered = df[focal_filter].copy()
            print(f"Found {len(df_filtered):,} events involving {focal_country}")
            
            # Filter by event codes if provided
            if EVENT_CODES:
                df_filtered = df_filtered[df_filtered['EventBaseCode'].isin(EVENT_CODES)]
                print(f"After event code filtering: {len(df_filtered):,} events")
            
            # Create relationship column: always "FocalCountry-OtherCountry"
            def get_relationship_pair(row):
                if row['Actor1Geo_CountryCode'] == focal_country:
                    return f"{focal_country}-{row['Actor2Geo_CountryCode']}"
                else:
                    return f"{focal_country}-{row['Actor1Geo_CountryCode']}"
            
            df_filtered['RelationshipPair'] = df_filtered.apply(get_relationship_pair, axis=1)
            df_filtered['FocalCountry'] = focal_country
            
            # Convert SQLDATE to proper datetime and ensure year filtering
            df_filtered['Date'] = pd.to_datetime(df_filtered['SQLDATE'].astype(str), format='%Y%m%d')
            df_filtered = df_filtered[df_filtered['Date'].dt.year == YEAR]
            
            all_country_data.append(df_filtered)
        
        combined_df = pd.concat(all_country_data, ignore_index=True)
        print(f"\nTotal combined events: {len(combined_df):,}")
        
        # Save raw data (filtered version)
        random_str = "".join(random.choices(string.ascii_letters + string.digits, k=8))
        filename = Path(os.getcwd()) / f"comparison_{random_str}.xlsx"
        combined_df.to_excel(filename, index=False, engine="xlsxwriter", engine_kwargs={'options': {'strings_to_urls': False}})
        print(f"Filtered raw data saved to: {filename}")
        
        # Calculate daily scores
        daily_scores = calculate_negativity_scores(combined_df, FOCAL_COUNTRIES, YEAR)
        
        # Create PDF report
        pdf_path = create_comparison_pdf_report(daily_scores, FOCAL_COUNTRIES, YEAR, top_common=10)
        
        # Print summary
        print(f"\nComparison summary for {FOCAL_COUNTRIES[0]} vs {FOCAL_COUNTRIES[1]} in {YEAR}:")
        common_df = find_common_counterparts(daily_scores, FOCAL_COUNTRIES)
        if not common_df.empty:
            common_df = common_df.sort_values('total_negativity', ascending=False)
            for i, row in common_df.head(5).iterrows():
                print(f"{i+1}. {row['counterpart']}: "
                      f"{FOCAL_COUNTRIES[0]} negativity={row[f'{FOCAL_COUNTRIES[0]}_avg_negativity']:.3f}, "
                      f"{FOCAL_COUNTRIES[1]} negativity={row[f'{FOCAL_COUNTRIES[1]}_avg_negativity']:.3f}")
        
        # Save daily scores for correlation with cyber attacks
        daily_scores.to_csv(f'{"_".join(FOCAL_COUNTRIES)}_negativity_scores_{YEAR}.csv', index=False)
        print(f"\nDaily negativity scores saved to {'_'.join(FOCAL_COUNTRIES)}_negativity_scores_{YEAR}.csv")
        
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user! Exiting gracefully...")
    except Exception as e:
        print(f"\n\nError during execution: {e}")