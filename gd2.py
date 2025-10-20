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
from datetime import datetime, timedelta
import gc

# FIPS to ISO2 country code mapping (same as before)
FIPS_TO_ISO2 = {
    'AF': 'AF', 'AX': 'AX', 'AL': 'AL', 'DZ': 'DZ', 'AS': 'AS', 'AD': 'AD', 'AO': 'AO', 'AI': 'AI',
    'AQ': 'AQ', 'AG': 'AG', 'AR': 'AR', 'AM': 'AM', 'AW': 'AW', 'AU': 'AU', 'AT': 'AT', 'AZ': 'AZ',
    # ... (keep your existing FIPS_TO_ISO2 mapping)
    'UA': 'UA', 'AE': 'AE', 'GB': 'GB', 'US': 'US', 'UM': 'UM', 'UY': 'UY', 'UZ': 'UZ', 'VU': 'VU',
    'VE': 'VE', 'VN': 'VN', 'VG': 'VG', 'VI': 'VI', 'WF': 'WF', 'EH': 'EH', 'YE': 'YE', 'ZM': 'ZM',
    'ZW': 'ZW'
}

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



def download_in_chunks(year, focal_countries, chunk_size_days=7, temp_dir='temp_data'):
    """
    Download GDELT data in smaller chunks to avoid memory issues
    Reuses existing chunk files if available
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Define date ranges for chunks
    start_date = datetime(year, 9, 1)
    end_date = datetime(year, 10, 18)
    current_date = start_date
    
    chunk_files = []
    
    while current_date <= end_date:
        chunk_end = min(current_date + timedelta(days=chunk_size_days-1), end_date)
        
        start_str = current_date.strftime('%Y-%m-%d')
        end_str = chunk_end.strftime('%Y-%m-%d')
        
        chunk_filename = f"chunk_{start_str}_{end_str}.pkl"
        chunk_path = os.path.join(temp_dir, chunk_filename)
        
        # CHECK IF CHUNK ALREADY EXISTS
        if os.path.exists(chunk_path):
            print(f"Found existing chunk: {chunk_filename}")
            chunk_files.append(chunk_path)
        else:
            print(f"Downloading chunk: {start_str} to {end_str}")
            
            try:
                # Download this chunk
                df_chunk = g1.Search([start_str, end_str], table='events', coverage=False)
                
                if len(df_chunk) > 0:
                    # Remove duplicates and filter columns immediately
                    df_chunk = df_chunk.drop_duplicates(subset=["SOURCEURL"], keep='last')
                    df_chunk = filter_and_convert_columns(df_chunk, focal_countries)
                    
                    # Save chunk to file
                    df_chunk.to_pickle(chunk_path)
                    chunk_files.append(chunk_path)
                    
                    print(f"  Saved chunk with {len(df_chunk):,} events to {chunk_filename}")
                else:
                    print(f"  No data for this chunk")
                
                # Clear memory
                del df_chunk
                gc.collect()
                
            except Exception as e:
                print(f"  Error downloading chunk {start_str} to {end_str}: {e}")
        
        # Move to next chunk
        current_date = chunk_end + timedelta(days=1)
    
    return chunk_files
    
def check_existing_chunks(year, temp_dir='temp_data'):
    """
    Check how many chunks already exist and their status
    """
    if not os.path.exists(temp_dir):
        return 0, []
    
    start_date = datetime(year, 9, 1)
    end_date = datetime(year, 10, 18)
    current_date = start_date
    
    existing_chunks = []
    missing_chunks = []
    
    while current_date <= end_date:
        chunk_end = min(current_date + timedelta(days=6), end_date)  # 7-day chunks
        
        start_str = current_date.strftime('%Y-%m-%d')
        end_str = chunk_end.strftime('%Y-%m-%d')
        
        chunk_filename = f"chunk_{start_str}_{end_str}.pkl"
        chunk_path = os.path.join(temp_dir, chunk_filename)
        
        if os.path.exists(chunk_path):
            # Check file size to see if it's valid
            file_size = os.path.getsize(chunk_path)
            if file_size > 100:  # Minimum reasonable file size
                existing_chunks.append(chunk_path)
            else:
                print(f"Warning: Small/empty chunk file: {chunk_filename} ({file_size} bytes)")
                missing_chunks.append((start_str, end_str))
        else:
            missing_chunks.append((start_str, end_str))
        
        current_date = chunk_end + timedelta(days=1)
    
    return existing_chunks, missing_chunks

def process_chunk_files(chunk_files, focal_countries, event_codes=None):
    """
    Process chunk files and combine filtered data
    """
    all_country_data = []
    
    for i, chunk_file in enumerate(chunk_files):
        print(f"Processing chunk {i+1}/{len(chunk_files)}: {os.path.basename(chunk_file)}")
        
        # Load chunk
        df_chunk = pd.read_pickle(chunk_file)
        
        # Process each focal country for this chunk
        for focal_country in focal_countries:
            # Filter for events where focal country is either Actor1 OR Actor2
            focal_filter = (
                (df_chunk['Actor1Geo_CountryCode'] == focal_country) | 
                (df_chunk['Actor2Geo_CountryCode'] == focal_country)
            )
            
            df_filtered = df_chunk[focal_filter].copy()
            
            # Filter by event codes if provided
            if event_codes:
                df_filtered = df_filtered[df_filtered['EventBaseCode'].isin(event_codes)]
            
            if len(df_filtered) > 0:
                # Create relationship column
                def get_relationship_pair(row):
                    if row['Actor1Geo_CountryCode'] == focal_country:
                        return f"{focal_country}-{row['Actor2Geo_CountryCode']}"
                    else:
                        return f"{focal_country}-{row['Actor1Geo_CountryCode']}"
                
                df_filtered['RelationshipPair'] = df_filtered.apply(get_relationship_pair, axis=1)
                df_filtered['FocalCountry'] = focal_country
                
                # Convert SQLDATE to proper datetime
                df_filtered['Date'] = pd.to_datetime(df_filtered['SQLDATE'].astype(str), format='%Y%m%d')
                
                all_country_data.append(df_filtered)
        
        # Clear memory
        del df_chunk, df_filtered
        gc.collect()
    
    if all_country_data:
        combined_df = pd.concat(all_country_data, ignore_index=True)
        print(f"Total combined events after processing: {len(combined_df):,}")
        return combined_df
    else:
        return pd.DataFrame()

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
    
    print(f"  Filtered columns from {len(df.columns)} to {len(df_filtered.columns)}")
    
    # Convert country codes from FIPS to ISO2
    def convert_country_code(code):
        if pd.isna(code) or code == '':
            return code
        return FIPS_TO_ISO2.get(code, code)
    
    # Convert both actor country codes
    df_filtered['Actor1Geo_CountryCode'] = df_filtered['Actor1Geo_CountryCode'].apply(convert_country_code)
    df_filtered['Actor2Geo_CountryCode'] = df_filtered['Actor2Geo_CountryCode'].apply(convert_country_code)
    
    return df_filtered

def calculate_negativity_scores_incremental(combined_df, focal_countries, year, batch_size=10000):
    """
    Calculate scores in batches to avoid memory issues
    """
    if len(combined_df) == 0:
        return pd.DataFrame()
    
    # Process in batches if dataframe is large
    if len(combined_df) > batch_size:
        print(f"Processing {len(combined_df):,} events in batches of {batch_size:,}")
        
        all_daily_scores = []
        num_batches = (len(combined_df) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(combined_df))
            
            print(f"  Processing batch {i+1}/{num_batches}: rows {start_idx:,} to {end_idx:,}")
            
            batch_df = combined_df.iloc[start_idx:end_idx].copy()
            batch_scores = calculate_negativity_scores_batch(batch_df, focal_countries, year)
            
            if not batch_scores.empty:
                all_daily_scores.append(batch_scores)
            
            # Clear memory
            del batch_df, batch_scores
            gc.collect()
        
        if all_daily_scores:
            return pd.concat(all_daily_scores, ignore_index=True)
        else:
            return pd.DataFrame()
    else:
        return calculate_negativity_scores_batch(combined_df, focal_countries, year)

def calculate_negativity_scores_batch(df, focal_countries, year):
    """
    Calculate scores for a single batch
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
    
    # Filter by year
    daily_relationships = daily_relationships[daily_relationships['Date'].dt.year == year]
    
    # Calculate scores
    daily_scores = pd.DataFrame()
    
    for focal_country in focal_countries:
        country_data = daily_relationships[daily_relationships['FocalCountry'] == focal_country].copy()
        
        for relationship in country_data['RelationshipPair'].unique():
            rel_data = country_data[country_data['RelationshipPair'] == relationship].copy()
            rel_data = rel_data.sort_values('Date')
            
            window = min(7, len(rel_data))
            
            # Calculate rolling statistics
            rel_data['AvgTone_Rolling_Mean'] = rel_data['Daily_AvgTone'].rolling(window, min_periods=1).mean()
            rel_data['AvgTone_Rolling_Std'] = rel_data['Daily_AvgTone'].rolling(window, min_periods=1).std()
            
            # Calculate scores
            rel_data['Negativity_Score'] = -rel_data['Daily_AvgTone']
            rel_data['Negativity_ZScore'] = -rel_data['AvgTone_Rolling_Mean'] / rel_data['AvgTone_Rolling_Std']
            rel_data['Negativity_ZScore'] = rel_data['Negativity_ZScore'].replace([np.inf, -np.inf], np.nan).fillna(0)
            rel_data['Composite_Negativity_Score'] = rel_data['Negativity_Score'] + rel_data['Negativity_ZScore']
            
            # New impact indices
            rel_data['Tone_Article_Index'] = rel_data['Daily_AvgTone'] * rel_data['Daily_NumArticles']
            rel_data['Negativity_Article_Index'] = rel_data['Negativity_Score'] * rel_data['Daily_NumArticles']
            
            daily_scores = pd.concat([daily_scores, rel_data])
    
    return daily_scores

# Keep the existing find_common_counterparts and create_comparison_pdf_report functions
# (they should work as-is with the incremental approach)

def cleanup_temp_files(chunk_files):
    """Clean up temporary chunk files"""
    for chunk_file in chunk_files:
        try:
            os.remove(chunk_file)
            print(f"Cleaned up: {chunk_file}")
        except:
            pass


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
        
        # REDUCE TICKS: Set major locator for dates (every 7 days)
        axes[0, 0].xaxis.set_major_locator(plt.MaxNLocator(8))
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
        
        # REDUCE TICKS
        axes[0, 1].xaxis.set_major_locator(plt.MaxNLocator(8))
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Plot 3: NEW - Tone-Article Impact Index
        for i, counterpart in enumerate(top_counterparts):
            for focal_country in focal_countries:
                rel_data = daily_scores[
                    (daily_scores['FocalCountry'] == focal_country) & 
                    (daily_scores['RelationshipPair'] == f"{focal_country}-{counterpart}")
                ]
                if not rel_data.empty:
                    color = 'red' if focal_country == focal_countries[0] else 'blue'
                    linestyle = '-' if focal_country == focal_countries[0] else '--'
                    axes[1, 0].plot(rel_data['Date'], rel_data['Tone_Article_Index'], 
                                   label=f'{focal_country}-{counterpart}', 
                                   color=color, linestyle=linestyle, linewidth=2,
                                   marker='o' if focal_country == focal_countries[0] else 's',
                                   markersize=4)
        
        axes[1, 0].set_title('Tone-Article Impact Index\n(AvgTone × NumArticles)', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Impact Index', fontsize=10)
        
        # REDUCE TICKS
        axes[1, 0].xaxis.set_major_locator(plt.MaxNLocator(8))
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Plot 4: NEW - Negativity-Article Impact Index
        for i, counterpart in enumerate(top_counterparts):
            for focal_country in focal_countries:
                rel_data = daily_scores[
                    (daily_scores['FocalCountry'] == focal_country) & 
                    (daily_scores['RelationshipPair'] == f"{focal_country}-{counterpart}")
                ]
                if not rel_data.empty:
                    color = 'red' if focal_country == focal_countries[0] else 'blue'
                    linestyle = '-' if focal_country == focal_countries[0] else '--'
                    axes[1, 1].plot(rel_data['Date'], rel_data['Negativity_Article_Index'], 
                                   label=f'{focal_country}-{counterpart}', 
                                   color=color, linestyle=linestyle, linewidth=2,
                                   marker='o' if focal_country == focal_countries[0] else 's',
                                   markersize=4)
        
        axes[1, 1].set_title('Negativity-Article Impact Index\n(Negativity × NumArticles)', 
                            fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Impact Index', fontsize=10)
        
        # REDUCE TICKS
        axes[1, 1].xaxis.set_major_locator(plt.MaxNLocator(8))
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
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
                    
                    # Plot both countries' Tone-Article Impact Index
                    if not data1.empty:
                        axes[row, col].plot(data1['Date'], data1['Tone_Article_Index'], 
                                          label=focal_countries[0], color='red', linewidth=2, marker='o')
                    if not data2.empty:
                        axes[row, col].plot(data2['Date'], data2['Tone_Article_Index'], 
                                          label=focal_countries[1], color='blue', linewidth=2, marker='s')
                    
                    axes[row, col].set_title(f'With {counterpart}\nTone-Article Impact Index', 
                                           fontsize=11, fontweight='bold')
                    axes[row, col].set_ylabel('Impact Index', fontsize=9)
                    
                    # REDUCE TICKS
                    axes[row, col].xaxis.set_major_locator(plt.MaxNLocator(6))
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
        
        # PAGE 3: Summary statistics with new indices
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Create summary table with new indices
        summary_data = []
        for relationship in top_counterparts:
            for focal_country in focal_countries:
                rel_data = daily_scores[
                    (daily_scores['FocalCountry'] == focal_country) & 
                    (daily_scores['RelationshipPair'] == f"{focal_country}-{relationship}")
                ]
                if not rel_data.empty:
                    summary_data.append([
                        f"{focal_country}-{relationship}",
                        f"{rel_data['Daily_EventCount'].sum():,}",
                        f"{rel_data['Daily_AvgTone'].mean():.3f}",
                        f"{rel_data['Daily_NumArticles'].sum():,}",
                        f"{rel_data['Tone_Article_Index'].mean():.3f}",
                        f"{rel_data['Negativity_Article_Index'].mean():.3f}"
                    ])
        
        table = ax.table(cellText=summary_data,
                        colLabels=['Relationship', 'Total Events', 'Avg Tone', 'Total Articles', 
                                 'Tone-Article Index', 'Negativity-Article Index'],
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)
        
        ax.set_title(f'Summary Statistics with Impact Indices: {" vs ".join(focal_countries)} ({year})', 
                    fontsize=16, fontweight='bold', pad=20)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"PDF report saved to: {pdf_path}")
    return pdf_path



# UPDATED MAIN EXECUTION
if __name__ == "__main__":
    # Initialize GDELT
    g1 = gdelt(version=1)
    
    # Parameters
    YEAR = 2025
    FOCAL_COUNTRIES = ['FR', 'UK']
    EVENT_CODES = ['012','016','0212','0214','0232','0233','0234','0243','0244','0252','0253','0254','0255','0256','026','027','028','0312','0314','032','0332','0333','0334','0354','0355','0356','036','037','038','039','046','050','051','052','053','054','055','056','057','06','060','061','062','063','064','071','072','073','074','075','0811','0812','0813','0814','082','083','0831','0832','0833','0834','0841','085','086','0861','0862','0863','087','0871','0872','0873','0874','092','093','094','1012','1014','102','1032','1033','1034','1041','1042','1043','1044','1052','1054','1055','1056','106','107','108','111','1121','1122','1123','1124','1125','113','114','115','116','121','1211','1212','122','1221','1222','1223','1224','123','1231','1232','1233','1234','124','1241','1242','1243','1244','1245','1246','125','126','127','128','129','130','131','1311','1312','1313','132','1321','1322','1323','1324','133','134','135','136','137','138','1381','1382','1383','1384','1385','139','140','141','1411','1412','1413','1414','142','1421','1422','1423','1424','143','1431','1432','1433','1434','144','1441','1442','1443','1444','145','1451','1452','1453','1454','150','151','152','153','154','155','16','160','161','162','1621','1622','1623','163','164','165','166','1661','1662','1663','1712','1721','1722','1723','1724','174','175','180','181','182','1821','1822','1823','183','1831','1832','1833','1834','184','185','186','190','191','192','193','194','195','1951','1952','196','200','201','202','203','204','2041','2042']
    
    chunk_files = []
    
    try:
        print(f"Starting incremental download and processing for {YEAR}...")
        print("Press Ctrl+C at any time to interrupt the process")
        
        # STEP 0: CHECK FOR EXISTING CHUNKS FIRST
        print("Checking for existing chunk files...")
        existing_chunks, missing_chunks = check_existing_chunks(YEAR)
        
        if existing_chunks:
            print(f"Found {len(existing_chunks)} existing chunk files:")
            for chunk in existing_chunks:
                file_size = os.path.getsize(chunk) / 1024 / 1024  # Size in MB
                print(f"  {os.path.basename(chunk)} ({file_size:.1f} MB)")
            
            if missing_chunks:
                print(f"\nMissing chunks for dates:")
                for start_str, end_str in missing_chunks:
                    print(f"  {start_str} to {end_str}")
                
                user_input = input("\nDo you want to: [1] Use existing chunks only, [2] Download missing chunks, [3] Redownload all? ")
                
                if user_input == "1":
                    chunk_files = existing_chunks
                    print("Using existing chunks only...")
                elif user_input == "2":
                    print("Downloading missing chunks...")
                    # We'll handle this in the download_in_chunks function
                    chunk_files = download_in_chunks(YEAR, FOCAL_COUNTRIES, chunk_size_days=7)
                else:  # "3" or any other input
                    print("Redownloading all chunks...")
                    # Clean up existing chunks
                    for chunk in existing_chunks:
                        try:
                            os.remove(chunk)
                            print(f"Removed: {os.path.basename(chunk)}")
                        except:
                            pass
                    chunk_files = download_in_chunks(YEAR, FOCAL_COUNTRIES, chunk_size_days=7)
            else:
                print("\nAll chunks already exist! Using existing files.")
                chunk_files = existing_chunks
        else:
            print("No existing chunks found. Starting fresh download...")
            chunk_files = download_in_chunks(YEAR, FOCAL_COUNTRIES, chunk_size_days=7)
        
        if not chunk_files:
            print("No chunk files available! Exiting.")
            exit()
        
        # STEP 2: Process chunks incrementally
        print(f"\nProcessing {len(chunk_files)} chunk files...")
        combined_df = process_chunk_files(chunk_files, FOCAL_COUNTRIES, EVENT_CODES)
        
        if combined_df.empty:
            print("No data after processing! Exiting.")
            exit()
        
        # STEP 3: Calculate scores incrementally
        print(f"\nCalculating scores for {len(combined_df):,} events...")
        daily_scores = calculate_negativity_scores_incremental(combined_df, FOCAL_COUNTRIES, YEAR)
        
        if daily_scores.empty:
            print("No scores calculated! Exiting.")
            exit()
        
        # STEP 4: Create PDF report
        print("\nCreating PDF report...")
        pdf_path = create_comparison_pdf_report(daily_scores, FOCAL_COUNTRIES, YEAR, top_common=10)
        
        # STEP 5: Save final results
        random_str = "".join(random.choices(string.ascii_letters + string.digits, k=8))
        
        # Save combined data
        combined_filename = Path(os.getcwd()) / f"combined_data_{random_str}.pkl"
        combined_df.to_pickle(combined_filename)
        print(f"Combined data saved to: {combined_filename}")
        
        # Save daily scores
        scores_filename = f'{"_".join(FOCAL_COUNTRIES)}_negativity_scores_{YEAR}_{random_str}.csv'
        daily_scores.to_csv(scores_filename, index=False)
        print(f"Daily scores saved to: {scores_filename}")
        
        # Print summary
        print(f"\nProcessing complete!")
        print(f"Total events processed: {len(combined_df):,}")
        print(f"Daily scores calculated: {len(daily_scores):,}")
        
        # Ask about cleanup
        cleanup = input("\nDo you want to keep the chunk files for future use? (y/n): ")
        if cleanup.lower() != 'y':
            print("Cleaning up temporary files...")
            cleanup_temp_files(chunk_files)
        else:
            print("Chunk files preserved in 'temp_data' folder")
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user! Exiting gracefully...")
        # Don't clean up chunks on interrupt so they can be reused
        print("Chunk files preserved for future use")
    except Exception as e:
        print(f"\n\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        print("Chunk files preserved for debugging")