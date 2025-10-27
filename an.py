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



def main():
    """
    SEQUENTIAL EXECUTION - Step by step process
    """
    global interrupted
    
    df = pd.read_excel('GLOB_UK_allcountries.xlsx', sheet_name='Sheet1')
    
    def global_aggregated_analysis(df):
        """Global analysis without relationship pairs and reduced data requirements"""
        
        print("="*70)
        print("GLOBAL AGGREGATED ANALYSIS")
        print("No Relationship Pairs - Reduced Data Requirements")
        print("="*70)
        
        # Clean data - focus on global patterns
        clean_data = df.dropna(subset=['Daily_AvgTone', 'event_count', 'Composite_Negativity_Score']).copy()
        clean_data['Date'] = pd.to_datetime(clean_data['Date'])
        clean_data = clean_data.sort_values('Date')
        
        results = {}
        
        # 1. GLOBAL DAILY DESCRIPTIVE ANALYSIS
        print("1. GLOBAL DAILY DESCRIPTIVE CORRELATIONS")
        print("-" * 50)
        
        # Aggregate by date (sum across all relationship pairs)
        daily_global = clean_data.groupby('Date').agg({
            'Daily_AvgTone': 'sum',  # Sum of all daily tones
            'Composite_Negativity_Score': 'mean',  # Average composite negativity
            'event_count': 'sum',  # Total events worldwide
            'FocalCountry': 'count'  # Number of relationship pairs
        }).reset_index()
        
        daily_global.columns = ['Date', 'Global_Daily_AvgTone_Sum', 'Global_Composite_Negativity_Mean', 
                               'Global_Event_Count_Sum', 'Relationship_Pairs_Count']
        
        print(f"Global daily data points: {len(daily_global)}")
        print(f"Date range: {daily_global['Date'].min()} to {daily_global['Date'].max()}")
        
        # Global daily descriptive correlations
        daily_tone_corr, daily_tone_p = stats.pearsonr(daily_global['Global_Daily_AvgTone_Sum'], 
                                                      daily_global['Global_Event_Count_Sum'])
        daily_comp_corr, daily_comp_p = stats.pearsonr(daily_global['Global_Composite_Negativity_Mean'], 
                                                      daily_global['Global_Event_Count_Sum'])
        
        print(f"Global Daily_AvgTone_Sum → Same day events: {daily_tone_corr:.4f} (p = {daily_tone_p:.4f})")
        print(f"Global Composite_Negativity_Mean → Same day events: {daily_comp_corr:.4f} (p = {daily_comp_p:.4f})")
        
        results['global_daily_descriptive'] = {
            'Daily_AvgTone_Sum': (daily_tone_corr, daily_tone_p, len(daily_global)),
            'Composite_Negativity_Mean': (daily_comp_corr, daily_comp_p, len(daily_global))
        }
        
        # 2. GLOBAL MONTHLY DESCRIPTIVE ANALYSIS (Reduced: min 5 days)
        print(f"\n2. GLOBAL MONTHLY DESCRIPTIVE CORRELATIONS")
        print("-" * 50)
        
        daily_global['YearMonth'] = daily_global['Date'].dt.to_period('M')
        monthly_global = daily_global.groupby('YearMonth').agg({
            'Global_Daily_AvgTone_Sum': 'sum',
            'Global_Composite_Negativity_Mean': 'mean',
            'Global_Event_Count_Sum': 'sum',
            'Date': 'count'
        }).reset_index()
        
        monthly_global.columns = ['YearMonth', 'Monthly_Daily_AvgTone_Sum', 'Monthly_Composite_Negativity_Mean', 
                                 'Monthly_Event_Count_Sum', 'Days_With_Data']
        
        # Reduced requirement: only 5 days per month
        monthly_global = monthly_global[monthly_global['Days_With_Data'] >= 5]
        
        print(f"Global monthly data points: {len(monthly_global)}")
        
        # Global monthly descriptive correlations
        monthly_tone_corr, monthly_tone_p = stats.pearsonr(monthly_global['Monthly_Daily_AvgTone_Sum'], 
                                                          monthly_global['Monthly_Event_Count_Sum'])
        monthly_comp_corr, monthly_comp_p = stats.pearsonr(monthly_global['Monthly_Composite_Negativity_Mean'], 
                                                          monthly_global['Monthly_Event_Count_Sum'])
        
        print(f"Global Monthly Daily_AvgTone_Sum → Same month events: {monthly_tone_corr:.4f} (p = {monthly_tone_p:.4f})")
        print(f"Global Monthly Composite_Negativity_Mean → Same month events: {monthly_comp_corr:.4f} (p = {monthly_comp_p:.4f})")
        
        results['global_monthly_descriptive'] = {
            'Daily_AvgTone_Sum': (monthly_tone_corr, monthly_tone_p, len(monthly_global)),
            'Composite_Negativity_Mean': (monthly_comp_corr, monthly_comp_p, len(monthly_global))
        }
        
        # 3. GLOBAL QUARTERLY DESCRIPTIVE ANALYSIS (Reduced: min 10 days)
        print(f"\n3. GLOBAL QUARTERLY DESCRIPTIVE CORRELATIONS")
        print("-" * 50)
        
        daily_global['YearQuarter'] = daily_global['Date'].dt.to_period('Q')
        quarterly_global = daily_global.groupby('YearQuarter').agg({
            'Global_Daily_AvgTone_Sum': 'sum',
            'Global_Composite_Negativity_Mean': 'mean',
            'Global_Event_Count_Sum': 'sum',
            'Date': 'count'
        }).reset_index()
        
        quarterly_global.columns = ['YearQuarter', 'Quarterly_Daily_AvgTone_Sum', 'Quarterly_Composite_Negativity_Mean', 
                                   'Quarterly_Event_Count_Sum', 'Days_With_Data']
        
        # Reduced requirement: only 10 days per quarter
        quarterly_global = quarterly_global[quarterly_global['Days_With_Data'] >= 10]
        
        print(f"Global quarterly data points: {len(quarterly_global)}")
        
        # Global quarterly descriptive correlations
        quarterly_tone_corr, quarterly_tone_p = stats.pearsonr(quarterly_global['Quarterly_Daily_AvgTone_Sum'], 
                                                              quarterly_global['Quarterly_Event_Count_Sum'])
        quarterly_comp_corr, quarterly_comp_p = stats.pearsonr(quarterly_global['Quarterly_Composite_Negativity_Mean'], 
                                                              quarterly_global['Quarterly_Event_Count_Sum'])
        
        print(f"Global Quarterly Daily_AvgTone_Sum → Same quarter events: {quarterly_tone_corr:.4f} (p = {quarterly_tone_p:.4f})")
        print(f"Global Quarterly Composite_Negativity_Mean → Same quarter events: {quarterly_comp_corr:.4f} (p = {quarterly_comp_p:.4f})")
        
        results['global_quarterly_descriptive'] = {
            'Daily_AvgTone_Sum': (quarterly_tone_corr, quarterly_tone_p, len(quarterly_global)),
            'Composite_Negativity_Mean': (quarterly_comp_corr, quarterly_comp_p, len(quarterly_global))
        }
        
        # 4. GLOBAL MONTHLY PREDICTIVE ANALYSIS
        print(f"\n4. GLOBAL MONTHLY PREDICTIVE CORRELATIONS")
        print("-" * 50)
        
        monthly_predictive = monthly_global.copy()
        monthly_predictive = monthly_predictive.sort_values('YearMonth')
        monthly_predictive['Next_Month_Events'] = monthly_predictive['Monthly_Event_Count_Sum'].shift(-1)
        monthly_predictive = monthly_predictive.dropna(subset=['Next_Month_Events'])
        
        # Global monthly predictive correlations
        monthly_pred_tone_corr, monthly_pred_tone_p = stats.pearsonr(monthly_predictive['Monthly_Daily_AvgTone_Sum'], 
                                                                   monthly_predictive['Next_Month_Events'])
        monthly_pred_comp_corr, monthly_pred_comp_p = stats.pearsonr(monthly_predictive['Monthly_Composite_Negativity_Mean'], 
                                                                   monthly_predictive['Next_Month_Events'])
        
        print(f"Global Monthly Daily_AvgTone_Sum → Next month events: {monthly_pred_tone_corr:.4f} (p = {monthly_pred_tone_p:.4f})")
        print(f"Global Monthly Composite_Negativity_Mean → Next month events: {monthly_pred_comp_corr:.4f} (p = {monthly_pred_comp_p:.4f})")
        
        results['global_monthly_predictive'] = {
            'Daily_AvgTone_Sum': (monthly_pred_tone_corr, monthly_pred_tone_p, len(monthly_predictive)),
            'Composite_Negativity_Mean': (monthly_pred_comp_corr, monthly_pred_comp_p, len(monthly_predictive))
        }
        
        # 5. COMPREHENSIVE GLOBAL RESULTS TABLE
        print(f"\n5. GLOBAL RESULTS SUMMARY")
        print("-" * 50)
        
        print(f"{'Analysis':<25} | {'Metric':<30} | {'Correlation':<12} | {'P-value':<10} | {'Significant':<12} | {'N':<6}")
        print("-" * 110)
        
        for analysis_type, metrics in results.items():
            for metric, (corr, p, n) in metrics.items():
                sig = "YES" if p < 0.05 else "NO"
                print(f"{analysis_type:<25} | {metric:<30} | {corr:11.4f}  | {p:9.4f}  | {sig:<12} | {n:<6}")
        
        # 6. GLOBAL TREND VISUALIZATION
        print(f"\n6. GLOBAL TREND VISUALIZATION")
        print("-" * 50)
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Global daily trends over time
        plt.subplot(2, 2, 1)
        plt.plot(daily_global['Date'], daily_global['Global_Daily_AvgTone_Sum'], 'b-', alpha=0.7, label='Daily Tone Sum')
        plt.plot(daily_global['Date'], daily_global['Global_Event_Count_Sum'] * 10, 'r-', alpha=0.7, label='Events × 10')
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.title('Global Daily Trends: Tone Sum vs Events')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Global monthly trends
        plt.subplot(2, 2, 2)
        if len(monthly_global) > 0:
            plt.plot(monthly_global['YearMonth'].astype(str), monthly_global['Monthly_Daily_AvgTone_Sum'], 
                    'b-o', alpha=0.7, label='Monthly Tone Sum')
            plt.plot(monthly_global['YearMonth'].astype(str), monthly_global['Monthly_Event_Count_Sum'], 
                    'r-s', alpha=0.7, label='Monthly Events')
            plt.xlabel('Month')
            plt.ylabel('Values')
            plt.title('Global Monthly Trends')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        # Plot 3: Correlation strength by timeframe
        plt.subplot(2, 2, 3)
        timeframes = ['Daily', 'Monthly', 'Quarterly']
        tone_corrs = [daily_tone_corr, monthly_tone_corr, quarterly_tone_corr]
        comp_corrs = [daily_comp_corr, monthly_comp_corr, quarterly_comp_corr]
        
        x = np.arange(len(timeframes))
        width = 0.35
        
        plt.bar(x - width/2, [abs(c) for c in tone_corrs], width, label='Daily_AvgTone', alpha=0.7)
        plt.bar(x + width/2, [abs(c) for c in comp_corrs], width, label='Composite_Negativity', alpha=0.7)
        plt.xlabel('Timeframe')
        plt.ylabel('Absolute Correlation')
        plt.title('Global Correlation Strength by Timeframe')
        plt.xticks(x, timeframes)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Global monthly predictive scatter
        plt.subplot(2, 2, 4)
        if len(monthly_predictive) >= 2:
            plt.scatter(monthly_predictive['Monthly_Daily_AvgTone_Sum'], monthly_predictive['Next_Month_Events'], 
                       alpha=0.6, s=50)
            plt.xlabel('Monthly Daily_AvgTone_Sum')
            plt.ylabel('Next Month Events')
            plt.title(f'Global Monthly Predictive\nr = {monthly_pred_tone_corr:.4f}')
            
            # Add trendline
            z = np.polyfit(monthly_predictive['Monthly_Daily_AvgTone_Sum'], monthly_predictive['Next_Month_Events'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(monthly_predictive['Monthly_Daily_AvgTone_Sum'].min(), 
                                 monthly_predictive['Monthly_Daily_AvgTone_Sum'].max(), 100)
            plt.plot(x_range, p(x_range), 'r-', alpha=0.8)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 7. GLOBAL KEY INSIGHTS
        print(f"\n7. GLOBAL KEY INSIGHTS")
        print("-" * 50)
        
        # Find strongest global relationships
        all_global_correlations = []
        for analysis_type, metrics in results.items():
            for metric, (corr, p, n) in metrics.items():
                if not np.isnan(corr) and not np.isnan(p):
                    all_global_correlations.append((analysis_type, metric, corr, p, n))
        
        if all_global_correlations:
            # Strongest global correlation
            strongest_global = max(all_global_correlations, key=lambda x: abs(x[2]))
            print(f"• Strongest global correlation: {strongest_global[0]} - {strongest_global[1]}")
            print(f"  Correlation: {strongest_global[2]:.4f} (p = {strongest_global[3]:.4f})")
            
            # Count significant global findings
            significant_global = [c for c in all_global_correlations if c[3] < 0.05]
            print(f"• Statistically significant global findings: {len(significant_global)}/{len(all_global_correlations)}")
            
            if significant_global:
                print("Significant global relationships:")
                for analysis, metric, corr, p, n in significant_global:
                    print(f"  - {analysis}: {metric} (r = {corr:.4f}, p = {p:.4f})")
            
            # Compare global vs relationship-specific patterns
            print(f"\n• Global analysis includes {len(daily_global)} days of aggregated data")
            print(f"• Monthly analysis includes {len(monthly_global)} months with reduced requirements")
            print(f"• Quarterly analysis includes {len(quarterly_global)} quarters with reduced requirements")
        
        # Data quality assessment
        print(f"\n8. DATA QUALITY ASSESSMENT")
        print("-" * 50)
        print(f"Daily data points: {len(daily_global)}")
        print(f"Monthly data points (≥5 days): {len(monthly_global)}")
        print(f"Quarterly data points (≥10 days): {len(quarterly_global)}")
        print(f"Date range: {daily_global['Date'].min()} to {daily_global['Date'].max()}")
        print(f"Average relationships per day: {daily_global['Relationship_Pairs_Count'].mean():.1f}")
        
        return results, daily_global, monthly_global, quarterly_global

    # Run the global aggregated analysis
    global_results, daily_global, monthly_global, quarterly_global = global_aggregated_analysis(df)


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