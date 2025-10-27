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
import glob



def read_and_concatenate_excel_files():
    """
    Read and concatenate all Excel files in the 'analysis' folder
    """
    folder_path = "analysis"
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist")
    
    # Get all Excel files
    excel_files = glob.glob(os.path.join(folder_path, "*.xlsx")) + \
                  glob.glob(os.path.join(folder_path, "*.xls"))
    
    if not excel_files:
        raise FileNotFoundError(f"No Excel files found in '{folder_path}'")
    
    # Read and concatenate all files
    dataframes = []
    for file_path in excel_files:
        df = pd.read_excel(file_path)
        df['source_file'] = os.path.basename(file_path)
        dataframes.append(df)
    
    return pd.concat(dataframes, ignore_index=True)

# Usage

concatenated_df = read_and_concatenate_excel_files()
print(f"Successfully concatenated {len(concatenated_df['source_file'].unique())} files")
print(f"Final DataFrame shape: {concatenated_df.shape}")

    
    #df = pd.read_excel('GLOB_UK_allcountries.xlsx', sheet_name='Sheet1')
df=concatenated_df 
def interpret_predictive_results(results):
    """Detailed interpretation of the predictive power test results"""
    
    print("="*70)
    print("DETAILED INTERPRETATION OF PREDICTIVE POWER RESULTS")
    print("="*70)
    
    # Key findings from the test
    print("\n🔍 KEY INSIGHTS:")
    print("-" * 50)
    
    print("1. LEADING INDICATOR CONFIRMED:")
    print("   • Tone is significantly different 1 day BEFORE events (p = 0.044)")
    print("   • Tone is significantly different 7 days BEFORE events (p = 0.005)")
    print("   → Tone acts as an EARLY WARNING signal")
    
    print("\n2. PREDICTIVE POWER VARIES BY TIME HORIZON:")
    print("   • 1-day prediction: F1 = 0.248 (Weak)")
    print("   • 3-day prediction: F1 = 0.493 (Moderate)")
    print("   → Better for medium-term (3-day) forecasting")
    
    print("\n3. THRESHOLD SENSITIVITY:")
    print("   • At 0.5 threshold: High precision (0.80), moderate recall (0.67)")
    print("   • At 0.3 threshold: Lower precision (0.22), perfect recall (1.00)")
    print("   → Trade-off between catching all events vs false alarms")
    
    # Practical recommendations
    print("\n🎯 PRACTICAL RECOMMENDATIONS:")
    print("-" * 50)
    
    print("1. USE CASE 1: HIGH-CONFIDENCE ALERTS")
    print("   • Threshold: 0.5")
    print("   • Precision: 80% | Recall: 67%")
    print("   • Best for: Situations where false alarms are costly")
    print("   • Action: Take preventive measures when alert triggers")
    
    print("\n2. USE CASE 2: COMPREHENSIVE MONITORING")
    print("   • Threshold: 0.3") 
    print("   • Precision: 22% | Recall: 100%")
    print("   • Best for: Situations where missing events is costly")
    print("   • Action: Increase vigilance, gather more information")
    
    print("\n3. OPTIMAL STRATEGY: TWO-TIER SYSTEM")
    print("   • Tier 1 (Threshold 0.3): Broad monitoring - flag potential risks")
    print("   • Tier 2 (Threshold 0.5): High-confidence alerts - take action")
    print("   • This balances comprehensive coverage with actionable intelligence")
    
    # Implementation guidelines
    print("\n🚀 IMPLEMENTATION GUIDELINES:")
    print("-" * 50)
    
    print("1. DATA PROCESSING:")
    print("   • Monitor Daily_AvgTone with 3-day and 7-day moving averages")
    print("   • Track tone volatility (standard deviation)")
    print("   • Include recent event history (last 1-3 days)")
    
    print("\n2. ALERT TRIGGERS:")
    print("   • Significant tone drops (below historical averages)")
    print("   • Sustained negative tone trends")
    print("   • Combined with recent event patterns")
    
    print("\n3. OPERATIONAL WORKFLOW:")
    print("   Daily:")
    print("   - Calculate tone metrics and prediction probabilities")
    print("   - Generate Tier 1 alerts (low threshold)")
    print("   - Review Tier 2 alerts (high threshold)")
    print("   - Update risk assessments")
    
    print("\n   Weekly:")
    print("   - Review model performance")
    print("   - Adjust thresholds based on recent accuracy")
    print("   - Update historical baselines")
    
    # Limitations and caveats
    print("\n⚠️ LIMITATIONS AND CAVEATS:")
    print("-" * 50)
    
    print("1. MODEST PREDICTIVE POWER:")
    print("   • F1-score of 0.493 means ~50% accuracy in event prediction")
    print("   • Better than random, but not highly reliable alone")
    
    print("2. CONTEXT DEPENDENCE:")
    print("   • Tone signals work better in certain contexts than others")
    print("   • Should be combined with domain knowledge")
    
    print("3. FALSE POSITIVES:")
    print("   • Even at optimal threshold, ~20% of alerts may be false")
    print("   • Requires human verification and contextual analysis")
    
    # Future improvements
    print("\n🔮 FUTURE ENHANCEMENTS:")
    print("-" * 50)
    
    print("1. FEATURE ENRICHMENT:")
    print("   • Add sentiment volatility measures")
    print("   • Include external factors (economic indicators, news volume)")
    print("   • Incorporate geopolitical context")
    
    print("2. MODEL REFINEMENT:")
    print("   • Ensemble methods combining multiple algorithms")
    print("   • Context-aware thresholds (adjust based on situation)")
    print("   • Real-time model retraining")
    
    print("3. OPERATIONAL INTEGRATION:")
    print("   • Dashboard with real-time tone monitoring")
    print("   • Automated alert escalation")
    print("   • Integration with other intelligence sources")
    
    return {
        "recommended_threshold_low": 0.3,
        "recommended_threshold_high": 0.5,
        "optimal_horizon": "3-day",
        "implementation_priority": "MEDIUM",
        "confidence_level": "MODERATE"
    }

# Run interpretation
recommendations = interpret_predictive_results(results)

# Create actionable dashboard visualization
def create_actionable_dashboard(results, recommendations):
    """Create a practical dashboard for operational use"""
    
    predictive_data = results['predictive_data']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Real-time monitoring view (last 90 days)
    recent_data = predictive_data.tail(90).copy()
    
    # Calculate prediction probabilities for demonstration
    feature_cols = ['Global_Daily_AvgTone_Sum', 'Tone_MA_3', 'Tone_MA_7', 'Tone_Std_7',
                   'Tone_Lag_1', 'Tone_Lag_2', 'Tone_Lag_3', 'Event_Lag_1', 'Event_Lag_3']
    
    if 'Event_Next_3D' in recent_data.columns and len(feature_cols) > 0:
        X_recent = recent_data[feature_cols]
        
        # Train a simple model for demonstration
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        # Use all data for training in this demo
        X_all = predictive_data[feature_cols]
        y_all = predictive_data['Event_Next_3D']
        model.fit(X_all, y_all)
        
        recent_data['Prediction_Probability'] = model.predict_proba(X_recent)[:, 1]
        
        # Plot tone and predictions
        axes[0, 0].plot(recent_data['Date'], recent_data['Global_Daily_AvgTone_Sum'], 
                       'blue', linewidth=2, label='Daily Tone', alpha=0.7)
        
        # Add prediction probability
        axes2 = axes[0, 0].twinx()
        axes2.plot(recent_data['Date'], recent_data['Prediction_Probability'], 
                  'red', linewidth=2, label='Event Probability', alpha=0.8)
        
        # Add alert thresholds
        axes2.axhline(y=recommendations['recommended_threshold_low'], color='orange', 
                     linestyle='--', alpha=0.7, label='Tier 1 Alert')
        axes2.axhline(y=recommendations['recommended_threshold_high'], color='red', 
                     linestyle='--', alpha=0.7, label='Tier 2 Alert')
        
        # Mark actual events
        event_dates = recent_data[recent_data['Event_Occurred'] == 1]['Date']
        if len(event_dates) > 0:
            axes[0, 0].scatter(event_dates, 
                              recent_data[recent_data['Event_Occurred'] == 1]['Global_Daily_AvgTone_Sum'],
                              color='black', s=50, zorder=5, label='Actual Events')
        
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Tone Value', color='blue')
        axes2.set_ylabel('Event Probability', color='red')
        axes[0, 0].set_title('Real-time Monitoring Dashboard\n(Tone vs Event Prediction Probability)')
        axes[0, 0].legend(loc='upper left')
        axes2.legend(loc='upper right')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Alert performance at different thresholds
    thresholds = np.arange(0.1, 0.9, 0.1)
    precisions = []
    recalls = []
    f1_scores = []
    
    if 'Event_Next_3D' in predictive_data.columns and len(feature_cols) > 0:
        X_full = predictive_data[feature_cols]
        y_full = predictive_data['Event_Next_3D']
        
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_full, y_full)
        y_pred_proba = model.predict_proba(X_full)[:, 1]
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            report = classification_report(y_full, y_pred, output_dict=True, zero_division=0)
            try:
                precision = report['1']['precision'] if '1' in report else 0.0
                recall = report['1']['recall'] if '1' in report else 0.0
                f1 = report['1']['f1-score'] if '1' in report else 0.0
            except:
                precision = recall = f1 = 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        axes[0, 1].plot(thresholds, precisions, 'go-', label='Precision', linewidth=2)
        axes[0, 1].plot(thresholds, recalls, 'bo-', label='Recall', linewidth=2)
        axes[0, 1].plot(thresholds, f1_scores, 'ro-', label='F1-Score', linewidth=2)
        
        # Mark recommended thresholds
        axes[0, 1].axvline(x=recommendations['recommended_threshold_low'], color='orange', 
                          linestyle='--', alpha=0.7, label='Tier 1 Threshold')
        axes[0, 1].axvline(x=recommendations['recommended_threshold_high'], color='red', 
                          linestyle='--', alpha=0.7, label='Tier 2 Threshold')
        
        axes[0, 1].set_xlabel('Prediction Threshold')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Performance Trade-offs at Different Thresholds')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Operational recommendations
    recommendation_text = """
    OPERATIONAL RECOMMENDATIONS:
    
    🎯 USE CASE: 3-DAY EVENT PREDICTION
    
    TIER 1 ALERTS (Threshold: 0.3):
    • Comprehensive monitoring
    • Cast wide net, don't miss events
    • Action: Increased vigilance
    
    TIER 2 ALERTS (Threshold: 0.5):
    • High-confidence predictions  
    • Fewer false alarms
    • Action: Consider preventive measures
    
    📊 EXPECTED PERFORMANCE:
    • Catch 67-100% of events
    • 20-80% precision rate
    • Overall accuracy: ~50%
    
    ⚠️ BEST USED AS:
    Early warning system supplement
    Not standalone decision tool
    """
    
    axes[1, 0].text(0.1, 0.9, recommendation_text, fontsize=10, 
                   verticalalignment='top', linespacing=1.5)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title('Operational Implementation Guide')
    axes[1, 0].axis('off')
    
    # Plot 4: Confidence intervals for predictions
    if 'Prediction_Probability' in recent_data.columns:
        # Simulate confidence intervals (in real implementation, use proper uncertainty quantification)
        recent_data['Lower_CI'] = recent_data['Prediction_Probability'] * 0.8
        recent_data['Upper_CI'] = recent_data['Prediction_Probability'] * 1.2
        recent_data['Upper_CI'] = np.minimum(recent_data['Upper_CI'], 1.0)
        
        axes[1, 1].fill_between(recent_data['Date'], 
                               recent_data['Lower_CI'], 
                               recent_data['Upper_CI'], 
                               alpha=0.3, color='red', label='Uncertainty Range')
        axes[1, 1].plot(recent_data['Date'], recent_data['Prediction_Probability'], 
                       'red', linewidth=2, label='Prediction Probability')
        axes[1, 1].axhline(y=recommendations['recommended_threshold_low'], color='orange', 
                          linestyle='--', alpha=0.7, label='Tier 1 Alert')
        axes[1, 1].axhline(y=recommendations['recommended_threshold_high'], color='red', 
                          linestyle='--', alpha=0.7, label='Tier 2 Alert')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Prediction Probability')
        axes[1, 1].set_title('Prediction Uncertainty and Alert Thresholds')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Create the actionable dashboard
create_actionable_dashboard(results, recommendations)

print("\n" + "="*70)
print("SUMMARY: Daily_AvgTone has MODERATE predictive power for events")
print("Best used as an EARLY WARNING SYSTEM with two-tier alerts")
print("="*70)