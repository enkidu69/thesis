import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from statsmodels.tsa.stattools import acf
import warnings
warnings.filterwarnings('ignore')


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
df=concatenated_df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

def test_tone_predictive_power(df):
    """Test if Daily_AvgTone can PREDICT event occurrence"""
    
    print("="*70)
    print("PREDICTIVE POWER TEST: Can Daily_AvgTone Predict Event Occurrence?")
    print("="*70)
    
    # Clean data
    clean_data = df.dropna(subset=['Daily_AvgTone', 'event_count']).copy()
    clean_data['Date'] = pd.to_datetime(clean_data['Date'])
    clean_data = clean_data.sort_values('Date')
    
    # Create complete date range
    start_date = clean_data['Date'].min()
    end_date = clean_data['Date'].max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Aggregate daily data
    daily_aggregated = clean_data.groupby('Date').agg({
        'Daily_AvgTone': 'sum',
        'event_count': 'sum'
    }).reset_index()
    
    daily_aggregated.columns = ['Date', 'Global_Daily_AvgTone_Sum', 'Global_Event_Count_Sum']
    
    # Create complete daily dataset
    daily_global = pd.DataFrame({'Date': all_dates})
    daily_global = daily_global.merge(daily_aggregated, on='Date', how='left')
    daily_global[['Global_Daily_AvgTone_Sum', 'Global_Event_Count_Sum']] = daily_global[['Global_Daily_AvgTone_Sum', 'Global_Event_Count_Sum']].fillna(0)
    
    # Create binary event indicator
    daily_global['Event_Occurred'] = (daily_global['Global_Event_Count_Sum'] > 0).astype(int)
    
    print(f"Analysis period: {start_date} to {end_date}")
    print(f"Total days: {len(daily_global)}")
    print(f"Days with events: {daily_global['Event_Occurred'].sum()} ({daily_global['Event_Occurred'].mean()*100:.1f}%)")
    
    # 1. TEST 1: Compare tone BEFORE events vs normal days
    print(f"\n1. TONE LEADING INDICATOR TEST")
    print("-" * 50)
    
    # Create lagged features to test if past tone predicts future events
    test_data = daily_global.copy().sort_values('Date')
    
    # Create multiple lagged tone features
    for lag in [1, 2, 3, 5, 7]:  # Test different lead times
        test_data[f'Tone_Lag_{lag}'] = test_data['Global_Daily_AvgTone_Sum'].shift(lag)
        test_data[f'Event_Lead_{lag}'] = test_data['Event_Occurred'].shift(-lag)
    
    test_data_clean = test_data.dropna()
    
    # Statistical test for each lag
    print("Testing if tone PRECEDES events (t-tests):")
    print(f"{'Lag':<10} | {'Tone Before Events':<20} | {'Tone Normal Days':<20} | {'P-value':<10} | {'Significant':<12}")
    print("-" * 90)
    
    for lag in [1, 2, 3, 5, 7]:
        tone_before_events = test_data_clean[test_data_clean[f'Event_Lead_{lag}'] == 1][f'Tone_Lag_{lag}']
        tone_normal_days = test_data_clean[test_data_clean[f'Event_Lead_{lag}'] == 0][f'Tone_Lag_{lag}']
        
        if len(tone_before_events) > 1 and len(tone_normal_days) > 1:
            t_stat, p_value = stats.ttest_ind(tone_before_events, tone_normal_days, equal_var=False)
            significant = "YES" if p_value < 0.05 else "NO"
            
            print(f"{lag:>2} days   | {tone_before_events.mean():>18.2f} | {tone_normal_days.mean():>18.2f} | {p_value:>9.4f} | {significant:>12}")
    
    # 2. TEST 2: Predictive Modeling with Time-Series Cross-Validation
    print(f"\n2. PREDICTIVE MODELING TEST")
    print("-" * 50)
    
    # Prepare features for prediction
    predictive_data = daily_global.copy().sort_values('Date')
    
    # Feature engineering focused on prediction
    predictive_data['Tone_MA_3'] = predictive_data['Global_Daily_AvgTone_Sum'].rolling(3).mean()
    predictive_data['Tone_MA_7'] = predictive_data['Global_Daily_AvgTone_Sum'].rolling(7).mean()
    predictive_data['Tone_Std_7'] = predictive_data['Global_Daily_AvgTone_Sum'].rolling(7).std()
    
    # Lagged tone features
    for lag in [1, 2, 3]:
        predictive_data[f'Tone_Lag_{lag}'] = predictive_data['Global_Daily_AvgTone_Sum'].shift(lag)
    
    # Event history
    predictive_data['Event_Lag_1'] = predictive_data['Event_Occurred'].shift(1)
    predictive_data['Event_Lag_3'] = predictive_data['Event_Occurred'].shift(3)
    
    # Target: Will an event occur in next X days?
    # Fix the boolean operation - convert to int first
    predictive_data['Event_Next_1D'] = predictive_data['Event_Occurred'].shift(-1).fillna(0).astype(int)
    
    # For 3-day prediction: event in next 1, 2, or 3 days
    event_next_1 = predictive_data['Event_Occurred'].shift(-1).fillna(0).astype(int)
    event_next_2 = predictive_data['Event_Occurred'].shift(-2).fillna(0).astype(int)
    event_next_3 = predictive_data['Event_Occurred'].shift(-3).fillna(0).astype(int)
    predictive_data['Event_Next_3D'] = ((event_next_1 + event_next_2 + event_next_3) > 0).astype(int)
    
    predictive_data_clean = predictive_data.dropna()
    
    # Features for prediction
    feature_cols = ['Global_Daily_AvgTone_Sum', 'Tone_MA_3', 'Tone_MA_7', 'Tone_Std_7',
                   'Tone_Lag_1', 'Tone_Lag_2', 'Tone_Lag_3', 'Event_Lag_1', 'Event_Lag_3']
    
    results_summary = {}
    
    for target_col, horizon in [('Event_Next_1D', '1-day'), ('Event_Next_3D', '3-day')]:
        print(f"\n--- Predicting events in next {horizon} ---")
        
        X = predictive_data_clean[feature_cols]
        y = predictive_data_clean[target_col]
        
        print(f"Event rate in target: {y.mean():.3f} ({y.sum()}/{len(y)} events)")
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        models = {
            'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        }
        
        for model_name, model in models.items():
            cv_scores = []
            cv_auc_scores = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Skip if no events in test set
                if y_test.sum() == 0:
                    continue
                
                # Scale features for logistic regression
                if model_name == 'Logistic Regression':
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    model.fit(X_train_scaled, y_train)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    model.fit(X_train, y_train)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Use optimal threshold based on precision-recall
                precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
                if len(thresholds) > 0:
                    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
                    optimal_threshold = thresholds[np.argmax(f1_scores)]
                    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
                else:
                    y_pred = (y_pred_proba >= 0.5).astype(int)
                
                # Calculate metrics
                try:
                    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                    f1 = report['1']['f1-score'] if '1' in report else 0.0
                except:
                    f1 = 0.0
                
                try:
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                except:
                    auc_score = 0.5
                
                cv_scores.append(f1)
                cv_auc_scores.append(auc_score)
            
            if cv_scores:  # Only if we have valid folds
                avg_f1 = np.mean(cv_scores)
                avg_auc = np.mean(cv_auc_scores)
                
                print(f"{model_name:20} | Avg F1: {avg_f1:.4f} | Avg AUC: {avg_auc:.4f}")
                
                results_summary[f'{model_name}_{horizon}'] = {
                    'f1': avg_f1,
                    'auc': avg_auc
                }
            else:
                print(f"{model_name:20} | No valid folds with events in test set")
    
    # 3. TEST 3: Baseline Comparison
    print(f"\n3. BASELINE COMPARISON TEST")
    print("-" * 50)
    
    # What if we just predict the majority class?
    if 'Event_Next_1D' in predictive_data_clean.columns:
        baseline_accuracy = 1 - predictive_data_clean['Event_Next_1D'].mean()
        baseline_f1 = 0.0  # F1 is 0 for always predicting no events
        
        print(f"Baseline (always predict no events):")
        print(f"Accuracy: {baseline_accuracy:.4f}")
        print(f"F1-Score: {baseline_f1:.4f}")
        
        # Compare with our models
        if results_summary:
            best_model_f1 = max([result['f1'] for result in results_summary.values()])
            improvement = best_model_f1 - baseline_f1
            
            print(f"\nBest model F1: {best_model_f1:.4f}")
            print(f"Improvement over baseline: {improvement:.4f}")
            print(f"Relative improvement: {improvement/max(baseline_f1, 0.001)*100:.1f}%")
        else:
            print("No valid model results to compare")
    else:
        print("No target column available for baseline comparison")
    
    # 4. TEST 4: Practical Significance Test
    print(f"\n4. PRACTICAL SIGNIFICANCE TEST")
    print("-" * 50)
    
    if 'Event_Next_1D' in predictive_data_clean.columns and len(feature_cols) > 0:
        X_full = predictive_data_clean[feature_cols]
        y_full = predictive_data_clean['Event_Next_1D']
        
        if y_full.sum() > 0:  # Only if we have events to predict
            # Use Random Forest for final analysis
            rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
            rf_model.fit(X_full, y_full)
            y_pred_proba = rf_model.predict_proba(X_full)[:, 1]
            
            # Test different thresholds
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
            print("Performance at different prediction thresholds:")
            print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
            print("-" * 50)
            
            for threshold in thresholds:
                y_pred = (y_pred_proba >= threshold).astype(int)
                report = classification_report(y_full, y_pred, output_dict=True, zero_division=0)
                try:
                    precision = report['1']['precision'] if '1' in report else 0.0
                    recall = report['1']['recall'] if '1' in report else 0.0
                    f1 = report['1']['f1-score'] if '1' in report else 0.0
                except:
                    precision = recall = f1 = 0.0
                
                print(f"{threshold:>9.1f} | {precision:>9.3f} | {recall:>9.3f} | {f1:>9.3f}")
        else:
            print("No events in target variable for threshold analysis")
    else:
        print("Insufficient data for practical significance test")
    
    # 5. FINAL VERDICT
    print(f"\n5. FINAL VERDICT: Can Daily_AvgTone Predict Events?")
    print("-" * 50)
    
    if results_summary:
        # Criteria for "useful" prediction
        useful_f1_threshold = 0.3
        useful_auc_threshold = 0.7
        
        best_overall_f1 = max([result['f1'] for result in results_summary.values()])
        best_overall_auc = max([result['auc'] for result in results_summary.values()])
        
        f1_useful = best_overall_f1 >= useful_f1_threshold
        auc_useful = best_overall_auc >= useful_auc_threshold
        
        print(f"Best F1-Score: {best_overall_f1:.4f} {'(USEFUL)' if f1_useful else '(NOT USEFUL)'}")
        print(f"Best AUC: {best_overall_auc:.4f} {'(USEFUL)' if auc_useful else '(NOT USEFUL)'}")
        
        if f1_useful or auc_useful:
            print("\nCONCLUSION: Daily_AvgTone CAN predict event occurrence")
            print("Recommended use: Early warning system with appropriate thresholds")
        else:
            print("\nCONCLUSION: Daily_AvgTone CANNOT reliably predict event occurrence")
            print("It correlates with event intensity but cannot forecast when events will happen")
        
        final_verdict = 'CAN predict' if (f1_useful or auc_useful) else 'CANNOT predict'
    else:
        print("No valid results - insufficient data for conclusion")
        final_verdict = 'INCONCLUSIVE'
        best_overall_f1 = 0.0
        best_overall_auc = 0.0
    
    return {
        'predictive_data': predictive_data_clean,
        'results_summary': results_summary,
        'final_verdict': final_verdict,
        'best_f1': best_overall_f1,
        'best_auc': best_overall_auc
    }

# Run the predictive power test
results = test_tone_predictive_power(df)

# Create visualization of the test results
def create_predictive_power_visualization(results):
    """Visualize whether tone can predict events"""
    
    predictive_data = results['predictive_data']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Tone distribution before events vs normal days
    if 'Event_Next_1D' in predictive_data.columns:
        tone_before_events_1d = predictive_data[predictive_data['Event_Next_1D'] == 1]['Global_Daily_AvgTone_Sum']
        tone_normal_days = predictive_data[predictive_data['Event_Next_1D'] == 0]['Global_Daily_AvgTone_Sum']
        
        if len(tone_before_events_1d) > 0 and len(tone_normal_days) > 0:
            axes[0, 0].hist(tone_normal_days, bins=50, alpha=0.7, label='No Events Next Day', color='blue', density=True)
            axes[0, 0].hist(tone_before_events_1d, bins=50, alpha=0.7, label='Events Next Day', color='red', density=True)
            axes[0, 0].set_xlabel('Daily Tone Sum')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].set_title('Tone Distribution: Events vs No Events Next Day')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Model performance comparison
    model_results = results['results_summary']
    if model_results:
        models_horizons = list(model_results.keys())
        f1_scores = [model_results[key]['f1'] for key in models_horizons]
        auc_scores = [model_results[key]['auc'] for key in models_horizons]
        
        x = np.arange(len(models_horizons))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, f1_scores, width, label='F1-Score', alpha=0.7)
        axes[0, 1].bar(x + width/2, auc_scores, width, label='AUC', alpha=0.7)
        axes[0, 1].set_xlabel('Model & Horizon')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Predictive Model Performance')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models_horizons, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add useful thresholds
        axes[0, 1].axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Useful F1 Threshold')
        axes[0, 1].axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Useful AUC Threshold')
    
    # Plot 3: Final verdict
    verdict = results['final_verdict']
    best_f1 = results['best_f1']
    
    axes[1, 0].text(0.5, 0.6, f'VERDICT: {verdict}', 
                   ha='center', va='center', fontsize=16, 
                   color='red' if 'CANNOT' in verdict else 'green',
                   weight='bold')
    axes[1, 0].text(0.5, 0.4, f'Best F1-Score: {best_f1:.3f}', 
                   ha='center', va='center', fontsize=12)
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title('Predictive Power Conclusion')
    axes[1, 0].axis('off')
    
    # Plot 4: Event pattern over time
    if 'Date' in predictive_data.columns and 'Event_Occurred' in predictive_data.columns:
        sample_data = predictive_data.tail(100)  # Last 100 days for clarity
        axes[1, 1].plot(sample_data['Date'], sample_data['Global_Daily_AvgTone_Sum'], 
                       'blue', alpha=0.7, label='Daily Tone')
        # Mark event days
        event_dates = sample_data[sample_data['Event_Occurred'] == 1]['Date']
        event_tones = sample_data[sample_data['Event_Occurred'] == 1]['Global_Daily_AvgTone_Sum']
        axes[1, 1].scatter(event_dates, event_tones, color='red', s=50, 
                          alpha=0.8, label='Event Days', zorder=5)
        
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Tone Value')
        axes[1, 1].set_title('Recent Tone Pattern and Events')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# Create visualization
create_predictive_power_visualization(results)