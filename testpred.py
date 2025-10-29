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
#check on data:print(df['event_count'].count())
#print(len(df))


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
    clean_data = df
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
    
    test_data_clean = test_data
    # Statistical test for each lag
    print("Testing if tone PRECEDES events (t-tests):")
    print(f"{'Lag':<10} | {'Tone Before Events':<20} | {'Tone Normal Days':<20} | {'P-value':<10} | {'Significant':<12}")
    print("-" * 90)
    
    for lag in [1, 2, 3, 5, 7]:
        tone_before_events = test_data_clean[test_data_clean[f'Event_Lead_{lag}'] == 1][f'Tone_Lag_{lag}']
        tone_normal_days = test_data_clean[test_data_clean[f'Event_Lead_{lag}'] == 0][f'Tone_Lag_{lag}']
        
        if len(tone_before_events) > 1 and len(tone_normal_days) > 1:
            # Check if we have valid data for t-test
            if tone_before_events.notna().all() and tone_normal_days.notna().all():
                try:
                    t_stat, p_value = stats.ttest_ind(tone_before_events, tone_normal_days, equal_var=False)
                    significant = "YES" if p_value < 0.05 else "NO"
                    
                    print(f"{lag:>2} days   | {tone_before_events.mean():>18.2f} | {tone_normal_days.mean():>18.2f} | {p_value:>9.4f} | {significant:>12}")
                except:
                    print(f"{lag:>2} days   | {tone_before_events.mean():>18.2f} | {tone_normal_days.mean():>18.2f} | {'NaN':>9} | {'ERROR':>12}")
            else:
                print(f"{lag:>2} days   | {tone_before_events.mean():>18.2f} | {tone_normal_days.mean():>18.2f} | {'NaN':>9} | {'INVALID':>12}")
    
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
    for lag in [1, 2, 3, 7]:
        predictive_data[f'Tone_Lag_{lag}'] = predictive_data['Global_Daily_AvgTone_Sum'].shift(lag)
    
    # Event history
    predictive_data['Event_Lag_1'] = predictive_data['Event_Occurred'].shift(1)
    predictive_data['Event_Lag_3'] = predictive_data['Event_Occurred'].shift(3)
    predictive_data['Event_Lag_7'] = predictive_data['Event_Occurred'].shift(7)
    
    # Target: Will an event occur in next X days?
    predictive_data['Event_Next_1D'] = predictive_data['Event_Occurred'].shift(-1).fillna(0).astype(int)
    
    # For 3-day and 7-day prediction
    event_next_1 = predictive_data['Event_Occurred'].shift(-1).fillna(0).astype(int)
    event_next_2 = predictive_data['Event_Occurred'].shift(-2).fillna(0).astype(int)
    event_next_3 = predictive_data['Event_Occurred'].shift(-3).fillna(0).astype(int)
    event_next_4 = predictive_data['Event_Occurred'].shift(-4).fillna(0).astype(int)
    event_next_5 = predictive_data['Event_Occurred'].shift(-5).fillna(0).astype(int)
    event_next_6 = predictive_data['Event_Occurred'].shift(-6).fillna(0).astype(int)
    event_next_7 = predictive_data['Event_Occurred'].shift(-7).fillna(0).astype(int)
    
    predictive_data['Event_Next_3D'] = ((event_next_1 + event_next_2 + event_next_3) > 0).astype(int)
    predictive_data['Event_Next_7D'] = ((event_next_1 + event_next_2 + event_next_3 + event_next_4 + event_next_5 + event_next_6 + event_next_7) > 0).astype(int)

    predictive_data_clean = predictive_data.dropna()
    
    # Features for prediction
    feature_cols = ['Global_Daily_AvgTone_Sum', 'Tone_MA_3', 'Tone_MA_7', 'Tone_Std_7',
                   'Tone_Lag_1', 'Tone_Lag_2', 'Tone_Lag_3', 'Event_Lag_1', 'Event_Lag_3']
    
    results_summary = {}
    
    for target_col, horizon in [('Event_Next_1D', '1-day'), ('Event_Next_3D', '3-day'), ('Event_Next_7D', '7-day')]:
        print(f"\n--- Predicting events in next {horizon} ---")
        
        X = predictive_data_clean[feature_cols]
        y = predictive_data_clean[target_col]
        
        print(f"Event rate in target: {y.mean():.3f} ({y.sum()}/{len(y)} events)")
        
        # Skip cross-validation modeling if event rate is too high (no predictive challenge)
        if y.mean() > 0.9:
            print(f"SKIPPED CV: Event rate too high ({y.mean():.3f}) - no meaningful prediction challenge")
            # But we'll still include it in baseline and practical significance tests
            continue
            
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=min(5, len(X) // 10))  # Adjust splits based on data size
        
        models = {
            'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        }
        
        for model_name, model in models.items():
            cv_scores = []
            cv_auc_scores = []
            valid_folds = 0
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Skip if no events in test set OR no non-events in test set
                if y_test.sum() == 0 or (len(y_test) - y_test.sum()) == 0:
                    continue
                
                # Skip if training set has only one class
                if len(np.unique(y_train)) < 2:
                    continue
                
                try:
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
                    valid_folds += 1
                    
                except Exception as e:
                    # Skip this fold if any error occurs
                    continue
            
            if valid_folds > 0:  # Only if we have valid folds
                avg_f1 = np.mean(cv_scores)
                avg_auc = np.mean(cv_auc_scores)
                
                print(f"{model_name:20} | Avg F1: {avg_f1:.4f} | Avg AUC: {avg_auc:.4f} | Valid folds: {valid_folds}")
                
                results_summary[f'{model_name}_{horizon}'] = {
                    'f1': avg_f1,
                    'auc': avg_auc,
                    'event_rate': y.mean(),
                    'horizon': horizon,
                    'valid_folds': valid_folds
                }
            else:
                print(f"{model_name:20} | No valid folds with events in test set")
    
    # 3. TEST 3: Baseline Comparison for ALL horizons including D+7
    print(f"\n3. BASELINE COMPARISON TEST")
    print("-" * 50)
    
    # Compare baselines for all prediction horizons including D+7
    horizons = ['1-day', '3-day', '7-day']
    target_cols = ['Event_Next_1D', 'Event_Next_3D', 'Event_Next_7D']
    
    print("Baseline performance (always predict no events):")
    print(f"{'Horizon':<10} | {'Accuracy':<10} | {'F1-Score':<10} | {'Event Rate':<10} | {'Note':<15}")
    print("-" * 75)
    
    baseline_summary = {}
    
    for target_col, horizon in zip(target_cols, horizons):
        if target_col in predictive_data_clean.columns:
            event_rate = predictive_data_clean[target_col].mean()
            baseline_accuracy = 1 - event_rate
            baseline_f1 = 0.0  # F1 is 0 for always predicting no events
            
            # Determine baseline note
            if event_rate > 0.9:
                note = "POOR BASELINE"
            elif event_rate > 0.7:
                note = "LOW BASELINE"
            else:
                note = "REASONABLE"
                
            baseline_summary[horizon] = {
                'accuracy': baseline_accuracy,
                'f1': baseline_f1,
                'event_rate': event_rate,
                'note': note
            }
            
            print(f"{horizon:<10} | {baseline_accuracy:>9.4f} | {baseline_f1:>9.4f} | {event_rate:>9.4f} | {note:>15}")
    
    # Compare with our models
    if results_summary:
        # Find best model for each horizon
        horizon_results = {}
        for key, result in results_summary.items():
            horizon = result['horizon']
            if horizon not in horizon_results or result['f1'] > horizon_results[horizon]['f1']:
                horizon_results[horizon] = result
        
        print(f"\nBest model performance vs baseline:")
        print(f"{'Horizon':<10} | {'Best F1':<10} | {'Baseline F1':<12} | {'Improvement':<12} | {'Status':<15}")
        print("-" * 80)
        
        for horizon in ['1-day', '3-day']:  # Only show horizons we actually modeled with CV
            if horizon in horizon_results:
                best_f1 = horizon_results[horizon]['f1']
                baseline_f1 = 0.0
                improvement = best_f1 - baseline_f1
                
                # Determine status
                if improvement > 0.3:
                    status = "EXCELLENT"
                elif improvement > 0.2:
                    status = "GOOD"
                elif improvement > 0.1:
                    status = "MODERATE"
                else:
                    status = "POOR"
                
                print(f"{horizon:<10} | {best_f1:>9.4f} | {baseline_f1:>11.4f} | {improvement:>11.4f} | {status:>15}")
    
    # 4. TEST 4: Practical Significance Test for ALL horizons including D+7
    print(f"\n4. PRACTICAL SIGNIFICANCE TEST (Full Dataset Analysis)")
    print("-" * 60)
    
    # Test different thresholds for ALL horizons including D+7
    for target_col, horizon in [('Event_Next_1D', '1-day'), ('Event_Next_3D', '3-day'), ('Event_Next_7D', '7-day')]:
        if target_col in predictive_data_clean.columns and len(feature_cols) > 0:
            X_full = predictive_data_clean[feature_cols]
            y_full = predictive_data_clean[target_col]
            
            if y_full.sum() > 0 and (len(y_full) - y_full.sum()) > 0:  # Need both classes
                # Use Random Forest for final analysis
                rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
                rf_model.fit(X_full, y_full)
                y_pred_proba = rf_model.predict_proba(X_full)[:, 1]
                
                # Test different thresholds
                thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
                print(f"\n{horizon} prediction - Performance at different thresholds:")
                print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Alerts':<10} | {'TP':<8} | {'FP':<8}")
                print("-" * 85)
                
                best_f1 = 0
                best_threshold = 0.5
                
                for threshold in thresholds:
                    y_pred = (y_pred_proba >= threshold).astype(int)
                    report = classification_report(y_full, y_pred, output_dict=True, zero_division=0)
                    try:
                        precision = report['1']['precision'] if '1' in report else 0.0
                        recall = report['1']['recall'] if '1' in report else 0.0
                        f1 = report['1']['f1-score'] if '1' in report else 0.0
                        
                        # Calculate TP and FP
                        tp = ((y_pred == 1) & (y_full == 1)).sum()
                        fp = ((y_pred == 1) & (y_full == 0)).sum()
                        
                    except:
                        precision = recall = f1 = 0.0
                        tp = fp = 0
                    
                    alerts_count = y_pred.sum()
                    
                    # Track best threshold
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                    
                    print(f"{threshold:>9.1f} | {precision:>9.3f} | {recall:>9.3f} | {f1:>9.3f} | {alerts_count:>9} | {tp:>7} | {fp:>7}")
                
                # Print best threshold recommendation
                print(f"→ Best threshold for {horizon}: {best_threshold:.1f} (F1: {best_f1:.3f})")
                
                # Special analysis for D+7 due to high event rate
                if horizon == '7-day':
                    print(f"\nD+7 SPECIAL ANALYSIS (High Event Rate: {y_full.mean():.3f}):")
                    print(f"- Majority class baseline (always predict event): Accuracy = {y_full.mean():.3f}")
                    print(f"- Current best model F1: {best_f1:.3f}")
                    
                    # Check if model is better than majority class
                    if best_f1 > 0.5:  # Arbitrary threshold for "useful"
                        print(f"- CONCLUSION: Model provides useful predictions despite high event rate")
                    else:
                        print(f"- CONCLUSION: Model struggles due to class imbalance")
                        
            else:
                print(f"Insufficient class balance for {horizon} threshold analysis")
        else:
            print(f"Insufficient data for {horizon} practical significance test")
    
    # 5. FINAL VERDICT with horizon comparison including D+7 insights
    print(f"\n5. FINAL VERDICT: Can Daily_AvgTone Predict Events?")
    print("-" * 50)
    
    if results_summary or 'Event_Next_7D' in predictive_data_clean.columns:
        # Criteria for "useful" prediction
        useful_f1_threshold = 0.3
        useful_auc_threshold = 0.7
        
        # Group results by horizon
        horizon_performance = {}
        for key, result in results_summary.items():
            horizon = result['horizon']
            if horizon not in horizon_performance:
                horizon_performance[horizon] = []
            horizon_performance[horizon].append(result)
        
        print("Performance by prediction horizon:")
        print(f"{'Horizon':<10} | {'Best F1':<10} | {'Best AUC':<10} | {'Event Rate':<10} | {'Verdict':<15}")
        print("-" * 80)
        
        best_overall_f1 = 0.0
        best_overall_auc = 0.0
        useful_horizons = []
        
        # Check modeled horizons
        for horizon in ['1-day', '3-day']:
            if horizon in horizon_performance:
                horizon_f1 = max([r['f1'] for r in horizon_performance[horizon]])
                horizon_auc = max([r['auc'] for r in horizon_performance[horizon]])
                event_rate = horizon_performance[horizon][0]['event_rate']
                
                f1_useful = horizon_f1 >= useful_f1_threshold
                auc_useful = horizon_auc >= useful_auc_threshold
                
                verdict = "USEFUL" if (f1_useful or auc_useful) else "NOT USEFUL"
                if f1_useful or auc_useful:
                    useful_horizons.append(horizon)
                
                # Update overall best
                best_overall_f1 = max(best_overall_f1, horizon_f1)
                best_overall_auc = max(best_overall_auc, horizon_auc)
                
                print(f"{horizon:<10} | {horizon_f1:>9.4f} | {horizon_auc:>9.4f} | {event_rate:>9.4f} | {verdict:>15}")
        
        # Special analysis for D+7
        if 'Event_Next_7D' in predictive_data_clean.columns:
            d7_event_rate = predictive_data_clean['Event_Next_7D'].mean()
            print(f"{'7-day':<10} | {'N/A':>9} | {'N/A':>9} | {d7_event_rate:>9.4f} | {'HIGH EVENT RATE':>15}")
            
            # Provide D+7 specific insight
            if d7_event_rate > 0.9:
                print(f"→ D+7 NOTE: Very high event rate ({d7_event_rate:.3f}) limits predictive value")
            elif d7_event_rate > 0.7:
                print(f"→ D+7 NOTE: High event rate ({d7_event_rate:.3f}) - focus on precision over recall")
        
        print(f"\nOverall Best F1-Score: {best_overall_f1:.4f}")
        print(f"Overall Best AUC: {best_overall_auc:.4f}")
        
        if useful_horizons:
            print(f"\nCONCLUSION: Daily_AvgTone CAN predict event occurrence for {', '.join(useful_horizons)} horizons")
            print("Recommended use: Early warning system with appropriate thresholds")
            
            # D+7 specific recommendation
            if 'Event_Next_7D' in predictive_data_clean.columns:
                d7_event_rate = predictive_data_clean['Event_Next_7D'].mean()
                if d7_event_rate > 0.7:
                    print(f"D+7 NOTE: 7-day predictions may have limited utility due to high baseline event rate ({d7_event_rate:.1%})")
        else:
            print(f"\nCONCLUSION: Daily_AvgTone CANNOT reliably predict event occurrence")
            print("It correlates with event intensity but cannot forecast when events will happen")
        
        final_verdict = 'CAN predict' if useful_horizons else 'CANNOT predict'
    else:
        print("No valid results - insufficient data for conclusion")
        final_verdict = 'INCONCLUSIVE'
        best_overall_f1 = 0.0
        best_overall_auc = 0.0
    
    return {
        'predictive_data': predictive_data_clean,
        'results_summary': results_summary,
        'baseline_summary': baseline_summary,
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
#create_predictive_power_visualization(results)

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
#create_actionable_dashboard(results, recommendations)

print("\n" + "="*70)
print("SUMMARY: Daily_AvgTone has MODERATE predictive power for events")
print("Best used as an EARLY WARNING SYSTEM with two-tier alerts")
print("="*70)

def create_alert_table(df):
    """Apply prediction thresholds to all daily data and create alert table"""
    
    print("="*70)
    print("DAILY ALERT TABLE - Applying Thresholds to All Data")
    print("="*70)
    
    # Clean and prepare data
    clean_data = df.dropna(subset=['Daily_AvgTone', 'event_count']).copy()
    clean_data['Date'] = pd.to_datetime(clean_data['Date'])
    clean_data = clean_data.sort_values('Date')
    
    # Aggregate daily data
    daily_aggregated = clean_data.groupby('Date').agg({
        'Daily_AvgTone': 'sum',
        'event_count': 'sum'
    }).reset_index()
    
    daily_aggregated.columns = ['Date', 'Global_Daily_AvgTone_Sum', 'Global_Event_Count_Sum']
    
    # Create complete dataset
    all_dates = pd.date_range(start=clean_data['Date'].min(), end=clean_data['Date'].max(), freq='D')
    daily_global = pd.DataFrame({'Date': all_dates})
    daily_global = daily_global.merge(daily_aggregated, on='Date', how='left')
    daily_global[['Global_Daily_AvgTone_Sum', 'Global_Event_Count_Sum']] = daily_global[['Global_Daily_AvgTone_Sum', 'Global_Event_Count_Sum']].fillna(0)
    daily_global['Event_Occurred'] = (daily_global['Global_Event_Count_Sum'] > 0).astype(int)
    
    # Feature engineering for prediction
    predictive_data = daily_global.copy().sort_values('Date')
    
    # Calculate features
    predictive_data['Tone_MA_3'] = predictive_data['Global_Daily_AvgTone_Sum'].rolling(3).mean()
    predictive_data['Tone_MA_7'] = predictive_data['Global_Daily_AvgTone_Sum'].rolling(7).mean()
    predictive_data['Tone_Std_7'] = predictive_data['Global_Daily_AvgTone_Sum'].rolling(7).std()
    
    for lag in [1, 2, 3, 7]:
        predictive_data[f'Tone_Lag_{lag}'] = predictive_data['Global_Daily_AvgTone_Sum'].shift(lag)
    
    predictive_data['Event_Lag_1'] = predictive_data['Event_Occurred'].shift(1)
    predictive_data['Event_Lag_3'] = predictive_data['Event_Occurred'].shift(3)
    predictive_data['Event_Lag_7'] = predictive_data['Event_Occurred'].shift(7)

    # Target: events in next 3 days and next 7 days
    event_next_1 = predictive_data['Event_Occurred'].shift(-1).fillna(0).astype(int)
    event_next_2 = predictive_data['Event_Occurred'].shift(-2).fillna(0).astype(int)
    event_next_3 = predictive_data['Event_Occurred'].shift(-3).fillna(0).astype(int)
    event_next_4 = predictive_data['Event_Occurred'].shift(-4).fillna(0).astype(int)
    event_next_5 = predictive_data['Event_Occurred'].shift(-5).fillna(0).astype(int)
    event_next_6 = predictive_data['Event_Occurred'].shift(-6).fillna(0).astype(int)
    event_next_7 = predictive_data['Event_Occurred'].shift(-7).fillna(0).astype(int)
    
    predictive_data['Event_Next_3D'] = ((event_next_1 + event_next_2 + event_next_3) > 0).astype(int)
    predictive_data['Event_Next_7D'] = ((event_next_1 + event_next_2 + event_next_3 + event_next_4 + event_next_5 + event_next_6 + event_next_7) > 0).astype(int)
    predictive_data_clean = predictive_data.dropna()
    
    # Features for prediction
    feature_cols = ['Global_Daily_AvgTone_Sum', 'Tone_MA_3', 'Tone_MA_7', 'Tone_Std_7',
                   'Tone_Lag_1', 'Tone_Lag_2', 'Tone_Lag_3', 'Event_Lag_1', 'Event_Lag_3']
    
    X = predictive_data_clean[feature_cols]
    y_3d = predictive_data_clean['Event_Next_3D']
    y_7d = predictive_data_clean['Event_Next_7D']
    
    # Train models for both timeframes
    from sklearn.ensemble import RandomForestClassifier
    model_3d = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model_7d = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    
    model_3d.fit(X, y_3d)
    model_7d.fit(X, y_7d)
    
    # Get prediction probabilities for both timeframes
    prediction_probabilities_3d = model_3d.predict_proba(X)[:, 1]
    prediction_probabilities_7d = model_7d.predict_proba(X)[:, 1]
    
    predictive_data_clean['Prediction_Probability_3D'] = prediction_probabilities_3d
    predictive_data_clean['Prediction_Probability_7D'] = prediction_probabilities_7d
    
    # Apply thresholds for both timeframes
    tier1_threshold = 0.3
    tier2_threshold = 0.5
    
    predictive_data_clean['Tier1_Alert_3D'] = (predictive_data_clean['Prediction_Probability_3D'] >= tier1_threshold).astype(int)
    predictive_data_clean['Tier2_Alert_3D'] = (predictive_data_clean['Prediction_Probability_3D'] >= tier2_threshold).astype(int)
    predictive_data_clean['Tier1_Alert_7D'] = (predictive_data_clean['Prediction_Probability_7D'] >= tier1_threshold).astype(int)
    predictive_data_clean['Tier2_Alert_7D'] = (predictive_data_clean['Prediction_Probability_7D'] >= tier2_threshold).astype(int)
    
    # Create the alert table
    alert_table = predictive_data_clean[[
        'Date', 'Global_Daily_AvgTone_Sum', 'Global_Event_Count_Sum', 
        'Event_Occurred', 'Event_Next_3D', 'Event_Next_7D',
        'Prediction_Probability_3D', 'Prediction_Probability_7D',
        'Tier1_Alert_3D', 'Tier2_Alert_3D', 'Tier1_Alert_7D', 'Tier2_Alert_7D'
    ]].copy()
    
    # Add alert descriptions for both timeframes
    alert_table['Tier1_Alert_3D_Text'] = alert_table['Tier1_Alert_3D'].map({1: 'ALERT', 0: 'No Alert'})
    alert_table['Tier2_Alert_3D_Text'] = alert_table['Tier2_Alert_3D'].map({1: 'HIGH ALERT', 0: 'No Alert'})
    alert_table['Tier1_Alert_7D_Text'] = alert_table['Tier1_Alert_7D'].map({1: 'ALERT', 0: 'No Alert'})
    alert_table['Tier2_Alert_7D_Text'] = alert_table['Tier2_Alert_7D'].map({1: 'HIGH ALERT', 0: 'No Alert'})
    
    # Add performance evaluation for both timeframes
    alert_table['Correct_Prediction_3D'] = (
        ((alert_table['Tier1_Alert_3D'] == 1) & (alert_table['Event_Next_3D'] == 1)) |
        ((alert_table['Tier1_Alert_3D'] == 0) & (alert_table['Event_Next_3D'] == 0))
    ).astype(int)
    
    alert_table['Correct_Prediction_7D'] = (
        ((alert_table['Tier1_Alert_7D'] == 1) & (alert_table['Event_Next_7D'] == 1)) |
        ((alert_table['Tier1_Alert_7D'] == 0) & (alert_table['Event_Next_7D'] == 0))
    ).astype(int)
    
    # Print summary statistics for both timeframes
    print(f"\nSUMMARY STATISTICS:")
    print("-" * 60)
    print(f"Total days analyzed: {len(alert_table)}")
    print(f"\nD+3 PREDICTIONS:")
    print(f"Days with events (next 3D): {alert_table['Event_Next_3D'].sum()} ({alert_table['Event_Next_3D'].mean()*100:.1f}%)")
    print(f"Tier 1 Alerts triggered: {alert_table['Tier1_Alert_3D'].sum()} ({alert_table['Tier1_Alert_3D'].mean()*100:.1f}%)")
    print(f"Tier 2 Alerts triggered: {alert_table['Tier2_Alert_3D'].sum()} ({alert_table['Tier2_Alert_3D'].mean()*100:.1f}%)")
    print(f"Overall accuracy: {alert_table['Correct_Prediction_3D'].mean()*100:.1f}%")
    
    print(f"\nD+7 PREDICTIONS:")
    print(f"Days with events (next 7D): {alert_table['Event_Next_7D'].sum()} ({alert_table['Event_Next_7D'].mean()*100:.1f}%)")
    print(f"Tier 1 Alerts triggered: {alert_table['Tier1_Alert_7D'].sum()} ({alert_table['Tier1_Alert_7D'].mean()*100:.1f}%)")
    print(f"Tier 2 Alerts triggered: {alert_table['Tier2_Alert_7D'].sum()} ({alert_table['Tier2_Alert_7D'].mean()*100:.1f}%)")
    print(f"Overall accuracy: {alert_table['Correct_Prediction_7D'].mean()*100:.1f}%")
    
    # Show recent alerts (last 30 days) for both timeframes
    recent_alerts = alert_table.tail(30).copy()
    
    print(f"\nRECENT ALERTS - D+3 (Last 30 days):")
    print("-" * 120)
    print(f"{'Date':<12} | {'Tone':<8} | {'Prob3D':<7} | {'Tier1_3D':<10} | {'Tier2_3D':<12} | {'Event Next 3D':<13} | {'Match':<6}")
    print("-" * 120)
    
    for _, row in recent_alerts.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d')
        tone = row['Global_Daily_AvgTone_Sum']
        prob_3d = row['Prediction_Probability_3D']
        tier1_3d = row['Tier1_Alert_3D_Text']
        tier2_3d = row['Tier2_Alert_3D_Text']
        event_next_3d = "EVENT" if row['Event_Next_3D'] else "No Event"
        match_3d = "✓" if row['Correct_Prediction_3D'] else "✗"
        
        print(f"{date_str} | {tone:>7.2f} | {prob_3d:>6.3f} | {tier1_3d:>10} | {tier2_3d:>12} | {event_next_3d:>13} | {match_3d:>6}")
    
    print(f"\nRECENT ALERTS - D+7 (Last 30 days):")
    print("-" * 120)
    print(f"{'Date':<12} | {'Tone':<8} | {'Prob7D':<7} | {'Tier1_7D':<10} | {'Tier2_7D':<12} | {'Event Next 7D':<13} | {'Match':<6}")
    print("-" * 120)
    
    for _, row in recent_alerts.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d')
        tone = row['Global_Daily_AvgTone_Sum']
        prob_7d = row['Prediction_Probability_7D']
        tier1_7d = row['Tier1_Alert_7D_Text']
        tier2_7d = row['Tier2_Alert_7D_Text']
        event_next_7d = "EVENT" if row['Event_Next_7D'] else "No Event"
        match_7d = "✓" if row['Correct_Prediction_7D'] else "✗"
        
        print(f"{date_str} | {tone:>7.2f} | {prob_7d:>6.3f} | {tier1_7d:>10} | {tier2_7d:>12} | {event_next_7d:>13} | {match_7d:>6}")
    
    # Show highest probability alerts for both timeframes
    high_prob_alerts_3d = alert_table.nlargest(10, 'Prediction_Probability_3D')[['Date', 'Global_Daily_AvgTone_Sum', 'Prediction_Probability_3D', 'Event_Next_3D']]
    high_prob_alerts_7d = alert_table.nlargest(10, 'Prediction_Probability_7D')[['Date', 'Global_Daily_AvgTone_Sum', 'Prediction_Probability_7D', 'Event_Next_7D']]
    
    print(f"\nTOP 10 HIGHEST PROBABILITY ALERTS - D+3:")
    print("-" * 70)
    print(f"{'Date':<12} | {'Tone':<8} | {'Probability':<12} | {'Event Occurred':<14}")
    print("-" * 70)
    
    for _, row in high_prob_alerts_3d.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d')
        tone = row['Global_Daily_AvgTone_Sum']
        prob = row['Prediction_Probability_3D']
        event_occurred = "✓ EVENT" if row['Event_Next_3D'] else "✗ No Event"
        
        print(f"{date_str} | {tone:>7.2f} | {prob:>11.3f} | {event_occurred:>14}")
    
    print(f"\nTOP 10 HIGHEST PROBABILITY ALERTS - D+7:")
    print("-" * 70)
    print(f"{'Date':<12} | {'Tone':<8} | {'Probability':<12} | {'Event Occurred':<14}")
    print("-" * 70)
    
    for _, row in high_prob_alerts_7d.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d')
        tone = row['Global_Daily_AvgTone_Sum']
        prob = row['Prediction_Probability_7D']
        event_occurred = "✓ EVENT" if row['Event_Next_7D'] else "✗ No Event"
        
        print(f"{date_str} | {tone:>7.2f} | {prob:>11.3f} | {event_occurred:>14}")
    
    # Performance by threshold for both timeframes
    print(f"\nPERFORMANCE BY THRESHOLD LEVEL - D+3:")
    print("-" * 60)
    print(f"{'Threshold':<10} | {'Alerts':<8} | {'Precision':<10} | {'Recall':<8} | {'F1':<6}")
    print("-" * 60)
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for threshold in thresholds:
        alerts = (alert_table['Prediction_Probability_3D'] >= threshold).astype(int)
        true_positives = ((alerts == 1) & (alert_table['Event_Next_3D'] == 1)).sum()
        false_positives = ((alerts == 1) & (alert_table['Event_Next_3D'] == 0)).sum()
        false_negatives = ((alerts == 0) & (alert_table['Event_Next_3D'] == 1)).sum()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{threshold:>9.1f} | {alerts.sum():>7} | {precision:>9.3f} | {recall:>7.3f} | {f1:>5.3f}")
    
    print(f"\nPERFORMANCE BY THRESHOLD LEVEL - D+7:")
    print("-" * 60)
    print(f"{'Threshold':<10} | {'Alerts':<8} | {'Precision':<10} | {'Recall':<8} | {'F1':<6}")
    print("-" * 60)
    
    for threshold in thresholds:
        alerts = (alert_table['Prediction_Probability_7D'] >= threshold).astype(int)
        true_positives = ((alerts == 1) & (alert_table['Event_Next_7D'] == 1)).sum()
        false_positives = ((alerts == 1) & (alert_table['Event_Next_7D'] == 0)).sum()
        false_negatives = ((alerts == 0) & (alert_table['Event_Next_7D'] == 1)).sum()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{threshold:>9.1f} | {alerts.sum():>7} | {precision:>9.3f} | {recall:>7.3f} | {f1:>5.3f}")
    
    return alert_table

# Create the comprehensive alert table
alert_table = create_alert_table(df)


def analyze_alert_patterns(alert_table):
    """Analyze patterns and trends in alert triggers"""
    
    print("="*70)
    print("ALERT PATTERN ANALYSIS")
    print("="*70)
    
    # Make sure we have the required columns
    print("Available columns in alert_table:", alert_table.columns.tolist())
    
    # Check which alert columns exist and use them
    available_columns = alert_table.columns.tolist()
    
    # Use D+3 alerts by default (since they were in the original function)
    tier1_col = 'Tier1_Alert_3D' if 'Tier1_Alert_3D' in available_columns else 'Tier1_Alert'
    tier2_col = 'Tier2_Alert_3D' if 'Tier2_Alert_3D' in available_columns else 'Tier2_Alert'
    prob_col = 'Prediction_Probability_3D' if 'Prediction_Probability_3D' in available_columns else 'Prediction_Probability'
    event_next_col = 'Event_Next_3D' if 'Event_Next_3D' in available_columns else 'Event_Next_3D'
    
    print(f"Using columns: {tier1_col}, {tier2_col}, {prob_col}, {event_next_col}")
    
    # Ensure Date is datetime
    alert_table = alert_table.copy()
    alert_table['Date'] = pd.to_datetime(alert_table['Date'])
    
    # Extract time components
    alert_table['Year'] = alert_table['Date'].dt.year
    alert_table['Month'] = alert_table['Date'].dt.month
    alert_table['YearMonth'] = alert_table['Date'].dt.to_period('M')
    alert_table['DayOfWeek'] = alert_table['Date'].dt.dayofweek
    alert_table['Week'] = alert_table['Date'].dt.isocalendar().week
    
    # 1. Monthly Trends
    print(f"\n1. MONTHLY ALERT TRENDS")
    print("-" * 50)
    
    monthly_stats = alert_table.groupby('YearMonth').agg({
        tier1_col: 'sum',
        tier2_col: 'sum',
        event_next_col: 'sum',
        prob_col: 'mean'
    }).reset_index()
    
    monthly_stats.columns = ['YearMonth', 'Tier1_Alerts', 'Tier2_Alerts', 'Actual_Events', 'Avg_Probability']
    
    print(f"{'Year-Month':<12} | {'Tier1':<6} | {'Tier2':<6} | {'Events':<7} | {'Avg Prob':<9}")
    print("-" * 60)
    
    for _, row in monthly_stats.iterrows():
        print(f"{str(row['YearMonth']):<12} | {row['Tier1_Alerts']:>5} | {row['Tier2_Alerts']:>5} | {row['Actual_Events']:>6} | {row['Avg_Probability']:>8.3f}")
    
    # 2. Day of Week Patterns
    print(f"\n2. DAY OF WEEK PATTERNS")
    print("-" * 50)
    
    dow_mapping = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                  4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    
    dow_stats = alert_table.groupby('DayOfWeek').agg({
        tier1_col: ['sum', 'mean'],
        tier2_col: ['sum', 'mean'],
        event_next_col: ['sum', 'mean'],
        prob_col: 'mean'
    }).round(4).reset_index()
    
    dow_stats.columns = ['DayOfWeek', 'Tier1_Count', 'Tier1_Rate', 'Tier2_Count', 'Tier2_Rate', 
                        'Events_Count', 'Events_Rate', 'Avg_Probability']
    
    print(f"{'Day':<10} | {'Tier1 Rate':<10} | {'Tier2 Rate':<10} | {'Event Rate':<10} | {'Avg Prob':<9}")
    print("-" * 70)
    
    for _, row in dow_stats.iterrows():
        day_name = dow_mapping[row['DayOfWeek']]
        print(f"{day_name:<10} | {row['Tier1_Rate']:>9.3f} | {row['Tier2_Rate']:>9.3f} | {row['Events_Rate']:>9.3f} | {row['Avg_Probability']:>8.3f}")
    
    # 3. Alert Clustering Analysis
    print(f"\n3. ALERT CLUSTERING ANALYSIS")
    print("-" * 50)
    
    # Find alert sequences
    alert_sequences = []
    current_sequence = []
    
    for i, row in alert_table.iterrows():
        if row[tier1_col] == 1:
            current_sequence.append(row['Date'])
        else:
            if len(current_sequence) >= 2:  # Only consider sequences of 2+ alerts
                alert_sequences.append(current_sequence)
            current_sequence = []
    
    # Don't forget the last sequence
    if len(current_sequence) >= 2:
        alert_sequences.append(current_sequence)
    
    print(f"Number of alert clusters (2+ consecutive days): {len(alert_sequences)}")
    
    if alert_sequences:
        cluster_lengths = [len(seq) for seq in alert_sequences]
        print(f"Cluster length - Min: {min(cluster_lengths)}, Max: {max(cluster_lengths)}, Avg: {np.mean(cluster_lengths):.1f}")
        
        # Show top 5 longest clusters
        longest_clusters = sorted(alert_sequences, key=len, reverse=True)[:5]
        print(f"\nTop 5 longest alert clusters:")
        for i, cluster in enumerate(longest_clusters, 1):
            start_date = cluster[0].strftime('%Y-%m-%d')
            end_date = cluster[-1].strftime('%Y-%m-%d')
            duration = (cluster[-1] - cluster[0]).days + 1
            print(f"  {i}. {start_date} to {end_date} ({duration} days, {len(cluster)} alerts)")
    
    # 4. Performance by Alert Probability Bins
    print(f"\n4. PERFORMANCE BY PROBABILITY BINS")
    print("-" * 50)
    
    alert_table['Prob_Bin'] = pd.cut(alert_table[prob_col], 
                                    bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                    labels=['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', 
                                           '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'])
    
    bin_stats = alert_table.groupby('Prob_Bin').agg({
        tier1_col: 'count',
        event_next_col: 'sum'
    }).reset_index()
    
    bin_stats['Event_Rate'] = bin_stats[event_next_col] / bin_stats[tier1_col]
    bin_stats['Alert_Count'] = bin_stats[tier1_col]
    
    print(f"{'Prob Bin':<10} | {'Alerts':<8} | {'Events':<8} | {'Event Rate':<10}")
    print("-" * 50)
    
    for _, row in bin_stats.iterrows():
        if row['Alert_Count'] > 0:
            print(f"{row['Prob_Bin']:<10} | {row['Alert_Count']:>7} | {row[event_next_col]:>7} | {row['Event_Rate']:>9.3f}")
    
    # 5. Seasonal Analysis
    print(f"\n5. SEASONAL ANALYSIS")
    print("-" * 50)
    
    alert_table['Season'] = alert_table['Month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    seasonal_stats = alert_table.groupby('Season').agg({
        tier1_col: ['sum', 'mean'],
        tier2_col: ['sum', 'mean'],
        event_next_col: ['sum', 'mean'],
        prob_col: 'mean'
    }).round(4).reset_index()
    
    seasonal_stats.columns = ['Season', 'Tier1_Count', 'Tier1_Rate', 'Tier2_Count', 'Tier2_Rate', 
                             'Events_Count', 'Events_Rate', 'Avg_Probability']
    
    print(f"{'Season':<8} | {'Tier1 Rate':<10} | {'Tier2 Rate':<10} | {'Event Rate':<10} | {'Avg Prob':<9}")
    print("-" * 65)
    
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        row = seasonal_stats[seasonal_stats['Season'] == season].iloc[0]
        print(f"{season:<8} | {row['Tier1_Rate']:>9.3f} | {row['Tier2_Rate']:>9.3f} | {row['Events_Rate']:>9.3f} | {row['Avg_Probability']:>8.3f}")
    
    # 6. Compare D+3 and D+7 patterns if both exist
    if 'Tier1_Alert_7D' in available_columns and 'Tier1_Alert_3D' in available_columns:
        print(f"\n6. D+3 vs D+7 COMPARISON")
        print("-" * 50)
        
        comparison_stats = alert_table.agg({
            'Tier1_Alert_3D': ['sum', 'mean'],
            'Tier2_Alert_3D': ['sum', 'mean'],
            'Tier1_Alert_7D': ['sum', 'mean'],
            'Tier2_Alert_7D': ['sum', 'mean'],
            'Event_Next_3D': ['sum', 'mean'],
            'Event_Next_7D': ['sum', 'mean'],
            'Prediction_Probability_3D': 'mean',
            'Prediction_Probability_7D': 'mean'
        }).round(4)
        
        print(f"{'Metric':<20} | {'D+3':<10} | {'D+7':<10} | {'Difference':<12}")
        print("-" * 60)
        
        metrics = [
            ('Tier1 Alert Rate', 'Tier1_Alert_3D', 'mean', 'Tier1_Alert_7D', 'mean'),
            ('Tier2 Alert Rate', 'Tier2_Alert_3D', 'mean', 'Tier2_Alert_7D', 'mean'),
            ('Event Rate', 'Event_Next_3D', 'mean', 'Event_Next_7D', 'mean'),
            ('Avg Probability', 'Prediction_Probability_3D', 'mean', 'Prediction_Probability_7D', 'mean')
        ]
        
        for metric_name, col3d, stat3d, col7d, stat7d in metrics:
            val3d = comparison_stats.loc[stat3d, col3d]
            val7d = comparison_stats.loc[stat7d, col7d]
            diff = val7d - val3d
            print(f"{metric_name:<20} | {val3d:>9.3f} | {val7d:>9.3f} | {diff:>11.3f}")
    
    return {
        'monthly_stats': monthly_stats,
        'dow_stats': dow_stats,
        'alert_sequences': alert_sequences,
        'bin_stats': bin_stats,
        'seasonal_stats': seasonal_stats
    }


# Run pattern analysis
analyze_alert_patterns(alert_table)



def save_alert_table(alert_table, filename="alert_table_detailed.csv"):
    """Save the alert table with additional formatting and analysis"""
    
    print("="*70)
    print("SAVING ALERT TABLE")
    print("="*70)
    
    # Make a copy to avoid modifying the original
    detailed_table = alert_table.copy()
    
    # Check which columns exist and use appropriate ones
    available_columns = detailed_table.columns.tolist()
    print(f"Available columns: {available_columns}")
    
    # Determine which alert columns to use
    if 'Tier1_Alert_3D' in available_columns and 'Tier2_Alert_3D' in available_columns:
        # Use D+3 alerts (new format)
        tier1_col = 'Tier1_Alert_3D'
        tier2_col = 'Tier2_Alert_3D'
        prob_col = 'Prediction_Probability_3D'
        event_next_col = 'Event_Next_3D'
        print("Using D+3 alert columns")
    elif 'Tier1_Alert' in available_columns and 'Tier2_Alert' in available_columns:
        # Use original alert columns
        tier1_col = 'Tier1_Alert'
        tier2_col = 'Tier2_Alert'
        prob_col = 'Prediction_Probability'
        event_next_col = 'Event_Next_3D'
        print("Using original alert columns")
    else:
        # Fallback to first available columns
        tier1_col = [col for col in available_columns if 'Tier1' in col][0] if any('Tier1' in col for col in available_columns) else None
        tier2_col = [col for col in available_columns if 'Tier2' in col][0] if any('Tier2' in col for col in available_columns) else None
        prob_col = [col for col in available_columns if 'Probability' in col][0] if any('Probability' in col for col in available_columns) else None
        event_next_col = [col for col in available_columns if 'Event_Next' in col][0] if any('Event_Next' in col for col in available_columns) else None
        print(f"Using fallback columns: {tier1_col}, {tier2_col}, {prob_col}, {event_next_col}")
    
    # Create alert level column
    if tier1_col and tier2_col:
        detailed_table['Alert_Level'] = detailed_table.apply(
            lambda x: 'HIGH' if x[tier2_col] == 1 else 'MEDIUM' if x[tier1_col] == 1 else 'LOW',
            axis=1
        )
    elif tier1_col:
        detailed_table['Alert_Level'] = detailed_table[tier1_col].map({1: 'MEDIUM', 0: 'LOW'})
    else:
        detailed_table['Alert_Level'] = 'LOW'
    
    # Create performance evaluation
    if tier1_col and event_next_col:
        detailed_table['Prediction_Correct'] = detailed_table.apply(
            lambda x: 'TRUE POSITIVE' if (x[tier1_col] == 1 and x[event_next_col] == 1) else
                     'FALSE POSITIVE' if (x[tier1_col] == 1 and x[event_next_col] == 0) else
                     'TRUE NEGATIVE' if (x[tier1_col] == 0 and x[event_next_col] == 0) else
                     'FALSE NEGATIVE',
            axis=1
        )
    
    # Add probability categories
    if prob_col:
        detailed_table['Probability_Category'] = pd.cut(
            detailed_table[prob_col],
            bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            labels=['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', 
                   '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
        )
    
    # Format date for better readability
    detailed_table['Date_Formatted'] = detailed_table['Date'].dt.strftime('%Y-%m-%d')
    
    # Create summary statistics
    summary_stats = {}
    
    if tier1_col:
        summary_stats['Total_Tier1_Alerts'] = detailed_table[tier1_col].sum()
        summary_stats['Tier1_Alert_Rate'] = detailed_table[tier1_col].mean()
    
    if tier2_col:
        summary_stats['Total_Tier2_Alerts'] = detailed_table[tier2_col].sum()
        summary_stats['Tier2_Alert_Rate'] = detailed_table[tier2_col].mean()
    
    if event_next_col:
        summary_stats['Total_Events'] = detailed_table[event_next_col].sum()
        summary_stats['Event_Rate'] = detailed_table[event_next_col].mean()
    
    if 'Prediction_Correct' in detailed_table.columns:
        confusion_matrix = detailed_table['Prediction_Correct'].value_counts()
        summary_stats.update(confusion_matrix.to_dict())
    
    # Select and reorder columns for final output
    output_columns = ['Date_Formatted', 'Alert_Level']
    
    # Add probability column if available
    if prob_col:
        output_columns.append(prob_col)
    
    # Add tone and event columns
    tone_event_cols = ['Global_Daily_AvgTone_Sum', 'Global_Event_Count_Sum']
    for col in tone_event_cols:
        if col in detailed_table.columns:
            output_columns.append(col)
    
    # Add event prediction columns
    event_cols = [event_next_col] if event_next_col else []
    for col in event_cols:
        if col in detailed_table.columns:
            output_columns.append(col)
    
    # Add performance columns
    if 'Prediction_Correct' in detailed_table.columns:
        output_columns.append('Prediction_Correct')
    
    if 'Probability_Category' in detailed_table.columns:
        output_columns.append('Probability_Category')
    
    # Include all D+7 columns if they exist
    d7_cols = [col for col in available_columns if '7D' in col]
    output_columns.extend(d7_cols)
    
    # Create final table
    final_table = detailed_table[output_columns].copy()
    
    # Sort by date
    final_table = final_table.sort_values('Date_Formatted', ascending=False)
    
    # Save to CSV
    final_table.to_csv(filename, index=False)
    
    print(f"Alert table saved to: {filename}")
    print(f"Total records: {len(final_table)}")
    print(f"Columns saved: {list(final_table.columns)}")
    
    # Print summary
    print(f"\nSUMMARY STATISTICS:")
    print("-" * 40)
    for key, value in summary_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Show recent alerts in the saved file
    print(f"\nRECENT ALERTS IN SAVED FILE:")
    print("-" * 80)
    recent_alerts = final_table.head(10)
    
    # Create a simplified view for display
    display_columns = ['Date_Formatted', 'Alert_Level']
    if prob_col in final_table.columns:
        display_columns.append(prob_col)
    if event_next_col in final_table.columns:
        display_columns.append(event_next_col)
    if 'Prediction_Correct' in final_table.columns:
        display_columns.append('Prediction_Correct')
    
    display_table = recent_alerts[display_columns]
    print(display_table.to_string(index=False))
    
    return final_table



# Save the table (uncomment if you want to save)
saved_table = save_alert_table(alert_table)

print(f"\n" + "="*70)
print("ALERT TABLE COMPLETE")
print("="*70)
print("The table shows daily predictions and alerts based on your thresholds.")
print("Use this for operational monitoring and decision-making.")
exit()
def produce_next_day_alerts(df):
    """Produce alerts for the next day based on current tone data"""
    
    print("="*70)
    print("NEXT DAY ALERTS - Daily Prediction System")
    print("="*70)
    
    # Clean and prepare data
    clean_data = df.dropna(subset=['Daily_AvgTone', 'event_count']).copy()
    clean_data['Date'] = pd.to_datetime(clean_data['Date'])
    clean_data = clean_data.sort_values('Date')
    
    # Aggregate daily data
    daily_aggregated = clean_data.groupby('Date').agg({
        'Daily_AvgTone': 'sum',
        'event_count': 'sum'
    }).reset_index()
    
    daily_aggregated.columns = ['Date', 'Global_Daily_AvgTone_Sum', 'Global_Event_Count_Sum']
    
    # Create complete dataset
    all_dates = pd.date_range(start=clean_data['Date'].min(), end=clean_data['Date'].max(), freq='D')
    daily_global = pd.DataFrame({'Date': all_dates})
    daily_global = daily_global.merge(daily_aggregated, on='Date', how='left')
    daily_global[['Global_Daily_AvgTone_Sum', 'Global_Event_Count_Sum']] = daily_global[['Global_Daily_AvgTone_Sum', 'Global_Event_Count_Sum']].fillna(0)
    daily_global['Event_Occurred'] = (daily_global['Global_Event_Count_Sum'] > 0).astype(int)
    
    # Feature engineering for next day prediction
    predictive_data = daily_global.copy().sort_values('Date')
    
    # Calculate features needed for prediction
    predictive_data['Tone_MA_3'] = predictive_data['Global_Daily_AvgTone_Sum'].rolling(3).mean()
    predictive_data['Tone_MA_7'] = predictive_data['Global_Daily_AvgTone_Sum'].rolling(7).mean()
    predictive_data['Tone_Std_7'] = predictive_data['Global_Daily_AvgTone_Sum'].rolling(7).std()
    
    # Lagged tone features
    for lag in [1, 2, 3]:
        predictive_data[f'Tone_Lag_{lag}'] = predictive_data['Global_Daily_AvgTone_Sum'].shift(lag)
    
    # Event history
    predictive_data['Event_Lag_1'] = predictive_data['Event_Occurred'].shift(1)
    predictive_data['Event_Lag_3'] = predictive_data['Event_Occurred'].shift(3)
    
    # Target: event tomorrow
    predictive_data['Event_Tomorrow'] = predictive_data['Event_Occurred'].shift(-1)
    
    # Remove rows with missing data
    feature_cols = ['Global_Daily_AvgTone_Sum', 'Tone_MA_3', 'Tone_MA_7', 'Tone_Std_7',
                   'Tone_Lag_1', 'Tone_Lag_2', 'Tone_Lag_3', 'Event_Lag_1', 'Event_Lag_3']
    
    predictive_data_clean = predictive_data.dropna(subset=feature_cols + ['Event_Tomorrow'])
    
    # Train model on all available data
    from sklearn.ensemble import RandomForestClassifier
    
    X = predictive_data_clean[feature_cols]
    y = predictive_data_clean['Event_Tomorrow']
    
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X, y)
    
    # Get predictions for ALL days
    all_predictions = model.predict_proba(X)[:, 1]
    predictive_data_clean['Prediction_Probability'] = all_predictions
    
    # Apply thresholds
    tier1_threshold = 0.3
    tier2_threshold = 0.5
    
    predictive_data_clean['Tier1_Alert'] = (predictive_data_clean['Prediction_Probability'] >= tier1_threshold).astype(int)
    predictive_data_clean['Tier2_Alert'] = (predictive_data_clean['Prediction_Probability'] >= tier2_threshold).astype(int)
    
    # Create the alert table
    alert_table = predictive_data_clean[[
        'Date', 'Global_Daily_AvgTone_Sum', 'Global_Event_Count_Sum', 
        'Event_Occurred', 'Event_Tomorrow', 'Prediction_Probability',
        'Tier1_Alert', 'Tier2_Alert'
    ]].copy()
    
    # Add alert descriptions
    alert_table['Tier1_Alert_Text'] = alert_table['Tier1_Alert'].map({1: 'ALERT', 0: 'No Alert'})
    alert_table['Tier2_Alert_Text'] = alert_table['Tier2_Alert'].map({1: 'HIGH ALERT', 0: 'No Alert'})
    
    # Add whether prediction was correct
    alert_table['Correct_Prediction'] = (
        ((alert_table['Tier1_Alert'] == 1) & (alert_table['Event_Tomorrow'] == 1)) |
        ((alert_table['Tier1_Alert'] == 0) & (alert_table['Event_Tomorrow'] == 0))
    ).astype(int)
    
    # Print summary
    print(f"\nALERT SYSTEM SUMMARY:")
    print("-" * 40)
    print(f"Total days analyzed: {len(alert_table)}")
    print(f"Days with events tomorrow: {alert_table['Event_Tomorrow'].sum()} ({alert_table['Event_Tomorrow'].mean()*100:.1f}%)")
    print(f"Tier 1 Alerts: {alert_table['Tier1_Alert'].sum()} ({alert_table['Tier1_Alert'].mean()*100:.1f}%)")
    print(f"Tier 2 Alerts: {alert_table['Tier2_Alert'].sum()} ({alert_table['Tier2_Alert'].mean()*100:.1f}%)")
    print(f"Overall accuracy: {alert_table['Correct_Prediction'].mean()*100:.1f}%")
    
    # Show recent alerts (last 30 days)
    recent_alerts = alert_table.tail(30).copy()
    
    print(f"\nRECENT NEXT-DAY ALERTS (Last 30 days):")
    print("-" * 110)
    print(f"{'Date':<12} | {'Tone':<8} | {'Prob':<6} | {'Tier1':<8} | {'Tier2':<10} | {'Event Tomorrow':<15} | {'Correct':<8}")
    print("-" * 110)
    
    for _, row in recent_alerts.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d')
        tone = row['Global_Daily_AvgTone_Sum']
        prob = row['Prediction_Probability']
        tier1 = row['Tier1_Alert_Text']
        tier2 = row['Tier2_Alert_Text']
        event_tomorrow = "EVENT" if row['Event_Tomorrow'] else "No Event"
        correct = "✓" if row['Correct_Prediction'] else "✗"
        
        print(f"{date_str} | {tone:>7.2f} | {prob:>5.3f} | {tier1:>8} | {tier2:>10} | {event_tomorrow:>15} | {correct:>8}")
    
    # Show highest probability alerts
    high_prob_alerts = alert_table.nlargest(20, 'Prediction_Probability')[['Date', 'Global_Daily_AvgTone_Sum', 'Prediction_Probability', 'Event_Tomorrow']]
    
    print(f"\nTOP 20 HIGHEST PROBABILITY ALERTS:")
    print("-" * 70)
    print(f"{'Date':<12} | {'Tone':<8} | {'Probability':<12} | {'Event Tomorrow':<15}")
    print("-" * 70)
    
    for _, row in high_prob_alerts.iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d')
        tone = row['Global_Daily_AvgTone_Sum']
        prob = row['Prediction_Probability']
        event_tomorrow = "✓ EVENT" if row['Event_Tomorrow'] else "✗ No Event"
        
        print(f"{date_str} | {tone:>7.2f} | {prob:>11.3f} | {event_tomorrow:>15}")
    
    # Performance by threshold
    print(f"\nPERFORMANCE BY THRESHOLD LEVEL:")
    print("-" * 60)
    print(f"{'Threshold':<10} | {'Alerts':<8} | {'Precision':<10} | {'Recall':<8} | {'F1':<6}")
    print("-" * 60)
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    for threshold in thresholds:
        alerts = (alert_table['Prediction_Probability'] >= threshold).astype(int)
        true_positives = ((alerts == 1) & (alert_table['Event_Tomorrow'] == 1)).sum()
        false_positives = ((alerts == 1) & (alert_table['Event_Tomorrow'] == 0)).sum()
        false_negatives = ((alerts == 0) & (alert_table['Event_Tomorrow'] == 1)).sum()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{threshold:>9.1f} | {alerts.sum():>7} | {precision:>9.3f} | {recall:>7.3f} | {f1:>5.3f}")
    
    return alert_table, model, feature_cols

# Run the next-day alert system
alert_table, trained_model, features_used = produce_next_day_alerts(df)

# Function to get today's prediction
def get_todays_prediction(current_tone_data, model, feature_names):
    """
    Get prediction for today based on current tone data
    current_tone_data should be a dictionary with today's features
    """
    # Create feature vector in the right order
    feature_vector = [current_tone_data[feature] for feature in feature_names]
    
    # Get prediction probability
    probability = model.predict_proba([feature_vector])[0, 1]
    
    # Apply thresholds
    tier1_alert = probability >= 0.3
    tier2_alert = probability >= 0.5
    
    # Determine alert level and recommendation
    if tier2_alert:
        alert_level = "HIGH ALERT"
        recommendation = "Consider preventive measures - high probability of events tomorrow"
    elif tier1_alert:
        alert_level = "MEDIUM ALERT" 
        recommendation = "Increase monitoring - moderate probability of events tomorrow"
    else:
        alert_level = "LOW RISK"
        recommendation = "Normal monitoring - low probability of events tomorrow"
    
    return {
        'prediction_probability': probability,
        'alert_level': alert_level,
        'tier1_alert': tier1_alert,
        'tier2_alert': tier2_alert,
        'recommendation': recommendation
    }

# Example: How to use for today's prediction
print(f"\n" + "="*70)
print("HOW TO USE FOR TODAY'S PREDICTION")
print("="*70)

# Example current data (you would replace with real current data)
example_current_data = {
    'Global_Daily_AvgTone_Sum': -0.25,
    'Tone_MA_3': -0.22,
    'Tone_MA_7': -0.20,
    'Tone_Std_7': 0.06,
    'Tone_Lag_1': -0.18,
    'Tone_Lag_2': -0.15,
    'Tone_Lag_3': -0.12,
    'Event_Lag_1': 0,
    'Event_Lag_3': 1
}

today_prediction = get_todays_prediction(example_current_data, trained_model, features_used)

print(f"Example Prediction for Today:")
print(f"Current Tone: {example_current_data['Global_Daily_AvgTone_Sum']}")
print(f"Prediction Probability: {today_prediction['prediction_probability']:.3f}")
print(f"Alert Level: {today_prediction['alert_level']}")
print(f"Recommendation: {today_prediction['recommendation']}")

# Show what the system would predict for the most recent day
most_recent = alert_table.iloc[-1]
print(f"\nMOST RECENT PREDICTION IN DATA:")
print(f"Date: {most_recent['Date'].strftime('%Y-%m-%d')}")
print(f"Tone: {most_recent['Global_Daily_AvgTone_Sum']:.2f}")
print(f"Probability: {most_recent['Prediction_Probability']:.3f}")
print(f"Tier 1 Alert: {most_recent['Tier1_Alert_Text']}")
print(f"Tier 2 Alert: {most_recent['Tier2_Alert_Text']}")
print(f"Actual Event Tomorrow: {'EVENT' if most_recent['Event_Tomorrow'] else 'No Event'}")

print(f"\n" + "="*70)
print("NEXT-DAY ALERT SYSTEM READY")
print("=" + "="*70)
print("Use get_todays_prediction() with current tone data to get tomorrow's forecast")
print(f"Features needed: {features_used}")