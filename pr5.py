import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from imblearn.over_sampling import SMOTE
import warnings
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
import glob
import pickle

warnings.filterwarnings('ignore')

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def _save_pickle(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def _save_csv(df, path):
    ensure_dir(os.path.dirname(path))
    df.to_excel(path, index=False)

def calculate_confusion_matrix_metrics(y_true, y_pred):
    """Calculate TP, FP, FN, TN from true and predicted values"""
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    return TP, FP, FN, TN

def apply_temporal_smoothing(probabilities, window=3):
    """Apply moving average smoothing to probabilities"""
    return pd.Series(probabilities).rolling(window=window, center=True, min_periods=1).mean().values

def find_balanced_threshold(y_true, y_proba, balance_method="combined_max", max_fp_tp_ratio=0.6):
    """
    Find optimal threshold ensuring:
    1. Precision, Recall, F1 > 0.5
    2. TP, FP, FN, TN > 0 (No zero values in confusion matrix)
    3. Optimizes for: Precision, F1, and (TP - FP)
    """
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Remove the last values which are undefined
    precision_vals = precision_vals[:-1]
    recall_vals = recall_vals[:-1]
    
    # Pre-calculate totals
    n_positives = y_true.sum()
    n_negatives = len(y_true) - n_positives
    total_samples = len(y_true)
    
    viable_thresholds = []
    
    for i, (prec, rec, th) in enumerate(zip(precision_vals, recall_vals, thresholds)):
        # Avoid division by zero in F1 calculation
        if prec <= 1e-8 or rec <= 1e-8:
            continue
            
        # --- ESTIMATE CONFUSION MATRIX ---
        tp_est = rec * n_positives
        fp_est = tp_est * ((1.0 / prec) - 1.0)
        fn_est = n_positives - tp_est
        tn_est = n_negatives - fp_est
        
        # Calculate F1
        f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
        
        # --- STRICT CONSTRAINTS ---
        
        # 1. Base Quality: Precision & F1 > 0.5
        if prec <= 0.5 or f1 <= 0.5:
            continue
            
        # 2. Confusion Matrix Completeness: No Zeros Allowed
        # We use < 0.5 to check for effective integer 0 given floating point estimates
        if tp_est < 0.5 or fp_est < 0.5 or fn_est < 0.5 or tn_est < 0.5:
            continue
            
        # --- CALCULATE SCORES ---
        
        # Net Benefit (TP vs FP), normalized to -1 to 1 range
        tp_fp_diff_norm = (tp_est - fp_est) / total_samples
        
        # Combined Score: Sum of F1, Precision, and Net Benefit
        # This rewards high accuracy/precision while pushing for more TPs than FPs
        combined_score = f1 + prec + tp_fp_diff_norm
        
        viable_thresholds.append({
            'threshold': th,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'tp': tp_est,
            'fp': fp_est,
            'combined_score': combined_score
        })
    
    if not viable_thresholds:
        # Fallback: Use Geometric Mean if strict constraints fail
        geometric_mean_scores = np.sqrt(precision_vals * recall_vals)
        best_idx = np.argmax(geometric_mean_scores)
        if len(thresholds) > best_idx:
            return thresholds[best_idx], precision_vals[best_idx], recall_vals[best_idx], geometric_mean_scores[best_idx]
        else:
            return 0.5, 0.0, 0.0, 0.0
    
    # Convert to DataFrame
    viable_df = pd.DataFrame(viable_thresholds)
    
    # Maximize the combined score
    best_row = viable_df.loc[viable_df['combined_score'].idxmax()]
    
    return best_row['threshold'], best_row['precision'], best_row['recall'], best_row['f1']

def adaptive_threshold_optimization(y_true, y_proba, horizon):
    """
    Adaptive threshold optimization wrapper.
    """
    threshold, precision, recall, f1 = find_balanced_threshold(
        y_true, y_proba, balance_method="combined_max"
    )
    return threshold, precision, recall, f1

def train_with_smote(X_train, y_train, model):
    """Apply SMOTE for severe class imbalance"""
    if len(np.unique(y_train)) < 2:
        return model.fit(X_train, y_train)
    
    # Only apply SMOTE if positive class is very small
    pos_ratio = y_train.sum() / len(y_train)
    if pos_ratio < 0.1:  # Less than 10% positive
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, y_train.sum()-1))
            X_res, y_res = smote.fit_resample(X_train, y_train)
            return model.fit(X_res, y_res)
        except Exception:
            # Fallback if SMOTE fails
            return model.fit(X_train, y_train)
    else:
        return model.fit(X_train, y_train)

def apply_models_with_manual_threshold(trained_models, test_df, feature_cols, manual_threshold):
    """Apply all trained models with manual threshold to test data"""
    manual_predictions = test_df[['Date']].copy()
    
    for horizon in trained_models.keys():
        for model_name, model_obj in trained_models[horizon].items():
            # Get probabilities using the model
            if model_name == "Logistic Regression":
                _, scaler, lr_model = model_obj
                X_test_s = scaler.transform(test_df[feature_cols])
                probabilities = lr_model.predict_proba(X_test_s)[:, 1]
            elif model_name == "XGBoost":
                probabilities = model_obj.predict_proba(test_df[feature_cols].values)[:, 1]
            else:  # Random Forest
                probabilities = model_obj.predict_proba(test_df[feature_cols])[:, 1]
            
            # Apply temporal smoothing
            probabilities_smoothed = apply_temporal_smoothing(probabilities, window=3)
            
            # Create predictions with manual threshold
            predictions = (probabilities_smoothed >= manual_threshold).astype(int)
            
            # Add to results
            col_prefix = f"Manual_{horizon.replace('-','')}_{model_name.replace(' ', '_')}"
            manual_predictions[f'{col_prefix}_Prob'] = probabilities_smoothed
            manual_predictions[f'{col_prefix}_Alert'] = predictions
            # manual_predictions[f'{col_prefix}_Threshold'] = manual_threshold
    
    return manual_predictions

def read_and_concatenate_excel_files():
    """
    Read and concatenate all Excel files in the 'analysis' folder
    """
    folder_path = "analysis"
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist")
    
    # Get all Excel files
    excel_files = glob.glob(os.path.join(folder_path, "aggregated*.xlsx")) + \
                  glob.glob(os.path.join(folder_path, "singular*.xlsx"))
    
    if not excel_files:
        raise FileNotFoundError(f"No Excel files found in '{folder_path}'")
    
    # Read and concatenate all files
    dataframes = []
    for file_path in excel_files:
        df = pd.read_excel(file_path)
        df['source_file'] = os.path.basename(file_path)
        dataframes.append(df)
        name=str(os.path.basename(file_path))
        print(name)
    return pd.concat(dataframes, ignore_index=True)

# ==============================================================================
# DATA LOADING AND PREPROCESSING
# ==============================================================================

desk = os.getcwd()
path = os.path.join(desk, 'analysis')

# Robustly load attack data
attack_files = glob.glob(os.path.join(path, 'cyberevents*.xlsx'))
if not attack_files:
    raise FileNotFoundError("No attack files matching 'cyberevents*.xlsx' were found.")
attacks_list = [pd.read_excel(f) for f in attack_files]
attacks = pd.concat(attacks_list, ignore_index=True)

# Robustly determine the run name from a related file
name_files = glob.glob(os.path.join(path, '*cyber_events*.xlsx'))
if name_files:
    # Use the first matching file to derive the name
    base_name = os.path.basename(name_files[0])
    # Assuming format is 'PREFIX_cyber_events.xlsx', extract PREFIX
    name = base_name.replace('_cyber_events.xlsx', '')
else:
    # Fallback if no naming file is found
    name = "run_results"
    print("Warning: No '*_cyber_events.xlsx' file found to derive a run name.")

print(f"Determinated run name as: '{name}'")

concatenated_df = read_and_concatenate_excel_files()

print(f"Successfully concatenated {len(concatenated_df['source_file'].unique())} files")
print(f"Final DataFrame shape: {concatenated_df.shape}")
df = concatenated_df

# ==============================================================================
# SCENARIO CONFIGURATION
# ==============================================================================

scenarios = {
    "GoldsteinScale": "df['GoldsteinScale']",
    "NumArticles": "df['NumArticles']",
    "AvgTone": "df['AvgTone']",
    "AvgTone_X_NumArticles": "df['AvgTone']*df['NumArticles']",
    "AvgTone_X_NumArticles_X_GoldsteinScale": "df['AvgTone']*df['NumArticles']*df['GoldsteinScale']",
    "AvgTone_RollingMean": "df['AvgTone'].rolling(window, min_periods=1).mean()",
    "GoldsteinScale_RollingMean": "df['GoldsteinScale'].rolling(window, min_periods=1).mean()",
    "AvgTone_RollingMedian": "df['AvgTone'].rolling(window, min_periods=1).median()",
    "GoldsteinScale_RollingMedian": "df['GoldsteinScale'].rolling(window, min_periods=1).median()",
    "AvgTone_X_NumArticles_RollingMean": "(df['AvgTone']*df['NumArticles']).rolling(window, min_periods=1).mean()",
    "AvgTone_X_NumArticles_RollingMedian": "(df['AvgTone']*df['NumArticles']).rolling(window, min_periods=1).median()",
    "AvgTone_X_NumArticles_X_GoldsteinScale_RollingMean": "(df['AvgTone']*df['NumArticles']*df['GoldsteinScale']).rolling(window, min_periods=1).mean()",
    "AvgTone_X_NumArticles_X_GoldsteinScale_RollingMedian": "(df['AvgTone']*df['NumArticles']*df['GoldsteinScale']).rolling(window, min_periods=1).median()",
    
    # NEW Z-SCORE SCENARIOS
    "GoldsteinScale_Zscore": "((df['GoldsteinScale'] - df['GoldsteinScale'].rolling(window, min_periods=1).mean()) / df['GoldsteinScale'].rolling(window, min_periods=1).std())",
    "AvgTone_Zscore": "((df['AvgTone'] - df['AvgTone'].rolling(window, min_periods=1).mean()) / df['AvgTone'].rolling(window, min_periods=1).std())",
    "AvgTone_X_NumArticles_Zscore": "((df['AvgTone']*df['NumArticles']) - (df['AvgTone']*df['NumArticles']).rolling(window, min_periods=1).mean()) / (df['AvgTone']*df['NumArticles']).rolling(window, min_periods=1).std()",
    "AvgTone_X_NumArticles_X_GoldsteinScale_Zscore": "((df['AvgTone']*df['NumArticles']*df['GoldsteinScale']) - (df['AvgTone']*df['NumArticles']*df['GoldsteinScale']).rolling(window, min_periods=1).mean()) / (df['AvgTone']*df['NumArticles']*df['GoldsteinScale']).rolling(window, min_periods=1).std()"
}

aggregations = ["mean", "median", "sum"]
all_results = []
run_data_cache = {}

# ==============================================================================
# BASE FEATURE ENGINEERING
# ==============================================================================

# 1. Process attacks data once
attacks["event_date"] = pd.to_datetime(attacks["event_date"])
attacks = attacks.sort_values("event_date").reset_index(drop=True)
attacks_daily = attacks.groupby("event_date", as_index=False).agg({"event_count": "sum"}).rename(
    columns={"event_date": "Date", "event_count": "Global_Event_Count_Sum"}
)

# 2. Process base df data once
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

# 3. Create a complete date range DataFrame
start_date = min(df["Date"].min(), attacks_daily["Date"].min())
end_date = max(df["Date"].max(), attacks_daily["Date"].max())
all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
base_daily_df = pd.DataFrame({"Date": all_dates})

# 4. Merge attack data and create event-based features
base_daily_df = base_daily_df.merge(attacks_daily, on="Date", how="outer")
base_daily_df["Global_Event_Count_Sum"] = base_daily_df["Global_Event_Count_Sum"].fillna(0)
base_daily_df["Event_Occurred"] = (base_daily_df["Global_Event_Count_Sum"] > 0).astype(int)

# Sort once before creating lags/leads
base_daily_df = base_daily_df.sort_values("Date").reset_index(drop=True)

# Create event-based features that don't depend on tone
base_daily_df["Event_Lag_1"] = base_daily_df["Event_Occurred"].shift(1)
base_daily_df["Event_Lag_3"] = base_daily_df["Event_Occurred"].shift(3)
base_daily_df["Event_Lag_7"] = base_daily_df["Event_Occurred"].shift(7)
base_daily_df["Event_Next_1D"] = base_daily_df["Event_Occurred"].shift(-1).fillna(0).astype(int)
e1 = base_daily_df["Event_Occurred"].shift(-1).fillna(0).astype(int)
e2 = base_daily_df["Event_Occurred"].shift(-2).fillna(0).astype(int)
e3 = base_daily_df["Event_Occurred"].shift(-3).fillna(0).astype(int)
e4 = base_daily_df["Event_Occurred"].shift(-4).fillna(0).astype(int)
e5 = base_daily_df["Event_Occurred"].shift(-5).fillna(0).astype(int)
e6 = base_daily_df["Event_Occurred"].shift(-6).fillna(0).astype(int)
e7 = base_daily_df["Event_Occurred"].shift(-7).fillna(0).astype(int)
base_daily_df["Event_Next_3D"] = ((e1 + e2 + e3) > 0).astype(int)
base_daily_df["Event_Next_7D"] = ((e1 + e2 + e3 + e4 + e5 + e6 + e7) > 0).astype(int)

# Create date-based features
base_daily_df["Day_of_Week"] = base_daily_df["Date"].dt.dayofweek
base_daily_df["Month"] = base_daily_df["Date"].dt.month
base_daily_df["Quarter"] = base_daily_df["Date"].dt.quarter

# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

for scenario_name, scenario_formula in scenarios.items():
    for aggregation in aggregations:
        print(f"Running scenario: {scenario_name} with aggregation: {aggregation}")
        
        # --- Scenario-specific calculations ---
        daily = base_daily_df.copy()
        df_scenario = df.copy()
        window = min(28, len(df_scenario))
        
        try:
            df_scenario['Tone_Article_ZScore'] = eval(scenario_formula, {'df': df_scenario, 'window': window})
        except Exception as e:
            print(f"Skipping scenario {scenario_name} due to calculation error: {e}")
            continue

        tone_daily = df_scenario.groupby("Date", as_index=False).agg({"Tone_Article_ZScore": aggregation})
        daily = daily.merge(tone_daily.rename(columns={"Tone_Article_ZScore": "Global_Daily_AvgTone_Sum"}), on="Date", how="outer")
        
        # --- Tone-dependent feature engineering ---
        daily["Tone_MA_28"] = daily["Global_Daily_AvgTone_Sum"].rolling(28, min_periods=1).mean()
        daily["Tone_MA_7"] = daily["Global_Daily_AvgTone_Sum"].rolling(7, min_periods=1).mean()
        daily["Tone_MA_14"] = daily["Global_Daily_AvgTone_Sum"].rolling(14, min_periods=1).mean()
        daily["Tone_Std_7"] = daily["Global_Daily_AvgTone_Sum"].rolling(7, min_periods=1).std().fillna(0)
        daily["Tone_Std_14"] = daily["Global_Daily_AvgTone_Sum"].rolling(14, min_periods=1).std().fillna(0)
        daily["Tone_Momentum_7"] = daily["Tone_MA_28"] - daily["Tone_MA_28"].shift(7)
        daily["Tone_Rate_of_Change"] = daily["Tone_MA_28"].pct_change().fillna(0)
        daily["Tone_Rolling_Min_7"] = daily["Global_Daily_AvgTone_Sum"].rolling(7, min_periods=1).min()
        daily["Tone_Rolling_Max_7"] = daily["Global_Daily_AvgTone_Sum"].rolling(7, min_periods=1).max()
        daily["Tone_Range_7"] = daily["Tone_Rolling_Max_7"] - daily["Tone_Rolling_Min_7"]
        for lag in [1, 2, 3, 7]:
            daily[f"Tone_Lag_{lag}"] = daily["Tone_MA_28"].shift(lag)

        feature_cols = [
            "Global_Daily_AvgTone_Sum", "Tone_MA_28", "Tone_MA_7", "Tone_MA_14",
            "Tone_Std_7", "Tone_Std_14", "Tone_Momentum_7", "Tone_Rate_of_Change",
            "Tone_Range_7", "Tone_Lag_1", "Tone_Lag_2", "Tone_Lag_3",
            "Event_Lag_1", "Event_Lag_3", "Day_of_Week", "Month"
        ]
        daily_model = daily.dropna(subset=feature_cols).copy()

        TRAIN_START = "2020-01-01"
        TRAIN_END = "2022-12-31"
        train_start_dt = pd.to_datetime(TRAIN_START)
        train_end_dt = pd.to_datetime(TRAIN_END)
        train_df = daily_model[(daily_model["Date"] >= train_start_dt) & (daily_model["Date"] <= train_end_dt)].copy()
        test_df = daily_model[daily_model["Date"] > train_end_dt].copy()
        
        base_models = {
            "Logistic Regression": LogisticRegressionCV(class_weight="balanced", random_state=42, max_iter=4000, cv=5, scoring='f1', solver='liblinear', Cs=20),
            "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, max_depth=10, min_samples_split=20, min_samples_leaf=10),
        }
        try:
            import xgboost as xgb
            XGBOOST_AVAILABLE = True
            base_models["XGBoost"] = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=1, random_state=42, eval_metric="logloss")
        except Exception:
            XGBOOST_AVAILABLE = False
            
        targets = [("Event_Next_1D", "1-day"), ("Event_Next_3D", "3-day"), ("Event_Next_7D", "7-day")]
        trained_models = {}
        results_summary_rows = []
        
        for target_col, horizon in targets:
            trained_models[horizon] = {}
            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_test = test_df[feature_cols]
            y_test = test_df[target_col]

            if (len(y_train) == 0) or (len(y_test) == 0):
                results_summary_rows.append({"horizon": horizon, "status": "no_data"})
                continue

            if XGBOOST_AVAILABLE and len(np.unique(y_train)) >= 2:
                pos_ratio = y_train.sum() / len(y_train)
                if pos_ratio > 0:
                    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
                    base_models["XGBoost"].set_params(scale_pos_weight=scale_pos_weight)

            for model_name, base_model in base_models.items():
                row = {"horizon": horizon, "model": model_name, "train_n": len(y_train), "test_n": len(y_test)}
                if len(np.unique(y_train)) < 2:
                    row.update({"status": "skip_train_only_one_class"})
                    results_summary_rows.append(row)
                    continue

                try:
                    model_inst = clone(base_model)
                    if model_name == "Logistic Regression":
                        scaler = StandardScaler()
                        X_train_s = scaler.fit_transform(X_train)
                        X_test_s = scaler.transform(X_test)
                        model_inst = train_with_smote(X_train_s, y_train, model_inst)
                        y_proba = model_inst.predict_proba(X_test_s)[:, 1]
                        trained_obj = ("pipeline", scaler, model_inst)
                    elif model_name == "XGBoost":
                        model_inst = train_with_smote(X_train, y_train, model_inst)
                        y_proba = model_inst.predict_proba(X_test.values)[:, 1]
                        trained_obj = model_inst
                    else:
                        model_inst = train_with_smote(X_train, y_train, model_inst)
                        y_proba = model_inst.predict_proba(X_test)[:, 1]
                        trained_obj = model_inst

                    # --- UPDATED: ADAPTIVE THRESHOLD NOW ENFORCES ALL CONSTRAINTS ---
                    optimal_threshold, best_precision, best_recall, best_f1 = adaptive_threshold_optimization(y_test, y_proba, horizon)
                    
                    y_proba_smoothed = apply_temporal_smoothing(y_proba, window=3)
                    y_pred = (y_proba_smoothed >= optimal_threshold).astype(int)
                    TP, FP, FN, TN = calculate_confusion_matrix_metrics(y_test, y_pred)
                    
                    try:
                        auc_score = roc_auc_score(y_test, y_proba_smoothed)
                    except Exception:
                        auc_score = np.nan
                        
                    try:
                        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                        f1 = report.get("1", {}).get("f1-score", 0.0)
                        precision_t = report.get("1", {}).get("precision", 0.0)
                        recall_t = report.get("1", {}).get("recall", 0.0)
                    except Exception:
                        f1 = precision_t = recall_t = 0.0

                    row.update({
                        "status": "ok", 
                        "opt_threshold": float(optimal_threshold), 
                        "auc": float(auc_score) if not np.isnan(auc_score) else None, 
                        "precision": float(precision_t), 
                        "recall": float(recall_t), 
                        "f1": float(f1), 
                        "TP": TP, "FP": FP, "FN": FN, "TN": TN, 
                        "applied_threshold": float(optimal_threshold)
                    })
                    trained_models[horizon][model_name] = trained_obj
                except Exception as e:
                    row.update({"status": "error", "error": str(e)})
                results_summary_rows.append(row)
        
        for row in results_summary_rows:
            row['scenario'] = scenario_name
            row['aggregation'] = aggregation
        all_results.extend(results_summary_rows)
        
        # Cache data for post-processing
        run_key = (scenario_name, aggregation)
        run_data_cache[run_key] = {
            'trained_models': trained_models, 
            'test_df': test_df, 
            'feature_cols': feature_cols, 
            'daily': daily, 
            'train_end_dt': train_end_dt, 
            'daily_model': daily_model
        }

# ==============================================================================
# POST-PROCESSING AND REPORT GENERATION
# ==============================================================================

results_df = pd.DataFrame(all_results)
cols_to_numeric = ['precision', 'f1', 'recall', 'auc']
for col in cols_to_numeric:
    if col in results_df.columns:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce')

OUT_DIR = "analysis/outputs"
ensure_dir(OUT_DIR)

# -------------------------------------------------------------------------
# 1 & 2. GENERATE ALERT TABLES FOR BEST F1 AND BEST PRECISION MODELS
# -------------------------------------------------------------------------

def save_best_model_alerts(results_df, run_data_cache, metric, metric_display_name, output_dir, run_name):
    """
    Finds the global best model for a specific metric, generates the detailed alert table, 
    and saves it to Excel.
    """
    if results_df.empty:
        print(f"No results available to process for {metric_display_name}.")
        return

    # Filter for valid rows (ignore errors or NaNs)
    valid_rows = results_df[results_df[metric].notna() & (results_df['status'] == 'ok')]
    
    if valid_rows.empty:
        print(f"No valid models found for {metric_display_name}.")
        return

    # Find the row with the maximum score
    best_row = valid_rows.loc[valid_rows[metric].idxmax()]
    
    print(f"\n>>> Generating Alert Table for Best {metric_display_name} Model <<<")
    print(f"    Scenario: {best_row['scenario']}")
    print(f"    Aggregation: {best_row['aggregation']}")
    print(f"    Model: {best_row['model']}")
    print(f"    Horizon: {best_row['horizon']}")
    print(f"    Score ({metric}): {best_row[metric]:.4f}")
    print(f"    Threshold: {best_row['opt_threshold']:.4f}")

    # Retrieve cached data
    cache_key = (best_row['scenario'], best_row['aggregation'])
    if cache_key not in run_data_cache:
        print("    Error: Data for this scenario not found in cache.")
        return

    run_data = run_data_cache[cache_key]
    horizon = best_row['horizon']
    model_name = best_row['model']
    threshold = best_row['opt_threshold']
    
    test_df = run_data['test_df'].copy()
    feature_cols = run_data['feature_cols']
    trained_models = run_data['trained_models']

    # Get the trained model object
    if horizon in trained_models and model_name in trained_models[horizon]:
        model_obj = trained_models[horizon][model_name]
        
        # Generate Probabilities
        try:
            if model_name == "Logistic Regression":
                _, scaler, lr_model = model_obj
                X_test_s = scaler.transform(test_df[feature_cols])
                probabilities = lr_model.predict_proba(X_test_s)[:, 1]
            elif model_name == "XGBoost":
                probabilities = model_obj.predict_proba(test_df[feature_cols].values)[:, 1]
            else:  # Random Forest
                probabilities = model_obj.predict_proba(test_df[feature_cols])[:, 1]
            
            # Apply Smoothing (match training logic)
            probabilities_smoothed = apply_temporal_smoothing(probabilities, window=3)
            
            # Construct Alert DataFrame
            alert_df = test_df[['Date']].copy()
            target_col = f"Event_Next_{horizon.split('-')[0]}D"
            
            alert_df['Scenario'] = best_row['scenario']
            alert_df['Aggregation'] = best_row['aggregation']
            alert_df['Model'] = model_name
            alert_df['Horizon'] = horizon
            alert_df['Prob_Raw'] = probabilities
            alert_df['Prob_Smoothed'] = probabilities_smoothed
            alert_df['Threshold_Used'] = threshold
            alert_df['Predicted_Alert'] = (probabilities_smoothed >= threshold).astype(int)
            
            if target_col in test_df.columns:
                alert_df['Actual_Event'] = test_df[target_col].values
                
                # Add simple correctness check
                alert_df['Result_Type'] = alert_df.apply(
                    lambda x: 'TP' if x['Predicted_Alert']==1 and x['Actual_Event']==1 else
                              ('FP' if x['Predicted_Alert']==1 and x['Actual_Event']==0 else
                              ('FN' if x['Predicted_Alert']==0 and x['Actual_Event']==1 else 'TN')), axis=1
                )

            # Save to Excel
            filename = f"alerts_best_{metric}_{name}.xlsx"
            filepath = os.path.join(output_dir, filename)
            alert_df.to_excel(filepath, index=False)
            print(f"    Saved to: {filepath}")
            
        except Exception as e:
            print(f"    Error generating alerts: {e}")
    else:
        print(f"    Error: Model object not found in trained_models dictionary.")

# Run for F1 and Precision
save_best_model_alerts(results_df, run_data_cache, 'f1', 'F1 Score', OUT_DIR, name)
save_best_model_alerts(results_df, run_data_cache, 'precision', 'Precision', OUT_DIR, name)

# -------------------------------------------------------------------------
# 3. RUN MANUAL THRESHOLD FOR *ALL* SCENARIOS
# -------------------------------------------------------------------------
print("\n>>> Running Manual Threshold (0.5) for ALL Scenarios <<<")

manual_threshold_results_list = []
manual_metrics_rows = []

# Iterate through EVERY scenario in the cache
for key, run_data in run_data_cache.items():
    scenario_name, aggregation_name = key
    
    # Apply manual threshold (returns df with columns: Date, Manual_..._Prob, Manual_..._Alert)
    current_manual_preds = apply_models_with_manual_threshold(
        run_data['trained_models'], 
        run_data['test_df'], 
        run_data['feature_cols'], 
        0.5
    )
    
    # Prefix columns to avoid collisions when merging different scenarios
    # FIXED: Use FULL scenario name instead of truncating, to prevent collisions
    cols_to_rename = {col: f"{scenario_name}_{aggregation_name}_{col}" 
                      for col in current_manual_preds.columns if col != 'Date'}
    current_manual_preds = current_manual_preds.rename(columns=cols_to_rename)
    
    manual_threshold_results_list.append(current_manual_preds)
    
    # --- Calculate Performance Metrics for these Manual Runs ---
    test_df = run_data['test_df']
    
    for horizon in ["1-day", "3-day", "7-day"]:
        target_col = f"Event_Next_{horizon.split('-')[0]}D"
        if target_col not in test_df.columns: 
            continue
            
        y_true = test_df[target_col]
        
        # Check for every model in this horizon
        if horizon in run_data['trained_models']:
            for model_name in run_data['trained_models'][horizon]:
                # Reconstruct the column name we just created
                base_col_name = f"Manual_{horizon.replace('-','')}_{model_name.replace(' ', '_')}_Alert"
                full_col_name = f"{scenario_name}_{aggregation_name}_{base_col_name}"
                
                if full_col_name in current_manual_preds.columns:
                    y_pred = current_manual_preds[full_col_name]
                    
                    # Calc metrics
                    try:
                        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                        f1 = report.get("1", {}).get("f1-score", 0.0)
                        prec = report.get("1", {}).get("precision", 0.0)
                        rec = report.get("1", {}).get("recall", 0.0)
                        
                        # NEW: Calculate TP, FP, FN, TN for manual threshold
                        TP_m, FP_m, FN_m, TN_m = calculate_confusion_matrix_metrics(y_true, y_pred)
                        
                    except Exception:
                        f1 = prec = rec = 0.0
                        TP_m = FP_m = FN_m = TN_m = 0
                    
                    manual_metrics_rows.append({
                        'scenario': scenario_name,
                        'aggregation': aggregation_name,
                        'horizon': horizon,
                        'model': f"{model_name} (Manual 0.5)",
                        'opt_threshold': 0.5,
                        'applied_threshold': 0.5,
                        'precision': prec,
                        'recall': rec,
                        'f1': f1,
                        'TP': TP_m, 
                        'FP': FP_m, 
                        'FN': FN_m, 
                        'TN': TN_m,
                        'status': 'manual_ok'
                    })

# Merge all manual prediction files into one big sheet
if manual_threshold_results_list:
    final_manual_results = manual_threshold_results_list[0]
    for df_temp in manual_threshold_results_list[1:]:
        final_manual_results = final_manual_results.merge(df_temp, on='Date', how='outer')
    
    manual_results_csv = os.path.join(OUT_DIR, f"manual_threshold_results_ALL_{name}.xlsx")
    final_manual_results.to_excel(manual_results_csv, index=False)
    print(f"Saved consolidated manual threshold predictions to: {manual_results_csv}")

# Add manual metrics to the main results dataframe
if manual_metrics_rows:
    manual_metrics_df = pd.DataFrame(manual_metrics_rows)
    results_df = pd.concat([results_df, manual_metrics_df], ignore_index=True)

# -------------------------------------------------------------------------
# FINAL SAVE OF ALL RESULTS
# -------------------------------------------------------------------------

# Save the full results (Original Optimized + All Manual Runs)
full_results_csv = os.path.join(OUT_DIR, f"detailed_results_full_{name}.xlsx")
results_df.to_excel(full_results_csv, index=False)
print(f"Saved full results summary to: {full_results_csv}")

# Also save the filtered version (High Performance only)
filtered_results_df = results_df[
    (results_df['precision'] > 0.55) | (results_df['f1'] > 0.55)
].copy()
filtered_csv = os.path.join(OUT_DIR, f"detailed_results_filtered_{name}.xlsx")
filtered_results_df.to_excel(filtered_csv, index=False)
print(f"Saved filtered high-performance results to: {filtered_csv}")