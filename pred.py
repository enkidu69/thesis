# (Full file content with improvements integrated)
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
from imblearn.over_sampling import SMOTE
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
    excel_files = glob.glob(os.path.join(folder_path, "aggregated*.xlsx")) + \
                  glob.glob(os.path.join(folder_path, "aggregated*.xls"))
    
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

# Usage
desk = os.getcwd()
path = desk+ '\\analysis\\'
files = glob.glob(path+'cyberevents*.xlsx')
files2=glob.glob(path+'aggregated*.xlsx')
#print(files)

for file in files:
    attacks = pd.read_excel(str(file))
    #print(path+str(file))

for file in files2:
    name=str(file)

name=name[58:-5]
print(name)
#exit()
concatenated_df=read_and_concatenate_excel_files()

print(f"Successfully concatenated {len(concatenated_df['source_file'].unique())} files")
print(f"Final DataFrame shape: {concatenated_df.shape}")
df=concatenated_df



"""
Standalone script to train alert models (using 28-day moving average) and produce
D+1, D+3, D+7 alerts + verification and metrics. Flattened runtime version (no user functions).
This file:
 - trains models (LogisticRegression, RandomForest, optionally XGBoost)
 - applies models to post-train dates (or all dates if configured)
 - computes alerts and verification (TP/FP/FN/TN/TN) and persists outputs
 - computes alert efficiency and business metrics only on test data (post-train)
 - marks for each test row whether the event (1D/3D/7D) was anticipated by at least one Tier1 alert
   according to a conservative lookback definition.
How to use:
 - Edit the CONFIGURATION section to point to your input files or to set DataFrame variables.
 - Run: python testpred.py
 - Outputs (models, CSVs, pickles, metrics) are written to OUT_DIR (default 'outputs').
Notes:
 - Uses sklearn.clone to ensure each trained model is an independent estimator.
 - Aggregates Daily_AvgTone per day with mean (not sum).
 - Does NOT fill tone NaNs prior to rolling MA to preserve variance.
 - Anticipation logic: for 1D, checks the same prediction row; for 3D/7D, checks whether any Tier1 alert
   of the corresponding horizon occurred in the lookback window prior to the earliest possible event day (conservative).
"""

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings("ignore")

# optional xgboost
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# -------------------------
# CONFIGURATION (edit here)
# -------------------------
# Provide either file paths OR define DataFrames `df` and `attacks` in this script before running.

TONE_CSV = None  # e.g. "data/aggregated_tone.csv"
ATTACKS_CSV = None  # e.g. "data/cyberevents.csv" or .xlsx

TRAIN_START = "2015-01-01"
TRAIN_END = "2022-12-31"

OUT_DIR = "analysis\\outputs"
MODEL_ALGO_PREFERENCE = "Logistic Regression" if XGBOOST_AVAILABLE else "Random Forest"  # prefer XGBoost if available
APPLY_TO = "test"  # 'test' or 'all' (but metrics summaries will be computed only on test)
TIER_THRESHOLD = 0.472565017937055# single threshold for Tier1/Tier2 (simplified)


# Business/cost defaults (editable)
COST_PER_MISSED_EVENT = 100000
COST_PER_FALSE_ALERT = 200
BENEFIT_OF_CATCHING_EVENT = 50000
COST_OF_RESPONSE = 2000
SYSTEM_MAINTENANCE_COST = 10000

# -------------------------
# HELPERS
# -------------------------
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



# IMPROVED FUNCTIONS
# -------------------------
# Add this function to find the best model for each horizon
def find_best_models_per_horizon(results_summary_rows):
    """Find the best model for each horizon based on F1 score"""
    best_models = {}
    for horizon in ["1-day", "3-day", "7-day"]:
        horizon_rows = [r for r in results_summary_rows if r.get('horizon') == horizon and r.get('status') == 'ok']
        if horizon_rows:
            best_row = max(horizon_rows, key=lambda x: x.get('f1', 0))
            best_models[horizon] = {
                'model_name': best_row['model'],
                'threshold': best_row['opt_threshold'],
                'f1': best_row['f1'],
                'precision': best_row['precision'],
                'recall': best_row['recall']
            }
    return best_models

# Add this function to apply the best models to test data
def apply_best_models_to_test(best_models, trained_models, test_df, feature_cols):
    """Apply the best models with their optimal thresholds to test data"""
    best_predictions = test_df[['Date']].copy()
    
    for horizon, best_info in best_models.items():
        model_name = best_info['model_name']
        optimal_threshold = best_info['threshold']
        
        if horizon in trained_models and model_name in trained_models[horizon]:
            model_obj = trained_models[horizon][model_name]
            
            # Get probabilities using the best model
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
            
            # Create predictions with optimal threshold
            predictions = (probabilities_smoothed >= optimal_threshold).astype(int)
            
            # Add to results
            best_predictions[f'Best_{horizon}_Prob'] = probabilities_smoothed
            best_predictions[f'Best_{horizon}_Alert'] = predictions
            best_predictions[f'Best_{horizon}_Threshold'] = optimal_threshold
            best_predictions[f'Best_{horizon}_Model'] = model_name
    
    return best_predictions

# Add this function to create next day predictions
def create_next_day_predictions(best_models, trained_models, daily_model, feature_cols):
    """Create next day predictions using the best models"""
    # Get the most recent data (last available day)
    latest_data = daily_model.iloc[-1:].copy()
    
    next_day_predictions = []
    
    for horizon, best_info in best_models.items():
        model_name = best_info['model_name']
        optimal_threshold = best_info['threshold']
        
        if horizon in trained_models and model_name in trained_models[horizon]:
            model_obj = trained_models[horizon][model_name]
            
            # Get probability for next day
            if model_name == "Logistic Regression":
                _, scaler, lr_model = model_obj
                X_latest_s = scaler.transform(latest_data[feature_cols])
                probability = lr_model.predict_proba(X_latest_s)[:, 1][0]
            elif model_name == "XGBoost":
                probability = model_obj.predict_proba(latest_data[feature_cols].values)[:, 1][0]
            else:  # Random Forest
                probability = model_obj.predict_proba(latest_data[feature_cols])[:, 1][0]
            
            # Create alert decision
            alert = 1 if probability >= optimal_threshold else 0
            
            next_day_predictions.append({
                'Horizon': horizon,
                'Model': model_name,
                'Probability': probability,
                'Threshold': optimal_threshold,
                'Alert': alert,
                'Date': latest_data['Date'].iloc[0],
                'Prediction_Date': latest_data['Date'].iloc[0] + pd.Timedelta(days=1)
            })
    
    return pd.DataFrame(next_day_predictions)

def calculate_confusion_matrix_metrics(y_true, y_pred):
    """Calculate TP, FP, FN, TN from true and predicted values"""
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    return TP, FP, FN, TN

def find_balanced_threshold(y_true, y_proba, balance_method="f1", min_precision=0.3, min_recall=0.1):
    """
    Find optimal threshold that balances precision and recall
    """
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Remove the last values which are undefined
    precision_vals = precision_vals[:-1]
    recall_vals = recall_vals[:-1]
    
    viable_thresholds = []
    
    for i, (prec, rec, th) in enumerate(zip(precision_vals, recall_vals, thresholds)):
        # Skip if below minimum requirements
        if prec < min_precision or rec < min_recall:
            continue
            
        # Calculate different balance metrics
        f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
        geometric_mean = np.sqrt(prec * rec)
        precision_recall_diff = abs(prec - rec)
        f2 = (1 + 2**2) * (prec * rec) / (4 * prec + rec + 1e-8)  # F2-score (favors recall)
        f0_5 = (1 + 0.5**2) * (prec * rec) / (0.25 * prec + rec + 1e-8)  # F0.5-score (favors precision)
        
        viable_thresholds.append({
            'threshold': th,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'geometric_mean': geometric_mean,
            'precision_recall_diff': precision_recall_diff,
            'f2': f2,
            'f0_5': f0_5
        })
    
    if not viable_thresholds:
        # Fallback: use geometric mean of all thresholds
        print("  No viable thresholds found with minimum requirements, using geometric mean fallback")
        geometric_mean_scores = np.sqrt(precision_vals * recall_vals)
        best_idx = np.argmax(geometric_mean_scores)
        return thresholds[best_idx], precision_vals[best_idx], recall_vals[best_idx], geometric_mean_scores[best_idx]
    
    # Convert to DataFrame for easier sorting
    viable_df = pd.DataFrame(viable_thresholds)
    
    if balance_method == "f1":
        best_row = viable_df.loc[viable_df['f1'].idxmax()]
    elif balance_method == "geometric_mean":
        best_row = viable_df.loc[viable_df['geometric_mean'].idxmax()]
    elif balance_method == "closest":
        best_row = viable_df.loc[viable_df['precision_recall_diff'].idxmin()]
    elif balance_method == "f_beta":
        # Use F2 if recall is more important, F0.5 if precision is more important
        avg_recall = viable_df['recall'].mean()
        avg_precision = viable_df['precision'].mean()
        if avg_recall < 0.3:  # If recall is very low, favor it
            best_row = viable_df.loc[viable_df['f2'].idxmax()]
        elif avg_precision < 0.3:  # If precision is very low, favor it
            best_row = viable_df.loc[viable_df['f0_5'].idxmax()]
        else:
            best_row = viable_df.loc[viable_df['f1'].idxmax()]
    else:
        best_row = viable_df.loc[viable_df['f1'].idxmax()]
    
    return best_row['threshold'], best_row['precision'], best_row['recall'], best_row['f1']

def adaptive_threshold_optimization(y_true, y_proba, horizon):
    """
    Adaptive threshold optimization that adjusts strategy based on data characteristics
    """
    # Analyze class distribution
    positive_ratio = y_true.mean()
    n_positives = y_true.sum()
    
    print(f"  Class balance: {positive_ratio:.3f} positive ratio, {n_positives} positive samples")
    
    # Initial optimization
    if n_positives < 10:  # Very few positive examples
        print("  Strategy: Few positives - favoring recall")
        threshold, precision, recall, f1 = find_balanced_threshold(
            y_true, y_proba, balance_method="f2", min_precision=0.2, min_recall=0.05
        )
    elif positive_ratio < 0.1:  # Imbalanced data
        print("  Strategy: Imbalanced data - balanced approach")
        threshold, precision, recall, f1 = find_balanced_threshold(
            y_true, y_proba, balance_method="f_beta", min_precision=0.3, min_recall=0.1
        )
    else:  # Balanced case - start with F1 maximization
        print("  Strategy: Balanced data - maximizing F1")
        threshold, precision, recall, f1 = find_balanced_threshold(
            y_true, y_proba, balance_method="f1", min_precision=0.3, min_recall=0.1
        )
    
    # Check if we need rebalancing
    if abs(precision - recall) > 0.15:  # If still unbalanced
        print(f"  Large P-R gap ({abs(precision-recall):.3f}) - rebalancing...")
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_proba)
        precision_vals = precision_vals[:-1]
        recall_vals = recall_vals[:-1]
        
        # Find threshold that minimizes |precision - recall| among good candidates
        balanced_candidates = []
        for i, (p, r, t) in enumerate(zip(precision_vals, recall_vals, thresholds)):
            if p >= 0.3 and r >= 0.1:  # Minimum requirements
                balance_score = 1.0 - abs(p - r)  # Higher is better
                f1_score = 2 * (p * r) / (p + r + 1e-8)
                balanced_candidates.append((t, p, r, f1_score, balance_score))
        
        if balanced_candidates:
            # Sort by balance score first, then F1 score
            balanced_candidates.sort(key=lambda x: (x[4], x[3]), reverse=True)
            new_threshold, new_precision, new_recall, new_f1, balance_score = balanced_candidates[0]
            
            # Only switch if the balance improvement is significant
            if abs(new_precision - new_recall) < abs(precision - recall) - 0.05:
                threshold, precision, recall, f1 = new_threshold, new_precision, new_recall, new_f1
                print(f"  Rebalanced: precision={precision:.3f}, recall={recall:.3f}, diff={abs(precision-recall):.3f}")
            else:
                print(f"  Keeping original: precision={precision:.3f}, recall={recall:.3f}, diff={abs(precision-recall):.3f}")
    
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

def apply_temporal_smoothing(probabilities, window=3):
    """Apply moving average smoothing to probabilities"""
    return pd.Series(probabilities).rolling(window=window, center=True, min_periods=1).mean().values

def create_ensemble_prediction(trained_models, X, feature_cols, horizon):
    """Create ensemble prediction from all available models"""
    predictions = []
    weights = []
    
    for model_name, model_obj in trained_models[horizon].items():
        proba = _apply_trained(model_obj, model_name, X[feature_cols])
        predictions.append(proba)
        weights.append(1.0)  # Equal weights
    
    ensemble_proba = np.average(predictions, axis=0, weights=weights)
    return ensemble_proba
# -------------------------
# LOAD DATA
# -------------------------
if TONE_CSV is not None and ATTACKS_CSV is not None:
    # try reading tone
    try:
        df = pd.read_csv(TONE_CSV, parse_dates=["Date"])
    except Exception:
        df = pd.read_excel(TONE_CSV)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
    # try reading attacks
    try:
        attacks = pd.read_csv(ATTACKS_CSV, parse_dates=["event_date"])
    except Exception:
        attacks = pd.read_excel(ATTACKS_CSV)
        if "event_date" in attacks.columns:
            attacks["event_date"] = pd.to_datetime(attacks["event_date"])

# Validate data presence
if "df" not in globals() or "attacks" not in globals():
    raise RuntimeError(
        "Input data not found. Set TONE_CSV and ATTACKS_CSV, or define DataFrames `df` and `attacks` in the script."
    )

# -------------------------
# PREPARE DAILY SERIES
# -------------------------
df = df.copy()
if "Date" not in df.columns:
    raise RuntimeError("Tone DataFrame must contain a 'Date' column")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)




attacks = attacks.copy()
if "event_date" not in attacks.columns:
    raise RuntimeError("Attacks DataFrame must contain an 'event_date' column")
attacks["event_date"] = pd.to_datetime(attacks["event_date"])
attacks = attacks.sort_values("event_date").reset_index(drop=True)


# Aggregate
tone_daily = df.groupby("Date", as_index=False).agg({"Daily_AvgTone": "mean"})
attacks_daily = attacks.groupby("event_date", as_index=False).agg({"event_count": "sum"}).rename(
    columns={"event_date": "Date"}
)




# Continuous date index
start_date = min(tone_daily["Date"].min(), attacks_daily["Date"].min())
end_date = max(tone_daily["Date"].max(), attacks_daily["Date"].max())
all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
daily = pd.DataFrame({"Date": all_dates})


#daily checked
#attacks checked here: OK


# Merge; fill event counts with 0; keep tone NaNs (do not fill before rolling)
daily = daily.merge(tone_daily.rename(columns={"Daily_AvgTone": "Global_Daily_AvgTone_Sum"}), on="Date", how="outer")




daily = daily.merge(attacks_daily.rename(columns={"event_count": "Global_Event_Count_Sum"}), on="Date", how="outer")



daily["Global_Event_Count_Sum"] = daily["Global_Event_Count_Sum"].fillna(0)
daily["Event_Occurred"] = (daily["Global_Event_Count_Sum"] > 0).astype(int)

# -------------------------
# IMPROVED FEATURE ENGINEERING
# -------------------------
daily = daily.sort_values("Date").reset_index(drop=True)
daily["Tone_MA_28"] = daily["Global_Daily_AvgTone_Sum"].rolling(28, min_periods=1).mean()
daily["Tone_MA_7"] = daily["Global_Daily_AvgTone_Sum"].rolling(7, min_periods=1).mean()
daily["Tone_MA_14"] = daily["Global_Daily_AvgTone_Sum"].rolling(14, min_periods=1).mean()
daily["Tone_Std_7"] = daily["Global_Daily_AvgTone_Sum"].rolling(7, min_periods=1).std().fillna(0)
daily["Tone_Std_14"] = daily["Global_Daily_AvgTone_Sum"].rolling(14, min_periods=1).std().fillna(0)

# Add momentum and trend features
daily["Tone_Momentum_7"] = daily["Tone_MA_28"] - daily["Tone_MA_28"].shift(7)
daily["Tone_Rate_of_Change"] = daily["Tone_MA_28"].pct_change().fillna(0)

# Add volatility features
daily["Tone_Rolling_Min_7"] = daily["Global_Daily_AvgTone_Sum"].rolling(7, min_periods=1).min()
daily["Tone_Rolling_Max_7"] = daily["Global_Daily_AvgTone_Sum"].rolling(7, min_periods=1).max()
daily["Tone_Range_7"] = daily["Tone_Rolling_Max_7"] - daily["Tone_Rolling_Min_7"]

# Add day of week and month features
daily["Day_of_Week"] = daily["Date"].dt.dayofweek
daily["Month"] = daily["Date"].dt.month
daily["Quarter"] = daily["Date"].dt.quarter

for lag in [1, 2, 3, 7]:
    daily[f"Tone_Lag_{lag}"] = daily["Tone_MA_28"].shift(lag)
daily["Event_Lag_1"] = daily["Event_Occurred"].shift(1)
daily["Event_Lag_3"] = daily["Event_Occurred"].shift(3)
daily["Event_Lag_7"] = daily["Event_Occurred"].shift(7)

daily["Event_Next_1D"] = daily["Event_Occurred"].shift(-1).fillna(0).astype(int)
e1 = daily["Event_Occurred"].shift(-1).fillna(0).astype(int)
e2 = daily["Event_Occurred"].shift(-2).fillna(0).astype(int)
e3 = daily["Event_Occurred"].shift(-3).fillna(0).astype(int)
e4 = daily["Event_Occurred"].shift(-4).fillna(0).astype(int)
e5 = daily["Event_Occurred"].shift(-5).fillna(0).astype(int)
e6 = daily["Event_Occurred"].shift(-6).fillna(0).astype(int)
e7 = daily["Event_Occurred"].shift(-7).fillna(0).astype(int)
daily["Event_Next_3D"] = ((e1 + e2 + e3) > 0).astype(int)
daily["Event_Next_7D"] = ((e1 + e2 + e3 + e4 + e5 + e6 + e7) > 0).astype(int)

# Updated feature columns with new features
feature_cols = [
    "Global_Daily_AvgTone_Sum", "Tone_MA_28", "Tone_MA_7", "Tone_MA_14",
    "Tone_Std_7", "Tone_Std_14", "Tone_Momentum_7", "Tone_Rate_of_Change",
    "Tone_Range_7", "Tone_Lag_1", "Tone_Lag_2", "Tone_Lag_3",
    "Event_Lag_1", "Event_Lag_3", "Day_of_Week", "Month"
]

daily_model = daily.dropna(subset=feature_cols).copy()

# -------------------------
# TRAIN / TEST SPLIT
# -------------------------
train_start_dt = pd.to_datetime(TRAIN_START)
train_end_dt = pd.to_datetime(TRAIN_END)
train_df = daily_model[(daily_model["Date"] >= train_start_dt) & (daily_model["Date"] <= train_end_dt)].copy()
test_df = daily_model[daily_model["Date"] > train_end_dt].copy()

ensure_dir(OUT_DIR)
models_dir = os.path.join(OUT_DIR, "models")
alerts_dir = os.path.join(OUT_DIR, "alerts")
ensure_dir(models_dir)
ensure_dir(alerts_dir)

train_csv = os.path.join(OUT_DIR, f"train_df_{TRAIN_START.replace('-','')}_{TRAIN_END.replace('-','')}.csv")
test_csv = os.path.join(OUT_DIR, f"test_df_post_{TRAIN_END.replace('-','')}.csv")
#removed save
#_save_csv(train_df, train_csv)
#_save_csv(test_df, test_csv)

diag = {
    "train_rows": len(train_df),
    "test_rows": len(test_df),
    "train_start": str(train_df["Date"].min()) if len(train_df) > 0 else None,
    "train_end": str(train_df["Date"].max()) if len(train_df) > 0 else None,
    "test_start": str(test_df["Date"].min()) if len(test_df) > 0 else None,
    "test_end": str(test_df["Date"].max()) if len(test_df) > 0 else None,
}
print(diag)

#pd.DataFrame([diag]).to_csv(os.path.join(OUT_DIR, "training_diag.csv"), index=False)

# -------------------------
# IMPROVED MODEL TRAINING
# -------------------------

from sklearn.linear_model import LogisticRegressionCV


# Enhanced model configurations
base_models = {
    "Logistic Regression": LogisticRegressionCV(
        class_weight="balanced",
        random_state=42,
        max_iter=4000,
        cv=5,
        scoring='f1',
        solver='liblinear',
        Cs=20
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced", 
        random_state=42,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10
    ),
}
if XGBOOST_AVAILABLE:
    base_models["XGBoost"] = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=1,
        random_state=42,
        eval_metric="logloss"
    )

targets = [("Event_Next_1D", "1-day"), ("Event_Next_3D", "3-day"), ("Event_Next_7D", "7-day")]

trained_models = {}
results_summary_rows = []

for target_col, horizon in targets:
    trained_models[horizon] = {}
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    if (len(y_train) == 0) and (len(y_test) == 0):
        results_summary_rows.append({"horizon": horizon, "status": "no_data"})
        continue

    # Calculate class weight for XGBoost
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
                
            else:  # Random Forest
                model_inst = train_with_smote(X_train, y_train, model_inst)
                y_proba = model_inst.predict_proba(X_test)[:, 1]
                trained_obj = model_inst

            # Apply adaptive threshold optimization for ALL models
            optimal_threshold, best_precision, best_recall, best_f1 = adaptive_threshold_optimization(
                y_test, y_proba, horizon
            )
            
            print(f"\n{horizon} - {model_name}:")
            print(f"  Optimal threshold: {optimal_threshold:.3f}")
            print(f"  Precision: {best_precision:.3f}, Recall: {best_recall:.3f}, F1: {best_f1:.3f}")
            print(f"  Precision-Recall difference: {abs(best_precision - best_recall):.3f}")
            
            # Apply temporal smoothing and threshold
            y_proba_smoothed = apply_temporal_smoothing(y_proba, window=3)
            y_pred = (y_proba_smoothed >= optimal_threshold).astype(int)

            # Calculate confusion matrix metrics
            TP, FP, FN, TN = calculate_confusion_matrix_metrics(y_test, y_pred)

            try:
                auc = roc_auc_score(y_test, y_proba_smoothed)
            except Exception:
                auc = np.nan
                
            try:
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                f1 = report.get("1", {}).get("f1-score", 0.0)
                precision_t = report.get("1", {}).get("precision", 0.0)
                recall_t = report.get("1", {}).get("recall", 0.0)
            except Exception:
                f1 = precision_t = recall_t = 0.0

# In the results summary section, ensure the threshold is included:
            row.update(
                {
                    "status": "ok",
                    "opt_threshold": float(optimal_threshold),  # This should already be there
                    "auc": float(auc) if not np.isnan(auc) else None,
                    "precision": float(precision_t),
                    "recall": float(recall_t),
                    "f1": float(f1),
                    "TP": TP,
                    "FP": FP,
                    "FN": FN,
                    "TN": TN,
                    "applied_threshold": float(optimal_threshold)  # Explicitly include
                }
            )
            print(f"  Final: TP={TP}, FP={FP}, FN={FN}, TN={TN}")
                  
            model_fname = f"model_{horizon.replace(' ','_')}_{model_name.replace(' ','_')}.pkl"
            model_path = os.path.join(models_dir, model_fname)
            row["model_path"] = model_path

            trained_models[horizon][model_name] = trained_obj

        except Exception as e:
            row.update({"status": "error", "error": str(e)})
            print(f"  Error: {str(e)}")

        results_summary_rows.append(row)
# ... (rest of the code remains the same until diagnostics section)



results_df = pd.DataFrame(results_summary_rows)
results_csv = os.path.join(OUT_DIR, f"results_{name}_summary_{TRAIN_START.replace('-','')}_{TRAIN_END.replace('-','')}.xlsx")
_save_csv(results_df, results_csv)
#_save_pickle(results_summary_rows, os.path.join(models_dir, f"results_summary_{TRAIN_START.replace('-','')}_{TRAIN_END.replace('-','')}.pkl"))

# trainer dict compatibility
trainer = {
    "daily": daily,
    "daily_model": daily_model,
    "train_df": train_df,
    "test_df": test_df,
    "feature_cols": feature_cols,
    "trained_models": trained_models,
    "results_summary": results_summary_rows,
}

# -------------------------
# APPLY MODELS & PRODUCE ALERTS (post-train)
# -------------------------
if APPLY_TO == "test":
    apply_mask = daily["Date"] > train_end_dt
else:
    apply_mask = pd.Series(True, index=daily.index)

apply_df = daily.loc[apply_mask].copy().sort_values("Date").reset_index(drop=True)
#NL missing one day due to missing data at the end of JUNE 2025!!!

#apply_df.to_excel('dailyaft.xlsx', index=False, engine="xlsxwriter", engine_kwargs={'options': {'strings_to_urls': False}})

#exit()

apply_df = apply_df.dropna(subset=feature_cols).copy()

apply_df["Prob_1D"] = np.nan
apply_df["Prob_3D"] = np.nan
apply_df["Prob_7D"] = np.nan


def _apply_trained(trained_obj, model_name, X):
    if trained_obj is None:
        return np.zeros(len(X))
    if model_name == "Logistic Regression":
        _, scaler, lr_model = trained_obj
        Xs = scaler.transform(X)
        return lr_model.predict_proba(Xs)[:, 1]
    elif model_name == "XGBoost":
        return trained_obj.predict_proba(X.values)[:, 1]
    else:
        return trained_obj.predict_proba(X)[:, 1]


for horizon, prob_col in [("1-day", "Prob_1D"), ("3-day", "Prob_3D"), ("7-day", "Prob_7D")]:
    models_here = trained_models.get(horizon, {})
    chosen = None
    chosen_name = None
    if MODEL_ALGO_PREFERENCE in models_here:
        chosen_name = MODEL_ALGO_PREFERENCE
        chosen = models_here[chosen_name]
    elif len(models_here) > 0:
        chosen_name = list(models_here.keys())[0]
        chosen = models_here[chosen_name]
    if chosen is None:
        continue
    
    # Get base probabilities
    base_probs = _apply_trained(chosen, chosen_name, apply_df[feature_cols])
    
    # Apply temporal smoothing
    smoothed_probs = apply_temporal_smoothing(base_probs, window=3)
    
    # Optionally use ensemble if multiple models available
    if len(models_here) > 1:
        ensemble_probs = create_ensemble_prediction(trained_models, apply_df, feature_cols, horizon)
        # Blend base model with ensemble
        apply_df[prob_col] = 0.7 * smoothed_probs + 0.3 * ensemble_probs
    else:
        apply_df[prob_col] = smoothed_probs

tier1 = TIER_THRESHOLD
tier2 = TIER_THRESHOLD  # single tier

apply_df["Tier1_Alert_1D"] = (apply_df["Prob_1D"] >= tier1).astype(int)
#apply_df["Tier2_Alert_1D"] = (apply_df["Prob_1D"] >= tier2).astype(int)
apply_df["Tier1_Alert_3D"] = (apply_df["Prob_3D"] >= tier1).astype(int)
#apply_df["Tier2_Alert_3D"] = (apply_df["Prob_3D"] >= tier2).astype(int)
apply_df["Tier1_Alert_7D"] = (apply_df["Prob_7D"] >= tier1).astype(int)
#apply_df["Tier2_Alert_7D"] = (apply_df["Prob_7D"] >= tier2).astype(int)

# Merge apply_df with test_df to ensure summaries are computed only on test data
# We'll merge on Date and keep only rows present in test_df (post-train test set)



merged = apply_df.merge(test_df[["Date", "Event_Next_1D", "Event_Next_3D", "Event_Next_7D"]], on="Date", how="inner", suffixes=("", "_true"))

# If merged is empty, the test set had no rows after train_end (we handle gracefully)
if merged.empty:
    print("Warning: No rows in merged predictions vs test labels (no post-train rows). Metrics will be empty.")

# Add TN columns explicitly for verification per horizon (using Tier1 decisions)
merged["TN_1D"] = ((merged["Tier1_Alert_1D"] == 0) & (merged["Event_Next_1D"] == 0)).astype(int)
merged["TN_3D"] = ((merged["Tier1_Alert_3D"] == 0) & (merged["Event_Next_3D"] == 0)).astype(int)
merged["TN_7D"] = ((merged["Tier1_Alert_7D"] == 0) & (merged["Event_Next_7D"] == 0)).astype(int)

# Also compute TP/FP/FN in merged (redundant but clearer)
merged["TP_1D"] = ((merged["Tier1_Alert_1D"] == 1) & (merged["Event_Next_1D"] == 1)).astype(int)
merged["FP_1D"] = ((merged["Tier1_Alert_1D"] == 1) & (merged["Event_Next_1D"] == 0)).astype(int)
merged["FN_1D"] = ((merged["Tier1_Alert_1D"] == 0) & (merged["Event_Next_1D"] == 1)).astype(int)

merged["TP_3D"] = ((merged["Tier1_Alert_3D"] == 1) & (merged["Event_Next_3D"] == 1)).astype(int)
merged["FP_3D"] = ((merged["Tier1_Alert_3D"] == 1) & (merged["Event_Next_3D"] == 0)).astype(int)
merged["FN_3D"] = ((merged["Tier1_Alert_3D"] == 0) & (merged["Event_Next_3D"] == 1)).astype(int)

merged["TP_7D"] = ((merged["Tier1_Alert_7D"] == 1) & (merged["Event_Next_7D"] == 1)).astype(int)
merged["FP_7D"] = ((merged["Tier1_Alert_7D"] == 1) & (merged["Event_Next_7D"] == 0)).astype(int)
merged["FN_7D"] = ((merged["Tier1_Alert_7D"] == 0) & (merged["Event_Next_7D"] == 1)).astype(int)

# -------------------------
# ANTICIPATION COLUMNS: mark whether an event was anticipated by at least 1 Tier1 alert
# -------------------------
# We'll construct a prediction-alert lookup (preds) covering the test prediction rows only (dates in merged)
preds = apply_df[["Date", "Tier1_Alert_1D", "Tier1_Alert_3D", "Tier1_Alert_7D"]].copy().set_index("Date").sort_index()


merged['Tier1_Alert_1D_prior'] = merged['Tier1_Alert_1D'].shift(1).rolling(window=1, min_periods=1).max()
merged['Tier1_Alert_Flag1D'] = (merged['Event_Occurred'] == 1) & (merged['Tier1_Alert_1D_prior'] > 0)


# Add earliest possible event dates for merged rows
merged["EventDate_earliest"] = pd.to_datetime(merged["Date"]) + pd.Timedelta(days=1)

# Anticipated_3D: check lookback 3 days for Tier1_Alert_3D


merged['Tier1_Alert_3D_prior'] = merged['Tier1_Alert_3D'].shift(1).rolling(window=3, min_periods=3).max()
merged['Tier1_Alert_Flag3D'] = (merged['Event_Occurred'] == 1) & (merged['Tier1_Alert_3D_prior'] > 0)

# Anticipated_7D: check lookback 7 days for Tier1_Alert_7D

merged['Tier1_Alert_7D_prior'] = merged['Tier1_Alert_7D'].shift(1).rolling(window=7, min_periods=7).max()
merged['Tier1_Alert_Flag7D'] = (merged['Event_Occurred'] == 1) & (merged['Tier1_Alert_7D_prior'] > 0)


# drop helper column
merged.drop(columns=["EventDate_earliest"], inplace=True)

# -------------------------
# Persist merged alert table (test-only) with TN/TP/FP/FN and Anticipated_* columns
# -------------------------
timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
alert_csv = os.path.join(alerts_dir, f"next_day_{name}_posttrain_{TRAIN_END.replace('-','')}_{timestamp}.xlsx")
_save_csv(merged, alert_csv)

# -------------------------
# METRICS: compute on merged (i.e., test-only)
# -------------------------
def calculate_alert_efficiency_on_df(df_local, prob_col, true_col, threshold=0.3, cost_per_missed_event=10000, cost_per_false_alert=100):
    """
    Correctly calculate TP, FP, FN, TN for temporal prediction horizons
    TP: Alert issued AND event occurred within horizon
    FP: Alert issued BUT no event occurred within horizon  
    FN: Event occurred BUT no alert issued for that horizon
    TN: No alert issued AND no event occurred within horizon
    """
    total_days = len(df_local)
    if total_days == 0:
        return {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "total_days": 0, "precision": None, "recall": None, "f1_score": None, "alert_rate": None, "false_alert_ratio": None, "missed_event_cost": None, "false_alert_cost": None, "Events_occurred":0,"Predicted1D": 0,"Predicted3D": 0,"Predicted7D": 0,"prediction7Dpercentage": 0}
    
    # Get predictions based on threshold
    preds = (df_local[prob_col] >= threshold).astype(int)
    truth = df_local[true_col].astype(int)
    
    # Calculate confusion matrix components
    TP = int(((preds == 1) & (truth == 1)).sum())
    FP = int(((preds == 1) & (truth == 0)).sum())
    FN = int(((preds == 0) & (truth == 1)).sum())
    TN = int(((preds == 0) & (truth == 0)).sum())
    
    # Validation: Total events should equal TP + FN
    total_events = int(truth.sum())
    if (TP + FN) != total_events:
        print(f"WARNING: TP({TP}) + FN({FN}) = {TP+FN} but total events = {total_events}")
    
    # Validation: Total days should equal TP + FP + FN + TN
    if (TP + FP + FN + TN) != total_days:
        print(f"WARNING: Sum of confusion matrix ({TP+FP+FN+TN}) != total days ({total_days})")
    
    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    alert_rate = (TP + FP) / total_days
    false_alert_ratio = FP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    # Calculate costs
    missed_event_cost = FN * cost_per_missed_event
    false_alert_cost = FP * cost_per_false_alert
    
    # Get additional metrics
    Events_occurred = df_local['Event_Occurred'].sum()
    Predicted1D = df_local['Tier1_Alert_Flag1D'].value_counts().get(True, 0)
    Predicted3D = df_local['Tier1_Alert_Flag3D'].value_counts().get(True, 0)
    Predicted7D = df_local['Tier1_Alert_Flag7D'].value_counts().get(True, 0)
    prediction7Dpercentage = Predicted7D / Events_occurred if Events_occurred > 0 else 0
    
    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN, 
        "total_days": total_days, 
        "precision": precision, "recall": recall, "f1_score": f1_score, 
        "alert_rate": alert_rate, "false_alert_ratio": false_alert_ratio, 
        "missed_event_cost": missed_event_cost, "false_alert_cost": false_alert_cost,
        "Events_occurred": Events_occurred,
        "Predicted1D": Predicted1D, "Predicted3D": Predicted3D, "Predicted7D": Predicted7D,
        "prediction7Dpercentage": prediction7Dpercentage
    }

def business_efficiency(metrics_dict, benefit_of_catching_event=50000, cost_of_response=2000, response_cost=2000, opportunity_cost=500, alert_fatigue_cost=100, cost_per_FN=50000, system_maintenance_cost=10000):
    TP = metrics_dict.get("TP", 0)
    FP = metrics_dict.get("FP", 0)
    FN = metrics_dict.get("FN", 0)
    value_per_TP = (benefit_of_catching_event - cost_of_response)
    cost_per_FP = (response_cost + opportunity_cost + alert_fatigue_cost)
    net_value = (TP * value_per_TP) - (FP * cost_per_FP) - (FN * cost_per_FN)
    denom = (FP * cost_per_FP + system_maintenance_cost)
    ROI = (TP * value_per_TP) / denom if denom > 0 else None
    return {"value_per_TP": value_per_TP, "cost_per_FP": cost_per_FP, "net_value": net_value, "ROI": ROI}

# Compute per-horizon metrics on merged (test-only)
metrics_summaries = {}
horizon_map = {
    "1-day": ("Prob_1D", "Event_Next_1D"),
    "3-day": ("Prob_3D", "Event_Next_3D"),
    "7-day": ("Prob_7D", "Event_Next_7D"),
}
for horizon, (prob_col, true_col) in horizon_map.items():
    if prob_col not in merged.columns or true_col not in merged.columns:
        metrics_summaries[horizon] = {"error": f"missing {prob_col} or {true_col} in merged test dataset"}
        continue
    #print(merged)
    eff = calculate_alert_efficiency_on_df(merged, prob_col, true_col, threshold=tier1, cost_per_missed_event=COST_PER_MISSED_EVENT, cost_per_false_alert=COST_PER_FALSE_ALERT)
    biz = business_efficiency(eff, benefit_of_catching_event=BENEFIT_OF_CATCHING_EVENT, cost_of_response=COST_OF_RESPONSE, response_cost=COST_OF_RESPONSE, opportunity_cost=500, alert_fatigue_cost=100, cost_per_FN=COST_PER_MISSED_EVENT, system_maintenance_cost=SYSTEM_MAINTENANCE_COST)
    metrics_summaries[horizon] = {"efficiency": eff}
    #metrics_summaries[horizon] = {"efficiency": eff, "business": biz}

# Persist metrics summary CSV (one row per horizon)
timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
metrics_rows = []
for horizon, d in metrics_summaries.items():
    eff = d.get("efficiency", {})
    biz = d.get("business", {})
    row = {
        "name": name,
        "horizon": horizon,
        "threshold": TIER_THRESHOLD,
        "TP": eff.get("TP"),
        "FP": eff.get("FP"),
        "FN": eff.get("FN"),
        "TN": eff.get("TN"),
        "events": eff.get("Events_occurred"),
        "Predicted3D": eff.get("Predicted3D"),
        "Predicted7D": eff.get("Predicted7D"),
        "prediction7Dpercentage": eff.get("prediction7Dpercentage"),
        "total_days": eff.get("total_days"),
        "precision": eff.get("precision"),
        "recall": eff.get("recall"),
        "f1": eff.get("f1_score")
    }
    metrics_rows.append(row)
metrics_df = pd.DataFrame(metrics_rows)
metrics_csv = os.path.join(alerts_dir, f"alert_{name}_summary_test_{TRAIN_END.replace('-','')}_{timestamp}.xlsx")
_save_csv(metrics_df, metrics_csv)
#_save_pickle(metrics_summaries, os.path.join(alerts_dir, f"alert_metrics_summary_testonly_{TRAIN_END.replace('-','')}_{timestamp}.pkl"))

# Persist verification summary (TP/FP/FN/TN) per horizon (test-only)
verif_summary = {
    "1-day": {"tp": int(merged["TP_1D"].sum()), "fp": int(merged["FP_1D"].sum()), "fn": int(merged["FN_1D"].sum()), "tn": int(merged["TN_1D"].sum())},
    "3-day": {"tp": int(merged["TP_3D"].sum()), "fp": int(merged["FP_3D"].sum()), "fn": int(merged["FN_3D"].sum()), "tn": int(merged["TN_3D"].sum())},
    "7-day": {"tp": int(merged["TP_7D"].sum()), "fp": int(merged["FP_7D"].sum()), "fn": int(merged["FN_7D"].sum()), "tn": int(merged["TN_7D"].sum())},
}
#_save_pickle(verif_summary, os.path.join(models_dir, f"verification_summary_testonly_{TRAIN_END.replace('-','')}_{timestamp}.pkl"))


# After the model training section, add:

# -------------------------
# FIND AND APPLY BEST MODELS
# -------------------------

# Find best models for each horizon
best_models = find_best_models_per_horizon(results_summary_rows)

print("\n" + "="*80)
print("BEST MODELS SELECTION")
print("="*80)
for horizon, best_info in best_models.items():
    print(f"{horizon.upper():8} | Model: {best_info['model_name']:20} | "
          f"Threshold: {best_info['threshold']:.3f} | "
          f"F1: {best_info['f1']:.3f} | "
          f"Precision: {best_info['precision']:.3f} | "
          f"Recall: {best_info['recall']:.3f}")

# Apply best models to test data
best_predictions_test = apply_best_models_to_test(best_models, trained_models, test_df, feature_cols)

# Merge with original test data for comprehensive analysis
enhanced_test_results = test_df.merge(best_predictions_test, on='Date', how='left')

# Create next day predictions
next_day_predictions = create_next_day_predictions(best_models, trained_models, daily_model, feature_cols)

# -------------------------
# SAVE ENHANCED RESULTS
# -------------------------

# Save detailed results with thresholds
detailed_results_df = pd.DataFrame(results_summary_rows)
detailed_results_csv = os.path.join(OUT_DIR, f"detailed_results_{name}_all_models_{TRAIN_START.replace('-','')}_{TRAIN_END.replace('-','')}.xlsx")
_save_csv(detailed_results_df, detailed_results_csv)

# Save best models summary
best_models_df = pd.DataFrame([
    {
        'horizon': horizon,
        'best_model': info['model_name'],
        'optimal_threshold': info['threshold'],
        'f1_score': info['f1'],
        'precision': info['precision'],
        'recall': info['recall']
    }
    for horizon, info in best_models.items()
])
best_models_csv = os.path.join(OUT_DIR, f"best_models_summary_{name}_{TRAIN_START.replace('-','')}_{TRAIN_END.replace('-','')}.xlsx")
_save_csv(best_models_df, best_models_csv)

# Save enhanced test results with best model predictions
enhanced_test_csv = os.path.join(OUT_DIR, f"enhanced_test_results_{name}_{TRAIN_START.replace('-','')}_{TRAIN_END.replace('-','')}.xlsx")
_save_csv(enhanced_test_results, enhanced_test_csv)

# Save next day predictions
next_day_csv = os.path.join(OUT_DIR, f"next_day_predictions_{name}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.xlsx")
_save_csv(next_day_predictions, next_day_csv)

# -------------------------
# ENHANCED DIAGNOSTICS
# -------------------------

print("\n" + "="*80)
print("ENHANCED TEST RESULTS WITH BEST MODELS")
print("="*80)

# Calculate performance metrics for best models on test data
for horizon in best_models.keys():
    prob_col = f'Best_{horizon}_Prob'
    alert_col = f'Best_{horizon}_Alert'
    
    if prob_col in enhanced_test_results.columns:
        true_col = f'Event_Next_{horizon.split("-")[0]}D'
        
        if true_col in enhanced_test_results.columns:
            # Calculate metrics
            y_true = enhanced_test_results[true_col]
            y_pred = enhanced_test_results[alert_col]
            
            TP, FP, FN, TN = calculate_confusion_matrix_metrics(y_true, y_pred)
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n{horizon.upper()} - Best Model ({best_models[horizon]['model_name']}):")
            print(f"  TP: {TP:3d} | FP: {FP:3d} | FN: {FN:3d} | TN: {TN:3d}")
            print(f"  Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")
            print(f"  Optimal Threshold: {best_models[horizon]['threshold']:.3f}")

print("\n" + "="*80)
print("NEXT DAY PREDICTIONS")
print("="*80)
print(next_day_predictions.to_string(index=False))

print("\n" + "="*80)
print("FILES GENERATED")
print("="*80)
print(f"1. Detailed Results: {detailed_results_csv}")
print(f"2. Best Models Summary: {best_models_csv}")
print(f"3. Enhanced Test Results: {enhanced_test_csv}")
print(f"4. Next Day Predictions: {next_day_csv}")
