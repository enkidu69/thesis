# (Full file content with edits applied: models now use only the 28-day moving average and other MAs removed)
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
    
    return pd.concat(dataframes, ignore_index=True)

# Usage
desk = os.getcwd()
path = desk+ '\\analysis\\'
files = glob.glob(path+'cyberevents*.xlsx')
#print(files)

for file in files:
    attacks = pd.read_excel(str(file))
    #print(path+str(file))

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

TRAIN_START = "2018-01-01"
TRAIN_END = "2022-12-31"

OUT_DIR = "analysis\\outputs"
MODEL_ALGO_PREFERENCE = "Logistic Regression" if XGBOOST_AVAILABLE else "Random Forest"  # prefer XGBoost if available
APPLY_TO = "test"  # 'test' or 'all' (but metrics summaries will be computed only on test)
TIER_THRESHOLD = 0.5  # single threshold for Tier1/Tier2 (simplified)

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
    df.to_csv(path, index=False)


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
# FEATURE ENGINEERING
# -------------------------
daily = daily.sort_values("Date").reset_index(drop=True)
daily["Tone_MA_28"] = daily["Global_Daily_AvgTone_Sum"].rolling(28, min_periods=1).mean()
daily["Tone_Std_7"] = daily["Global_Daily_AvgTone_Sum"].rolling(7, min_periods=1).std().fillna(0)

for lag in [1, 2, 3, 7]:
    daily[f"Tone_Lag_{lag}"] = daily["Global_Daily_AvgTone_Sum"].shift(lag)
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

feature_cols = [
    "Global_Daily_AvgTone_Sum",
    "Tone_MA_28",
    "Tone_Std_7",
    "Tone_Lag_1",
    "Tone_Lag_2",
    "Tone_Lag_3",
    "Event_Lag_1",
    "Event_Lag_3",
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
_save_csv(train_df, train_csv)
_save_csv(test_df, test_csv)

diag = {
    "train_rows": len(train_df),
    "test_rows": len(test_df),
    "train_start": str(train_df["Date"].min()) if len(train_df) > 0 else None,
    "train_end": str(train_df["Date"].max()) if len(train_df) > 0 else None,
    "test_start": str(test_df["Date"].min()) if len(test_df) > 0 else None,
    "test_end": str(test_df["Date"].max()) if len(test_df) > 0 else None,
}
pd.DataFrame([diag]).to_csv(os.path.join(OUT_DIR, "training_diag.csv"), index=False)

# -------------------------
# MODEL TRAINING
# -------------------------
base_models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
}
if XGBOOST_AVAILABLE:
    xgb_template = xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="logloss", verbosity=0)
    base_models["XGBoost"] = xgb_template

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
                model_inst.fit(X_train_s, y_train)
                y_proba = model_inst.predict_proba(X_test_s)[:, 1]
                trained_obj = ("pipeline", scaler, model_inst)
            elif model_name == "XGBoost":
                model_inst.fit(X_train.values, y_train.values)
                y_proba = model_inst.predict_proba(X_test.values)[:, 1]
                trained_obj = model_inst
            else:
                model_inst.fit(X_train, y_train)
                y_proba = model_inst.predict_proba(X_test)[:, 1]
                trained_obj = model_inst

            precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_proba)
            if len(thresholds) > 0:
                f1_scores = 2 * (precision_vals[:-1] * recall_vals[:-1]) / (precision_vals[:-1] + recall_vals[:-1] + 1e-8)
                opt_th = thresholds[np.argmax(f1_scores)]
            else:
                opt_th = 0.5
            y_pred = (y_proba >= opt_th).astype(int)

            try:
                auc = roc_auc_score(y_test, y_proba)
            except Exception:
                auc = np.nan
            try:
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                f1 = report.get("1", {}).get("f1-score", 0.0)
                precision_t = report.get("1", {}).get("precision", 0.0)
                recall_t = report.get("1", {}).get("recall", 0.0)
            except Exception:
                f1 = precision_t = recall_t = 0.0

            row.update(
                {
                    "status": "ok",
                    "opt_threshold": float(opt_th),
                    "auc": float(auc) if not np.isnan(auc) else None,
                    "precision": float(precision_t),
                    "recall": float(recall_t),
                    "f1": float(f1),
                }
            )

            model_fname = f"model_{horizon.replace(' ','_')}_{model_name.replace(' ','_')}.pkl"
            model_path = os.path.join(models_dir, model_fname)
            _save_pickle(trained_obj, model_path)
            row["model_path"] = model_path

            trained_models[horizon][model_name] = trained_obj

        except Exception as e:
            row.update({"status": "error", "error": str(e)})

        results_summary_rows.append(row)

results_df = pd.DataFrame(results_summary_rows)
results_csv = os.path.join(OUT_DIR, f"results_summary_{TRAIN_START.replace('-','')}_{TRAIN_END.replace('-','')}.csv")
_save_csv(results_df, results_csv)
_save_pickle(results_summary_rows, os.path.join(models_dir, f"results_summary_{TRAIN_START.replace('-','')}_{TRAIN_END.replace('-','')}.pkl"))

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
    apply_df[prob_col] = _apply_trained(chosen, chosen_name, apply_df[feature_cols])

tier1 = TIER_THRESHOLD
tier2 = TIER_THRESHOLD  # single tier

apply_df["Tier1_Alert_1D"] = (apply_df["Prob_1D"] >= tier1).astype(int)
apply_df["Tier2_Alert_1D"] = (apply_df["Prob_1D"] >= tier2).astype(int)
apply_df["Tier1_Alert_3D"] = (apply_df["Prob_3D"] >= tier1).astype(int)
apply_df["Tier2_Alert_3D"] = (apply_df["Prob_3D"] >= tier2).astype(int)
apply_df["Tier1_Alert_7D"] = (apply_df["Prob_7D"] >= tier1).astype(int)
apply_df["Tier2_Alert_7D"] = (apply_df["Prob_7D"] >= tier2).astype(int)

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

# For 1D: event_date = prediction Date + 1; an event is anticipated if the same prediction row had Tier1_Alert_1D==1
merged["Anticipated_1D"] = merged.apply(
    lambda r: bool(r["Tier1_Alert_1D"] == 1) if r.get("Event_Next_1D", 0) == 1 else False, axis=1
)





# For 3D and 7D: conservative check using earliest possible event day = prediction Date + 1.
# If any Tier1 alert of corresponding horizon exists in the lookback window [event_day - lookback, event_day - 1],
# we mark the event as anticipated. The lookback equals the horizon window length (3 or 7).
def any_prior_alert_for_event(event_date, lookback_days, alert_col, preds_indexed):
    start = (pd.to_datetime(event_date) - pd.Timedelta(days=lookback_days)).normalize()
    end = (pd.to_datetime(event_date) - pd.Timedelta(days=1)).normalize()
    if end < start:
        return False
    try:
        window = preds_indexed.loc[start:end]
    except KeyError:
        return False
    if window.empty:
        return False
    return bool((window[alert_col] == 1).any())

# Add earliest possible event dates for merged rows
merged["EventDate_earliest"] = pd.to_datetime(merged["Date"]) + pd.Timedelta(days=1)

# Anticipated_3D: check lookback 3 days for Tier1_Alert_3D
merged["Anticipated_3D"] = merged.apply(
    lambda r: any_prior_alert_for_event(r["EventDate_earliest"], lookback_days=3, alert_col="Tier1_Alert_3D", preds_indexed=preds)
    if r.get("Event_Next_3D", 0) == 1
    else False,
    axis=1,
)

# Anticipated_7D: check lookback 7 days for Tier1_Alert_7D
merged["Anticipated_7D"] = merged.apply(
    lambda r: any_prior_alert_for_event(r["EventDate_earliest"], lookback_days=7, alert_col="Tier1_Alert_7D", preds_indexed=preds)
    if r.get("Event_Next_7D", 0) == 1
    else False,
    axis=1,
)

# drop helper column
merged.drop(columns=["EventDate_earliest"], inplace=True)

# -------------------------
# Persist merged alert table (test-only) with TN/TP/FP/FN and Anticipated_* columns
# -------------------------
timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
alert_csv = os.path.join(alerts_dir, f"next_day_alerts_posttrain_{TRAIN_END.replace('-','')}_{timestamp}.csv")
_save_csv(merged, alert_csv)

# -------------------------
# METRICS: compute on merged (i.e., test-only)
# -------------------------
def calculate_alert_efficiency_on_df(df_local, prob_col, true_col, threshold=0.3, cost_per_missed_event=10000, cost_per_false_alert=100):
    total_days = len(df_local)
    if total_days == 0:
        return {"TP": 0, "FP": 0, "FN": 0, "TN": 0, "total_days": 0, "precision": None, "recall": None, "f1_score": None, "alert_rate": None, "false_alert_ratio": None, "missed_event_cost": None, "false_alert_cost": None}
    preds = (df_local[prob_col] >= threshold).astype(int)
    truth = df_local[true_col].astype(int)
    TP = int(((preds == 1) & (truth == 1)).sum())
    FP = int(((preds == 1) & (truth == 0)).sum())
    FN = int(((preds == 0) & (truth == 1)).sum())
    TN = int(((preds == 0) & (truth == 0)).sum())
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    alert_rate = (TP + FP) / total_days
    false_alert_ratio = FP / (TP + FP) if (TP + FP) > 0 else 0.0
    missed_event_cost = FN * cost_per_missed_event
    false_alert_cost = FP * cost_per_false_alert
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN, "total_days": total_days, "precision": precision, "recall": recall, "f1_score": f1_score, "alert_rate": alert_rate, "false_alert_ratio": false_alert_ratio, "missed_event_cost": missed_event_cost, "false_alert_cost": false_alert_cost}

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
    eff = calculate_alert_efficiency_on_df(merged, prob_col, true_col, threshold=tier1, cost_per_missed_event=COST_PER_MISSED_EVENT, cost_per_false_alert=COST_PER_FALSE_ALERT)
    biz = business_efficiency(eff, benefit_of_catching_event=BENEFIT_OF_CATCHING_EVENT, cost_of_response=COST_OF_RESPONSE, response_cost=COST_OF_RESPONSE, opportunity_cost=500, alert_fatigue_cost=100, cost_per_FN=COST_PER_MISSED_EVENT, system_maintenance_cost=SYSTEM_MAINTENANCE_COST)
    metrics_summaries[horizon] = {"efficiency": eff, "business": biz}

# Persist metrics summary CSV (one row per horizon)
timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
metrics_rows = []
for horizon, d in metrics_summaries.items():
    eff = d.get("efficiency", {})
    biz = d.get("business", {})
    row = {
        "horizon": horizon,
        "TP": eff.get("TP"),
        "FP": eff.get("FP"),
        "FN": eff.get("FN"),
        "TN": eff.get("TN"),
        "total_days": eff.get("total_days"),
        "precision": eff.get("precision"),
        "recall": eff.get("recall"),
        "f1": eff.get("f1_score"),
        "alert_rate": eff.get("alert_rate"),
        "false_alert_ratio": eff.get("false_alert_ratio"),
        "missed_event_cost": eff.get("missed_event_cost"),
        "false_alert_cost": eff.get("false_alert_cost"),
        "net_value": biz.get("net_value"),
        "ROI": biz.get("ROI"),
    }
    metrics_rows.append(row)
metrics_df = pd.DataFrame(metrics_rows)
metrics_csv = os.path.join(alerts_dir, f"alert_metrics_summary_testonly_{TRAIN_END.replace('-','')}_{timestamp}.csv")
_save_csv(metrics_df, metrics_csv)
_save_pickle(metrics_summaries, os.path.join(alerts_dir, f"alert_metrics_summary_testonly_{TRAIN_END.replace('-','')}_{timestamp}.pkl"))

# Persist verification summary (TP/FP/FN/TN) per horizon (test-only)
verif_summary = {
    "1-day": {"tp": int(merged["TP_1D"].sum()), "fp": int(merged["FP_1D"].sum()), "fn": int(merged["FN_1D"].sum()), "tn": int(merged["TN_1D"].sum())},
    "3-day": {"tp": int(merged["TP_3D"].sum()), "fp": int(merged["FP_3D"].sum()), "fn": int(merged["FN_3D"].sum()), "tn": int(merged["TN_3D"].sum())},
    "7-day": {"tp": int(merged["TP_7D"].sum()), "fp": int(merged["FP_7D"].sum()), "fn": int(merged["FN_7D"].sum()), "tn": int(merged["TN_7D"].sum())},
}
_save_pickle(verif_summary, os.path.join(models_dir, f"verification_summary_testonly_{TRAIN_END.replace('-','')}_{timestamp}.pkl"))

# -------------------------
# DIAGNOSTICS (print)
# -------------------------
print("Artifacts saved to:", OUT_DIR)
print(" - Train CSV:", train_csv)
print(" - Test CSV:", test_csv)
print(" - Results summary CSV:", results_csv)
print(" - Alert CSV (test-only rows):", alert_csv)
print(" - Metrics CSV (test-only):", metrics_csv)
print("Trained models saved in:", models_dir)

print("\nMODEL AVAILABILITY:")
for h, models in trained_models.items():
    print("Horizon:", h, "->", list(models.keys()))

if len(test_df) > 0:
    X_test = test_df[feature_cols]
    print("\nPROBABILITY DISTRIBUTIONS ON TEST SET:")
    for h, models in trained_models.items():
        for name, obj in models.items():
            if name == "Logistic Regression":
                scaler = obj[1]
                clf = obj[2]
                probs = clf.predict_proba(scaler.transform(X_test))[:, 1]
            elif name == "XGBoost":
                probs = obj.predict_proba(X_test.values)[:, 1]
            else:
                probs = obj.predict_proba(X_test)[:, 1]
            print(f"{h} / {name}: min={probs.min():.6f} mean={probs.mean():.6f} max={probs.max():.6f} unique={len(np.unique(np.round(probs,6)))}")

print("\nTEST-ONLY verification summary (TP/FP/FN/TN):")
print(verif_summary)

# End of script.