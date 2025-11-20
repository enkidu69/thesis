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
from sklearn.linear_model import LogisticRegressionCV
from sklearn.base import clone
import warnings
import pickle
warnings.filterwarnings('ignore')

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
            manual_predictions[f'{col_prefix}_Threshold'] = manual_threshold
    
    return manual_predictions

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

desk = os.getcwd()
path = desk+ '\\analysis\\'
files = glob.glob(path+'cyberevents*.xlsx')
files2=glob.glob(path+'*cyber_events*.xlsx')

for file in files:
    attacks = pd.read_excel(str(file))

for file in files2:
    name=str(file)

name=name[58:-5]
print(name)

concatenated_df=read_and_concatenate_excel_files()

print(f"Successfully concatenated {len(concatenated_df['source_file'].unique())} files")
print(f"Final DataFrame shape: {concatenated_df.shape}")
df=concatenated_df

scenarios = {
    "GoldsteinScale": "df['GoldsteinScale']",
    "NumArticles": "df['NumArticles']",
    "AvgTone": "df['AvgTone']",
    "AvgTone_X_NumArticles": "df['AvgTone']*df['NumArticles']",
    "AvgTone_X_NumArticles_X_GoldsteinScale": "df['AvgTone']*df['NumArticles']*df['GoldsteinScale']",
    "AvgTone_RollingMean": "df['AvgTone'].rolling(window, min_periods=0).mean()",
    "GoldsteinScale_RollingMean": "df['GoldsteinScale'].rolling(window, min_periods=0).mean()",
    "AvgTone_RollingMedian": "df['AvgTone'].rolling(window, min_periods=0).median()",
    "GoldsteinScale_RollingMedian": "df['GoldsteinScale'].rolling(window, min_periods=0).median()",
}

aggregations = ["mean", "median", "sum"]

all_results = []
run_data_cache = {}

for scenario_name, scenario_formula in scenarios.items():
    for aggregation in aggregations:
        print(f"Running scenario: {scenario_name} with aggregation: {aggregation}")
        
        # This is the former `run_analysis` function, now inlined
        # -------------------------
        # PREPARE DAILY SERIES
        # -------------------------
        df_copy = df.copy()
        df_copy["Date"] = pd.to_datetime(df_copy["Date"])
        df_copy = df_copy.sort_values("Date").reset_index(drop=True)

        attacks_copy = attacks.copy()
        attacks_copy["event_date"] = pd.to_datetime(attacks_copy["event_date"])
        attacks_copy = attacks_copy.sort_values("event_date").reset_index(drop=True)

        window = min(28, len(df_copy))

        df_copy['Tone_Article_ZScore'] = eval(scenario_formula, {'df': df_copy, 'window': window})

        tone_daily = df_copy.groupby("Date", as_index=False).agg({"Tone_Article_ZScore": aggregation})
        attacks_daily = attacks_copy.groupby("event_date", as_index=False).agg({"event_count": "sum"}).rename(
            columns={"event_date": "Date"}
        )

        start_date = min(tone_daily["Date"].min(), attacks_daily["Date"].min())
        end_date = max(tone_daily["Date"].max(), attacks_daily["Date"].max())
        all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
        daily = pd.DataFrame({"Date": all_dates})

        daily = daily.merge(tone_daily.rename(columns={"Tone_Article_ZScore": "Global_Daily_AvgTone_Sum"}), on="Date", how="outer")
        daily = daily.merge(attacks_daily.rename(columns={"event_count": "Global_Event_Count_Sum"}), on="Date", how="outer")

        daily["Global_Event_Count_Sum"] = daily["Global_Event_Count_Sum"].fillna(0)
        daily["Event_Occurred"] = (daily["Global_Event_Count_Sum"] > 0).astype(int)
        
        daily = daily.sort_values("Date").reset_index(drop=True)
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

        feature_cols = [
            "Global_Daily_AvgTone_Sum", "Tone_MA_28", "Tone_MA_7", "Tone_MA_14",
            "Tone_Std_7", "Tone_Std_14", "Tone_Momentum_7", "Tone_Rate_of_Change",
            "Tone_Range_7", "Tone_Lag_1", "Tone_Lag_2", "Tone_Lag_3",
            "Event_Lag_1", "Event_Lag_3", "Day_of_Week", "Month"
        ]
        daily_model = daily.dropna(subset=feature_cols).copy()

        TRAIN_START = "2018-01-01"
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

            if (len(y_train) == 0) and (len(y_test) == 0):
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

                    optimal_threshold, best_precision, best_recall, best_f1 = adaptive_threshold_optimization(y_test, y_proba, horizon)
                    y_proba_smoothed = apply_temporal_smoothing(y_proba, window=3)
                    y_pred = (y_proba_smoothed >= optimal_threshold).astype(int)
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

                    row.update({"status": "ok", "opt_threshold": float(optimal_threshold), "auc": float(auc) if not np.isnan(auc) else None, "precision": float(precision_t), "recall": float(recall_t), "f1": float(f1), "TP": TP, "FP": FP, "FN": FN, "TN": TN, "applied_threshold": float(optimal_threshold)})
                    trained_models[horizon][model_name] = trained_obj
                except Exception as e:
                    row.update({"status": "error", "error": str(e)})
                results_summary_rows.append(row)
        
        for row in results_summary_rows:
            row['scenario'] = scenario_name
            row['aggregation'] = aggregation
        all_results.extend(results_summary_rows)
        
        run_key = (scenario_name, aggregation)
        run_data_cache[run_key] = {'trained_models': trained_models, 'test_df': test_df, 'feature_cols': feature_cols, 'daily': daily, 'train_end_dt': train_end_dt, 'daily_model': daily_model}

results_df = pd.DataFrame(all_results)
results_df['precision'] = pd.to_numeric(results_df['precision'], errors='coerce')
results_df['f1'] = pd.to_numeric(results_df['f1'], errors='coerce')
filtered_results_df = results_df[(results_df['precision'] > 0.55) | (results_df['f1'] > 0.55)].copy()

OUT_DIR = "analysis/outputs"
ensure_dir(OUT_DIR)
results_csv = os.path.join(OUT_DIR, name+"_detailed_results.xlsx")
filtered_results_df.to_excel(results_csv, index=False)

manual_threshold_results = []
best_3day_model = results_df[results_df['horizon'] == '3-day'].nlargest(1, 'f1')
best_7day_model = results_df[results_df['horizon'] == '7-day'].nlargest(1, 'f1')

best_models_to_run = []
if not best_3day_model.empty:
    best_models_to_run.append(best_3day_model.iloc[0])
if not best_7day_model.empty:
    best_models_to_run.append(best_7day_model.iloc[0])

for best_model_row in best_models_to_run:
    scenario = best_model_row['scenario']
    aggregation = best_model_row['aggregation']
    run_data = run_data_cache[(scenario, aggregation)]
    manual_predictions = apply_models_with_manual_threshold(run_data['trained_models'], run_data['test_df'], run_data['feature_cols'], 0.5)
    manual_threshold_results.append(manual_predictions)

if manual_threshold_results:
    final_manual_results = pd.concat(manual_threshold_results, axis=1)
    final_manual_results = final_manual_results.loc[:,~final_manual_results.columns.duplicated()]
    manual_results_csv = os.path.join(OUT_DIR, name+"_manual_threshold_results.xlsx")
    final_manual_results.to_excel(manual_results_csv, index=False)
