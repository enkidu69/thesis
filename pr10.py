import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from imblearn.over_sampling import SMOTE
import warnings
import os
import glob
import pickle
import random

# Try importing XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed. Skipping XGB models.")

warnings.filterwarnings('ignore')

# ==============================================================================
# GLOBAL SEED CONTROL
# ==============================================================================
SEED = 42

def set_seed(seed=42):
    """Sets the seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Apply seed immediately
set_seed(SEED)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def _save_csv(df, path):
    ensure_dir(os.path.dirname(path))
    df.to_excel(path, index=False)

def apply_temporal_smoothing(probabilities, window=3):
    # center=False prevents future leakage
    return pd.Series(probabilities).rolling(window=window, center=False, min_periods=1).mean().values

def calculate_lookback_coverage(alert_series, horizon_days):
    """
    CORE LOGIC FUNCTION:
    Calculates whether a day is 'covered' by an alert in the PRIOR x days.
    Logic: Shift(1) to exclude today, then Roll(horizon) to sum previous days.
    """
    return alert_series.shift(1).rolling(window=horizon_days, min_periods=1).sum().fillna(0)

def calculate_row_based_metrics_with_lookback(df, horizon_days):
    """
    Applies user-specific logic for row-by-row classification using shared core logic.
    """
    df = df.copy()
    
    # Use the shared core logic function
    df['Prior_Alerts_Sum'] = calculate_lookback_coverage(df['Predicted_Alert'], horizon_days)
    
    # Define Masks
    mask_event = df['Event_Occurred'] > 0
    mask_no_event = df['Event_Occurred'] == 0
    mask_alert_active = df['Prior_Alerts_Sum'] > 0
    mask_no_alert = df['Prior_Alerts_Sum'] == 0
    
    # Assign Result Types
    df['Result_Type'] = 'ERR' # Default
    df.loc[mask_event & mask_alert_active, 'Result_Type'] = 'TP'
    df.loc[mask_event & mask_no_alert, 'Result_Type'] = 'FN'
    df.loc[mask_no_event & mask_alert_active, 'Result_Type'] = 'FP'
    df.loc[mask_no_event & mask_no_alert, 'Result_Type'] = 'TN'
    
    return df

def calculate_coverage_metrics(y_true_daily, y_pred_alerts, horizon_days):
    """
    Metric Calculator used for Optimization.
    Uses the exact same logic as the Custom Report.
    """
    alert_series = pd.Series(y_pred_alerts)
    
    # Use the shared core logic function
    active_alerts_sum = calculate_lookback_coverage(alert_series, horizon_days)
    
    # If prior alerts > 0, we predict an event (Coverage = 1)
    y_pred_coverage = (active_alerts_sum > 0).astype(int)
    
    # Align indices
    y_true = y_true_daily.reset_index(drop=True)
    y_pred = y_pred_coverage.reset_index(drop=True)
    
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    
    try:
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    except:
        precision = recall = f1 = 0.0
        
    return TP, FP, FN, TN, precision, recall, f1, y_pred_coverage

def optimize_threshold_for_coverage(y_true_daily, y_proba_smoothed, horizon_days):
    """
    Iterates through potential thresholds to maximize F1 coverage.
    """
    best_threshold = 0.5
    best_f1 = -1
    # Check thresholds
    thresholds = np.arange(0.05, 0.96, 0.01)
    
    for thresh in thresholds:
        y_pred_binary = (y_proba_smoothed >= thresh).astype(int)
        _, _, _, _, _, _, f1, _ = calculate_coverage_metrics(y_true_daily, y_pred_binary, horizon_days)
        
        # Optimization logic: strictly maximize F1
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            
    return best_threshold

def train_with_smote(X_train, y_train, model):
    if len(np.unique(y_train)) < 2: return model.fit(X_train, y_train)
    pos_ratio = y_train.sum() / len(y_train)
    if pos_ratio < 0.1:
        try:
            # Random State applied for consistency
            smote = SMOTE(random_state=SEED, k_neighbors=min(5, y_train.sum()-1))
            X_res, y_res = smote.fit_resample(X_train, y_train)
            return model.fit(X_res, y_res)
        except: return model.fit(X_train, y_train)
    else: return model.fit(X_train, y_train)

def read_and_concatenate_excel_files():
    folder_path = "analysis"
    if not os.path.exists(folder_path): raise FileNotFoundError(f"Folder '{folder_path}' does not exist")
    excel_files = glob.glob(os.path.join(folder_path, "aggregated*.xlsx")) + glob.glob(os.path.join(folder_path, "singular*.xlsx"))
    if not excel_files: raise FileNotFoundError(f"No Excel files found in '{folder_path}'")
    dataframes = []
    for file_path in excel_files:
        df = pd.read_excel(file_path)
        dataframes.append(df)
        print(f"Loaded: {os.path.basename(file_path)}")
    return pd.concat(dataframes, ignore_index=True)

# ==============================================================================
# MAIN PROCESSING
# ==============================================================================

# 1. Load Data
desk = os.getcwd()
path = os.path.join(desk, 'analysis')
attack_files = glob.glob(os.path.join(path, 'cyberevents*.xlsx'))
if not attack_files: raise FileNotFoundError("No attack files found.")
attacks_list = [pd.read_excel(f) for f in attack_files]
attacks = pd.concat(attacks_list, ignore_index=True)

name_files = glob.glob(os.path.join(path, '*cyber_events*.xlsx'))
name = os.path.basename(name_files[0]).replace('_cyber_events.xlsx', '') if name_files else "run_results"

df = read_and_concatenate_excel_files()

# 2. Prepare Base Timeline
attacks["event_date"] = pd.to_datetime(attacks["event_date"])
attacks_daily = attacks.groupby("event_date", as_index=False).agg({"event_count": "sum"}).rename(columns={"event_date": "Date", "event_count": "Global_Event_Count_Sum"})

df["Date"] = pd.to_datetime(df["Date"])
start_date = min(df["Date"].min(), attacks_daily["Date"].min())
end_date = max(df["Date"].max(), attacks_daily["Date"].max())
base_daily_df = pd.DataFrame({"Date": pd.date_range(start=start_date, end=end_date, freq="D")})
base_daily_df = base_daily_df.merge(attacks_daily, on="Date", how="outer")
base_daily_df["Global_Event_Count_Sum"] = base_daily_df["Global_Event_Count_Sum"].fillna(0)
base_daily_df["Event_Occurred"] = (base_daily_df["Global_Event_Count_Sum"] > 0).astype(int)
base_daily_df = base_daily_df.sort_values("Date").reset_index(drop=True)

# Targets
e_shifts = [base_daily_df["Event_Occurred"].shift(-i).fillna(0).astype(int) for i in range(1, 8)]
base_daily_df["Event_Next_1D"] = base_daily_df["Event_Occurred"].shift(-1).fillna(0).astype(int)
base_daily_df["Event_Next_3D"] = (sum(e_shifts[:3]) > 0).astype(int)
base_daily_df["Event_Next_7D"] = (sum(e_shifts) > 0).astype(int)

# Daily Article Counts and Z-Score
daily_articles = df.groupby("Date", as_index=False)["NumArticles"].sum()
base_daily_df = base_daily_df.merge(daily_articles, on="Date", how="left").fillna(0)
base_daily_df["Article_Count_ZScore"] = (base_daily_df["NumArticles"] - base_daily_df["NumArticles"].rolling(30).mean()) / base_daily_df["NumArticles"].rolling(30).std()
base_daily_df["Article_Count_ZScore"] = base_daily_df["Article_Count_ZScore"].fillna(0)

# 3. Scenarios
scenarios = {
    "GoldsteinScale": "df['GoldsteinScale']",
    "NumArticles": "df['NumArticles']",
    "AvgTone": "df['AvgTone']",
    "AvgTone_X_NumArticles": "df['AvgTone']*df['NumArticles']",
    "AvgTone_X_NumArticles_X_GoldsteinScale": "df['AvgTone']*df['NumArticles']*df['GoldsteinScale']",
    "AvgTone_RollingMean": "df['AvgTone'].rolling(window, min_periods=1).mean()",
    "NumArticles_RollingMean": "df['NumArticles'].rolling(window, min_periods=1).mean()",
    "NumArticles_RollingMedian": "df['NumArticles'].rolling(window, min_periods=1).median()",
    "GoldsteinScale_RollingMean": "df['GoldsteinScale'].rolling(window, min_periods=1).mean()",
    "AvgTone_RollingMedian": "df['AvgTone'].rolling(window, min_periods=1).median()",
    "GoldsteinScale_RollingMedian": "df['GoldsteinScale'].rolling(window, min_periods=1).median()",
    "AvgTone_X_NumArticles_RollingMean": "(df['AvgTone']*df['NumArticles']).rolling(window, min_periods=1).mean()",
    "AvgTone_X_NumArticles_RollingMedian": "(df['AvgTone']*df['NumArticles']).rolling(window, min_periods=1).median()",
    "AvgTone_X_NumArticles_X_GoldsteinScale_RollingMean": "(df['AvgTone']*df['NumArticles']*df['GoldsteinScale']).rolling(window, min_periods=1).mean()",
    "AvgTone_X_NumArticles_X_GoldsteinScale_RollingMedian": "(df['AvgTone']*df['NumArticles']*df['GoldsteinScale']).rolling(window, min_periods=1).median()",
    # Z-SCORE SCENARIOS
    "GoldsteinScale_Zscore": "((df['GoldsteinScale'] - df['GoldsteinScale'].rolling(window, min_periods=1).mean()) / df['GoldsteinScale'].rolling(window, min_periods=1).std())",
    "AvgTone_Zscore": "((df['AvgTone'] - df['AvgTone'].rolling(window, min_periods=1).mean()) / df['AvgTone'].rolling(window, min_periods=1).std())",
    "AvgTone_X_NumArticles_Zscore": "((df['AvgTone']*df['NumArticles']) - (df['AvgTone']*df['NumArticles']).rolling(window, min_periods=1).mean()) / (df['AvgTone']*df['NumArticles']).rolling(window, min_periods=1).std()",
    "AvgTone_X_NumArticles_X_GoldsteinScale_Zscore": "((df['AvgTone']*df['NumArticles']*df['GoldsteinScale']) - (df['AvgTone']*df['NumArticles']*df['GoldsteinScale']).rolling(window, min_periods=1).mean()) / (df['AvgTone']*df['NumArticles']*df['GoldsteinScale']).rolling(window, min_periods=1).std()",
    "NumArticles_Zscore": "((df['NumArticles']) - (df['NumArticles']).rolling(window, min_periods=1).mean()) / (df['NumArticles']).rolling(window, min_periods=1).std()",
    "NumMentions_Zscore": "((df['NumMentions']) - (df['NumMentions']).rolling(window, min_periods=1).mean()) / (df['NumMentions']).rolling(window, min_periods=1).std()",
    "NumMentions": "df['NumMentions']",
    "NumSources_Zscore": "((df['NumSources']) - (df['NumSources']).rolling(window, min_periods=1).mean()) / (df['NumSources']).rolling(window, min_periods=1).std()",
    "NumSources": "df['NumSources']",
    "NumSourcesXNumMentionsXNumArticles_Zscore": "((df['NumSources']*df['NumMentions']*df['NumArticles']) - (df['NumSources']*df['NumMentions']*df['NumArticles']).rolling(window, min_periods=1).mean()) / (df['NumSources']*df['NumMentions']*df['NumArticles']).rolling(window, min_periods=1).std()",
    "NumSourcesXNumMentionsXNumArticles": "df['NumSources']*df['NumMentions']*df['NumArticles']",
    "NumMentions_RollingMean": "df['NumMentions'].rolling(window, min_periods=1).mean()",
    "NumSources_RollingMean": "df['NumSources'].rolling(window, min_periods=1).mean()",
    "NumSourcesXNumMentionsXNumArticlesXAvgTone_RollingMean": "(df['NumSources']*df['NumMentions']*df['NumArticles']*df['AvgTone']).rolling(window, min_periods=1).mean()"
}
aggregations = ["sum", "mean", "median"]

all_results = []
run_data_cache = {}

print("\n>>> STARTING TRAINING LOOP <<<")

for scenario_name, scenario_formula in scenarios.items():
    for aggregation in aggregations:
        print(f"Processing: {scenario_name} ({aggregation})")
        
        daily = base_daily_df.copy()
        df_scenario = df.copy()
        window = min(30, len(df_scenario))
        
        try:
            df_scenario['Tone_Article_ZScore'] = eval(scenario_formula, {'df': df_scenario, 'window': window})
        except Exception as e:
            all_results.append({"scenario": scenario_name, "aggregation": aggregation, "horizon": "N/A", "model": "N/A", "status": "error", "error": f"Formula Eval Error: {str(e)}"})
            continue

        tone_daily = df_scenario.groupby("Date", as_index=False).agg({"Tone_Article_ZScore": aggregation})
        daily = daily.merge(tone_daily.rename(columns={"Tone_Article_ZScore": "Global_Daily_AvgTone_Sum"}), on="Date", how="outer")
        
        daily["Global_Daily_AvgTone_Sum"] = daily["Global_Daily_AvgTone_Sum"].fillna(0)
        daily["Tone_MA_7"] = daily["Global_Daily_AvgTone_Sum"].rolling(7, min_periods=1).mean().fillna(0)
        daily["Tone_Std_7"] = daily["Global_Daily_AvgTone_Sum"].rolling(7, min_periods=1).std().fillna(0)
        daily["Tone_Rate_of_Change"] = daily["Tone_MA_7"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        
        feature_cols = ["Global_Daily_AvgTone_Sum", "Tone_MA_7", "Tone_Std_7", "Tone_Rate_of_Change", "Article_Count_ZScore"]
        
        train_df = daily[(daily["Date"] >= "2018-01-01") & (daily["Date"] <= "2024-12-31")].copy()
        test_df = daily[daily["Date"] > "2024-12-31"].copy()
        
        if len(train_df) == 0 or len(test_df) == 0: 
            all_results.append({"scenario": scenario_name, "aggregation": aggregation, "horizon": "N/A", "model": "N/A", "status": "error", "error": "Insufficient Data"})
            continue
        
        # Models with Seed Control
        base_models = {
            "Logistic Regression": LogisticRegressionCV(class_weight="balanced", cv=TimeSeriesSplit(n_splits=5), max_iter=1000, random_state=SEED),
            "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", max_depth=8, random_state=SEED),
            "Isolation Forest": IsolationForest(contamination=0.1, random_state=SEED)
        }
        
        if XGB_AVAILABLE:
            base_models["XGBoost"] = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=1, random_state=SEED, eval_metric="logloss")
        
        trained_models_horizon = {}
        
        for target_col, horizon in [("Event_Next_1D", "1-day"), ("Event_Next_3D", "3-day"), ("Event_Next_7D", "7-day")]:
            horizon_int = int(horizon.split('-')[0])
            trained_models_horizon[horizon] = {}
            
            y_train = train_df[target_col]
            y_test_target = test_df[target_col]
            X_train = train_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
            X_test = test_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
            y_train_daily = train_df["Event_Occurred"]
            
            if len(np.unique(y_train)) < 2: 
                all_results.append({"scenario": scenario_name, "aggregation": aggregation, "horizon": horizon, "model": "N/A", "status": "error", "error": "Train data has only 1 class"})
                continue
            
            if "XGBoost" in base_models and len(np.unique(y_train)) >= 2:
                pos_ratio = y_train.sum() / len(y_train)
                if pos_ratio > 0:
                    scale_pos_weight = ((len(y_train) - y_train.sum()) / y_train.sum()) * 0.2
                    base_models["XGBoost"].set_params(scale_pos_weight=scale_pos_weight)

            for model_name, base_model in base_models.items():
                try:
                    model = clone(base_model)
                    
                    if model_name == "Isolation Forest":
                        model.fit(X_train)
                        # Anomaly scoring: lower = abnormal. Negate so higher = abnormal (like prob)
                        y_train_raw = -model.decision_function(X_train)
                        y_test_raw = -model.decision_function(X_test)
                        scaler = MinMaxScaler()
                        y_train_proba = scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
                        y_test_proba = scaler.transform(y_test_raw.reshape(-1, 1)).flatten()
                        trained_obj = ("pipeline", scaler, model)
                    elif model_name == "Logistic Regression":
                        scaler = StandardScaler()
                        X_train_s = scaler.fit_transform(X_train)
                        X_test_s = scaler.transform(X_test)
                        model = train_with_smote(X_train_s, y_train, model)
                        y_train_proba = model.predict_proba(X_train_s)[:, 1]
                        y_test_proba = model.predict_proba(X_test_s)[:, 1]
                        trained_obj = ("pipeline", scaler, model)
                    elif model_name == "XGBoost":
                        model = train_with_smote(X_train, y_train, model)
                        y_train_proba = model.predict_proba(X_train.values)[:, 1]
                        y_test_proba = model.predict_proba(X_test.values)[:, 1]
                        trained_obj = model
                    else:
                        model = train_with_smote(X_train, y_train, model)
                        y_train_proba = model.predict_proba(X_train)[:, 1]
                        y_test_proba = model.predict_proba(X_test)[:, 1]
                        trained_obj = model
                    
                    y_train_smooth = apply_temporal_smoothing(y_train_proba)
                    y_test_smooth = apply_temporal_smoothing(y_test_proba)
                    
                    # Optimize Threshold using PRIOR lookback logic
                    opt_thresh = optimize_threshold_for_coverage(y_train_daily, y_train_smooth, horizon_int)
                    y_pred_binary = (y_test_smooth >= opt_thresh).astype(int)
                    
                    y_true_daily_test = test_df["Event_Occurred"]
                    TP, FP, FN, TN, prec, rec, f1, coverage_mask = calculate_coverage_metrics(y_true_daily_test, y_pred_binary, horizon_int)
                    
                    # --- NEW: Calculate AUPRC ---
                    try:
                        auprc_val = average_precision_score(y_test_target, y_test_smooth)
                    except:
                        auprc_val = 0.0

                    try: auc_val = roc_auc_score(y_test_target, y_test_smooth)
                    except: auc_val = 0.5
                    
                    status = "ok" if auc_val >= 0.6 else "filtered_low_auc"
                    if status == "ok":
                         print(f"  [OK] {horizon} {model_name}: AUC={auc_val:.3f}, AUPRC={auprc_val:.3f}, F1={f1:.3f}")
                    else:
                         print(f"  [Skip] {horizon} {model_name}: Low AUC ({auc_val:.3f})")

                    all_results.append({
                        "scenario": scenario_name, "aggregation": aggregation, "horizon": horizon, "model": model_name,
                        "status": status, "auc": auc_val, "auprc": auprc_val, "f1": f1, "precision": prec, "recall": rec,
                        "TP": TP, "FP": FP, "FN": FN, "TN": TN, "opt_threshold": opt_thresh
                    })
                    trained_models_horizon[horizon][model_name] = trained_obj
                    
                except Exception as e:
                    all_results.append({"scenario": scenario_name, "aggregation": aggregation, "horizon": horizon, "model": model_name, "status": "error", "error": str(e)})
                    print(f"  Error {model_name}: {e}")
                    
        run_data_cache[(scenario_name, aggregation)] = {'trained_models': trained_models_horizon, 'test_df': test_df, 'feature_cols': feature_cols}

# 4. Save Results
results_df = pd.DataFrame(all_results)
OUT_DIR = "analysis/outputs"
ensure_dir(OUT_DIR)
_save_csv(results_df, os.path.join(OUT_DIR, f"detailed_results_coverage_based_{name}.xlsx"))

# 5. Alert Generation Functions (With ALL columns + Row Logic)
def generate_alert_table_full(run_data_cache, scenario, aggregation, horizon, model_name, threshold, output_dir, file_prefix):
    cache_key = (scenario, aggregation)
    if cache_key not in run_data_cache: 
        print(f"  Error: {cache_key} not found in cache.")
        return
    
    data = run_data_cache[cache_key]
    test_df = data['test_df'].copy() 
    
    if horizon not in data['trained_models'] or model_name not in data['trained_models'][horizon]:
        print(f"  Error: Model {model_name} for {horizon} not found.")
        return

    model_obj = data['trained_models'][horizon][model_name]
    feature_cols = data['feature_cols']
    
    # ---------------------------------------------------------
    # CORRECTED PREDICTION LOGIC FOR ISOLATION FOREST
    # ---------------------------------------------------------
    if isinstance(model_obj, tuple):
        _, scaler, clf = model_obj
        
        if isinstance(clf, IsolationForest):
            # IsoForest: Raw Features -> Decision Function -> Scaler
            X_test_raw = test_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
            raw_scores = -clf.decision_function(X_test_raw)
            probs = scaler.transform(raw_scores.reshape(-1, 1)).flatten()
        elif model_obj[0] == "pipeline":
             # LR: Scaler -> Predict Proba
            X_test_scaled = scaler.transform(test_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0))
            probs = clf.predict_proba(X_test_scaled)[:, 1]
        else:
             probs = np.zeros(len(test_df))
             
    elif model_name == "XGBoost":
         probs = model_obj.predict_proba(test_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0).values)[:, 1]
    else:
        probs = model_obj.predict_proba(test_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0))[:, 1]
    
    probs_smooth = apply_temporal_smoothing(probs)
    alerts = (probs_smooth >= threshold).astype(int)
    
    # Coverage Status (Using Core Logic)
    horizon_int = int(horizon.split('-')[0])
    coverage_sum = calculate_lookback_coverage(pd.Series(alerts), horizon_int)
    coverage_active = (coverage_sum > 0)
    
    test_df["Predicted_Alert"] = alerts
    test_df["Alert_Active_In_Window"] = coverage_active.astype(bool)
    test_df["Prob_Smoothed"] = probs_smooth
    test_df["Threshold_Used"] = threshold
    test_df["Model_Used"] = model_name
    
    # --- NEW: Calculate AUPRC for this specific run ---
    # Find target column
    target_col = f"Event_Next_{horizon_int}D"
    if target_col in test_df.columns:
        try:
            run_auprc = average_precision_score(test_df[target_col], probs_smooth)
        except:
            run_auprc = 0.0
    else:
        run_auprc = 0.0
    
    # ADD TO SPREADSHEET
    test_df['Run_AUPRC'] = run_auprc
    
    # --- APPLY ROW-BASED LOGIC ---
    test_df = calculate_row_based_metrics_with_lookback(test_df, horizon_int)
    
    # --- REPORT STATS ---
    counts = test_df['Result_Type'].value_counts()
    print(f"  Stats for {model_name} ({horizon}):")
    print(f"    TP: {counts.get('TP', 0)}")
    print(f"    FP: {counts.get('FP', 0)}")
    print(f"    FN: {counts.get('FN', 0)}")
    print(f"    TN: {counts.get('TN', 0)}")
    print(f"    Run AUPRC: {run_auprc:.4f}")
    
    # Save
    fname = f"{file_prefix}_{scenario}_{aggregation}_{horizon}_{model_name}.xlsx"
    _save_csv(test_df, os.path.join(output_dir, fname))
    print(f"  Saved full alert table: {fname}")

# Generate Best Model Alerts (NOW SELECTS BY AUPRC)
if not results_df.empty:
    valid_results = results_df[results_df['status'] == 'ok']
    if not valid_results.empty:
        # --- CHANGED SELECTION CRITERIA TO AUPRC ---
        best_model = valid_results.loc[valid_results['auprc'].idxmax()]
        print(f"\nBest Model by AUPRC: {best_model['scenario']} {best_model['model']} (AUPRC={best_model['auprc']:.3f}, F1={best_model['f1']:.3f})")
        generate_alert_table_full(
            run_data_cache, 
            best_model['scenario'], 
            best_model['aggregation'], 
            best_model['horizon'], 
            best_model['model'], 
            best_model['opt_threshold'], 
            OUT_DIR, 
            "best_model_alerts"
        )
    else:
        print("\nNo models passed the AUC >= 0.6 filter.")

# ==============================================================================
# 6. CUSTOM SCENARIO RUNNER
# ==============================================================================
print("\n>>> RUNNING CUSTOM SCENARIO <<<")

CUSTOM_SCENARIO = "NumArticles_RollingMedian"         
CUSTOM_AGG      = "sum"                  
CUSTOM_HORIZON  = "7-day"                
CUSTOM_MODEL    = "Random Forest"              
CUSTOM_THRESH   = 0.54                   

generate_alert_table_full(
    run_data_cache, 
    CUSTOM_SCENARIO, 
    CUSTOM_AGG, 
    CUSTOM_HORIZON, 
    CUSTOM_MODEL, 
    CUSTOM_THRESH, 
    OUT_DIR, 
    "CUSTOM_RUN"
)