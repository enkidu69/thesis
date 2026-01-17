import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from imblearn.over_sampling import SMOTE
import warnings
import os
import glob
import pickle

# Try importing XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not installed. Skipping XGB models.")

warnings.filterwarnings('ignore')

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

def calculate_row_based_metrics_with_lookback(df, horizon_int):
    """
    Applies Forward-Looking logic for row-by-row classification.
    - Alert Active: Did we predict an alert recently?
    - Event Imminent: Is an event actually coming up (or happening now)?
    
    Returns: DataFrame with correct 'Result_Type' where early warnings are TPs.
    """
    df = df.copy()
    
    # 1. Define the Window
    window_size = horizon_int + 1

    # 2. Determine if we are "Under Alert" (Coverage)
    # If I predicted an alert today or in the last 'horizon' days, the alarm is ON.
    df['Alert_Active'] = df['Predicted_Alert'].rolling(window=window_size, min_periods=1).sum() > 0
    
    # 3. Determine if an Event is "Valid" (Imminent)
    # We use the Target column corresponding to the horizon.
    # If horizon is 3, we look at 'Event_Next_3D'.
    target_col = f'Event_Next_{horizon_int}D'
    
    event_occurred = df['Event_Occurred']==1
    not_event_occurred = df['Event_Occurred']==0

    
    mask_event_coming = df[target_col] == 1
    mask_no_event_coming = df[target_col] == 0
    
    mask_alert_on = df['Alert_Active'] == True
    mask_alert_off = df['Alert_Active'] == False
    
    predicted = df['Predicted_Alert'] == 1
    not_predicted = df['Predicted_Alert'] == 0
    # 4. Assign Result Types
    df['Result_Type'] = 'ERR' 

    # TP: The Alarm is ON, and an Event is Coming (or here)
    df.loc[mask_alert_on & mask_event_coming, 'Result_Type'] = 'TP'


    # FP: The Alarm is ON, but NO Event is coming
    df.loc[mask_alert_on & mask_no_event_coming, 'Result_Type'] = 'FP'

    # FN: The Alarm is OFF, but an Event IS Coming
    df.loc[mask_alert_off & mask_event_coming, 'Result_Type'] = 'FN'

    # TN: The Alarm is OFF, and NO Event is coming
    df.loc[mask_alert_off & mask_no_event_coming, 'Result_Type'] = 'TN'
    
        #TP mask alert is on and event is present current override FPs 
    df.loc[mask_alert_on & event_occurred, 'Result_Type'] = 'TP'


    return df

def calculate_coverage_metrics(y_true_daily, y_pred_alerts, horizon_days, y_target):
    """
    Standard Coverage Metric Calculator.
    - y_true_daily: Actual events happening TODAY (df['Event_Occurred'])
    - y_pred_alerts: Raw predicted alerts (df['Predicted_Alert'])
    - y_target: Future target events (df['Event_Next_XD'])
    """
    # 1. Setup Window
    window_size = horizon_days + 1
    
    # 2. Calculate Prediction Coverage (Alert Active)
    alert_series = pd.Series(y_pred_alerts)
    active_alerts_sum = alert_series.rolling(window=window_size, min_periods=1).sum()
    y_pred_coverage = (active_alerts_sum > 0).astype(int)
    
    # 3. Align Indices
    y_true_today = y_true_daily.reset_index(drop=True)
    y_target_future = y_target.reset_index(drop=True)
    y_pred = y_pred_coverage.reset_index(drop=True)
    
    # ---------------------------------------------------------
    # 4. APPLY THE OVERRIDE LOGIC HERE
    # The "Truth" is 1 if an event is in the Future OR Today.
    # This matches: df.loc[mask_alert_on & event_occurred, 'Result_Type'] = 'TP'
    # ---------------------------------------------------------
    y_truth_inclusive = ((y_true_today == 1) | (y_target_future == 1)).astype(int)

    # 5. Calculate Metrics using the Inclusive Truth
    TP = int(((y_pred == 1) & (y_truth_inclusive == 1)).sum())
    FP = int(((y_pred == 1) & (y_truth_inclusive == 0)).sum())
    FN = int(((y_pred == 0) & (y_truth_inclusive == 1)).sum())
    TN = int(((y_pred == 0) & (y_truth_inclusive == 0)).sum())
    
    # 6. Calculate Rates
    try:
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    except:
        precision = recall = f1 = 0.0
        
    return TP, FP, FN, TN, precision, recall, f1, y_pred_coverage
    
def optimize_threshold_for_coverage(y_true_daily, y_proba_smoothed, horizon_days,y_target):
    """
    Iterates through potential thresholds to maximize F1 coverage.
    """
    best_threshold = 0.5
    best_f1 = -1
    thresholds = np.arange(0.05, 0.96, 0.01)
    
    for thresh in thresholds:
        y_pred_binary = (y_proba_smoothed >= thresh).astype(int)
        _, _, _, _, _, _, f1, _ = calculate_coverage_metrics(y_true_daily, y_pred_binary, horizon_days, y_target)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            
    return best_threshold

def train_with_smote(X_train, y_train, model):
    if len(np.unique(y_train)) < 2: return model.fit(X_train, y_train)
    pos_ratio = y_train.sum() / len(y_train)
    if pos_ratio < 0.1:
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, y_train.sum()-1))
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
    "GoldsteinScale_RollingMean": "df['GoldsteinScale'].rolling(window, min_periods=1).mean()"
    
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
        
        base_models = {
            "Logistic Regression": LogisticRegressionCV(class_weight="balanced", cv=TimeSeriesSplit(n_splits=5), max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", max_depth=8),
            "Isolation Forest": IsolationForest(contamination=0.1, random_state=42)
        }
        
        if XGB_AVAILABLE:
            base_models["XGBoost"] = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=1, random_state=42, eval_metric="logloss")
        
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
                    
                    opt_thresh = optimize_threshold_for_coverage(y_train_daily, y_train_smooth, horizon_int, y_train) # <--- Passed y_train
                    y_pred_binary = (y_test_smooth >= opt_thresh).astype(int)
                    
                    y_true_daily_test = test_df["Event_Occurred"]
                    #y_target = test_df[target_col]
                
                    #calc metrics to be fixed
                    #print(y_target)
                    #exit()
                    y_target_inclusive = ((test_df['Event_Occurred'] == 1) | (test_df[target_col] == 1)).astype(int)
                    y_target=y_target_inclusive
                    TP, FP, FN, TN, prec, rec, f1, coverage_mask = calculate_coverage_metrics(
                        y_true_daily_test, 
                        y_pred_binary, 
                        horizon_int, 
                        y_target_inclusive  # <--- The fix
)                    
                    try: auc_val = roc_auc_score(y_test_target, y_test_smooth)
                    except: auc_val = 0.5
                    
                    status = "ok" if auc_val >= 0.6 else "filtered_low_auc"
                    if status == "ok":
                         print(f"  [OK] {horizon} {model_name}: AUC={auc_val:.3f}, F1={f1:.3f}")
                    else:
                         print(f"  [Skip] {horizon} {model_name}: Low AUC ({auc_val:.3f})")

                    all_results.append({
                        "scenario": scenario_name, "aggregation": aggregation, "horizon": horizon, "model": model_name,
                        "status": status, "auc": auc_val, "f1": f1, "precision": prec, "recall": rec,
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
    
    if isinstance(model_obj, tuple):
        tag = model_obj[0]
        if tag == "pipeline":
            _, scaler, clf = model_obj
            probs_raw = clf.predict_proba(scaler.transform(test_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)))[:, 1]
            probs = probs_raw
        elif isinstance(model_obj[2], IsolationForest):
            _, scaler, clf = model_obj
            raw_scores = -clf.decision_function(test_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0))
            probs = scaler.transform(raw_scores.reshape(-1, 1)).flatten()
        else:
            probs = np.zeros(len(test_df))
    elif model_name == "XGBoost":
         probs = model_obj.predict_proba(test_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0).values)[:, 1]
    else:
        probs = model_obj.predict_proba(test_df[feature_cols].replace([np.inf, -np.inf], 0).fillna(0))[:, 1]
    
    probs_smooth = apply_temporal_smoothing(probs)
    alerts = (probs_smooth >= threshold).astype(int)
    
    horizon_int = int(horizon.split('-')[0])
    window_size = horizon_int + 1
    coverage_active = pd.Series(alerts).rolling(window=window_size, min_periods=1).sum() > 0
    
    test_df["Predicted_Alert"] = alerts
    test_df["Alert_Active_In_Window"] = coverage_active.astype(bool)
    test_df["Prob_Smoothed"] = probs_smooth
    test_df["Threshold_Used"] = threshold
    test_df["Model_Used"] = model_name
    
    # --- APPLY NEW ROW-BASED LOGIC ---
    test_df = calculate_row_based_metrics_with_lookback(test_df, horizon_int)
    
    # --- REPORT STATS ---
    counts = test_df['Result_Type'].value_counts()
    print(f"  Stats for {model_name} ({horizon}):")
    print(f"    TP: {counts.get('TP', 0)}")
    print(f"    FP: {counts.get('FP', 0)}")
    print(f"    FN: {counts.get('FN', 0)}")
    print(f"    TN: {counts.get('TN', 0)}")
    
    # Save
    fname = f"{file_prefix}_{scenario}_{aggregation}_{horizon}_{model_name}.xlsx"
    _save_csv(test_df, os.path.join(output_dir, fname))
    print(f"  Saved full alert table: {fname}")

# Generate Best Model Alerts
if not results_df.empty:
    valid_results = results_df[results_df['status'] == 'ok']
    if not valid_results.empty:
        best_f1 = valid_results.loc[valid_results['f1'].idxmax()]
        print(f"\nBest Model by Coverage F1: {best_f1['scenario']} {best_f1['model']} (F1={best_f1['f1']:.3f})")
        generate_alert_table_full(
            run_data_cache, 
            best_f1['scenario'], 
            best_f1['aggregation'], 
            best_f1['horizon'], 
            best_f1['model'], 
            best_f1['opt_threshold'], 
            OUT_DIR, 
            "best_model_alerts"
        )
    else:
        print("\nNo models passed the AUC >= 0.6 filter.")

# ==============================================================================
# 6. CUSTOM SCENARIO RUNNER
# ==============================================================================
print("\n>>> RUNNING CUSTOM SCENARIO <<<")

CUSTOM_SCENARIO = "GoldsteinScale_RollingMean"         
CUSTOM_AGG      = "sum"                  
CUSTOM_HORIZON  = "3-day"                
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