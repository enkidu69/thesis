import pandas as pd
import yfinance as yf
import numpy as np
import time
import random
from datetime import datetime, timedelta

# --- CONFIGURATION ---
INPUT_FILE = "market_tickers.csv"
OUTPUT_FILE = "nison_expert_signals_dated.csv"

# --- 1. ROBUST DOWNLOADER ---
def safe_download(ticker, period="5y", interval="1d"):
    """
    Downloads data with retry logic to handle Rate Limits.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            time.sleep(random.uniform(0.1, 1.0)) 
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True, threads=False)
            
            if df.empty:
                return df
            
            if interval != "1d": 
                df = df.reset_index()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0] if c[0] != 'Datetime' else 'Datetime' for c in df.columns]
                    close_col = [c for c in df.columns if 'Close' in str(c)]
                    if close_col: df['Close'] = df[close_col[0]]
                df = df.set_index('Datetime')
                
            return df
        except Exception as e:
            time.sleep(2)
    return pd.DataFrame()

# --- 2. TECHNICAL CALCULATOR ---
def calculate_technicals(df):
    df = df.copy()
    
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA8'] = df['Close'].ewm(span=8, adjust=False).mean()
    
    k = df['Close'].ewm(span=12, adjust=False).mean()
    d = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = k - d
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['ROC_5'] = df['Close'].pct_change(periods=5) * 100
    
    df['Prev_Close'] = df['Close'].shift(1)
    df['tr1'] = df['High'] - df['Low']
    df['tr2'] = (df['High'] - df['Prev_Close']).abs()
    df['tr3'] = (df['Low'] - df['Prev_Close']).abs()
    df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    df.drop(['Prev_Close', 'tr1', 'tr2', 'tr3', 'TR'], axis=1, inplace=True)

    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    
    std = df['Close'].rolling(window=20).std()
    sma20 = df['Close'].rolling(window=20).mean()
    df['UpperBB'] = df['EMA21'] + (2 * std)
    df['LowerBB'] = df['EMA21'] - (2 * std)
    df['BandWidth'] = (df['UpperBB'] - df['LowerBB']) / df['EMA21'].replace(0, np.nan)
    df['Z_Score'] = (df['Close'] - sma20) / std
    
    df['AvgVol'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['Volume'] / df['AvgVol']
    df['Body'] = abs(df['Close'] - df['Open'])
    df['AvgBody'] = df['Body'].rolling(window=20).mean()
    df['UpperShadow'] = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['LowerShadow'] = df[['Close', 'Open']].min(axis=1) - df['Low']

    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['Rolling_VWAP'] = (typical_price * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
    
    df['Prev_SMA50'] = df['SMA50'].shift(1)
    df['Prev_SMA200'] = df['SMA200'].shift(1)
    df['Golden_Cross'] = (df['SMA50'] > df['SMA200']) & (df['Prev_SMA50'] <= df['Prev_SMA200'])
    
    return df

# --- 3. EXPERT SIGNALS ---
def get_paul_tudor_jones_status(df):
    if len(df) < 200: return "N/A"
    curr = df.iloc[-1]
    
    if curr['Close'] > curr['SMA200']:
        return "BULLISH (>200MA)"
    else:
        return "BEARISH (<200MA)"

def get_trend_status(df):
    if len(df) < 21: return "N/A"
    
    last_close = df['Close'].iloc[-1]
    last_ema21 = df['EMA21'].iloc[-1]
    last_macd_hist = df['MACD_Hist'].iloc[-1]
    last_rsi = df['RSI'].iloc[-1]
    
    if last_close > last_ema21 and last_macd_hist > 0 and last_rsi > 50:
        status = "STRONG GO"
    elif last_close > last_ema21 and last_macd_hist > 0:
        status = "GO"
    else:
        status = "NO GO"
        
    for i in range(len(df)-1, -1, -1):
        c = df['Close'].iloc[i]
        e = df['EMA21'].iloc[i]
        m = df['MACD_Hist'].iloc[i]
        r = df['RSI'].iloc[i]
        
        if status == "STRONG GO" and not (c > e and m > 0 and r > 50):
            return f"STRONG GO (Since {df.index[i+1].strftime('%Y-%m-%d')})"
        elif status == "GO" and not (c > e and m > 0):
            return f"GO (Since {df.index[i+1].strftime('%Y-%m-%d')})"
        elif status == "NO GO" and (c > e and m > 0):
            return f"NO GO (Since {df.index[i+1].strftime('%Y-%m-%d')})"
            
    return status

def check_buy_signal(df, squeeze_status):
    if len(df) < 20: return "NO"
    curr = df.iloc[-1]
    ema_cross_up = curr['EMA8'] > curr['EMA21']
    momentum_up = curr['ROC_5'] > 0
    rsi_healthy = 45 < curr['RSI'] < 65
    above_vwap = curr['Close'] > curr['Rolling_VWAP']
    catalyst = (curr['Vol_Ratio'] > 1.2) or ("YES" in squeeze_status)
    
    if ema_cross_up and momentum_up and rsi_healthy and above_vwap and catalyst:
        return "YES (Momentum + Fuel)"
    return "NO"

def check_divergence(df):
    if len(df) < 30: return "No"
    window = df.iloc[-30:]
    min_price_idx = window['Close'].idxmin()
    min_rsi_idx = window['RSI'].idxmin()
    curr = df.iloc[-1]
    
    if (curr.name - min_rsi_idx).days > 5 and (curr.name - min_price_idx).days < 3:
        if curr['RSI'] > window.loc[min_rsi_idx]['RSI']:
            return "YES (Bullish)"
    return "No"

def check_squeeze(df):
    if len(df) < 130: return "No"
    curr_width = df.iloc[-1]['BandWidth']
    six_month_min = df['BandWidth'].rolling(window=126).min().iloc[-1]
    if curr_width <= (six_month_min * 1.1):
        return "YES (Volatility Squeeze)"
    return "No"

def check_patterns_full(ticker, df):
    if len(df) < 30: return "None", 0
    
    c0 = df.iloc[-1]
    c1 = df.iloc[-2]
    c2 = df.iloc[-3]
    c3 = df.iloc[-4]
    c4 = df.iloc[-5]
    
    avg_body = c0['AvgBody']
    body0 = c0['Body']
    body1 = c1['Body']

    is_white = c0['Close'] > c0['Open']
    is_black = c0['Close'] < c0['Open']
    prev_white = c1['Close'] > c1['Open']
    prev_black = c1['Close'] < c1['Open']
    
    patterns = []
    
    if (c0['LowerShadow'] > 2 * body0) and (c0['UpperShadow'] < 0.2 * body0):
        if c1['Close'] < df.iloc[-10]['Close']: patterns.append("Hammer 10")
    if (c0['LowerShadow'] > 2 * body0) and (c0['UpperShadow'] < 0.2 * body0):
        if c0['Close'] < df.iloc[-3]['Close']: patterns.append("Hammer 3")
    if (c0['UpperShadow'] > 2 * body0) and (c0['LowerShadow'] < 0.2 * body0):
        if c1['Close'] < df.iloc[-10]['Close']: patterns.append("Inverted Hammer (Unconfirmed)")
    if (c1['UpperShadow'] > 2 * body1) and (c1['LowerShadow'] < 0.2 * body1):
        if c0['Close'] < df.iloc[-3]['Close'] and c0['Close']>c1['Close']: patterns.append("Confirmed Inverted Hammer")

    is_doji = body0 <= (avg_body * 0.1)
    if is_doji and (c0['Open'] >= c0['High']*0.999) and (c0['LowerShadow'] > 2 * body0):
        patterns.append("Dragonfly Doji")
    if is_white and (c0['Open'] == c0['Low']) and (body0 > avg_body * 1.5):
        patterns.append("Bullish Belt-Hold")
    if prev_black and is_white and (c0['Close'] > c1['Open']) and (c0['Open'] < c1['Close']) and c1['Close'] < df.iloc[-7]['Close']and c1['Close'] < df.iloc[-3]['Close']:
        patterns.append("Bullish Engulfing")
    if prev_black and is_white and (c0['Close'] < c1['Open']) and (c0['Open'] > c1['Close']):
        patterns.append("Bullish Harami")
    if abs(c0['Low'] - c1['Low']) < (c0['Close'] * 0.002):
        patterns.append("Tweezers Bottom")
    if prev_black and is_white and (c0['Open'] < c1['Low']):
        if abs(c0['Close'] - c1['Close']) < (c0['Close'] * 0.002): patterns.append("Counter Attack Bullish")
    if prev_black and is_white and abs(c0['Open'] - c1['Open']) < (c0['Close'] * 0.002):
        patterns.append("Bullish Separating Lines")
    if c0['Low'] > c1['High']:
        patterns.append("Rising Window")

    if (c2['Close'] > c2['Open']) and prev_white and is_black:
        gap_exists = c1['Low'] > c2['High']
        opens_inside = (c0['Open'] < c1['Close']) and (c0['Open'] > c1['Open'])
        closes_in_gap = (c0['Close'] < c1['Open']) and (c0['Close'] > c2['High'])
        if gap_exists and opens_inside and closes_in_gap: patterns.append("Upward Gapping Tasuki")

    if prev_white and is_white:
        gap_exists = c1['Low'] > c2['High']
        similar_open = abs(c0['Open'] - c1['Open']) < (c0['Close'] * 0.002)
        if gap_exists and similar_open: patterns.append("Upgap Side-by-Side White Lines")

    if c3['Low'] > c4['High']: 
        if abs(c1['Close'] - c2['Close']) < avg_body: patterns.append("High Price Gapping Play (Watch)")

    if (c2['Close'] < c2['Open']) and (c2['Body'] > avg_body):
        if c1['Body'] < (avg_body * 0.6):
            if is_white and (c0['Close'] > (c2['Close'] + c2['Body']*0.5)): patterns.append("Morning Star pattern")

    if (c4['Close'] > c4['Open']) and (c4['Body'] > avg_body):
        if is_white and (c0['Close'] > c4['Close']): patterns.append("Rising Three Methods")

    small_bodies = all(df.iloc[-i]['Body'] < avg_body for i in range(2, 6))
    if small_bodies and is_white and (c0['Low'] > c1['High']): patterns.append("Frypan Bottom")

    if (c4['Close'] < c4['Open']) and (c4['Body'] > avg_body):
        consolidation = all(df.iloc[-i]['Body'] < avg_body for i in range(2, 5))
        if consolidation and is_white and (c0['Body'] > avg_body): patterns.append("Tower Bottom")

    supports = []
    if (c0['Low'] < c0['SMA50']) and (c0['Close'] > c0['SMA50']): supports.append("50MA")
    if (c0['Low'] < c0['SMA200']) and (c0['Close'] > c0['SMA200']): supports.append("200MA")
    
    vol_conf = " (High Vol)" if c0['Volume'] > (c0['AvgVol'] * 1.5) else ""
    pattern_str = ", ".join(patterns)
    if supports and patterns: pattern_str += f" [Supp: {','.join(supports)}]"
    pattern_str += vol_conf

    return pattern_str if pattern_str else "None", c0['Vol_Ratio']

# --- 4. THE APEX FILTER (EMPIRICAL BACKTEST INTEGRATION) ---
def check_apex_combinations(pattern_str, rsi, rel_vol, z_score):
    """
    Cross-references current structural patterns with the four empirically proven 
    winning combinations derived from the historical backtest.
    """
    if pattern_str == "None":
        return "NO"
        
    # 1. Exhaustion Hammer (72.5% Win Rate)
    if "Hammer 10" in pattern_str and rsi < 40 and 1.0 <= rel_vol <= 1.5 and z_score < -1:
        return "YES (Exhaustion Hammer)"
        
    # 2. Gap Acceleration (80.0% Win Rate)
    if "Upward Gapping Tasuki" in pattern_str and 40 <= rsi <= 60 and rel_vol > 1.5 and z_score > 1:
        return "YES (Gap Acceleration)"
        
    # 3. Morning Star Friction (75.0% Win Rate)
    if "Morning Star pattern" in pattern_str and 40 <= rsi <= 60 and rel_vol < 1.0 and -1 <= z_score <= 1:
        return "YES (Morning Star Friction)"
        
    # 4. Rubber Band Snap (68.8% Win Rate)
    if "Dragonfly Doji" in pattern_str and rsi < 40 and 1.0 <= rel_vol <= 1.5 and z_score < -1:
        return "YES (Rubber Band Snap)"
        
    return "NO"

# --- 5. MAIN EXECUTION ---
def main():
    print("--- QUANTITATIVE SCANNER ---")
    
    try:
        df_tickers = pd.read_csv(INPUT_FILE)
        col = next((c for c in df_tickers.columns if 'ticker' in c.lower()), df_tickers.columns[0])
        tickers = df_tickers[col].dropna().astype(str).str.strip().tolist()
    except:
        print(f"Error: {INPUT_FILE} not found. Please create it with a 'ticker' column.")
        return

    print(f"Downloading data for {len(tickers)} tickers...")
    try:
        data = yf.download(tickers, period="5y", group_by='ticker', auto_adjust=True, threads=True)
    except Exception as e:
        print(f"Download Error: {e}")
        return
    
    results = []
    avail = data.columns.levels[0] if isinstance(data.columns, pd.MultiIndex) else [tickers[0]]
    if len(tickers) == 1: avail = [tickers[0]]

    print("Analyzing structural patterns and applying empirical Apex Filters...")
    
    for ticker in avail:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df_t = data[ticker].copy().dropna()
            else:
                df_t = data.copy().dropna()

            if len(df_t) < 200: continue
            
            df_t = calculate_technicals(df_t)
            
            c0 = df_t.iloc[-1]
            signal_date = df_t.index[-1].strftime('%Y-%m-%d')
            
            squeeze = check_squeeze(df_t)
            buy_signal = check_buy_signal(df_t, squeeze)
            divergence = check_divergence(df_t)
            patterns, vol_ratio = check_patterns_full(ticker, df_t)
            trend_status = get_trend_status(df_t)
            ptj_status = get_paul_tudor_jones_status(df_t)
            
            golden_cross = "Yes" if c0['Golden_Cross'] else "No"
            if c0['SMA50'] > c0['SMA200'] and golden_cross == "No":
                golden_cross = df_t[df_t['Golden_Cross']].index[-1].strftime('%Y-%m-%d') if not df_t[df_t['Golden_Cross']].empty else "No"

            # Execute Empirical Filter
            apex_setup = check_apex_combinations(patterns, c0['RSI'], vol_ratio, c0['Z_Score'])

            # Vault Door Geometry (L&S Spread Protection)
            atr_14 = c0['ATR_14']
            stop_loss = c0['Low'] - (atr_14 * 1.5)
            risk_per_share = c0['Close'] - stop_loss
            take_profit = c0['Close'] + (risk_per_share * 2)

            res = {
                "Date": signal_date,
                "Ticker": ticker,
                "Price": round(c0['Close'], 2),
                "Apex Setup": apex_setup,
                "Buy Signal": buy_signal,
                "Vault Door (SL)": round(stop_loss, 2),
                "Target Price (TP)": round(take_profit, 2),
                "Patterns": patterns,
                "Daily Trend": trend_status,
                "Bullish Divergence": divergence,
                "Volatility Squeeze": squeeze,
                "Golden Cross": golden_cross,
                "PTJ Status": ptj_status,
                "RSI": round(c0['RSI'], 1),
                "ROC(5)": round(c0['ROC_5'], 2),
                "Z-Score": round(c0['Z_Score'], 2),
                "Rel Vol": round(vol_ratio, 1)
            }
            results.append(res)
            
            if apex_setup != "NO":
                print(f"🚨 APEX TARGET LOCATED: {ticker} -> {apex_setup}")
            elif buy_signal != "NO" or patterns != "None":
                print(f"[{signal_date}] Alert: {ticker} -> Buy Signal: {buy_signal} | Pattern: {patterns}")
            
        except Exception as e:
            continue

    if results:
        df_res = pd.DataFrame(results)
        # Reorder columns to put Apex Setup at the front
        cols = ['Date', 'Ticker', 'Price', 'Apex Setup', 'Buy Signal', 'Vault Door (SL)', 'Target Price (TP)','Patterns', 'ROC(5)', 'RSI', 'Z-Score', 'Rel Vol', 'Daily Trend', 'Bullish Divergence', 'Volatility Squeeze', 'PTJ Status', 'Golden Cross']
        df_res = df_res[[c for c in cols if c in df_res.columns]]
        
        # Sort the final board: Put the confirmed Apex Setups at the absolute top
        df_res = df_res.sort_values(by=['Apex Setup'], ascending=False)
        
        df_res.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ Analysis complete. Execution board updated: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()