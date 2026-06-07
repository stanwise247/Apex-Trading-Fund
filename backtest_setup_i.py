"""
backtest_setup_i.py  — Task 4 (research only, no code changes)
"""
import os, sys, warnings, pickle
import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/stanleywise/Desktop/apex')

DATABASE_URL        = os.environ['DATABASE_URL']
XGB_LONG_THRESHOLD  = 0.58
LR_LONG_THRESHOLD   = 0.58
LR_SHORT_THRESHOLD  = 1.0 - LR_LONG_THRESHOLD
SESSION_START       = 13
SESSION_END_MNQ     = 20
SESSION_END_ES      = 19
ALLOWED_DAYS        = {1, 2, 3}
TARGET_RR           = 2.5
STOP_ATR_MULT       = 1.5
MAX_BARS_FORWARD    = 60

def get_conn():
    return psycopg2.connect(DATABASE_URL)

def load_ohlcv(symbol):
    conn = get_conn(); cur = conn.cursor()
    cur.execute(
        'SELECT ts, open, high, low, close, volume FROM ohlcv '
        'WHERE symbol=%s AND timeframe=%s AND ts > 1000000 ORDER BY ts ASC',
        (symbol, '5min'))
    rows = cur.fetchall(); conn.close()
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows, columns=['ts','open','high','low','close','volume'])
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    df['dt'] = pd.to_datetime(df['ts'], unit='s', utc=True)
    return df.dropna(subset=['open','close']).reset_index(drop=True)

def load_regime(symbol):
    conn = get_conn(); cur = conn.cursor()
    cur.execute(
        'SELECT EXTRACT(EPOCH FROM timestamp)::float AS ts, regime, confidence '
        'FROM regime_log WHERE symbol=%s ORDER BY timestamp ASC', (symbol,))
    rows = cur.fetchall(); conn.close()
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows, columns=['ts','regime','confidence'])
    df['ts'] = df['ts'].astype(float)
    return df

def load_models(symbol):
    conn = get_conn(); cur = conn.cursor()
    models = {}
    for sfx in ('short','long'):
        cur.execute('SELECT model_bytes FROM ml_models WHERE model_name=%s',
                    (f'apex_xi_{symbol}_{sfx}',))
        row = cur.fetchone()
        if row:
            try: models[sfx] = pickle.loads(bytes(row[0]))
            except Exception as e: print(f'  load {sfx} error: {e}')
    conn.close()
    xgb_s = models['short']['xgb'] if 'short' in models else None
    xgb_l = models['long']['xgb']  if 'long'  in models else None
    ref   = models.get('short') or models.get('long')
    lr    = ref['lr']     if ref else None
    sc    = ref['scaler'] if ref else None
    return xgb_s, xgb_l, lr, sc

def calc_atr(df, period=14):
    h, l, pc = df['high'], df['low'], df['close'].shift(1)
    tr = pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.rolling(period).mean()

def simulate(df, df_regime, xgb_s, xgb_l, lr, sc, variant, symbol):
    from setup_i_mathematical import calculate_features_i
    warnings.filterwarnings('ignore')
    sess_end = SESSION_END_MNQ if symbol == 'MNQ' else SESSION_END_ES

    X_all    = calculate_features_i(df)
    atr      = calc_atr(df,14).values
    highs    = df['high'].values; lows = df['low'].values; closes = df['close'].values
    ts_vals  = df['ts'].values
    dow_vals = df['dt'].dt.dayofweek.values
    hour_vals= df['dt'].dt.hour.values
    n = len(df); trades = []

    for i in range(50, n - MAX_BARS_FORWARD):
        if dow_vals[i] not in ALLOWED_DAYS: continue
        if hour_vals[i] < SESSION_START or hour_vals[i] >= sess_end: continue
        X_i = X_all[i:i+1]
        if np.isnan(X_i).any(): continue

        regime_label, regime_conf = 'TRENDING', 0.5
        if not df_regime.empty:
            idx = np.searchsorted(df_regime['ts'].values, float(ts_vals[i]), side='right') - 1
            if idx >= 0:
                regime_label = df_regime['regime'].iloc[idx]
                regime_conf  = float(df_regime['confidence'].iloc[idx])

        if variant in ('regime','regime_l3') and regime_label != 'TRENDING': continue

        l3_mult = 1.0
        if variant == 'regime_l3' and regime_label == 'TRENDING':
            if regime_conf >= 0.65:   l3_mult = 1.5
            elif regime_conf >= 0.55: l3_mult = 1.25
            elif regime_conf < 0.35:  l3_mult = 0.75

        try:
            X_s      = sc.transform(X_i)
            lr_prob  = float(lr.predict_proba(X_s)[0,1])
            sp       = float(xgb_s.predict_proba(X_s)[0,1]) if xgb_s else 0.0
            lp       = float(xgb_l.predict_proba(X_s)[0,1]) if xgb_l else 0.0
        except Exception: continue

        if   sp > XGB_LONG_THRESHOLD and lr_prob < LR_SHORT_THRESHOLD: direction = 'short'
        elif lp > XGB_LONG_THRESHOLD and lr_prob > LR_LONG_THRESHOLD:  direction = 'long'
        else: continue

        a = float(atr[i]) if not np.isnan(atr[i]) else 0
        if a <= 0: continue
        entry  = closes[i]
        stop   = entry - STOP_ATR_MULT*a if direction=='long' else entry + STOP_ATR_MULT*a
        target = entry + TARGET_RR*STOP_ATR_MULT*a if direction=='long' else entry - TARGET_RR*STOP_ATR_MULT*a

        outcome = None
        for k in range(i+1, min(i+MAX_BARS_FORWARD, n)):
            if direction == 'long':
                if highs[k] >= target: outcome='win'; break
                if lows[k]  <= stop:  outcome='loss'; break
            else:
                if lows[k]  <= target: outcome='win'; break
                if highs[k] >= stop:   outcome='loss'; break
        if outcome is None: continue

        r = (TARGET_RR if outcome=='win' else -1.0) * l3_mult
        trades.append({'ts': float(ts_vals[i]),
                       'month': pd.Timestamp(df['dt'].iloc[i]).strftime('%Y-%m'),
                       'direction': direction, 'outcome': outcome, 'r': r,
                       'regime': regime_label, 'sp': round(sp,3), 'lp': round(lp,3), 'lr': round(lr_prob,3)})
    return pd.DataFrame(trades)

def metrics(trades):
    if trades.empty:
        return {'signals':0,'spm':0,'wr':0,'sharpe':0,'maxdd':0,'total_r':0,'pf':0,'pm':0,'tm':0}
    n = len(trades); wins = (trades['r']>0).sum(); total_r = trades['r'].sum()
    monthly = trades.groupby('month')['r'].sum()
    sharpe  = (monthly.mean()/(monthly.std()+1e-9))*np.sqrt(12)
    maxdd   = float((trades['r'].cumsum()-trades['r'].cumsum().cummax()).min())
    wins_r  = trades.loc[trades['r']>0,'r'].sum()
    loss_r  = abs(trades.loc[trades['r']<0,'r'].sum())
    pf = wins_r/loss_r if loss_r>0 else float('inf')
    return {'signals':n,'spm':round(n/max(len(monthly),1),1),'wr':round(wins/n*100,1),
            'sharpe':round(float(sharpe),2),'maxdd':round(maxdd,2),'total_r':round(float(total_r),2),
            'pf':round(float(pf),2) if np.isfinite(float(pf)) else 0,
            'pm':int((monthly>0).sum()),'tm':len(monthly)}

def run(symbol):
    print(f'\n{"="*60}')
    print(f'SETUP I BACKTEST — {symbol}')
    print(f'{"="*60}')
    print('Loading models...')
    xgb_s, xgb_l, lr, sc = load_models(symbol)
    if xgb_s is None and xgb_l is None: print(f'No models — abort'); return
    print(f'  xgb_short={"OK" if xgb_s else "missing"} xgb_long={"OK" if xgb_l else "missing"}')
    df = load_ohlcv(symbol)
    if df.empty: print(f'No data'); return
    print(f'Loaded {len(df):,} bars | {df["dt"].iloc[0].date()} → {df["dt"].iloc[-1].date()}')
    months = (df["dt"].iloc[-1]-df["dt"].iloc[0]).days/30.44
    print(f'~{months:.1f} months | regime rows: {len(load_regime(symbol)):,}')
    df_regime = load_regime(symbol)

    print(f'\n{"Variant":<14}{"Signals":<10}{"Sig/Mo":<10}{"WR%":<8}{"Sharpe":<10}{"MaxDD":<10}{"TotalR":<10}{"PF":<8}"+Mo/Tot"')
    print('-'*90)
    for v in ('raw','regime','regime_l3'):
        t = simulate(df.copy(), df_regime, xgb_s, xgb_l, lr, sc, v, symbol)
        m = metrics(t)
        print(f'{v:<14}{m["signals"]:<10}{m["spm"]:<10}{m["wr"]:<8}{m["sharpe"]:<10}{m["maxdd"]:<10}{m["total_r"]:<10}{m["pf"]:<8}{m["pm"]}/{m["tm"]}')
        if not t.empty:
            s = t.head(2)
            for _,row in s.iterrows():
                print(f'  eg: {row["month"]} {row["direction"]} sp={row["sp"]} lp={row["lp"]} lr={row["lr"]} → {row["outcome"]} ({row["r"]:+.2f}R)')

if __name__=='__main__':
    for sym in ('MNQ','ES'): run(sym)
    print('\nTask 4 complete — research only, no code changes.')
