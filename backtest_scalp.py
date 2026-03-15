"""
APEX Scalp Backtester v2 — backtest_scalp.py
Pre-computes scan_scalp() once, then optimises rapidly.
"""
import sqlite3, json, logging, numpy as np, pandas as pd
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Optional, Dict, List
from multiprocessing import Pool, cpu_count
import itertools, sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('APEX.ScalpBT')

DB_PATH   = 'apex_market.db'
NY_TZ     = ZoneInfo('America/New_York')
POINT_VAL = 20.0
COMM      = 5.0
BALANCE   = 10000

def build_session_windows(symbol):
    # All instruments: test windows from pre-market through close
    # No max duration — let the data find the best window naturally
    times = [
        (4,0),(4,30),(5,0),(5,30),(6,0),(6,30),
        (7,0),(7,30),(8,0),(8,30),(9,0),(9,30),
        (10,0),(10,30),(11,0),(11,30),(12,0),(12,30),
        (13,0),(13,30),(14,0),(14,30),(15,0),(15,30),(16,0),
    ]
    windows = []
    for i,(sh,sm) in enumerate(times):
        for (eh,em) in times[i+1:]:
            dur = (eh*60+em)-(sh*60+sm)
            if dur < 30: continue   # minimum 30 min
            if dur > 720: continue  # maximum 12hrs (full session)
            windows.append({'label':f'{sh:02d}:{sm:02d}-{eh:02d}:{em:02d}ET','start':sh*60+sm,'end':eh*60+em,'start_h':sh,'start_m':sm,'end_h':eh,'end_m':em})
    return windows

DOW_ALLOWED  = {'tue_wed_thu':{1,2,3},'mon_fri_out':{0,1,2,3},'all_week':{0,1,2,3,4}}
DOW_EXCLUDED = {'tue_wed_thu':{0,4,5,6},'mon_fri_out':{4,5,6},'all_week':{5,6}}

def load_tf(symbol,tf,limit=10000):
    try:
        conn=sqlite3.connect(DB_PATH)
        df=pd.read_sql_query('SELECT ts,open,high,low,close,volume FROM ohlcv WHERE symbol=? AND timeframe=? ORDER BY ts DESC LIMIT ?',conn,params=(symbol,tf,limit))
        conn.close()
        if len(df)<50: return None
        df=df.sort_values('ts').reset_index(drop=True)
        df['dt']=pd.to_datetime(df['ts'],unit='s',utc=True).dt.tz_convert(NY_TZ)
        for c in ['open','high','low','close','volume']: df[c]=df[c].astype(float)
        return df
    except Exception as e:
        logger.error(f'load_tf {symbol} {tf}: {e}'); return None

def load_vix():
    try:
        conn=sqlite3.connect(DB_PATH)
        df=pd.read_sql_query("SELECT ts,close FROM ohlcv WHERE symbol='VIX' AND timeframe='1day' ORDER BY ts ASC",conn)
        conn.close()
        df['date']=pd.to_datetime(df['ts'],unit='s',utc=True).dt.date
        return dict(zip(df['date'],df['close'].astype(float)))
    except: return {}

def precompute_signals(symbol,df_5min,df_15min,df_1h,df_4h,df_1d,min_score_floor=50):
    from strategy_scalp import scan_scalp
    n=len(df_5min); signals=[None]*n
    logger.info(f'Pre-computing scan_scalp() on {n} bars for {symbol}...')
    for i in range(50,n-5):
        if i%1000==0: logger.info(f'  {i}/{n} ({i/n*100:.0f}%) — {sum(1 for s in signals[:i] if s)} signals so far')
        row=df_5min.iloc[i]; ts=int(row['ts']); dt=row['dt']
        if dt.weekday()>=5: continue
        def sl(df,nb=150):
            if df is None: return None
            sub=df[df['ts']<=ts].tail(nb); return sub if len(sub)>20 else None
        try:
            setup=scan_scalp(df_1min=None,df_5min=df_5min.iloc[max(0,i-200):i+1],df_15min=sl(df_15min),df_1hour=sl(df_1h),df_4hour=sl(df_4h),df_1day=sl(df_1d,100),symbol=symbol,min_score=min_score_floor)
        except Exception as e:
            logger.debug(f'scan_scalp bar {i}: {e}'); continue
        if setup:
            signals[i]={'ts':ts,'dt':dt,'direction':setup.direction,'score':setup.score,'entry':setup.entry,'stop':setup.stop,'risk_pts':setup.risk_pts,'t_mins':dt.hour*60+dt.minute,'dow':dt.weekday(),'date':dt.date()}
    found=sum(1 for s in signals if s)
    logger.info(f'Pre-computation complete: {found} signals across {n} bars')
    return signals

def simulate_trade(entry,stop,target,direction,future_bars,max_bars=60):
    is_long=direction=='long'
    for bar in future_bars[:max_bars]:
        bh,bl=bar[1],bar[2]
        if is_long:
            if bl<=stop: return 'loss',stop
            if bh>=target: return 'win',target
        else:
            if bh>=stop: return 'loss',stop
            if bl<=target: return 'win',target
    lc=future_bars[min(max_bars-1,len(future_bars)-1)][3] if len(future_bars)>0 else entry
    return ('timeout_win',lc) if (is_long and lc>entry) or (not is_long and lc<entry) else ('timeout_loss',lc)

def simulate_account(trades,balance=10000,risk_pct=1.5):
    if not trades: return {'sharpe':-99,'total_return':-99,'win_rate':0,'max_dd':0,'trades':0,'trades_per_year':0}
    equity=[balance];rets=[];wins=0;gw=0.0;gl=0.0;peak=balance;max_dd=0.0
    for t in trades:
        bal=equity[-1]; rp=abs(t['entry']-t['stop'])
        if rp<=0: continue
        size=(bal*risk_pct/100)/(rp*POINT_VAL)
        pnl=((t['exit_px']-t['entry']) if t['direction']=='long' else (t['entry']-t['exit_px']))*POINT_VAL*size-COMM
        bal+=pnl; equity.append(bal); rets.append(pnl/equity[-2] if equity[-2]>0 else 0)
        if pnl>0: wins+=1; gw+=pnl
        else: gl+=abs(pnl)
        peak=max(peak,bal); max_dd=max(max_dd,(peak-bal)/peak*100)
    n=len(trades)
    if n==0: return {'sharpe':-99,'total_return':-99,'win_rate':0,'max_dd':0,'trades':0,'trades_per_year':0}
    wr=wins/n*100; r_arr=np.array(rets)
    sharpe=(np.mean(r_arr)/np.std(r_arr))*np.sqrt(252*390/max(n,1)) if len(r_arr)>1 and np.std(r_arr)>0 else 0
    span=(trades[-1]['ts']-trades[0]['ts'])/86400 if len(trades)>=2 else 0
    tpy=n/span*252 if span>5 else 0
    avg_wr=(gw/wins)/(balance*risk_pct/100) if wins>0 else 0
    avg_lr=(gl/(n-wins))/(balance*risk_pct/100) if n-wins>0 else 0
    return {'sharpe':round(sharpe,3),'total_return':round((equity[-1]-balance)/balance*100,2),'win_rate':round(wr,1),'max_dd':round(max_dd,2),'expectancy':round(wr/100*avg_wr-(1-wr/100)*avg_lr,4),'trades':n,'trades_per_year':round(tpy,1),'final_balance':round(equity[-1],2)}

def walk_forward_fast(signals,future_arr,sess,dow_key,params,vix_data,min_trades):
    sm=sess['start']; em=sess['end']
    allowed=DOW_ALLOWED[dow_key]; excluded=DOW_EXCLUDED[dow_key]
    ms=params['min_score']; rr=params['rr_ratio']; rp=params['risk_pct']; vx=params['vix_max']; mx=params['max_per_session']
    ft=future_arr[:,0]; trades=[]; last_ts=0; sc={}
    for sig in signals:
        if sig is None: continue
        if not (sm<=sig['t_mins']<em): continue
        if sig['dow'] in excluded or sig['dow'] not in allowed: continue
        if sig['score']<ms: continue
        vix=vix_data.get(sig['date'])
        if vix and vix>vx: continue
        if sig['ts']-last_ts<300: continue
        d=sig['date']
        if sc.get(d,0)>=mx: continue
        entry=sig['entry']; stop=sig['stop']; rpts=sig['risk_pts']
        if rpts<=0: continue
        il=sig['direction']=='long'
        target=(entry+rpts*rr) if il else (entry-rpts*rr)
        fi=np.searchsorted(ft,sig['ts'],side='right')
        ff=future_arr[fi:]
        if len(ff)<5: continue
        outcome,exit_px=simulate_trade(entry,stop,target,sig['direction'],ff)
        pnl=(exit_px-entry) if il else (entry-exit_px)
        trades.append({'ts':sig['ts'],'direction':sig['direction'],'entry':round(entry,2),'stop':round(stop,2),'target':round(target,2),'exit_px':round(exit_px,2),'outcome':outcome,'pnl_pts':round(pnl,2),'score':sig['score'],'risk_pct':rp})
        last_ts=sig['ts']; sc[d]=sc.get(d,0)+1
    return trades

_state={}
def pool_init(signals,future_arr,vix_data,min_trades):
    _state['s']=signals;_state['f']=future_arr;_state['v']=vix_data;_state['mt']=min_trades

def worker(args):
    sess,dow_key,params=args
    trades=walk_forward_fast(_state['s'],_state['f'],sess,dow_key,params,_state['v'],_state['mt'])
    metrics=simulate_account(trades,BALANCE,params['risk_pct'])
    if metrics['trades']<_state['mt'] or metrics['sharpe']<=0: return None
    return {'params':{**params,'session':sess['label'],'dow':dow_key},'session':dict(sess),'metrics':metrics}

GRIDS={
    'NQ':{'min_score':[55,60,65,70,75],'rr_ratio':[2.0,2.5,3.0],'risk_pct':[1.0,1.5],'vix_max':[20,25],'max_per_session':[1,2]},
    'ES':{'min_score':[55,60,65,70,75],'rr_ratio':[2.0,2.5,3.0],'risk_pct':[1.0,1.5],'vix_max':[20,25],'max_per_session':[1,2]},
    'GC':{'min_score':[55,60,65,70,75],'rr_ratio':[1.5,2.0,2.5],'risk_pct':[1.0,1.5],'vix_max':[20,25],'max_per_session':[1,2]},
}

def run_scalp_backtest(symbol='NQ',min_trades=10):
    logger.info(f'\n{"#"*60}\n  APEX Scalp Backtest — {symbol}\n  Real scan_scalp() pre-computed\n{"#"*60}')
    n_cores=min(cpu_count(),8); logger.info(f'  {n_cores} cores')
    df_5=load_tf(symbol,'5min',10000); df_15=load_tf(symbol,'15min',5000)
    df_1h=load_tf(symbol,'1hour',3000); df_4h=load_tf(symbol,'4hour',1000)
    df_1d=load_tf(symbol,'1day',500); vix=load_vix()
    for nm,df in [('5min',df_5),('15min',df_15),('1h',df_1h),('4h',df_4h)]:
        logger.info(f'  {nm}: {len(df) if df is not None else 0} bars')
    if df_5 is None or len(df_5)<200: logger.error('Need 200+ 5min bars'); return
    signals=precompute_signals(symbol,df_5,df_15,df_1h,df_4h,df_1d,min_score_floor=50)
    ns=sum(1 for s in signals if s)
    if ns<min_trades: logger.warning(f'Only {ns} signals — not enough'); return None
    logger.info(f'\n{ns} signals found — starting optimisation...')
    fa=df_5[['ts','high','low','close']].values
    sw=build_session_windows(symbol)
    grid=GRIDS[symbol]; keys=list(grid.keys())
    combos=[dict(zip(keys,c)) for c in itertools.product(*grid.values())]
    work=[(sess,dk,p) for sess in sw for dk in DOW_ALLOWED for p in combos]
    total=len(work); logger.info(f'  {total} combos on {n_cores} cores')
    results=[]; done=0
    with Pool(processes=n_cores,initializer=pool_init,initargs=(signals,fa,vix,min_trades)) as pool:
        for r in pool.imap_unordered(worker,work,chunksize=50):
            done+=1
            if r: results.append(r)
            if done%500==0:
                best=max(results,key=lambda x:x['metrics']['sharpe'])['metrics']['sharpe'] if results else 0
                logger.info(f'  {done}/{total} — Sharpe {best:.3f} ({len(results)} valid)')
    if not results: logger.warning('No valid results'); return None
    results.sort(key=lambda x:x['metrics']['sharpe'],reverse=True)
    best=results[0]; m=best['metrics']; p=best['params']; s=best['session']
    logger.info(f'\n{"="*60}\nBEST {symbol} SCALP:\n{"="*60}')
    logger.info(f"  Sharpe {m['sharpe']} | Return {m['total_return']}% | WR {m['win_rate']}% | DD {m['max_dd']}%")
    logger.info(f"  Trades/yr {m['trades_per_year']} | Session {s['label']} | DoW {p['dow']}")
    logger.info(f"  Params { {k:v for k,v in p.items() if k not in ('session','dow')} }")
    logger.info('\nTop 5:')
    for i,r in enumerate(results[:5]):
        m2=r['metrics'];p2=r['params']
        logger.info(f"  #{i+1}: Sharpe {m2['sharpe']} | WR {m2['win_rate']}% | DD {m2['max_dd']}% | {m2['trades_per_year']} tr/yr | {p2['session']} {p2['dow']}")
    out={'symbol':symbol,'timestamp':datetime.now().isoformat(),'best':best,'top10':results[:10],'total_tested':total,'valid_combos':len(results)}
    fname=f'backtest_scalp_results_{symbol}.json'
    with open(fname,'w') as f: json.dump(out,f,indent=2,default=str)
    logger.info(f'\n✅ Saved to {fname}')
    return out

if __name__=='__main__':
    symbol=next((a.split('=')[1] for a in sys.argv if a.startswith('--symbol=')),'NQ')
    min_trades=int(next((a.split('=')[1] for a in sys.argv if a.startswith('--min_trades=')),'30'))
    run_scalp_backtest(symbol=symbol,min_trades=min_trades)
