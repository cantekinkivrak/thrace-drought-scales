import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""Shared loaders reproducing the paper's primary sample and metrics (revision work)."""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

DATA='data'
SCALES=[1,3,6,9,12]
IDX=[f"{f}{s}" for f in ["SPI","SPEI"] for s in SCALES]
WIN=[6,7,8,9,10]
CORE=["corlu","edirne","kirklareli","luleburgaz","tekirdag","uzunkopru"]
ALL7=CORE+["ipsala"]

def load_monthly():
    vhi=pd.read_csv(f'{DATA}/trakya_district_vhi_vci_tci_monthly.csv')
    vhi['key']=vhi['name'].str.lower()
    vhi['VHI']=0.5*(vhi['VCI']+vhi['TCI'])          # paper: fixed alpha = 0.5 (never trust stored VHI)
    idx=pd.read_csv(f'{DATA}/trakya_spi_spei_flexible_1965_2024.csv')
    m=vhi.merge(idx.drop(columns='name'), on=['key','year','month'])
    return m

def core_sample(m=None, months=WIN, keys=CORE, ref='VHI'):
    m=load_monthly() if m is None else m
    s=m[m['key'].isin(keys) & m['month'].isin(months)].dropna(subset=[ref]+IDX)
    return s

def clean_pair(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); ok=~(np.isnan(a)|np.isnan(b)); return a[ok],b[ok]

def metric_table(frame, reference='VHI', indices=IDX, k=5, criteria=('pearson','spearman','MI','tercile_delta'),
                 rank_method='average'):
    rows=[]
    for name in indices:
        x,y=clean_pair(frame[reference],frame[name])
        if len(x)<30: continue
        q1,q3=np.quantile(y,[1/3,2/3])
        rows.append(dict(index=name,n=len(x),
            pearson=stats.pearsonr(x,y).statistic, spearman=stats.spearmanr(x,y).statistic,
            MI=mutual_info_regression(y.reshape(-1,1),x,random_state=0,n_neighbors=k)[0],
            tercile_delta=x[y>=q3].mean()-x[y<=q1].mean()))
    t=pd.DataFrame(rows)
    for c in criteria: t[f'{c}_score']=t[c].rank(method=rank_method)
    t['composite']=t[[f'{c}_score' for c in criteria]].sum(axis=1)
    return t.sort_values(['composite','pearson'],ascending=False).reset_index(drop=True)

def contingency(ref,det):
    ref=np.asarray(ref,bool); det=np.asarray(det,bool)
    h=int((ref&det).sum()); fa=int((~ref&det).sum()); mi=int((ref&~det).sum()); cn=int((~ref&~det).sum())
    n=h+fa+mi+cn
    exp=((h+fa)*(h+mi)+(mi+cn)*(fa+cn))/n
    return dict(hits=h,false_alarms=fa,misses=mi,correct_negatives=cn,n=n,
        POD=h/(h+mi) if h+mi else np.nan, FAR=fa/(h+fa) if h+fa else np.nan,
        CSI=h/(h+fa+mi) if h+fa+mi else np.nan, bias=(h+fa)/(h+mi) if h+mi else np.nan,
        HSS=(h+cn-exp)/(n-exp))

def moving_year_sample(rng, years, L=3):
    years=sorted(years); out=[]
    while len(out)<len(years):
        s=int(rng.integers(0,len(years))); out.extend(years[(s+o)%len(years)] for o in range(L))
    return out[:len(years)]

# ---------- yield panel ----------
def load_panel(crop):
    p=pd.read_csv(f'{DATA}/yield_panel_{crop}.csv').dropna(subset=['yield'])
    return p

def fe_design(d, idx, levels, trend='linear'):
    cols=[np.ones(len(d)), d[idx].to_numpy(float)]; names=['intercept',idx]
    if trend=='linear':
        cols.append(d['year_c'].to_numpy(float)); names.append('year_c')
    elif trend=='quadratic':
        cols += [d['year_c'].to_numpy(float), d['year_c'].to_numpy(float)**2]; names += ['year_c','year_c2']
    elif trend=='district':   # district-specific linear trends
        cols.append(d['year_c'].to_numpy(float)); names.append('year_c')
        for lv in levels[1:]:
            cols.append(((d['key']==lv).to_numpy(float))*d['year_c'].to_numpy(float)); names.append(f'trend_{lv}')
    elif trend=='none':
        pass
    for lv in levels[1:]:
        cols.append((d['key']==lv).to_numpy(float)); names.append(f'key_{lv}')
    return np.column_stack(cols), names

def fit_fe(d, idx, levels, trend='linear'):
    X,names=fe_design(d,idx,levels,trend); y=d['yield'].to_numpy(float)
    beta,*_=np.linalg.lstsq(X,y,rcond=None); res=y-X@beta; n,k=X.shape
    rss=float(res@res); aic=n*(np.log(2*np.pi)+1+np.log(rss/n))+2*k
    return dict(beta=beta,names=names,X=X,res=res,rss=rss,AIC=aic,n=n,k=k)

def loyo_rmse(d, idx, levels, trend='linear'):
    errs=[]
    for y in sorted(d['year'].unique()):
        tr=d[d.year!=y]; te=d[d.year==y]
        f=fit_fe(tr,idx,levels,trend); Xte,_=fe_design(te,idx,levels,trend)
        errs.extend((te['yield'].to_numpy()-Xte@f['beta']).tolist())
    return float(np.sqrt(np.mean(np.square(errs))))
