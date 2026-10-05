import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""Season-window and crop-mask scale selection under quality-controlled composites."""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from scipy import stats
from rev import metric_table, IDX, CORE
exec(open('H3_ipsala_scene_quality.py').read().split("def district_r")[0].split("# (1) coverage table")[0])  # loads sc, idx, V
full = sc.groupby('key')[f'NDVI_{V}_count'].transform('max'); sc['cov'] = sc[f'NDVI_{V}_count']/full; sc['l7off']=sc.slc_off==1
def comp(frame, var, scheme):
    d = frame[['key','year','month',f'NDVI_{var}',f'LST_{var}']].rename(columns={f'NDVI_{var}':'NDVI',f'LST_{var}':'LST'}).dropna(subset=['NDVI'])
    g = d.groupby(['key','year','month'])
    m = (g.agg(NDVI=('NDVI','max'),LST=('LST','mean')) if scheme=='max' else g.agg(NDVI=('NDVI','median'),LST=('LST','median'))).reset_index()
    cl = m.groupby(['key','month']).agg(nmin=('NDVI','min'),nmax=('NDVI','max'),tmin=('LST','min'),tmax=('LST','max')).reset_index()
    x = m.merge(cl,on=['key','month'])
    x['VHI'] = 0.5*((100*(x.NDVI-x.nmin)/(x.nmax-x.nmin).replace(0,np.nan)).clip(0,100) + (100*(x.tmax-x.LST)/(x.tmax-x.tmin).replace(0,np.nan)).clip(0,100))
    return x.merge(idx,on=['key','year','month'])
QC = {'paper (all scenes, max)': (lambda s: s, 'max'),
      'QC: no SLC-off, cov>=50%, max': (lambda s: s[(~s.l7off)&(s['cov']>=0.5)], 'max'),
      'median (all scenes)': (lambda s: s, 'median')}
WINS = {'Apr-Jun':[4,5,6],'Jun-Oct':[6,7,8,9,10],'Jul-Sep':[7,8,9],'Aug-Sep':[8,9]}
rows=[]
for qn,(f,sch) in QC.items():
    for var in ['v2018','wheat','sun']:
        x = comp(f(sc), var, sch)
        for wn,mo in WINS.items():
            s = x[x.key.isin(CORE)&x.month.isin(mo)].dropna(subset=['VHI']+IDX)
            t = metric_table(s).set_index('index')
            top = t.index[0]; sp = t[t.index.str.startswith('SPEI')]
            rows.append(dict(composite=qn, pixels=var, window=wn, n=len(s), winner=top, score=t.loc[top,'composite'],
                             **{f'r_{k}':t.loc[k,'pearson'] for k in ['SPEI1','SPEI3','SPEI6','SPEI9','SPEI12']}))
        print(qn, var, 'done', flush=True)
r=pd.DataFrame(rows); r.to_csv('data/revision_results/H4_windows_qc.csv',index=False)
pd.set_option('display.width',250); print(r.round(3).to_string())
