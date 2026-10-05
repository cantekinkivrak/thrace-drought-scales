import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
import numpy as np, pandas as pd
from rev import *
rng=np.random.default_rng(42)
def boot(crop,trend,B=600,L=3):
    d=load_panel(crop); d['yield']=d['yield'].astype(float); levels=sorted(d.key.unique())
    years=np.array(sorted(d.year.unique())); ny=len(years)
    sel=[]; d39=[]; dlong=[]
    for _ in range(B):
        starts=rng.integers(0,ny,size=int(np.ceil(ny/L)))
        idx=np.concatenate([[(s+i)%ny for i in range(L)] for s in starts])[:ny]
        by=years[idx]; tr=pd.concat([d[d.year==y] for y in by]); oob=d[~d.year.isin(set(by))]
        if oob.year.nunique()<3: continue
        r={}
        for s in SCALES:
            f=fit_fe(tr,f'SPEI{s}',levels,trend); X,_=fe_design(oob,f'SPEI{s}',levels,trend)
            r[s]=np.sqrt(np.mean((oob['yield'].to_numpy()-X@f['beta'])**2))
        sel.append(min(r,key=r.get)); d39.append(r[3]-r[9]); dlong.append(r[3]-min(r[6],r[9],r[12]))
    sel=np.array(sel); d39=np.array(d39); dlong=np.array(dlong)
    return dict(crop=crop,trend=trend,n_rep=len(sel),**{f'sel_{s}':100*np.mean(sel==s) for s in SCALES},
        d39_med=np.median(d39),d39_lo=np.percentile(d39,2.5),d39_hi=np.percentile(d39,97.5),P_9_beats_3=np.mean(d39>0),
        P_longer_beats_3=np.mean(dlong>0))
rows=[boot(c,t) for c in ['sunflower','wheat'] for t in ['linear','quadratic','district']]
t=pd.DataFrame(rows); t.to_csv('data/revision_results/A2_yield_scale_bootstrap.csv',index=False)
pd.set_option('display.width',200)
print(t.round(2).to_string(index=False))
print()
# yield trend shape: regional mean by year
d=load_panel('sunflower'); print('sunflower regional mean yield by year:'); print(d.groupby('year')['yield'].mean().round(0).to_string())
