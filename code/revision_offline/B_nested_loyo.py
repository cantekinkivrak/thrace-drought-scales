import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""R2 #8: nest the scale selection inside each leave-one-year-out fold."""
import numpy as np, pandas as pd
from rev import *
core=core_sample()
rows=[]; chosen=[]
for y in sorted(core.year.unique()):
    tr=core[core.year!=y]; te=core[core.year==y].copy()
    best=metric_table(tr).iloc[0]['index']            # selection made on training years only
    chosen.append(best)
    rate=(tr[best]<-1).mean(); thr=tr['VHI'].quantile(rate)
    te['drought']=te[best]<-1; te['detected']=te['VHI']<thr; te['chosen']=best
    rows.append(te)
nested=pd.concat(rows)
# fixed-SPEI3 reference (paper)
rows=[]
for y in sorted(core.year.unique()):
    tr=core[core.year!=y]; te=core[core.year==y].copy()
    rate=(tr['SPEI3']<-1).mean(); thr=tr['VHI'].quantile(rate)
    te['drought']=te['SPEI3']<-1; te['detected']=te['VHI']<thr; rows.append(te)
fixed=pd.concat(rows)
res=pd.DataFrame([dict(scheme='fixed SPEI-3 (paper)',**contingency(fixed.drought,fixed.detected)),
                  dict(scheme='nested selection',**contingency(nested.drought,nested.detected))])
res.to_csv('data/revision_results/B_nested_loyo_skill.csv',index=False)
print(res[['scheme','n','hits','false_alarms','misses','POD','FAR','CSI','bias','HSS']].round(3).to_string(index=False))
print('\nindex chosen within folds:', pd.Series(chosen).value_counts().to_dict())
