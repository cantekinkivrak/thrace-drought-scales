import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""R2 #5 + minor: composite as rank aggregation — sensitivity to scheme, dropped criteria, MI k."""
import numpy as np, pandas as pd
from rev import *
core=core_sample()
base=metric_table(core)
crit=['pearson','spearman','MI','tercile_delta']
def top3(t): return ' > '.join(t['index'].head(3))
rows=[]
# 1. baseline
rows.append(dict(scheme='sum of ranks (paper)',ranking=top3(base),SPEI3_pos=int(base.index[base['index']=='SPEI3'][0])+1,SPEI3_score=base.loc[base['index']=='SPEI3','composite'].iloc[0]))
# 2. mean of z-scores (cardinal, not ordinal)
z=base.copy()
for c in crit: z[c+'_z']=(z[c]-z[c].mean())/z[c].std(ddof=0)
z['zsum']=z[[c+'_z' for c in crit]].sum(axis=1); z=z.sort_values('zsum',ascending=False).reset_index(drop=True)
rows.append(dict(scheme='sum of z-scores',ranking=top3(z),SPEI3_pos=int(z.index[z['index']=='SPEI3'][0])+1,SPEI3_score=round(z.loc[z['index']=='SPEI3','zsum'].iloc[0],2)))
# 3. drop one criterion at a time
for drop in crit:
    keep=tuple(c for c in crit if c!=drop); t=metric_table(core,criteria=keep)
    rows.append(dict(scheme=f'drop {drop}',ranking=top3(t),SPEI3_pos=int(t.index[t['index']=='SPEI3'][0])+1,SPEI3_score=t.loc[t['index']=='SPEI3','composite'].iloc[0]))
# 4. only association criteria (Pearson+Spearman) vs only 'other' (MI+tercile)
for keep,lab in [(('pearson','spearman'),'Pearson+Spearman only'),(('MI','tercile_delta'),'MI+tercile only')]:
    t=metric_table(core,criteria=keep)
    rows.append(dict(scheme=lab,ranking=top3(t),SPEI3_pos=int(t.index[t['index']=='SPEI3'][0])+1,SPEI3_score=t.loc[t['index']=='SPEI3','composite'].iloc[0]))
# 5. MI k sensitivity
for k in [3,5,7,10,15]:
    t=metric_table(core,k=k)
    srt=t.sort_values('MI',ascending=False).reset_index(drop=True); mi_rank=int(srt.index[srt['index']=='SPEI3'][0])+1
    rows.append(dict(scheme=f'MI k={k}',ranking=top3(t),SPEI3_pos=int(t.index[t['index']=='SPEI3'][0])+1,SPEI3_score=t.loc[t['index']=='SPEI3','composite'].iloc[0],SPEI3_MI_rank=mi_rank))
out=pd.DataFrame(rows); out.to_csv('data/revision_results/C_rank_aggregation_sensitivity.csv',index=False)
pd.set_option('display.width',200); print(out.to_string(index=False))
def pos(c):
    srt=base.sort_values(c,ascending=False).reset_index(drop=True); return int(srt.index[srt['index']=='SPEI3'][0])+1
print('\nBaseline per-criterion ranks of SPEI3 (1=best):', {c:pos(c) for c in crit})
