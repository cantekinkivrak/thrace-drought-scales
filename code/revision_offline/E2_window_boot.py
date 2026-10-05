import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""Two-way block bootstrap (districts x 3-year blocks, as in the paper) of the composite ranking per window."""
import numpy as np, pandas as pd
from rev import *
m=load_monthly(); rng=np.random.default_rng(11)
def two_way(frame, B):
    years=sorted(frame.year.unique()); dists=sorted(frame.key.unique()); byd={d:frame[frame.key==d] for d in dists}
    wins=[]
    for b in range(B):
        ys=moving_year_sample(rng,years,3); ds=rng.choice(dists,len(dists),replace=True)
        parts=[]
        for d in ds:
            g=byd[d]; parts.append(g[g.year.isin(ys)])   # keep sampled years (with multiplicity via years list)
        # respect multiplicity of years
        samp=pd.concat([pd.concat([byd[d][byd[d].year==y] for y in ys]) for d in ds])
        t=metric_table(samp); wins.append(t['index'].iloc[0])
    return pd.Series(wins).value_counts(normalize=True).mul(100).round(1)
for label,months in [('Jun-Oct (paper)',[6,7,8,9,10]),('Jul-Sep',[7,8,9]),('Aug-Sep',[8,9]),('Apr-Jun',[4,5,6])]:
    s=core_sample(m,months=months)
    f=two_way(s,300)
    print(f'{label:16s} n={len(s):4d}  selection %:', f.to_dict())
