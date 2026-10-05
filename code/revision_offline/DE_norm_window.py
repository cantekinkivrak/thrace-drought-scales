import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""R2 #7b percentile normalisation; R2 #1 crop-season windows."""
import numpy as np, pandas as pd
from rev import *
m=load_monthly()

def renorm(m, lo, hi):
    """VCI/TCI from per-district per-calendar-month percentiles instead of min/max, clipped 0-100."""
    g=m.groupby(['key','month'])
    nlo=g['NDVI'].transform(lambda s: s.quantile(lo)); nhi=g['NDVI'].transform(lambda s: s.quantile(hi))
    tlo=g['LST'].transform(lambda s: s.quantile(lo));  thi=g['LST'].transform(lambda s: s.quantile(hi))
    x=m.copy()
    x['VCI']=(100*(x.NDVI-nlo)/(nhi-nlo)).clip(0,100); x['TCI']=(100*(thi-x.LST)/(thi-tlo)).clip(0,100)
    x['VHI']=0.5*(x.VCI+x.TCI); return x

rows=[]
def rec(label,frame,months=WIN):
    t=metric_table(core_sample(frame,months=months))
    r3=t.loc[t['index']=='SPEI3']; s3=t.loc[t['index']=='SPI3']; s9=t.loc[t['index']=='SPEI9']
    rows.append(dict(scenario=label,n=int(t.n.iloc[0]),top=t['index'].iloc[0],top_composite=t.composite.iloc[0],
        second=t['index'].iloc[1],SPEI3_r=r3.pearson.iloc[0],SPEI3_composite=r3.composite.iloc[0],
        SPI3_r=s3.pearson.iloc[0],SPEI9_r=s9.pearson.iloc[0],ranking=' > '.join(t['index'].head(4))))

# --- D: normalisation baselines ---
# sanity: reconstruct min/max VCI/TCI from NDVI/LST to confirm the stored columns are reproduced
chk=renorm(m,0.0,1.0)
print('min/max reconstruction matches stored VCI/TCI:',
      np.allclose(chk.VCI,m.VCI,atol=1e-6,equal_nan=True), np.allclose(chk.TCI,m.TCI,atol=1e-6,equal_nan=True))
rec('min/max extrema (paper)', m)
rec('2nd/98th percentiles', renorm(m,.02,.98))
rec('5th/95th percentiles', renorm(m,.05,.95))
rec('10th/90th percentiles', renorm(m,.10,.90))
# --- E: crop-season windows (same paper normalisation) ---
rec('window Jun-Oct (paper)', m, [6,7,8,9,10])
rec('window Jul-Sep (post-wheat, sunflower-dominated)', m, [7,8,9])
rec('window Aug-Sep', m, [8,9])
rec('window Apr-Jun (wheat-dominated)', m, [4,5,6])
rec('window May-Jun', m, [5,6])
out=pd.DataFrame(rows); out.to_csv('data/revision_results/DE_normalisation_window.csv',index=False)
pd.set_option('display.width',230)
print(out[['scenario','n','top','top_composite','second','SPEI3_r','SPI3_r','SPEI9_r','ranking']].round(3).to_string(index=False))
