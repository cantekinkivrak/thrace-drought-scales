import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""R2 #2 + minors: CIs for district couplings and for the n=7 coupling-vs-amplitude relation."""
import numpy as np, pandas as pd
from scipy import stats
from rev import *
m=load_monthly(); rng=np.random.default_rng(3)
win=m[m.month.isin(WIN)].dropna(subset=['VHI','SPEI3'])
rows=[]
for d in ALL7:
    g=win[win.key==d]; years=sorted(g.year.unique())
    r=stats.pearsonr(g.SPEI3,g.VHI).statistic
    bs=[]
    for _ in range(2000):
        ys=moving_year_sample(rng,years,3); s=pd.concat([g[g.year==y] for y in ys])
        bs.append(stats.pearsonr(s.SPEI3,s.VHI).statistic)
    lo,hi=np.percentile(bs,[2.5,97.5])
    # Fisher z (independent-obs approximation, for reference)
    z=np.arctanh(r); se=1/np.sqrt(len(g)-3); flo,fhi=np.tanh(z-1.96*se),np.tanh(z+1.96*se)
    rows.append(dict(district=d,n=len(g),r=r,block_lo=lo,block_hi=hi,fisher_lo=flo,fisher_hi=fhi))
ci=pd.DataFrame(rows); ci.to_csv('data/revision_results/F_district_coupling_ci.csv',index=False)
print(ci.round(3).to_string(index=False))

# plateau-vs-Ipsala difference, paired on years (same years resampled for all districts)
pl=[d for d in ALL7 if d!='ipsala']; years=sorted(win.year.unique()); diffs=[]
for _ in range(2000):
    ys=moving_year_sample(rng,years,3); s=pd.concat([win[win.year==y] for y in ys])
    rp=np.mean([stats.pearsonr(s[s.key==d].SPEI3,s[s.key==d].VHI).statistic for d in pl])
    ri=stats.pearsonr(s[s.key=='ipsala'].SPEI3,s[s.key=='ipsala'].VHI).statistic
    diffs.append(rp-ri)
print(f'\nplateau mean r − İpsala r: point {ci[ci.district!="ipsala"].r.mean()-ci[ci.district=="ipsala"].r.iloc[0]:.3f}, block-bootstrap 95% [{np.percentile(diffs,2.5):.3f}, {np.percentile(diffs,97.5):.3f}], P(diff<=0)={np.mean(np.array(diffs)<=0):.3f}')

# n=7 coupling vs seasonal NDVI amplitude
clim=m.groupby(['key','month'])['NDVI'].mean().unstack()
amp=(clim.max(axis=1)-clim.min(axis=1)).rename('amp')
cp=ci.set_index('district')['r']
df=pd.concat([cp,amp],axis=1).dropna()
r7,p7=stats.pearsonr(df.amp,df.r)
z=np.arctanh(r7); se=1/np.sqrt(len(df)-3)
print(f'\ncoupling ~ NDVI amplitude (n={len(df)}): r={r7:.2f}, p={p7:.2f}, Fisher 95% CI [{np.tanh(z-1.96*se):.2f}, {np.tanh(z+1.96*se):.2f}]')
rs=[]
for _ in range(5000):
    i=rng.integers(0,len(df),len(df)); 
    if df.amp.iloc[i].std()==0: continue
    rs.append(stats.pearsonr(df.amp.iloc[i],df.r.iloc[i]).statistic)
print(f'  bootstrap 95% CI [{np.percentile(rs,2.5):.2f}, {np.percentile(rs,97.5):.2f}]')
print('  amplitudes:', df.amp.round(3).to_dict())
