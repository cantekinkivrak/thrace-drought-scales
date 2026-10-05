import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""How much does the monthly NDVI-max operator rely on partial-coverage scenes, by district?"""
import numpy as np, pandas as pd
sc = pd.read_csv('data/rev_scene_means_1985_2024.csv.gz')
sc['date']=pd.to_datetime(sc.millis,unit='ms'); sc['year']=sc.date.dt.year; sc['month']=sc.date.dt.month
V='v2018'; sc['key']=sc.name.str.lower()
sc['cov']=sc[f'NDVI_{V}_count']/sc.groupby('key')[f'NDVI_{V}_count'].transform('max')
d=sc[sc.month.between(6,10)].dropna(subset=[f'NDVI_{V}_mean'])
rows=[]
for (k,y,m),g in d.groupby(['key','year','month']):
    if len(g)<2: continue
    i=g[f'NDVI_{V}_mean'].idxmax()
    rows.append(dict(key=k,year=y,month=m,n=len(g),infl=g[f'NDVI_{V}_mean'].max()-g[f'NDVI_{V}_mean'].median(),
                     cov_of_max=g.loc[i,'cov'],max_is_slcoff=int(g.loc[i,'slc_off']==1),
                     spread=g[f'NDVI_{V}_mean'].std()))
r=pd.DataFrame(rows)
out=r.groupby('key').agg(months=('n','size'),scenes_per_month=('n','median'),
    max_minus_median=('infl','median'),max_minus_median_p90=('infl',lambda s:s.quantile(.9)),
    share_max_from_cov_lt50=('cov_of_max',lambda s:(s<0.5).mean()),share_max_from_slcoff=('max_is_slcoff','mean'),
    within_month_sd=('spread','median')).round(3)
out.to_csv('data/revision_results/H6_max_inflation.csv'); print(out.to_string())
# stratify: monthly NDVI-max inflation vs coverage of the selected scene
r['covbin']=pd.cut(r.cov_of_max,[0,.2,.5,.8,1.01])
print(r.groupby(['covbin',r.key=='ipsala']).infl.median().unstack().round(3))
