import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""Mask-vintage, SLC-off and coverage sensitivity with the MEDIAN composite as reference (for SI Table)."""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
from scipy import stats
from rev import metric_table, IDX, WIN, CORE, ALL7
idx=pd.read_csv('data/trakya_spi_spei_flexible_1965_2024.csv').drop(columns=['name'])
sc=pd.read_csv('data/rev_scene_means_1985_2024.csv.gz'); sc=sc.rename(columns={c:c[:-5] for c in sc.columns if c.endswith('_mean')})
sc['date']=pd.to_datetime(sc.millis,unit='ms'); sc['year']=sc.date.dt.year; sc['month']=sc.date.dt.month; sc['key']=sc.name.str.lower()
sc['cov']=sc['NDVI_v2018_count']/sc.groupby('key')['NDVI_v2018_count'].transform('max')
VY=lambda y:'v1990' if y<1995 else 'v2000' if y<2004 else 'v2006' if y<2010 else 'v2012' if y<2016 else 'v2018'
def comp(f,var,scheme='median'):
    d=f[['key','year','month',f'NDVI_{var}',f'LST_{var}']].rename(columns={f'NDVI_{var}':'NDVI',f'LST_{var}':'LST'}).dropna(subset=['NDVI'])
    g=d.groupby(['key','year','month']); m=(g.agg(NDVI=('NDVI','median'),LST=('LST','median')) if scheme=='median' else g.agg(NDVI=('NDVI','max'),LST=('LST','mean'))).reset_index()
    cl=m.groupby(['key','month']).agg(a=('NDVI','min'),b=('NDVI','max'),c=('LST','min'),e=('LST','max')).reset_index(); x=m.merge(cl,on=['key','month'])
    x['VCI']=(100*(x.NDVI-x.a)/(x.b-x.a).replace(0,np.nan)).clip(0,100); x['TCI']=(100*(x.e-x.LST)/(x.e-x.c).replace(0,np.nan)).clip(0,100); x['VHI']=0.5*(x.VCI+x.TCI)
    return x.merge(idx,on=['key','year','month'])
def ev(x,label):
    s=x[x.key.isin(CORE)&x.month.isin(WIN)].dropna(subset=['VHI']+IDX); t=metric_table(s); r=t.set_index('index')
    s7=x[x.key.isin(ALL7)&x.month.isin(WIN)].dropna(subset=['VHI','SPEI3']); d={k:stats.pearsonr(g.VHI,g.SPEI3).statistic for k,g in s7.groupby('key')}
    pl=np.mean([d[k] for k in CORE])
    return dict(variant=label,n=len(s),winner=t.iloc[0]['index'],score=t.iloc[0].composite,r_SPEI3=r.loc['SPEI3','pearson'],r_SPEI9=r.loc['SPEI9','pearson'],r_SPI3=r.loc['SPI3','pearson'],r_ipsala=d['ipsala'],plateau=pl,gap=pl-d['ipsala'])
rows=[]
rows.append(ev(comp(sc,'v2018'),'median composite, CORINE 2018 (primary)'))
for v,lab in [('v2012','CORINE 2012'),('v2006','CORINE 2006'),('v2000','CORINE 2000'),('v1990','CORINE 1990'),('stable','stable class 211 in all five vintages'),('wc40','ESA WorldCover 2021 cropland'),('none','no mask (whole district)')]:
    rows.append(ev(comp(sc,v),f'median composite, {lab}'))
parts=[]
for y,g in sc.groupby('year'):
    vv=VY(y); parts.append(g[['key','year','month',f'NDVI_{vv}',f'LST_{vv}']].rename(columns={f'NDVI_{vv}':'NDVI_tm',f'LST_{vv}':'LST_tm'}))
rows.append(ev(comp(pd.concat(parts),'tm'),'median composite, time-matched CORINE vintage'))
rows.append(ev(comp(sc[sc.slc_off!=1],'v2018'),'median composite, SLC-off scenes excluded'))
rows.append(ev(comp(sc[sc.sensor!='L7'],'v2018'),'median composite, Landsat 7 excluded'))
rows.append(ev(comp(sc[sc['cov']>=0.5],'v2018'),'median composite, scenes with ≥50% valid cropland pixels'))
rows.append(ev(comp(sc,'v2018','max'),'NDVI-max / LST-mean composite (submitted version)'))
rows.append(ev(comp(sc[sc.slc_off!=1],'v2018','max'),'NDVI-max / LST-mean, SLC-off excluded'))
r=pd.DataFrame(rows); r.to_csv('data/revision_results/H7_median_sensitivity.csv',index=False); pd.set_option('display.width',250); print(r.round(3).to_string())
