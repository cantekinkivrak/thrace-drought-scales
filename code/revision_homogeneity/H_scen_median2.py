import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""Extra median-composite rows for Table S4b: anomaly gap fill, reference 1985-2024, station subsets."""
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'code')
import numpy as np, pandas as pd
from H_indices import add_pet_d, indices
from rev import metric_table, IDX, WIN, CORE, ALL7
from revision_analysis import fold
raw=pd.read_csv('data/trakya_TP_PET_1965_2024.csv'); raw['key']=raw.name.map(fold)
raw=raw[raw.key.isin(ALL7)][['name','key','year','month','temp','precip','lat']].copy()
vhi=pd.read_csv('data/trakya_district_vhi_vci_tci_monthly.csv'); vhi['key']=vhi.name.str.lower()
base=pd.read_csv('data/revision_results/H_indices_base.csv'); anom=pd.read_csv('data/revision_results/H_indices_anomfill.csv')
def ev(idx,label,keys=CORE):
    s=vhi.merge(idx.drop(columns=['name'],errors='ignore'),on=['key','year','month']); s=s[s.key.isin(keys)&s.month.isin(WIN)].dropna(subset=['VHI']+IDX)
    t=metric_table(s); r=t.set_index('index'); return dict(scenario=label,n=len(s),winner=t.iloc[0]['index'],winner_composite=t.iloc[0].composite,r_SPEI3=r.loc['SPEI3','pearson'],r_SPI3=r.loc['SPI3','pearson'],r_SPEI9=r.loc['SPEI9','pearson'])
rows=[ev(anom,'median VHI: climatology + anomaly gap fill')]
rows.append(ev(indices(add_pet_d(raw),ref_years=set(range(1985,2025))),'median VHI: reference 1985-2024'))
rows.append(ev(base,'median VHI: stations without temperature break (Çorlu, Lüleburgaz, Tekirdağ)',['corlu','luleburgaz','tekirdag']))
rows.append(ev(base,'median VHI: stations without precipitation break (Çorlu, Lüleburgaz, Uzunköprü)',['corlu','luleburgaz','uzunkopru']))
pd.DataFrame(rows).to_csv('data/revision_results/H_scen_median2.csv',index=False); print(pd.DataFrame(rows).round(3).to_string())
