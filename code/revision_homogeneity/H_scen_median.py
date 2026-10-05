import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""Homogeneity scenarios S1 and S1max re-evaluated with the MEDIAN composite VHI (for Table S4b)."""
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'code')
import numpy as np, pandas as pd
from H_indices import add_pet_d, indices
from rev import metric_table, IDX, WIN, CORE, ALL7
from revision_analysis import fold
from H_scenarios_adjust import adjust
raw=pd.read_csv('data/trakya_TP_PET_1965_2024.csv'); raw['key']=raw.name.map(fold)
raw=raw[raw.key.isin(ALL7)][['name','key','year','month','temp','precip','lat']].copy()
vhi=pd.read_csv('data/trakya_district_vhi_vci_tci_monthly.csv'); vhi['key']=vhi.name.str.lower()
base=pd.read_csv('data/revision_results/H_indices_base.csv')
adj_all=pd.read_csv('data/revision_results/H_adjustments_all.csv')
def apply_all(df):
    d=df.copy()
    for r in adj_all.itertuples():
        pre=(d.key==r.station)&(d.year<r.break_year); sh=r._asdict()['shift']
        if r.variable=='temperature': d.loc[pre,'temp']+=sh
        else: d.loc[pre,'precip']*=np.exp(sh)
    return d
def ev(idx,label):
    s=vhi.merge(idx.drop(columns=['name'],errors='ignore'),on=['key','year','month']); s=s[s.key.isin(CORE)&s.month.isin(WIN)].dropna(subset=['VHI']+IDX)
    t=metric_table(s); r=t.set_index('index'); return dict(scenario=label,n=len(s),winner=t.iloc[0]['index'],winner_composite=t.iloc[0].composite,r_SPEI3=r.loc['SPEI3','pearson'],r_SPI3=r.loc['SPI3','pearson'],r_SPEI9=r.loc['SPEI9','pearson'])
rows=[ev(base,'median VHI: submitted indices')]
rows.append(ev(indices(add_pet_d(adjust(raw))),'median VHI: step-adjusted flagged T and P'))
rows.append(ev(indices(add_pet_d(apply_all(raw))),'median VHI: step-adjusted every station'))
rows.append(ev(indices(add_pet_d(raw),ref_years=set(range(1995,2025))),'median VHI: reference 1995-2024'))
pd.DataFrame(rows).to_csv('data/revision_results/H_scen_median.csv',index=False); print(pd.DataFrame(rows).round(3).to_string())
