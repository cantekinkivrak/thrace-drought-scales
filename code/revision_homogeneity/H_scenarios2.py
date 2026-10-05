import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""Extra stress-test scenarios: (S1u) adjust the union of stations flagged by either relative-SNHT variant
(raw-difference/log-ratio [this revision] or z-scored [submitted version]); (S1max) adjust every station at its
relative break year regardless of significance."""
import sys, time; sys.path.insert(0,'.'); sys.path.insert(0,'code')
import numpy as np, pandas as pd
from H_indices import add_pet_d, indices
from rev import metric_table, IDX, WIN, CORE, ALL7
from revision_analysis import fold
t0=time.time()
raw = pd.read_csv('data/trakya_TP_PET_1965_2024.csv'); raw['key'] = raw.name.map(fold)
raw = raw[raw.key.isin(ALL7)][['name','key','year','month','temp','precip','lat']].copy()
vhi = pd.read_csv('data/trakya_district_vhi_vci_tci_monthly_maxcomposite.csv'); vhi['key']=vhi['name'].str.lower(); vhi['VHI']=0.5*(vhi.VCI+vhi.TCI)
base_idx = pd.read_csv('data/revision_results/H_indices_base.csv')
br = pd.read_csv('data/revision_results/H_snht_breaks.csv'); rel = br[(br.test_type=='relative')&(br.series=='annual')]
paper = pd.read_csv('data/revision_results/relative_snht_screen.csv')

def shift_at(station, variable, year):
    """relative shift (raw difference / log ratio) at a given break year, from the annual relative series"""
    d = raw.copy()
    if variable=='temperature':
        a = d.pivot_table(index='year',columns='key',values='temp',aggfunc='mean'); r = a[station]-a.drop(columns=station).mean(axis=1)
    else:
        a = d.pivot_table(index='year',columns='key',values='precip',aggfunc='sum'); r = np.log(a[station])-np.log(a.drop(columns=station)).mean(axis=1)
    return float(r[r.index>=year].mean()-r[r.index<year].mean())

def apply(df, adjustments):
    d=df.copy()
    for st,var,yr,sh in adjustments:
        pre=(d.key==st)&(d.year<yr)
        if var=='temperature': d.loc[pre,'temp']+=sh
        else: d.loc[pre,'precip']*=np.exp(sh)
    return d

def sample(idx, keys=CORE):
    m=vhi.merge(idx.drop(columns=['name'],errors='ignore'),on=['key','year','month'])
    return m[m.key.isin(keys)&m.month.isin(WIN)].dropna(subset=['VHI']+IDX)
def run(label, idx):
    t=metric_table(sample(idx)); r=t.set_index('index'); top=t.iloc[0]
    b=base_idx.merge(idx,on=['key','year','month'],suffixes=('_b','')); b=b[b.key.isin(CORE)&b.month.isin(WIN)&(b.year>=1985)]
    row=dict(scenario=label,n=int(top.n),winner=top['index'],winner_composite=top.composite,r_SPEI3=r.loc['SPEI3','pearson'],comp_SPEI3=r.loc['SPEI3','composite'],
             r_SPI3=r.loc['SPI3','pearson'],comp_SPI3=r.loc['SPI3','composite'],r_SPEI9=r.loc['SPEI9','pearson'],comp_SPEI9=r.loc['SPEI9','composite'],
             r_SPEI6=r.loc['SPEI6','pearson'],r_SPEI1=r.loc['SPEI1','pearson'],r_SPEI12=r.loc['SPEI12','pearson'],
             rms_change_SPEI3=float(np.sqrt(np.nanmean((b.SPEI3-b.SPEI3_b)**2))), corr_SPEI3_vs_base=float(np.corrcoef(*np.array(b[['SPEI3','SPEI3_b']].dropna()).T)[0,1]))
    print(f"[{label}] winner={row['winner']} ({row['winner_composite']}) r3={row['r_SPEI3']:.3f} r9={row['r_SPEI9']:.3f} {time.time()-t0:.0f}s",flush=True)
    return row, t.assign(scenario=label)

# union of flags (p<0.05 in either variant); break year from the raw-difference test when flagged there, else from the z-scored test
adj_u=[]
for var in ['temperature','precipitation']:
    for st in ALL7:
        mine=rel[(rel.variable==var)&(rel.station==st)].iloc[0]; pap=paper[(paper.variable==var)&(paper.station==st)].iloc[0]
        if mine.SNHT_p<0.05: adj_u.append((st,var,int(mine.break_year_first_of_new_segment),float(mine['shift'])))
        elif pap.permutation_p<0.05: yr=int(pap.candidate_break_year); adj_u.append((st,var,yr,shift_at(st,var,yr)))
adj_max=[(r.station,r.variable,int(r.break_year_first_of_new_segment),float(r._asdict()['shift'])) for r in rel.itertuples()]
pd.DataFrame(adj_u,columns=['station','variable','break_year','shift']).to_csv('data/revision_results/H_adjustments_union.csv',index=False)
pd.DataFrame(adj_max,columns=['station','variable','break_year','shift']).to_csv('data/revision_results/H_adjustments_all.csv',index=False)
print('union adjustments:',adj_u)
rows=[];tabs=[]
for label,adj in [('S1u step-adjusted: union of both SNHT variants',adj_u),('S1max step-adjusted: every station at its break year',adj_max)]:
    idx=indices(add_pet_d(apply(raw,adj))); r,t=run(label,idx); rows.append(r); tabs.append(t)
pd.DataFrame(rows).to_csv('data/revision_results/H_scenarios2_summary.csv',index=False); pd.concat(tabs).to_csv('data/revision_results/H_scenarios2_selection.csv',index=False)
print(pd.DataFrame(rows).round(3).to_string())
