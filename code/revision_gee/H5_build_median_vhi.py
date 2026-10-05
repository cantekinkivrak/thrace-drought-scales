import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""Primary VHI for the revision: monthly MEDIAN composite of per-scene district means (NDVI and LST),
CORINE-2018 class-211 mask, all scenes; VCI/TCI by within-district-month min/max over 1985-2024; VHI=0.5(VCI+TCI).
Output has the same layout as the archived trakya_district_vhi_vci_tci_monthly_maxcomposite.csv."""
import numpy as np, pandas as pd
sc = pd.read_csv('data/rev_scene_means_1985_2024.csv.gz')
sc['date'] = pd.to_datetime(sc.millis, unit='ms'); sc['year'] = sc.date.dt.year; sc['month'] = sc.date.dt.month
V = 'v2018'
d = sc[['name', 'year', 'month', f'NDVI_{V}_mean', f'LST_{V}_mean']].rename(columns={f'NDVI_{V}_mean': 'NDVI', f'LST_{V}_mean': 'LST'})
n_sc = d.groupby(['name', 'year', 'month']).size().rename('n_scenes')
n_valid = d.dropna(subset=['NDVI']).groupby(['name', 'year', 'month']).size().rename('n_valid')
m = d.dropna(subset=['NDVI']).groupby(['name', 'year', 'month']).agg(NDVI=('NDVI', 'median'), LST=('LST', 'median'))
grid = pd.MultiIndex.from_product([sorted(sc.name.unique()), range(1985, 2025), range(1, 13)], names=['name', 'year', 'month'])
m = m.reindex(grid).join(n_sc).join(n_valid).reset_index()
m[['n_scenes', 'n_valid']] = m[['n_scenes', 'n_valid']].fillna(0).astype(int)
m['date'] = pd.to_datetime(dict(year=m.year, month=m.month, day=1)).dt.strftime('%Y-%m-%d')
cl = m.groupby(['name', 'month']).agg(NDVI_min=('NDVI', 'min'), NDVI_max=('NDVI', 'max'), LST_min=('LST', 'min'), LST_max=('LST', 'max')).reset_index()
m = m.merge(cl, on=['name', 'month']).sort_values(['name', 'year', 'month'])
m['VCI'] = (100 * (m.NDVI - m.NDVI_min) / (m.NDVI_max - m.NDVI_min)).clip(0, 100)
m['TCI'] = (100 * (m.LST_max - m.LST) / (m.LST_max - m.LST_min)).clip(0, 100)
m['alpha'] = 0.5
m['VHI'] = 0.5 * (m.VCI + m.TCI)
m['composite'] = 'median'
cols = ['name', 'year', 'month', 'NDVI', 'LST', 'n_scenes', 'n_valid', 'date', 'NDVI_min', 'NDVI_max', 'LST_min', 'LST_max', 'VCI', 'TCI', 'alpha', 'VHI', 'composite']
m[cols].to_csv('data/trakya_district_vhi_vci_tci_monthly.csv', index=False)
print(m.shape, 'missing VHI:', m.VHI.isna().sum(), f'({100*m.VHI.isna().mean():.1f}%)')
print('Jun-Oct missing by district:', m[m.month.between(6,10)].groupby('name').VHI.apply(lambda s: s.isna().sum()).to_dict())
print('missing by month:', m.groupby('month').VHI.apply(lambda s: s.isna().sum()).to_dict())
a = pd.read_csv('data/trakya_district_vhi_vci_tci_monthly_maxcomposite.csv'); a['VHIc'] = 0.5*(a.VCI+a.TCI)
c = m.merge(a[['name','year','month','VHIc']], on=['name','year','month']).dropna(subset=['VHI','VHIc'])
print('corr(median VHI, submitted VHI):', round(np.corrcoef(c.VHI, c.VHIc)[0,1], 3), 'n', len(c))
for k, g in c[c.month.between(6,10)].groupby('name'): print(f'  {k:11s} r={np.corrcoef(g.VHI,g.VHIc)[0,1]:.3f}')
