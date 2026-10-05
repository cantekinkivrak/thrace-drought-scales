import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""Offline post-processing of the notebook-09 per-scene table (rev_scene_means_1985_2024.csv).
Produces, with the paper's own selection code (rev.metric_table = revision_analysis.metric_table semantics):
  (A) CORINE vintage / stable-211 / WorldCover / time-matched mask  -> scale selection + İpsala contrast
  (B) compositing alternatives (NDVI-max/LST-mean [paper], mean/mean, median/median, same-scene, max/max)
  (C) Landsat-7 SLC-off ablation (drop SLC-off scenes; drop all L7)
  (D) crop-specific VHI (wheat-like / sunflower-like phenology masks) x crop-season windows
Usage: python H2_gee_postprocess.py <scene_csv> [out_dir]
"""
import sys, os, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../revision_offline')
from rev import metric_table, IDX, WIN, CORE, ALL7

scene_csv = sys.argv[1] if len(sys.argv) > 1 else 'rev_scene_means_1985_2024.csv'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'data/revision_results'
os.makedirs(OUT, exist_ok=True)
idx = pd.read_csv('data/trakya_spi_spei_flexible_1965_2024.csv').drop(columns=['name'])
arch = pd.read_csv('data/trakya_district_vhi_vci_tci_monthly_maxcomposite.csv'); arch['key'] = arch['name'].str.lower()

sc = pd.read_csv(scene_csv)
sc = sc.rename(columns={c: c[:-5] for c in sc.columns if c.endswith('_mean')})   # v2 export: NDVI_x_mean -> NDVI_x (counts kept as *_count)
sc['date'] = pd.to_datetime(sc['millis'], unit='ms'); sc['year'] = sc.date.dt.year; sc['month'] = sc.date.dt.month
sc['key'] = sc['name'].str.lower()
VARIANTS = sorted({c[5:] for c in sc.columns if c.startswith('NDVI_') and not c.endswith('_count')})
print('mask variants:', VARIANTS, '| rows', len(sc), '| sensors', sc.sensor.value_counts().to_dict())
VINTAGE_OF_YEAR = lambda y: 'v1990' if y < 1995 else 'v2000' if y < 2004 else 'v2006' if y < 2010 else 'v2012' if y < 2016 else 'v2018'


def composite(frame, variant, scheme='paper'):
    """Monthly district composite from per-scene means. scheme: paper (NDVI max, LST mean), mean, median,
    samescene (LST from the scene with maximum NDVI), max (both max)."""
    d = frame[['key', 'year', 'month', 'date', f'NDVI_{variant}', f'LST_{variant}']].rename(
        columns={f'NDVI_{variant}': 'NDVI', f'LST_{variant}': 'LST'}).dropna(subset=['NDVI'])
    g = d.groupby(['key', 'year', 'month'])
    if scheme == 'paper':   m = g.agg(NDVI=('NDVI', 'max'), LST=('LST', 'mean'))
    elif scheme == 'mean':  m = g.agg(NDVI=('NDVI', 'mean'), LST=('LST', 'mean'))
    elif scheme == 'median': m = g.agg(NDVI=('NDVI', 'median'), LST=('LST', 'median'))
    elif scheme == 'max':   m = g.agg(NDVI=('NDVI', 'max'), LST=('LST', 'max'))
    elif scheme == 'samescene':
        m = d.loc[g['NDVI'].idxmax()].set_index(['key', 'year', 'month'])[['NDVI', 'LST']]
    m = m.reset_index()
    m['n_scenes'] = g.size().to_numpy()
    return m


def vhi_from_monthly(m):
    cl = m.groupby(['key', 'month']).agg(NDVI_min=('NDVI', 'min'), NDVI_max=('NDVI', 'max'),
                                         LST_min=('LST', 'min'), LST_max=('LST', 'max')).reset_index()
    x = m.merge(cl, on=['key', 'month'])
    dv = (x.NDVI_max - x.NDVI_min).replace(0, np.nan); dt = (x.LST_max - x.LST_min).replace(0, np.nan)
    x['VCI'] = (100 * (x.NDVI - x.NDVI_min) / dv).clip(0, 100)
    x['TCI'] = (100 * (x.LST_max - x.LST) / dt).clip(0, 100)
    x['VHI'] = 0.5 * (x.VCI + x.TCI)
    return x


def evaluate(m, label, months=WIN, keys=CORE):
    s = m.merge(idx, on=['key', 'year', 'month'])
    s = s[s.key.isin(keys) & s.month.isin(months)].dropna(subset=['VHI'] + IDX)
    t = metric_table(s); r = t.set_index('index')
    row = dict(variant=label, n=len(s), winner=t.iloc[0]['index'], winner_composite=t.iloc[0].composite,
               r_SPEI1=r.loc['SPEI1', 'pearson'], r_SPEI3=r.loc['SPEI3', 'pearson'], comp_SPEI3=r.loc['SPEI3', 'composite'],
               r_SPEI6=r.loc['SPEI6', 'pearson'], r_SPEI9=r.loc['SPEI9', 'pearson'], comp_SPEI9=r.loc['SPEI9', 'composite'],
               r_SPI3=r.loc['SPI3', 'pearson'])
    # district SPEI-3 coupling and plateau-İpsala gap
    s7 = m.merge(idx, on=['key', 'year', 'month']); s7 = s7[s7.key.isin(ALL7) & s7.month.isin(months)].dropna(subset=['VHI'] + IDX)
    dr = {k: stats.pearsonr(g.VHI, g.SPEI3).statistic for k, g in s7.groupby('key') if len(g) > 20}
    row['r_ipsala'] = dr.get('ipsala', np.nan); row['plateau_mean_r'] = np.mean([dr[k] for k in CORE if k in dr])
    row['gap'] = row['plateau_mean_r'] - row['r_ipsala']
    return row, t.assign(variant=label)


rows, tabs = [], []
# ---- validation: paper mask + paper scheme vs archived monthly file ----
m0 = vhi_from_monthly(composite(sc, 'v2018', 'paper'))
v = m0.merge(arch, on=['key', 'year', 'month'], suffixes=('', '_arch'))
print(f"validation vs archived (v2018, paper scheme): n={len(v)} NDVI r={np.corrcoef(v.NDVI, v.NDVI_arch)[0,1]:.4f} "
      f"LST r={np.corrcoef(v.LST, v.LST_arch)[0,1]:.4f} VHI r={np.corrcoef(v.VHI, 0.5*(v.VCI_arch+v.TCI_arch))[0,1]:.4f}")

# (A) masks
for var in VARIANTS:
    if var in ('wheat', 'sun'): continue
    r, t = evaluate(vhi_from_monthly(composite(sc, var, 'paper')), f'A mask={var}'); rows.append(r); tabs.append(t)
# time-matched vintage
parts = []
for y, g in sc.groupby('year'):
    vv = VINTAGE_OF_YEAR(y)
    parts.append(g[['key', 'year', 'month', 'date', f'NDVI_{vv}', f'LST_{vv}']].rename(columns={f'NDVI_{vv}': 'NDVI_tm', f'LST_{vv}': 'LST_tm'}))
sc_tm = pd.concat(parts)
r, t = evaluate(vhi_from_monthly(composite(sc_tm, 'tm', 'paper')), 'A mask=time-matched vintage'); rows.append(r); tabs.append(t)

# (B) compositing schemes on the paper mask
for scheme in ['paper', 'mean', 'median', 'samescene', 'max']:
    r, t = evaluate(vhi_from_monthly(composite(sc, 'v2018', scheme)), f'B composite={scheme}'); rows.append(r); tabs.append(t)

# (C) SLC-off ablation
r, t = evaluate(vhi_from_monthly(composite(sc[sc.slc_off != 1], 'v2018', 'paper')), 'C no SLC-off scenes'); rows.append(r); tabs.append(t)
r, t = evaluate(vhi_from_monthly(composite(sc[sc.sensor != 'L7'], 'v2018', 'paper')), 'C no Landsat-7 at all'); rows.append(r); tabs.append(t)
r, t = evaluate(vhi_from_monthly(composite(sc[sc.sensor.isin(['L8', 'L9'])], 'v2018', 'paper')), 'C Landsat-8/9 only (2013-2024)'); rows.append(r); tabs.append(t)

# (D) crop-specific VHI x crop-season window
WINDOWS = {'Jun-Oct (paper)': [6, 7, 8, 9, 10], 'Apr-Jun (wheat season)': [4, 5, 6], 'Jul-Sep (sunflower season)': [7, 8, 9]}
for var in ['v2018', 'wheat', 'sun']:
    if var not in VARIANTS: continue
    mv = vhi_from_monthly(composite(sc, var, 'paper'))
    for wl, months in WINDOWS.items():
        r, t = evaluate(mv, f'D pixels={var} window={wl}', months=months); rows.append(r); tabs.append(t)

res = pd.DataFrame(rows); res.to_csv(f'{OUT}/H2_gee_summary.csv', index=False)
pd.concat(tabs, ignore_index=True).to_csv(f'{OUT}/H2_gee_selection_tables.csv', index=False)
pd.set_option('display.width', 250); print(res.round(3).to_string())
