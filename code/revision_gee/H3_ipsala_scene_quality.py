import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""Why does the İpsala contrast shrink when SLC-off scenes or the max-NDVI composite are removed?
(1) per-scene valid-pixel coverage (v2018 mask) by district and sensor
(2) coverage-filtered composites (paper scheme) -> district r, plateau-İpsala gap, winner
(3) İpsala/plateau r by sub-period under paper vs no-SLC-off
(4) VCI vs TCI decomposition
(5) paired 3-yr block bootstrap CI of the gap under paper vs no-SLC-off vs median composite"""
import sys, os; sys.path.insert(0, '.'); sys.path.insert(0, 'gee')
import numpy as np, pandas as pd
from scipy import stats
from rev import metric_table, IDX, WIN, CORE, ALL7, moving_year_sample

idx = pd.read_csv('data/trakya_spi_spei_flexible_1965_2024.csv').drop(columns=['name'])
sc = pd.read_csv('data/rev_scene_means_1985_2024.csv.gz')
sc = sc.rename(columns={c: c[:-5] for c in sc.columns if c.endswith('_mean')})
sc['date'] = pd.to_datetime(sc.millis, unit='ms'); sc['year'] = sc.date.dt.year; sc['month'] = sc.date.dt.month
sc['key'] = sc['name'].str.lower()
V = 'v2018'
full = sc.groupby('key')[f'NDVI_{V}_count'].transform('max')
sc['cov'] = sc[f'NDVI_{V}_count'] / full
sc['l7off'] = (sc.slc_off == 1)

# (1) coverage table
g = sc[sc[f'NDVI_{V}_count'] > 0]
cov = g.groupby(['key', 'l7off'])['cov'].describe(percentiles=[.1, .5, .9])[['count', '10%', '50%', '90%']].round(2)
print('(1) valid-pixel coverage of scenes with any valid pixel (v2018 mask):\n', cov.to_string())
share_off = g[g.month.isin(WIN)].groupby('key').l7off.mean().round(2)
print('share of Jun-Oct scene-district rows that are SLC-off:', share_off.to_dict())

def composite(frame, scheme='paper', var=V):
    d = frame[['key', 'year', 'month', f'NDVI_{var}', f'LST_{var}']].rename(columns={f'NDVI_{var}': 'NDVI', f'LST_{var}': 'LST'}).dropna(subset=['NDVI'])
    gg = d.groupby(['key', 'year', 'month'])
    m = gg.agg(NDVI=('NDVI', 'max'), LST=('LST', 'mean')) if scheme == 'paper' else gg.agg(NDVI=('NDVI', 'median'), LST=('LST', 'median'))
    m = m.reset_index()
    cl = m.groupby(['key', 'month']).agg(nmin=('NDVI', 'min'), nmax=('NDVI', 'max'), tmin=('LST', 'min'), tmax=('LST', 'max')).reset_index()
    x = m.merge(cl, on=['key', 'month'])
    x['VCI'] = (100 * (x.NDVI - x.nmin) / (x.nmax - x.nmin).replace(0, np.nan)).clip(0, 100)
    x['TCI'] = (100 * (x.tmax - x.LST) / (x.tmax - x.tmin).replace(0, np.nan)).clip(0, 100)
    x['VHI'] = 0.5 * (x.VCI + x.TCI)
    return x.merge(idx, on=['key', 'year', 'month'])

def district_r(x, col='VHI', years=None):
    s = x[x.month.isin(WIN) & x.key.isin(ALL7)].dropna(subset=[col, 'SPEI3'])
    if years is not None: s = s[s.year.isin(years)]
    return {k: stats.pearsonr(gg[col], gg.SPEI3).statistic for k, gg in s.groupby('key') if len(gg) > 15}

def summarise(x, label):
    s = x[x.key.isin(CORE) & x.month.isin(WIN)].dropna(subset=['VHI'] + IDX)
    t = metric_table(s); d = district_r(x)
    pl = np.mean([d[k] for k in CORE])
    return dict(variant=label, n=len(s), winner=t.iloc[0]['index'], score=t.iloc[0].composite,
                r_SPEI3_pooled=t.set_index('index').loc['SPEI3', 'pearson'], r_ipsala=d['ipsala'], plateau=pl, gap=pl - d['ipsala'],
                **{f'r_{k}': d[k] for k in CORE})

rows = []
X = {}
X['paper'] = composite(sc); rows.append(summarise(X['paper'], 'paper (all scenes, NDVI max / LST mean)'))
for thr in [0.2, 0.5, 0.8]:
    x = composite(sc[sc['cov'] >= thr]); rows.append(summarise(x, f'coverage >= {thr:.0%}'))
X['nooff'] = composite(sc[~sc.l7off]); rows.append(summarise(X['nooff'], 'no SLC-off scenes'))
x = composite(sc[~sc.l7off | (sc['cov'] >= 0.5)]); rows.append(summarise(x, 'SLC-off kept only if coverage >= 50%'))
X['median'] = composite(sc, 'median'); rows.append(summarise(X['median'], 'median composite (all scenes)'))
res = pd.DataFrame(rows); res.to_csv('data/revision_results/H3_ipsala_scene_quality.csv', index=False)
pd.set_option('display.width', 250)
print('\n(2)\n', res.round(3).to_string())

# (3) sub-periods
sp = []
for lab, x in [('paper', X['paper']), ('no SLC-off', X['nooff']), ('median', X['median'])]:
    for per, yrs in [('1985-2002 (pre SLC-off)', range(1985, 2003)), ('2003-2012', range(2003, 2013)), ('2013-2024', range(2013, 2025))]:
        d = district_r(x, years=yrs); pl = np.mean([d[k] for k in CORE])
        sp.append(dict(composite=lab, period=per, r_ipsala=d['ipsala'], plateau=pl, gap=pl - d['ipsala']))
sp = pd.DataFrame(sp); sp.to_csv('data/revision_results/H3_ipsala_subperiods.csv', index=False); print('\n(3)\n', sp.round(3).to_string())

# (4) VCI vs TCI
dc = []
for lab, x in [('paper', X['paper']), ('no SLC-off', X['nooff'])]:
    for col in ['VCI', 'TCI', 'VHI']:
        d = district_r(x, col); dc.append(dict(composite=lab, component=col, r_ipsala=d['ipsala'], plateau=np.mean([d[k] for k in CORE])))
dc = pd.DataFrame(dc); dc.to_csv('data/revision_results/H3_ipsala_vci_tci.csv', index=False); print('\n(4)\n', dc.round(3).to_string())

# (5) paired block bootstrap of the gap
rng = np.random.default_rng(11); bs = []
for lab, x in [('paper', X['paper']), ('no SLC-off', X['nooff']), ('median', X['median'])]:
    w = x[x.month.isin(WIN) & x.key.isin(ALL7)].dropna(subset=['VHI', 'SPEI3']); years = sorted(w.year.unique()); diffs = []
    byy = {y: w[w.year == y] for y in years}
    for _ in range(1000):
        s = pd.concat([byy[y] for y in moving_year_sample(rng, years, 3)])
        d = {k: stats.pearsonr(gg.VHI, gg.SPEI3).statistic for k, gg in s.groupby('key')}
        diffs.append(np.mean([d[k] for k in CORE]) - d['ipsala'])
    d0 = district_r(x); g0 = np.mean([d0[k] for k in CORE]) - d0['ipsala']
    bs.append(dict(composite=lab, gap=g0, lo=np.percentile(diffs, 2.5), hi=np.percentile(diffs, 97.5), P_le0=np.mean(np.array(diffs) <= 0)))
bs = pd.DataFrame(bs); bs.to_csv('data/revision_results/H3_ipsala_gap_ci.csv', index=False); print('\n(5)\n', bs.round(3).to_string())
