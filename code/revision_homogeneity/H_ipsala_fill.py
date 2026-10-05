import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""Effect of the temperature gap-filling method on İpsala (the only station with a multi-month gap:
1976 Apr-Oct and 1978 Sep-Dec). PCHIP across a 7-month gap produces implausible summer values
(e.g. July 1976 = 7.4 °C).  Alternative: climatological-anomaly fill = İpsala monthly climatology +
mean anomaly of the other stations in that month (standard neighbour-based infilling).
Also: İpsala VHI-SPEI-3 coupling before/after its 2015 precipitation shift."""
import sys, time; sys.path.insert(0, '.'); sys.path.insert(0, 'code')
import numpy as np, pandas as pd
from scipy import stats
from H_indices import add_pet_d, indices, SCALES
from rev import metric_table, IDX, WIN, ALL7, CORE
from revision_analysis import fold
t0 = time.time()
raw = pd.read_csv('data/trakya_TP_PET_1965_2024.csv'); raw['key'] = raw.name.map(fold)
raw = raw[raw.key.isin(ALL7)][['name', 'key', 'year', 'month', 'temp', 'precip', 'lat']].copy()
gaps = [('ipsala', 1976, m) for m in range(4, 11)] + [('ipsala', 1978, m) for m in range(9, 13)] + \
       [('luleburgaz', 2009, 3), ('luleburgaz', 2009, 4), ('corlu', 2021, 3)]
g = raw.copy(); g['pchip'] = g['temp']
for k, y, m in gaps:
    g.loc[(g.key == k) & (g.year == y) & (g.month == m), 'temp'] = np.nan
# climatology (1965-2024 excluding gap months) and neighbour anomalies
clim = g.groupby(['key', 'month'])['temp'].mean().rename('clim').reset_index()
g = g.merge(clim, on=['key', 'month']); g['anom'] = g['temp'] - g['clim']
nb = g.groupby(['year', 'month'])['anom'].mean().rename('nb_anom').reset_index()  # gap station's own NaN excluded
g = g.merge(nb, on=['year', 'month'])
fill = g['temp'].isna(); g.loc[fill, 'temp'] = g.loc[fill, 'clim'] + g.loc[fill, 'nb_anom']
print(g.loc[fill, ['key', 'year', 'month', 'pchip', 'clim', 'nb_anom', 'temp']].round(2).to_string())
g[['key', 'year', 'month', 'pchip', 'temp']].loc[fill].round(2).to_csv('data/revision_results/H_ipsala_fill_values.csv', index=False)

vhi = pd.read_csv('data/trakya_district_vhi_vci_tci_monthly_maxcomposite.csv'); vhi['key'] = vhi['name'].str.lower()
vhi['VHI'] = 0.5 * (vhi['VCI'] + vhi['TCI'])

def district_r(idx, key, years=None):
    m = vhi.merge(idx.drop(columns=['name'], errors='ignore'), on=['key', 'year', 'month'])
    m = m[(m.key == key) & m.month.isin(WIN)].dropna(subset=['VHI'] + IDX)
    if years is not None: m = m[m.year.isin(years)]
    return len(m), {i: stats.pearsonr(m.VHI, m[i]).statistic for i in ['SPEI3', 'SPI3', 'SPEI9']}

base = pd.read_csv('data/revision_results/H_indices_base.csv') if __import__('os').path.exists('data/revision_results/H_indices_base.csv') else indices(add_pet_d(raw))
alt = indices(add_pet_d(g[['name', 'key', 'year', 'month', 'temp', 'precip', 'lat']]))
alt.to_csv('data/revision_results/H_indices_anomfill.csv', index=False)
rows = []
for label, idx in [('PCHIP (paper)', base), ('climatological-anomaly fill', alt)]:
    for key in ['ipsala', 'luleburgaz', 'corlu']:
        n, r = district_r(idx, key); rows.append(dict(fill=label, district=key, period='1985-2024', n=n, **r))
    n, r = district_r(idx, 'ipsala', range(1985, 2015)); rows.append(dict(fill=label, district='ipsala', period='1985-2014 (pre P-shift)', n=n, **r))
    n, r = district_r(idx, 'ipsala', range(2015, 2025)); rows.append(dict(fill=label, district='ipsala', period='2015-2024 (post P-shift)', n=n, **r))
    t = metric_table(vhi.merge(idx.drop(columns=['name'], errors='ignore'), on=['key', 'year', 'month'])
                     .query('key in @ALL7 and month in @WIN').dropna(subset=['VHI'] + IDX))
    rows.append(dict(fill=label, district='ALL7 composite', period='1985-2024', n=int(t.iloc[0].n),
                     SPEI3=t.set_index('index').loc['SPEI3', 'composite'], SPI3=t.set_index('index').loc['SPI3', 'composite'],
                     SPEI9=t.set_index('index').loc['SPEI9', 'composite']))
# how different are the İpsala SPEI-3 values in the analysis window?
b = base[base.key == 'ipsala'].merge(alt[alt.key == 'ipsala'], on=['key', 'year', 'month'], suffixes=('_p', '_a'))
b = b[(b.year >= 1985) & b.month.isin(WIN)]
print('İpsala SPEI-3 1985-2024 Jun-Oct: RMS change', np.sqrt(np.nanmean((b.SPEI3_p - b.SPEI3_a) ** 2)).round(4),
      'corr', np.corrcoef(*np.array(b[['SPEI3_p', 'SPEI3_a']].dropna()).T)[0, 1].round(5))
out = pd.DataFrame(rows); out.to_csv('data/revision_results/H_ipsala_fill_effect.csv', index=False)
print(out.round(3).to_string()); print('elapsed', time.time() - t0)
