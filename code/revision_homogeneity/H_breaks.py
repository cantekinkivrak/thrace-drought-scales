import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""Homogeneity diagnostics for the station series (R2 #6, R1 #2).
(1) Quantify PCHIP-filled temperature months from the raw MGM export.
(2) Absolute and relative SNHT (single shift, Alexandersson 1986) + Pettitt test on annual mean T,
    annual P and June-October seasonal T / P; break year, magnitude, permutation p.
Outputs: out/H_pchip_gaps.csv, out/H_snht_breaks.csv"""
import sys, re, numpy as np, pandas as pd
sys.path.insert(0, 'code'); from revision_analysis import fold
UP = '/mnt/user-data/uploads/paper 2'
rng = np.random.default_rng(437)

# ---------- (1) raw MGM temperature export: which months were missing? ----------
y = pd.read_excel(f'{UP}/202605047ED4-Aylık Ortalama Sıcaklık (°C).xlsx', header=None)
blocks = []
for i in range(len(y)):
    row = y.iloc[i].tolist()
    s = [v for v in row if isinstance(v, str) and 'İstasyon' in v]
    if s:
        name = s[0].split(':')[1].split('/')[0].strip()
        # header row 'Yıl/Ay' is 3 rows below; find its column
        hdr = y.iloc[i + 3].tolist(); c0 = [j for j, v in enumerate(hdr) if isinstance(v, str) and 'Yıl' in v][0]
        j = i + 4
        while j < len(y) and str(y.iloc[j, c0]).strip().isdigit():
            vals = [pd.to_numeric(y.iloc[j, c0 + m], errors='coerce') for m in range(1, 13)]
            blocks.append(dict(station=name, year=int(y.iloc[j, c0]), **{f'm{m}': vals[m - 1] for m in range(1, 13)}))
            j += 1
raw = pd.DataFrame(blocks)
raw = raw[raw.year.between(1965, 2024)]
long = raw.melt(id_vars=['station', 'year'], var_name='m', value_name='temp')
long['month'] = long['m'].str[1:].astype(int)
gaps = long[long.temp.isna()].sort_values(['station', 'year', 'month'])
summary = (long.groupby('station')['temp'].apply(lambda s: s.isna().sum()).rename('missing_months').reset_index())
summary['total_months'] = 60 * 12
summary['pct'] = 100 * summary.missing_months / summary.total_months
summary['missing_list'] = summary.station.map(
    lambda st: '; '.join(f'{r.year}-{r.month:02d}' for r in gaps[gaps.station == st].itertuples()))
# growing-season (Jun-Oct) gaps and gaps inside the VHI period (1985-2024)
summary['missing_JunOct'] = summary.station.map(lambda st: int(gaps[(gaps.station == st) & gaps.month.between(6, 10)].shape[0]))
summary['missing_1985_2024'] = summary.station.map(lambda st: int(gaps[(gaps.station == st) & (gaps.year >= 1985)].shape[0]))
summary.to_csv('data/revision_results/H_pchip_gaps.csv', index=False)
print(summary.drop(columns='missing_list').to_string()); print(gaps[['station', 'year', 'month']].to_string())

# check that the archived (filled) file equals raw where raw is present
arch = pd.read_csv('data/trakya_TP_PET_1965_2024.csv'); arch['key'] = arch.name.map(fold)
long['key'] = long.station.map(fold)
cmp = long.merge(arch[['key', 'year', 'month', 'temp']], on=['key', 'year', 'month'], suffixes=('_raw', '_arch'))
ok = cmp.dropna(subset=['temp_raw'])
print('stations matched:', sorted(cmp.key.unique()))
print('max |raw - archived| where raw present:', (ok.temp_raw - ok.temp_arch).abs().max())
filled = cmp[cmp.temp_raw.isna()]
print('filled values (archived) at raw gaps:\n', filled[['key', 'year', 'month', 'temp_arch']].to_string())

# ---------- (2) SNHT / Pettitt ----------
def snht_stat(z):
    n = len(z); best = (-1, None)
    for a in range(10, n - 9):
        t = a * z[:a].mean() ** 2 + (n - a) * z[a:].mean() ** 2
        if t > best[0]: best = (t, a)
    return best

def pettitt(x):
    n = len(x); r = pd.Series(x).rank().to_numpy(); U = [2 * r[:k].sum() - k * (n + 1) for k in range(1, n)]
    K = int(np.argmax(np.abs(U))); Kmax = abs(U[K])
    p = 2 * np.exp(-6 * Kmax ** 2 / (n ** 3 + n ** 2)); return int(K + 1), float(min(p, 1))

def test_series(s, label, variable, station, kind, B=4999):
    s = s.dropna(); z = ((s - s.mean()) / s.std(ddof=1)).to_numpy(); n = len(z)
    T0, a = snht_stat(z)
    perm = np.array([snht_stat(rng.permutation(z))[0] for _ in range(B)])
    p = (1 + (perm >= T0).sum()) / (B + 1)
    yr = int(s.index[a]); before = s.iloc[:a].mean(); after = s.iloc[a:].mean()
    K, pp = pettitt(s.to_numpy())
    return dict(variable=variable, series=label, station=station, test_type=kind, n_years=n, SNHT_T0=round(T0, 2),
                SNHT_p=round(p, 4), break_year_first_of_new_segment=yr, mean_before=round(before, 2), mean_after=round(after, 2),
                shift=round(after - before, 2), shift_pct=round(100 * (after - before) / abs(before), 1) if before != 0 else np.nan,
                pettitt_year=int(s.index[K]), pettitt_p=round(pp, 4))

arch = arch[arch.key != 'sariyer']
ann_T = arch.pivot_table(index='year', columns='key', values='temp', aggfunc='mean')
ann_P = arch.pivot_table(index='year', columns='key', values='precip', aggfunc='sum')
gs = arch[arch.month.between(6, 10)]
gs_T = gs.pivot_table(index='year', columns='key', values='temp', aggfunc='mean')
gs_P = gs.pivot_table(index='year', columns='key', values='precip', aggfunc='sum')
rows = []
for variable, tab in [('temperature', ann_T), ('precipitation', ann_P)]:
    for label, frame in [('annual', tab), ('Jun-Oct', gs_T if variable == 'temperature' else gs_P)]:
        for st in frame.columns:
            rows.append(test_series(frame[st], label, variable, st, 'absolute'))
            if variable == 'temperature':
                rel = frame[st] - frame.drop(columns=st).mean(axis=1)          # additive reference (°C)
            else:
                rel = np.log(frame[st]) - np.log(frame.drop(columns=st)).mean(axis=1)  # log-ratio reference
            r = test_series(rel, label, variable, st, 'relative')
            if variable == 'precipitation':  # express shift as percent ratio
                r['shift_pct'] = round(100 * (np.exp(r['shift']) - 1), 1)
            rows.append(r)
res = pd.DataFrame(rows)
res.to_csv('data/revision_results/H_snht_breaks.csv', index=False)
pd.set_option('display.width', 250)
print(res[(res.test_type == 'relative')].to_string())
print(res[(res.test_type == 'absolute')].to_string())
