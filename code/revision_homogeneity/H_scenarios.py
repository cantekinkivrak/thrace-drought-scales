import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""Scale-selection robustness to station inhomogeneities (R2 #6, R1 #2).
Scenarios re-standardise the indices with the paper's exact pipeline (H_indices.py) and re-run the
four-criterion selection on the primary sample (six plateau districts, Jun-Oct, complete case).
Outputs: out/H_scenarios_selection.csv (full metric tables), out/H_scenarios_summary.csv"""
import sys, time; sys.path.insert(0, '.'); sys.path.insert(0, 'code')
import numpy as np, pandas as pd
from H_indices import add_pet_d, indices, SCALES
from rev import metric_table, IDX, WIN, CORE, ALL7
from revision_analysis import fold
t0 = time.time()

raw = pd.read_csv('data/trakya_TP_PET_1965_2024.csv'); raw['key'] = raw.name.map(fold)
raw = raw[raw.key.isin(ALL7)][['name', 'key', 'year', 'month', 'temp', 'precip', 'lat']].copy()
vhi = pd.read_csv('data/trakya_district_vhi_vci_tci_monthly_maxcomposite.csv'); vhi['key'] = vhi['name'].str.lower()
vhi['VHI'] = 0.5 * (vhi['VCI'] + vhi['TCI'])
breaks = pd.read_csv('data/revision_results/H_snht_breaks.csv')
sig = breaks[(breaks.test_type == 'relative') & (breaks.series == 'annual') & (breaks.SNHT_p < 0.05)]
print(sig[['variable', 'station', 'break_year_first_of_new_segment', 'shift', 'shift_pct', 'SNHT_p']].to_string())


def sample(idx, keys=CORE, months=WIN):
    m = vhi.merge(idx.drop(columns=['name'], errors='ignore'), on=['key', 'year', 'month'])
    return m[m.key.isin(keys) & m.month.isin(months)].dropna(subset=['VHI'] + IDX)


def run(label, idx, keys=CORE, base=None):
    s = sample(idx, keys)
    t = metric_table(s); t.insert(0, 'scenario', label)
    top = t.iloc[0]
    r = t.set_index('index')
    row = dict(scenario=label, n=int(top.n), winner=top['index'], winner_composite=top.composite,
               r_SPEI3=r.loc['SPEI3', 'pearson'], comp_SPEI3=r.loc['SPEI3', 'composite'],
               r_SPI3=r.loc['SPI3', 'pearson'], comp_SPI3=r.loc['SPI3', 'composite'],
               r_SPEI9=r.loc['SPEI9', 'pearson'], comp_SPEI9=r.loc['SPEI9', 'composite'],
               r_SPEI6=r.loc['SPEI6', 'pearson'], r_SPEI1=r.loc['SPEI1', 'pearson'], r_SPEI12=r.loc['SPEI12', 'pearson'])
    if base is not None:  # how much did the SPEI-3 values themselves move?
        b = base.merge(idx, on=['key', 'year', 'month'], suffixes=('_b', ''))
        b = b[b.key.isin(keys) & b.month.isin(WIN) & (b.year >= 1985)]
        row['rms_change_SPEI3'] = float(np.sqrt(np.nanmean((b.SPEI3 - b.SPEI3_b) ** 2)))
        row['corr_SPEI3_vs_base'] = float(np.corrcoef(*np.array(b[['SPEI3', 'SPEI3_b']].dropna()).T)[0, 1])
    print(f"[{label}] n={row['n']} winner={row['winner']} ({row['winner_composite']}) r3={row['r_SPEI3']:.3f} "
          f"r9={row['r_SPEI9']:.3f} {time.time()-t0:.0f}s", flush=True)
    return row, t


def adjust(df, which=('temperature', 'precipitation')):
    d = df.copy()
    for b in sig.itertuples():
        if b.variable not in which: continue
        pre = (d.key == b.station) & (d.year < b.break_year_first_of_new_segment)
        if b.variable == 'temperature':
            d.loc[pre, 'temp'] = d.loc[pre, 'temp'] + b.shift          # bring pre-break segment to post-break level
        else:
            d.loc[pre, 'precip'] = d.loc[pre, 'precip'] * np.exp(b.shift)  # log-ratio shift -> multiplicative factor
    return d


rows, tables = [], []
base_idx = indices(add_pet_d(raw))
r, t = run('S0 baseline (reproduced)', base_idx); rows.append(r); tables.append(t)

for label, which in [('S1 step-adjusted: T and P', ('temperature', 'precipitation')),
                     ('S1a step-adjusted: T only', ('temperature',)),
                     ('S1b step-adjusted: P only', ('precipitation',))]:
    idx = indices(add_pet_d(adjust(raw, which)))
    r, t = run(label, idx, base=base_idx); rows.append(r); tables.append(t)

for label, yrs in [('S2 reference period 1985-2024', range(1985, 2025)), ('S3 reference period 1995-2024', range(1995, 2025))]:
    idx = indices(add_pet_d(raw), ref_years=set(yrs))
    r, t = run(label, idx, base=base_idx); rows.append(r); tables.append(t)

r, t = run('S4 stations without T break (corlu, luleburgaz, tekirdag)', base_idx, keys=['corlu', 'luleburgaz', 'tekirdag']); rows.append(r); tables.append(t)
r, t = run('S5 stations without P break (corlu, luleburgaz, uzunkopru)', base_idx, keys=['corlu', 'luleburgaz', 'uzunkopru']); rows.append(r); tables.append(t)
r, t = run('S6 stations with T break only (edirne, kirklareli, uzunkopru)', base_idx, keys=['edirne', 'kirklareli', 'uzunkopru']); rows.append(r); tables.append(t)

pd.DataFrame(rows).to_csv('data/revision_results/H_scenarios_summary.csv', index=False)
pd.concat(tables, ignore_index=True).to_csv('data/revision_results/H_scenarios_selection.csv', index=False)
base_idx.to_csv('data/revision_results/H_indices_base.csv', index=False)
print(pd.DataFrame(rows).round(3).to_string())
print('done', time.time() - t0)
