import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""Yield-panel sensitivity to the step-adjusted station series (uses the S1 adjusted indices).
Rebuilds the sunflower (August) and wheat (June) panels from a given index table and reports
within-district correlation, AIC and LOYO-RMSE by scale under the linear-trend spec."""
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'code')
import numpy as np, pandas as pd
from rev import load_panel, fit_fe, loyo_rmse, CORE, ALL7
from H_indices import add_pet_d, indices
from revision_analysis import fold

def panels_from(idx):
    out = {}
    for crop, month in [('sunflower', 8), ('wheat', 6)]:
        p = load_panel(crop)[['year', 'name', 'yield', 'key', 'year_c']]
        d = idx[idx.month == month][['key', 'year'] + [f'SPEI{k}' for k in [1, 3, 6, 9, 12]]]
        out[crop] = p.merge(d, on=['key', 'year']).dropna(subset=['yield'])
    return out

def table(panels, label):
    rows = []
    for crop, d in panels.items():
        levels = sorted(d.key.unique()); d = d.copy(); d['yield'] = d['yield'].astype(float)
        for k in [1, 3, 6, 9, 12]:
            f = fit_fe(d, f'SPEI{k}', levels, 'linear')
            rows.append(dict(scenario=label, crop=crop, index=f'SPEI{k}', n=f['n'], beta=f['beta'][1], AIC=f['AIC'],
                             LOYO_RMSE=loyo_rmse(d, f'SPEI{k}', levels, 'linear')))
    return pd.DataFrame(rows)

if __name__ == '__main__':
    base = pd.read_csv('data/revision_results/H_indices_base.csv')
    t0 = table(panels_from(base), 'S0 baseline')
    # sanity: archived panel values equal reproduced ones
    arch = load_panel('sunflower'); rep = panels_from(base)['sunflower'].merge(arch, on=['key', 'year'], suffixes=('', '_a'))
    print('sunflower panel SPEI9 max |diff| vs archived:', (rep.SPEI9 - rep.SPEI9_a).abs().max())
    raw = pd.read_csv('data/trakya_TP_PET_1965_2024.csv'); raw['key'] = raw.name.map(fold)
    raw = raw[raw.key.isin(ALL7)][['name', 'key', 'year', 'month', 'temp', 'precip', 'lat']]
    from H_scenarios_adjust import adjust  # noqa
    adj = indices(add_pet_d(adjust(raw)))
    t1 = table(panels_from(adj), 'S1 step-adjusted T and P')
    res = pd.concat([t0, t1]); res.to_csv('data/revision_results/H_yield_adjusted.csv', index=False)
    pd.set_option('display.width', 200); print(res.round(3).to_string())
