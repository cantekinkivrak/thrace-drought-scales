import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""Pre-detrended yield specification with the district-wise detrending re-estimated inside every leave-one-year-out fold
(no held-out year enters the detrending), for Table S7a."""
import numpy as np, pandas as pd
from rev import *
rows = []
for crop in ['sunflower', 'wheat']:
    d = load_panel(crop); levels = sorted(d.key.unique())
    for s in SCALES:
        idx = f'SPEI{s}'; x = d.dropna(subset=[idx]).copy(); x['yield'] = x['yield'].astype(float)
        errs = []
        for y in sorted(x.year.unique()):
            tr = x[x.year != y].copy(); te = x[x.year == y].copy()
            for k, g in tr.groupby('key'):
                b = np.polyfit(g.year_c, g['yield'], 1); m = g['yield'].mean()
                tr.loc[g.index, 'yield'] = g['yield'] - np.polyval(b, g.year_c) + m
                sel = te.key == k
                te.loc[sel, 'yield'] = te.loc[sel, 'yield'] - np.polyval(b, te.loc[sel, 'year_c']) + m
            f = fit_fe(tr, idx, levels, 'none'); Xte, _ = fe_design(te, idx, levels, 'none')
            errs.extend((te['yield'].to_numpy() - Xte @ f['beta']).tolist())
        rmse = float(np.sqrt(np.mean(np.square(errs))))
        rows.append(dict(crop=crop, index=idx, LOYO_RMSE_fold_detrended=rmse)); print(crop, idx, round(rmse, 2), flush=True)
pd.DataFrame(rows).to_csv('data/revision_results/A4_predetrend_loyo.csv', index=False)
