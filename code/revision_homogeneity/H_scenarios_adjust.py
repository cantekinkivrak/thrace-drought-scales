import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""Step-adjustment helper shared by H_scenarios.py and H_yield_adjusted.py (annual relative SNHT, p<0.05)."""
import numpy as np, pandas as pd
breaks = pd.read_csv('data/revision_results/H_snht_breaks.csv')
sig = breaks[(breaks.test_type == 'relative') & (breaks.series == 'annual') & (breaks.SNHT_p < 0.05)]

def adjust(df, which=('temperature', 'precipitation')):
    d = df.copy()
    for b in sig.itertuples():
        if b.variable not in which: continue
        pre = (d.key == b.station) & (d.year < b.break_year_first_of_new_segment)
        if b.variable == 'temperature':
            d.loc[pre, 'temp'] = d.loc[pre, 'temp'] + b.shift
        else:
            d.loc[pre, 'precip'] = d.loc[pre, 'precip'] * np.exp(b.shift)
    return d
