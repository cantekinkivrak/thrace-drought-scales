import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""Two-way block bootstrap (districts with replacement x circular 3-year year blocks; 1,000 replicates per window) of the
full four-criterion ranking within seasonal windows (median VHI), with paired Pearson-r differences between the leading
candidates. Same design as the primary bootstrap of Section 2.8; seed differs."""
import sys, numpy as np, pandas as pd
from scipy import stats
from rev import *
m = load_monthly()
WINS = {'Apr-Jun': [4, 5, 6], 'Jun-Oct': [6, 7, 8, 9, 10], 'Jul-Sep': [7, 8, 9], 'Aug-Sep': [8, 9]}
PAIRS = {'Apr-Jun': [('SPEI1', 'SPEI3'), ('SPEI1', 'SPI1')], 'Jun-Oct': [('SPEI3', 'SPEI9'), ('SPEI3', 'SPI3')],
         'Jul-Sep': [('SPEI9', 'SPEI3'), ('SPEI9', 'SPI9')], 'Aug-Sep': [('SPEI9', 'SPEI3'), ('SPEI9', 'SPI9')]}
B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
rows = []; freq_rows = []
for wn, mo in WINS.items():
    rng = np.random.default_rng(2026)
    s = m[m.key.isin(CORE) & m.month.isin(mo)].dropna(subset=['VHI'] + IDX); years = sorted(s.year.unique())
    byd = {k: g for k, g in s.groupby('key')}
    r0 = {i: stats.pearsonr(s.VHI, s[i]).statistic for i in IDX}
    t0 = metric_table(s)
    wins = []; rr_all = []
    for b in range(B):
        ks = rng.choice(CORE, len(CORE), replace=True); ys = moving_year_sample(rng, years, 3)
        x = pd.concat([pd.concat([byd[k][byd[k].year == y] for y in ys]) for k in ks])
        t = metric_table(x); wins.append(t['index'].iloc[0])
        rr_all.append({i: np.corrcoef(x.VHI, x[i])[0, 1] for i in IDX})
    rr = pd.DataFrame(rr_all); vc = pd.Series(wins).value_counts(normalize=True) * 100
    for i, f in vc.items(): freq_rows.append(dict(window=wn, index=i, pct=f))
    for a, bb in PAIRS[wn]:
        d = (rr[a] - rr[bb]).to_numpy()
        rows.append(dict(window=wn, n=len(s), B=B, winner=t0['index'].iloc[0], winner_score=t0['composite'].iloc[0],
                         winner_pct=vc.get(t0['index'].iloc[0], 0.0), second=vc.index[1] if len(vc) > 1 else '', second_pct=vc.iloc[1] if len(vc) > 1 else 0,
                         pair=f'{a}−{bb}', r_a=r0[a], r_b=r0[bb], diff=r0[a] - r0[bb], lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                         P_le0=float(np.mean(d <= 0)), frac_pos=float(np.mean(d > 0))))
        print(rows[-1], flush=True)
    print(wn, vc.round(1).to_dict(), flush=True)
pd.DataFrame(rows).to_csv('data/revision_results/W2_window_pairs.csv', index=False)
pd.DataFrame(freq_rows).to_csv('data/revision_results/W2_window_freq.csv', index=False)
print('done')
