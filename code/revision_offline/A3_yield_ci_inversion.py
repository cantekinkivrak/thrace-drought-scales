import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""Wild cluster bootstrap-t: p-value (restricted, Webb weights) AND a 95% interval obtained by inverting the same
restricted test over a grid of hypothesised coefficients (MacKinnon & Webb style), so that p < 0.05 <=> 0 outside the interval."""
import numpy as np, pandas as pd
from rev import *
rng = np.random.default_rng(7)
WEBB = np.array([-np.sqrt(1.5), -1, -np.sqrt(.5), np.sqrt(.5), 1, np.sqrt(1.5)])


def prep(d, idx, levels, trend):
    f = fit_fe(d, idx, levels, trend); j = f['names'].index(idx)
    X = f['X']; y = d['yield'].to_numpy(float); clus = d['key'].to_numpy()
    G = np.unique(clus); g_idx = [np.where(clus == g)[0] for g in G]
    bread = np.linalg.pinv(X.T @ X); n, k = X.shape; c = len(G) / (len(G) - 1) * (n - 1) / (n - k)
    def cr_se(res):
        meat = np.zeros((k, k))
        for ii in g_idx:
            s = (X[ii] * res[ii, None]).sum(0); meat += np.outer(s, s)
        V = bread @ (c * meat) @ bread; return np.sqrt(V[j, j])
    return f, j, X, y, g_idx, cr_se


def boot_p(f, j, X, y, g_idx, cr_se, beta0, B, seed):
    """p-value of H0: beta_j = beta0 from the restricted wild cluster bootstrap-t."""
    r = np.random.default_rng(seed)
    b_hat = f['beta'][j]; t_hat = (b_hat - beta0) / cr_se(f['res'])
    keep = [i for i in range(X.shape[1]) if i != j]; Xr = X[:, keep]
    y0 = y - beta0 * X[:, j]
    br, *_ = np.linalg.lstsq(Xr, y0, rcond=None); res_r = y0 - Xr @ br; yhat_r = Xr @ br + beta0 * X[:, j]
    ts = np.empty(B)
    W = r.choice(WEBB, size=(B, len(g_idx)))
    for b in range(B):
        ystar = yhat_r.copy()
        for gi, ii in enumerate(g_idx): ystar[ii] += W[b, gi] * res_r[ii]
        bs, *_ = np.linalg.lstsq(X, ystar, rcond=None); rs = ystar - X @ bs
        ts[b] = (bs[j] - beta0) / cr_se(rs)
    return float(np.mean(np.abs(ts) >= abs(t_hat)))


def invert(f, j, X, y, g_idx, cr_se, B=999, alpha=0.05):
    b_hat = f['beta'][j]; se = cr_se(f['res']); seed = 12345
    pf = lambda b0: boot_p(f, j, X, y, g_idx, cr_se, b0, B, seed)   # same weights for every beta0 -> smooth in beta0
    def edge(direction):
        lo, hi = b_hat, b_hat + direction * 8 * se
        if pf(hi) >= alpha:  # widen
            hi = b_hat + direction * 20 * se
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            if pf(mid) >= alpha: lo = mid
            else: hi = mid
        return lo
    return edge(-1), edge(+1), pf(0.0)


rows = []
for crop in ['sunflower', 'wheat']:
    d = load_panel(crop); levels = sorted(d.key.unique())
    variants = [('linear', d), ('district', d), ('quadratic', d)]
    dd = d.copy(); dd['yield'] = dd['yield'].astype(float)
    for k, g in dd.groupby('key'):
        b = np.polyfit(g.year_c, g['yield'], 1); dd.loc[g.index, 'yield'] = g['yield'] - np.polyval(b, g.year_c) + g['yield'].mean()
    variants.append(('pre-detrended (district)', dd))
    for trend, data in variants:
        tr = 'none' if trend.startswith('pre') else trend
        for s in SCALES:
            idx = f'SPEI{s}'; x = data.dropna(subset=[idx])
            f, j, X, y, g_idx, cr_se = prep(x, idx, levels, tr)
            lo, hi, p0 = invert(f, j, X, y, g_idx, cr_se)
            rows.append(dict(crop=crop, trend=trend, index=idx, coef=f['beta'][j], cr_se=cr_se(f['res']), wild_p=p0, inv_lo=lo, inv_hi=hi,
                             AIC=f['AIC'], LOYO_RMSE=loyo_rmse(x, idx, levels, tr)))
            print(crop, trend, idx, f"b={f['beta'][j]:.1f} p={p0:.3f} CI=[{lo:.1f},{hi:.1f}]", flush=True)
t = pd.DataFrame(rows); t.to_csv('data/revision_results/A3_yield_wild_inverted.csv', index=False)
