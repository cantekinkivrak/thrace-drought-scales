import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""Re-implementation of the paper's index pipeline (PCHIP fill -> Thornthwaite PET -> D ->
gamma SPI / AICc-flexible SPEI) so that alternative station series (step-adjusted, restricted
reference period) can be standardised identically.  Validated against the archived
trakya_spi_spei_flexible_1965_2024.csv in H_homogeneity.py."""
import sys, numpy as np, pandas as pd
from scipy import stats
from scipy.interpolate import PchipInterpolator
sys.path.insert(0, 'code')
from revision_analysis import fit_flexible_standardization, fold  # AICc-flexible SPEI (paper code)

SCALES = [1, 3, 6, 9, 12]
DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def daylight(lat, m):
    lat = np.radians(lat); J = np.array([15, 45, 74, 105, 135, 162, 198, 228, 258, 288, 318, 344])[m - 1]
    dec = 0.409 * np.sin(2 * np.pi / 365 * J - 1.39)
    return 24 / np.pi * np.arccos(np.clip(-np.tan(lat) * np.tan(dec), -1, 1))


def thornthwaite(tm, lat):
    I = np.sum((np.where(tm > 0, tm, 0) / 5) ** 1.514)
    a = 6.75e-7 * I ** 3 - 7.71e-5 * I ** 2 + 1.792e-2 * I + 0.49239
    pet = np.zeros(12)
    for i in range(12):
        if tm[i] > 0 and I > 0:
            pet[i] = 16 * (10 * tm[i] / I) ** a * (daylight(lat, i + 1) / 12) * (DAYS[i] / 30)
    return pet


def pchip_fill(df):
    df = df.sort_values(['name', 'year', 'month']).copy()
    for nm in df['name'].unique():
        mk = df['name'] == nm; y = df.loc[mk, 'temp'].to_numpy(float); idx = np.arange(len(y)); ok = np.isfinite(y)
        if ok.sum() < len(y):
            df.loc[mk, 'temp'] = PchipInterpolator(idx[ok], y[ok], extrapolate=True)(idx)
    return df


def add_pet_d(df):
    df = df.sort_values(['name', 'year', 'month']).copy()
    pa = []
    for (nm, yr), g in df.groupby(['name', 'year']):
        g = g.sort_values('month')
        pa.append(pd.DataFrame({'name': nm, 'year': yr, 'month': range(1, 13),
                                'PET': thornthwaite(g['temp'].to_numpy(float), g['lat'].iloc[0])}))
    df = df.drop(columns=[c for c in ['PET', 'D'] if c in df]).merge(pd.concat(pa, ignore_index=True),
                                                                    on=['name', 'year', 'month'], how='left')
    df['D'] = df['precip'] - df['PET']
    return df


def spi_gamma(x):
    x = np.asarray(x, float); out = np.full(len(x), np.nan); v = np.isfinite(x); xv = x[v]
    if len(xv) < 10: return out
    q = (xv == 0).mean(); nz = xv[xv > 0]
    if len(nz) < 4: return out
    A = np.log(nz.mean()) - np.log(nz).mean()
    sh = (1 + np.sqrt(1 + 4 * A / 3)) / (4 * A); sc = nz.mean() / sh
    H = np.clip(q + (1 - q) * stats.gamma.cdf(xv, a=sh, scale=sc), 1e-6, 1 - 1e-6)
    out[v] = stats.norm.ppf(H); return out


def spi_gamma_ref(x, ref_mask):
    """Gamma SPI whose parameters are fitted on the reference subset only (restricted-period variant)."""
    x = np.asarray(x, float); out = np.full(len(x), np.nan); v = np.isfinite(x)
    xr = x[v & ref_mask]
    if len(xr) < 10: return out
    q = (xr == 0).mean(); nz = xr[xr > 0]
    if len(nz) < 4: return out
    A = np.log(nz.mean()) - np.log(nz).mean()
    sh = (1 + np.sqrt(1 + 4 * A / 3)) / (4 * A); sc = nz.mean() / sh
    H = np.clip(q + (1 - q) * stats.gamma.cdf(x[v], a=sh, scale=sc), 1e-6, 1 - 1e-6)
    out[v] = stats.norm.ppf(H); return out


def spei_flex_ref(x, ref_mask):
    """AICc-flexible SPEI with the distribution fitted on the reference subset and applied to all values.
    With ref_mask all-True this reduces exactly to the paper's fit_flexible_standardization."""
    x = np.asarray(x, float); out = np.full(len(x), np.nan); v = np.isfinite(x)
    if ref_mask.all():
        z, _ = fit_flexible_standardization(x); return z
    from revision_analysis import DISTROS
    import warnings
    xr = x[v & ref_mask]
    if len(xr) < 20: return out
    best = None
    for name, dist in DISTROS.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                p = dist.fit(xr); lp = dist.logpdf(xr, *p); cdf = dist.cdf(xr, *p)
            if not (np.all(np.isfinite(lp)) and np.all(np.isfinite(cdf))): continue
            k = len(p); aicc = 2 * k - 2 * lp.sum() + 2 * k * (k + 1) / (len(xr) - k - 1)
            if best is None or aicc < best[0]: best = (aicc, dist, p)
        except Exception:
            continue
    if best is None:
        return out
    pr = np.clip(best[1].cdf(x[v], *best[2]), 1e-6, 1 - 1e-6)
    out[v] = stats.norm.ppf(pr); return out


def indices(df, ref_years=None):
    """SPI/SPEI at all scales per station, standardised per calendar month.
    ref_years: optional iterable of years used as the fitting reference (default: all years)."""
    rows = []
    for nm, g in df.groupby('name'):
        g = g.sort_values(['year', 'month']).copy(); base = g[['name', 'year', 'month']].copy()
        ref_mask_all = np.ones(len(g), bool) if ref_years is None else g['year'].isin(ref_years).to_numpy()
        for k in SCALES:
            accP = g['precip'].rolling(k, min_periods=k).sum().to_numpy()
            accD = g['D'].rolling(k, min_periods=k).sum().to_numpy()
            spi = np.full(len(g), np.nan); spei = np.full(len(g), np.nan)
            for m in range(1, 13):
                mk = (g['month'] == m).to_numpy()
                spi[mk] = spi_gamma_ref(accP[mk], ref_mask_all[mk]) if ref_years is not None else spi_gamma(accP[mk])
                spei[mk] = spei_flex_ref(accD[mk], ref_mask_all[mk])
            base[f'SPI{k}'] = spi; base[f'SPEI{k}'] = spei
        rows.append(base)
    out = pd.concat(rows, ignore_index=True); out['key'] = out['name'].map(fold)
    return out
