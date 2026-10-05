import sys as _s, os as _o; _R = _o.path.dirname(_o.path.abspath(__file__)); _s.path[:0] = [_R, _o.path.join(_R, '..', 'revision_offline'), _o.path.join(_R, '..')]  # run from the repository root: python code/<folder>/<script>.py
"""Regenerate Figures S1–S7 (station annual Jun–Oct SPI-3, SPEI-3 and median-composite VHI) and the annual r table."""
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from scipy import stats
mpl.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200})
m = pd.read_csv('data/trakya_district_vhi_vci_tci_monthly.csv'); m['key'] = m.name.str.lower()
idx = pd.read_csv('data/trakya_spi_spei_flexible_1965_2024.csv')
ORDER = [('edirne', 'Edirne', 'S1'), ('kirklareli', 'Kırklareli', 'S2'), ('luleburgaz', 'Lüleburgaz', 'S3'), ('corlu', 'Çorlu', 'S4'),
         ('tekirdag', 'Tekirdağ', 'S5'), ('uzunkopru', 'Uzunköprü', 'S6'), ('ipsala', 'İpsala', 'S7')]
rows = []
for key, name, tag in ORDER:
    v = m[(m.key == key) & m.month.between(6, 10)].groupby('year').agg(VHI=('VHI', 'mean'), nv=('VHI', 'count'))
    s = idx[(idx.key == key) & idx.month.between(6, 10)].groupby('year').agg(SPI3=('SPI3', 'mean'), SPEI3=('SPEI3', 'mean'))
    a = v.join(s).loc[1985:2024].dropna()
    r1 = stats.pearsonr(a.SPI3, a.VHI).statistic; r2 = stats.pearsonr(a.SPEI3, a.VHI).statistic
    rows.append(dict(key=key, district=name, years=len(a), r_SPI3=r1, r_SPEI3=r2, min_valid_months=int(a.nv.min())))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True, gridspec_kw=dict(hspace=0.25))
    fig.suptitle(f'Figure {tag}. {name}: annual June–October SPI-3, SPEI-3 and VHI', fontsize=13, weight='bold', y=0.97)
    ax1.plot(a.index, a.SPI3, '-o', ms=3.5, color='#1f77b4', label=f'SPI-3 (r with VHI = {r1:.3f})')
    ax1.plot(a.index, a.SPEI3, '-o', ms=3.5, color='#d9541e', label=f'SPEI-3 (r with VHI = {r2:.3f})')
    ax1.axhline(0, color='0.5', lw=0.8); ax1.axhline(-1, color='0.3', lw=1, ls='--', label='Drought reference = −1')
    ax1.set_ylabel('Standardized index'); ax1.set_ylim(-2.6, 2.6); ax1.legend(loc='upper left', ncol=3, frameon=False, fontsize=9)
    ax1.text(0.005, 0.94, '(a)', transform=ax1.transAxes, weight='bold')
    ax2.plot(a.index, a.VHI, '-o', ms=3.5, color='#0f9d6e', label='Landsat VHI (median composite)')
    ax2.axhline(40, color='0.3', lw=1, ls='--', label='VHI reference = 40'); ax2.set_ylim(0, 80); ax2.set_ylabel('Vegetation Health Index'); ax2.set_xlabel('Year')
    ax2.legend(loc='upper left', ncol=2, frameon=False, fontsize=9); ax2.text(0.005, 0.94, '(b)', transform=ax2.transAxes, weight='bold')
    for ax in (ax1, ax2): ax.axvspan(2023.5, 2024.5, color='#f6dfe3', alpha=0.6, zorder=0); ax.grid(alpha=0.2, ls=':')
    ax2.set_xlim(1984.5, 2025)
    role = 'primary plateau station' if key != 'ipsala' else 'irrigation-boundary sensitivity district'
    fig.text(0.99, 0.01, f'{role}; 1985–2024', ha='right', fontsize=8, color='0.4')
    fig.savefig(f'figures/SI/Fig{tag}_{key}.png', bbox_inches='tight', facecolor='white'); plt.close(fig)
t = pd.DataFrame(rows); t.to_csv('data/revision_results/SI_annual_r_median.csv', index=False); print(t.round(3).to_string(index=False))
