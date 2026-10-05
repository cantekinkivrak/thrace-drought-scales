"""Graphical abstract for the revised manuscript (season-dependent scales), same portrait format as the submitted one."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt, textwrap
from scipy import stats
from rev import load_monthly, IDX, CORE
mpl.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
m = load_monthly(); SC = [1, 3, 6, 9, 12]
wins = [('April–June (wheat-dominated canopy)', [4, 5, 6], '#2f6fb0', 'o'), ('June–October (primary window)', [6, 7, 8, 9, 10], '#1a7a52', 's'), ('July–September (sunflower-dominated canopy)', [7, 8, 9], '#b8781a', '^')]
pct = {'April–June (wheat-dominated canopy)': 'SPEI-1 · selected in 92%', 'June–October (primary window)': 'SPEI-3 · 68%', 'July–September (sunflower-dominated canopy)': 'SPEI-9 · 78% (3 months close behind)'}
fig = plt.figure(figsize=(8.27, 9.92), dpi=150)   # A4-like portrait, as the submitted abstract
fig.patch.set_facecolor('white')
title = ("Season-dependent drought accumulation scales for cropland monitoring and yield modelling in rainfed Turkish Thrace: "
         "a Landsat Vegetation Health Index evaluation")
fig.text(0.04, 0.965, textwrap.fill(title, 62), fontsize=16, weight='bold', va='top', linespacing=1.25)
fig.text(0.04, 0.845, 'Cantekin Kıvrak* and Mehmet Şener', fontsize=14, color='0.3', va='top')
fig.add_artist(plt.Line2D([0.04, 0.96], [0.815, 0.815], color='0.75', lw=1.2))
ax = fig.add_axes([0.13, 0.40, 0.82, 0.385])
for lab, mo, col, mk in wins:
    s = m[m.key.isin(CORE) & m.month.isin(mo)].dropna(subset=['VHI'] + IDX)
    r = [stats.pearsonr(s.VHI, s[f'SPEI{k}']).statistic for k in SC]
    ax.plot(SC, r, marker=mk, ms=8, color=col, lw=2.2, label=lab, markeredgecolor='white', markeredgewidth=1.2)
    kb = int(np.argmax(r)); ax.scatter([SC[kb]], [r[kb]], s=260, facecolor='none', edgecolor=col, lw=2.2, zorder=4)
    if lab.startswith('July'): ax.annotate(pct[lab], (SC[kb], r[kb]), xytext=(-14, 12), textcoords='offset points', ha='right', fontsize=10.5, color=col, weight='bold')
    elif lab.startswith('April'): ax.annotate(pct[lab], (SC[kb], r[kb]), xytext=(12, 10), textcoords='offset points', fontsize=10.5, color=col, weight='bold')
    else: ax.annotate(pct[lab], (SC[kb], r[kb]), xytext=(10, -20), textcoords='offset points', fontsize=10.5, color=col, weight='bold')
ax.axhline(0, color='0.75', lw=0.9)
ax.set_xticks(SC); ax.set_xlabel('SPEI accumulation scale (months)', fontsize=12.5); ax.set_ylabel('VHI–SPEI Pearson r', fontsize=12.5)
ax.set_ylim(-0.25, 0.72); ax.tick_params(labelsize=11); ax.grid(axis='y', color='0.92', lw=0.8); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=10.5, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=1, title='Canopy window — circled: best scale; label: two-way block-bootstrap selection frequency (1,000 replicates)', title_fontsize=9.5)
caption = ("Ten SPI and SPEI scales were scored against a 30 m, cropland-masked Landsat Vegetation Health Index over rainfed Turkish "
           "Thrace, 1985–2024. SPEI-3 best tracks June–October canopy stress, but the preferred scale varies with the season — one month "
           "in the wheat-dominated spring, nine months (with three close behind) in the sunflower-dominated late summer. Sunflower yield "
           "responds to SPEI without an identifiable best scale, and an apparent decoupling of the irrigation-boundary district proves to be a "
           "compositing artefact: median, not maximum, compositing of district Landsat series.")
fig.text(0.04, 0.205, textwrap.fill(caption, 96), fontsize=11.3, va='top', linespacing=1.35)
fig.savefig('figures/Graphical_Abstract.png', dpi=150, facecolor='white'); print('ok')
