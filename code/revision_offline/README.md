# code/revision_offline — analyses that run from the derived data

All scripts are run from the repository root (`python code/revision_offline/<script>.py`), read
`data/` and write to `data/revision_results/`. `rev.py` holds the shared loaders (VHI is always
recomputed as 0.5·VCI + 0.5·TCI from `data/trakya_district_vhi_vci_tci_monthly.csv`, the median
composite). Fixed random seeds; run times on a laptop in parentheses.

| Script | Produces | Used in |
|---|---|---|
| `A_yield.py` (~10 min) | `A_yield_wildcluster_trends.csv` — district fixed-effects yield models under four trend specifications, wild cluster bootstrap-t p-values (Webb weights, restricted null, 4,999 replicates) | Table S7a (p, AIC) |
| `A3_yield_ci_inversion.py` (~40 min) | `A3_yield_wild_inverted.csv` — the same, with 95% intervals obtained by inverting the restricted test | Table 3, Table S7a |
| `A2_yield_boot_specs.py` (~10 min) | `A2_yield_scale_bootstrap.csv` — year-block (circular, 3-yr) scale-selection bootstrap, ~600 replicates per specification | Table S7b, Section 3.6 |
| `A4_predetrend_loyo.py` (seconds) | `A4_predetrend_loyo.csv` — LOYO RMSE of the pre-detrended specification with the detrending re-estimated inside every fold | Table S7a (LOYO column) |
| `B_nested_loyo.py` (~5 min) | `B_nested_loyo_skill.csv` — leave-one-year-out categorical agreement with the index re-selected inside every fold | Table 4 cross-check |
| `C_rank_sens.py` (seconds) | `C_rank_aggregation_sensitivity.csv` — rank-aggregation rules and MI neighbourhoods k = 3–15 | Table S5 |
| `DE_norm_window.py` (seconds) | `DE_normalisation_window.csv` — percentile VCI/TCI normalization; seasonal windows | Tables S3(d), S6 |
| `E2_window_boot.py` (~5 min) | console only — 300-replicate window bootstrap (superseded by W2) | — |
| `W2_window_boot.py` (~20 min) | `W2_window_pairs.csv`, `W2_window_freq.csv` — 1,000-replicate two-way block bootstrap per seasonal window with paired Δr intervals | Table S6, Section 3.6, Figure 6a labels |
| `F_cis.py` (~3 min) | `F_district_coupling_ci.csv` — 2,000-replicate circular block-bootstrap intervals of district VHI–SPEI-3 correlations | Table S1, Section 3.3 |
| `fig7.py` | `figures/Fig7_categorical_agreement.png` (needs `categorical_skill_by_district.csv` from `revision_analysis.py`) | Figure 7 |
| `si_figs.py` | `figures/SI/FigS1–S7_*.png`, `SI_annual_r_median.csv` | Figures S1–S7, Table S1 |
| `graphical_abstract.py` | `figures/Graphical_Abstract.png` | — |

The primary tables and Figures 1–4 and 8 come from `code/revision_analysis.py` (run from the
repository root; it reads `data/` and writes `revision_outputs/`).
