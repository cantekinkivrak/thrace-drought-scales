# code/revision_homogeneity — station homogeneity and re-standardization scenarios

These scripts start from the raw monthly station series
(`data/trakya_TP_PET_1965_2024.csv`: name, year, month, temp, precip, lat), which are **not
redistributed** (Turkish State Meteorological Service licence; see LICENSE-DATA.md). With that file
in place they run from the repository root in this order and write to `data/revision_results/`:

| Script | Produces | Used in |
|---|---|---|
| `H_validate.py` | checks that `H_indices.py` reproduces the archived `data/trakya_spi_spei_flexible_1965_2024.csv` exactly | — |
| `H_breaks.py` | `H_pchip_gaps.csv` (gap inventory), `H_snht_breaks.csv` (relative and absolute SNHT with permutation p, Pettitt; annual and June–October) | Section 3.1, Table S4a |
| `H_scenarios_adjust.py` | `H_adjustments_union.csv`, `H_adjustments_all.csv` — step adjustments derived from the flags | — |
| `H_scenarios.py`, `H_scenarios2.py` | `H_scenarios_summary.csv`, `H_scenarios2_summary.csv` (+ `_selection`) — step-adjusted, restricted-reference and station-subset scale selection (maximum composite) | Table S4b (a) |
| `H_scen_median.py`, `H_scen_median2.py` | `H_scen_median.csv`, `H_scen_median2.csv` — the same treatments and the alternative gap fill with the median composite | Table S4b (b), Section 2.2.2 |
| `H_ipsala_fill.py` | `H_ipsala_fill_effect.csv`, `H_indices_anomfill.csv` — climatology + neighbour-anomaly gap fill and its effect | Section 2.2.2 |
| `H_yield_adjusted.py` | `H_yield_adjusted.csv` — yield panels with step-adjusted indices | Section 3.1 |

The deposited copies of all these result tables are in `data/revision_results/`, so the tables of
the paper can be inspected without the raw series.
