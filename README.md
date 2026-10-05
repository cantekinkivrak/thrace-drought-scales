# Season-dependent drought accumulation scales in rainfed Turkish Thrace

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21934057.svg)](https://doi.org/10.5281/zenodo.21934057)
[![License: MIT](https://img.shields.io/badge/Code%20licence-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data%20licence-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA.md)

Code and derived data accompanying:

> Kıvrak, C. and Şener, M. (2026) Season-dependent drought accumulation scales for cropland
> monitoring and yield modelling in rainfed Turkish Thrace: a Landsat Vegetation Health Index
> evaluation. *International Journal of Climatology* (revised version under review, manuscript 2684634).

Archived releases: <https://doi.org/10.5281/zenodo.21934057> (all versions). **v1.1.0** is the
release accompanying the revised manuscript; v1.0.x accompanied the original submission
(see `CHANGELOG.md` for what changed and why).

This archive reproduces the statistical results reported in the paper and in Tables S1–S3 and S5–S7
of the Supporting Information from the derived, analysis-ready data included here. The Google Earth
Engine notebooks that generate those derived products from the raw Landsat archive are included,
but re-running them requires a Google Earth Engine account. The homogeneity screen and the
step-adjusted, restricted-reference and gap-fill index recomputations (Section 3.1, Table S4) are
scripted here but start from the raw station records, which are not redistributed (see below).

---

## Headline results this archive reproduces

| Quantity | Value |
|---|---|
| Common complete-case sample (June–October, six plateau districts, 1985–2024) | n = 1,183 |
| Best-scoring index, consensus rank score (four criteria, maximum 40) | SPEI-3, 40/40 |
| SPEI-3 Pearson correlation with cropland VHI (median composite) | r = 0.521 |
| SPEI-9 (second) / SPI-3 (third) | 34 (r = 0.474) / 33 (r = 0.478) |
| Two-way block-bootstrap selection frequency for SPEI-3 (1,000 replicates) | 67.7 % |
| Leave-one-year-out categorical agreement with SPEI-3 < −1, Heidke skill score | HSS = 0.253 [0.164, 0.323] |
| Seasonal windows: April–June / July–September | SPEI-1 (92 %) / SPEI-9 (78 %; SPEI-3 not statistically separated) |
| Sunflower yield (district fixed effects, wild cluster bootstrap) | associated with August SPEI at every scale; **no best scale identified** |
| İpsala contrast under maximum-NDVI vs median compositing | r = 0.22 vs 0.45 (compositing artefact, Section 3.5) |

Run `python code/verify_headline_numbers.py` to regenerate the first six rows from `data/`.

---

## Repository layout

```
code/                        analysis notebooks and scripts (see below)
code/revision_offline/       revision analyses that run from the derived data (yield panels, windows, CIs, figures)
code/revision_gee/           post-processing of the per-scene Landsat table (compositing, masks, scene filters)
code/revision_homogeneity/   station homogeneity screen and re-standardization scenarios (need raw MGM data)
data/                        derived, analysis-ready data (CSV) + district boundaries
data/revision_results/       result tables behind Tables S1–S7 and Figures 4–7 of the revised paper
figures/                     publication figures as deposited (figures/SI: Figures S1–S7)
```

### `code/`

| File | What it does |
|---|---|
| `01_landsat_vhi.ipynb` | Landsat 5/7/8/9 Collection-2 compositing, cross-sensor calibration, CORINE-211 cropland masking, district-level NDVI/LST, VCI/TCI/VHI (submitted-version pipeline). **Requires Google Earth Engine.** |
| `09_gee_revision_extractions.ipynb` | Revision extraction: per-scene district means of NDVI and LST under ten masks (CORINE 1990–2018, their intersection, ESA WorldCover, two phenology masks, none) with valid-pixel counts; land-cover fractions; phenology-mask areas; September-greenness signature. **Requires Google Earth Engine.** Output: `data/rev_*.csv`. |
| `02_spi_spei.ipynb` | Station precipitation and temperature → SPI and SPEI at 1, 3, 6, 9 and 12 months, with AICc-selected candidate distributions |
| `03_scale_selection.ipynb` | The four-criterion scale-selection framework and consensus rank score |
| `04_pet_sensitivity_yield_panel.ipynb` | Oudin-PET sensitivity arm and the district × year yield panel |
| `05_vhi_drought_maps.ipynb`, `05b_…_standalone.ipynb` | Spatial VHI drought maps |
| `06_publication_figures.ipynb` | Publication figures of the submitted version |
| `07_statistical_robustness.ipynb` | Two-way block bootstrap, clustered standard errors, categorical skill scores, distribution-fit diagnostics |
| `08_resolution_sensitivity.ipynb` | 30 m vs 250/500/900 m, masked vs unmasked aggregation arms (Figure 9) |
| `revision_analysis.py` | Consolidated analysis driver (index fitting, scale selection, bootstrap, categorical agreement, lag profiles, yield panel, figures 1–4 and 8). Its first stage fits the SPEI distributions from the raw station series, so running it end-to-end needs the MGM file (`data/derived/` layout). |
| `reproduce_from_derived.py` | **Runs everything after the index-fitting stage from the deposited files** (indices, median VHI, yield panels): Table 2, lag profiles, sensitivity scenarios, two-way block bootstrap, leave-one-out rankings, categorical agreement, yield panels, Figures 1–4 and 8 → `revision_outputs/`. About 10–15 minutes. |
| `regenerate_selected_figures.py` | Re-renders selected figures from the derived CSVs |
| `verify_headline_numbers.py` | Standalone reproduction of the headline numbers above |
| `revision_offline/` | `A_yield.py`, `A2_yield_boot_specs.py`, `A3_yield_ci_inversion.py`, `A4_predetrend_loyo.py` (yield panels: four trend specifications, year-block scale-selection bootstrap, wild cluster bootstrap-t with test-inversion intervals, fold-internal detrending; Table 3, S7); `B_nested_loyo.py` (fold-internal index selection, Table 4); `C_rank_sens.py` (rank-aggregation and MI-k sensitivity, Table S5); `DE_norm_window.py`, `E2_window_boot.py`, `W2_window_boot.py` (percentile normalization, seasonal windows, 1,000-replicate window bootstrap with paired Δr; Tables S3, S6); `F_cis.py` (district correlation intervals, Table S1); `fig7.py`, `si_figs.py`; `rev.py` (shared loaders). See the README in that folder for the run order. |
| `revision_gee/` | `H2_gee_postprocess.py` (mask vintages, compositing operators, SLC-off ablation, crop masks), `H3_ipsala_scene_quality.py` (coverage filters, plateau–İpsala paired bootstrap), `H4_windows_qc.py`, `H5_build_median_vhi.py` (builds the primary median-composite series from the per-scene table), `H6_max_inflation.py`, `H7_median_sensitivity.py` (Table S3), `fig56.py` (Figures 5–6), `build_nb09.py` (generates notebook 09). |
| `revision_homogeneity/` | `H_indices.py` (exact re-implementation of the index pipeline), `H_validate.py`, `H_breaks.py` (gap inventory, SNHT/Pettitt; Table S4a), `H_scenarios*.py`, `H_scen_median*.py` (step-adjusted, restricted-reference, station-subset and gap-fill scenarios; Table S4b), `H_ipsala_fill.py`, `H_yield_adjusted.py`. **These need `data/trakya_TP_PET_1965_2024.csv` (raw MGM station series), which is not redistributed.** |

All scripts are run from the repository root, e.g. `python code/revision_offline/A3_yield_ci_inversion.py`; they
read from `data/` and write their tables to `data/revision_results/` (the deposited copies are overwritten).

### `data/`

| File | Contents |
|---|---|
| `trakya_district_vhi_vci_tci_monthly.csv` | **Primary series of the revised paper:** monthly district-level cropland NDVI, LST, VCI, TCI and VHI (α = 0.5), 7 districts × 1985–2024, **monthly median composite** of the per-scene district means (column `n_valid` = scenes per month) |
| `trakya_district_vhi_vci_tci_monthly_maxcomposite.csv` | The maximum-NDVI / mean-LST composite of the submitted version (v1.0.x primary series), kept for the comparisons of Section 3.5 and Table S3 |
| `rev_scene_means_1985_2024.csv.gz` | Per-scene district means of NDVI and LST under ten masks with valid-pixel counts, sensor and SLC-off flags (44,121 scene–district rows); every compositing, mask and scene-filter variant in Table S3 is rebuilt from this table |
| `rev_district_landcover_fractions.csv`, `rev_phenology_fractions.csv`, `rev_september_green.csv` | District land-cover shares by CORINE vintage and irrigation indicators; per-year phenology-mask areas; September-greenness area of class-211 pixels (Table S2) |
| `trakya_spi_spei_flexible_1965_2024.csv` | SPI and SPEI at five accumulation scales, AICc-flexible distribution choice; the `SPEIe*` columns are the **empirical-standardization** sensitivity variant |
| `trakya_spei_oudin_flexible_1965_2024.csv` | SPEI with Oudin PET (`SPEIo*`; `SPEIoe*` = empirical standardization), the PET sensitivity arm of Section 3.4 |
| `scale_selection_multicriteria.csv` | The four-criterion ranking table (Table 2, median composite) |
| `sensitivity_summary.csv` | Scale selection repeated under the alternative specifications of Section 2.8 |
| `bootstrap_selection_frequency.csv` | Two-way block bootstrap selection frequencies (Figure 4) |
| `categorical_skill_summary.csv` | POD, FAR, CSI, bias and HSS with confidence intervals (Table 4) |
| `distribution_fit_diagnostics.csv` | Per-station, per-scale candidate-distribution fit diagnostics |
| `trakya_resolution_sensitivity.csv` | 30 m / 250 / 500 / 900 m, masked / unmasked aggregation arms (Figure 9; maximum composite, Landsat 8/9 era) |
| `yield_panel_wheat.csv`, `yield_panel_sunflower.csv` | District × year yield panels joined to SPEI |
| `revision_results/` | Result tables behind Tables S1–S7 and Figures 4–7 (file names match the scripts that produce them) |
| `spatial/Trakya_Merged.*` | District boundaries used for zonal aggregation |
| `spatial/station_info.xlsx` | Station names, coordinates and elevations |

**Note on the meteorological inputs.** The monthly station precipitation and temperature series
supplied by the Turkish State Meteorological Service (MGM) are **not** redistributed here: MGM
retains all rights in those records and they were provided to the authors for research use only.
What is provided instead are the standardized indices computed from them, which are dimensionless
anomalies and are all that the published analyses use. Everything in the main text and in
Tables S1–S3 and S5–S7 reproduces from the files included here; the gap inventory, the homogeneity
tests and the step-adjusted, restricted-reference and gap-fill recomputations of Table S4 require
the raw series, which must be requested from MGM (https://www.mgm.gov.tr).

**Note on α.** The Vegetation Health Index is VHI = α·VCI + (1 − α)·TCI with a fixed α = 0.5
throughout; every script recomputes VHI from VCI and TCI with that value.

---

## Reproducing the results

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python code/verify_headline_numbers.py              # headline numbers, < 1 minute
python code/reproduce_from_derived.py               # primary tables and Figures 1-4, 8 (median composite), ~15 minutes
python code/revision_offline/A3_yield_ci_inversion.py   # etc.; see code/*/README.md for the run order
```

The notebooks and scripts assume the working directory is the repository root.

---

## Licence

- **Code** (`code/`): MIT — see `LICENSE`.
- **Derived data** (`data/`): CC BY 4.0 — see `LICENSE-DATA.md`, which also sets out the terms
  attaching to the underlying MGM and TÜİK records.

## Third-party data sources

| Source | Product | Access |
|---|---|---|
| Turkish State Meteorological Service (MGM) | Daily/monthly station precipitation and temperature | On request from MGM |
| Turkish Statistical Institute (TÜİK) | District-level wheat and sunflower yield | On request from TÜİK |
| U.S. Geological Survey | Landsat 5/7/8/9 Collection 2 Level-2 | Public |
| Copernicus Land Monitoring Service / EEA | CORINE Land Cover 1990, 2000, 2006, 2012, 2018 (class 211 and irrigated classes) | Public |
| ESA | WorldCover 2021 (cropland class, mask sensitivity) | Public |
| NASA LP DAAC | LGRIP30 V001 (irrigation indicator, Table S2) | Public |
| Google Earth Engine | Processing platform | Account required |
