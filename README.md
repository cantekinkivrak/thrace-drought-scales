# Purpose-dependent drought accumulation scales in rainfed Turkish Thrace

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21931123.svg)](https://doi.org/10.5281/zenodo.21931123)
[![License: MIT](https://img.shields.io/badge/Code%20licence-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data%20licence-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA.md)

Code and derived data accompanying:

> Kıvrak, C. and Şener, M. (2026) Purpose-dependent drought accumulation scales for canopy
> monitoring and yield prediction in rainfed Turkish Thrace: a cropland Landsat Vegetation
> Health Index evaluation. *International Journal of Climatology* (submitted).

Archived release: <https://doi.org/10.5281/zenodo.21931123>

This archive reproduces every statistical result reported in the paper from the derived,
analysis-ready data included here. The Google Earth Engine notebooks that generate those derived
products from the raw Landsat archive are also included, but re-running them requires a Google
Earth Engine account.

---

## Headline results this archive reproduces

| Quantity | Value |
|---|---|
| Common complete-case sample (June–October, six plateau districts, 1985–2024) | n = 1,183 |
| Best-scoring index, four-criterion composite | SPEI-3, composite 40/40 |
| SPEI-3 Pearson correlation with cropland VHI | r = 0.520 |
| SPI-3 (second) / SPEI-9 (third) | 34 (r = 0.483) / 32 (r = 0.464) |
| Two-way block bootstrap selection frequency for SPEI-3 | 63.7 % |
| Cross-validated drought detection, Heidke skill score | HSS = 0.297 |
| Best-scoring index for sunflower yield (lowest AICc / LOYO-RMSE) | SPEI-9 |

Run `python code/verify_headline_numbers.py` to regenerate the first five rows from `data/`.

---

## Repository layout

```
code/       analysis notebooks and scripts (see below)
data/       derived, analysis-ready data (CSV) + district boundaries
figures/    publication figures as deposited
```

### `code/`

| File | What it does |
|---|---|
| `01_landsat_vhi.ipynb` | Landsat 5/7/8/9 Collection-2 compositing, cross-sensor calibration, CORINE-211 cropland masking, district-level NDVI/LST, VCI/TCI/VHI. **Requires Google Earth Engine.** |
| `02_spi_spei.ipynb` | Station precipitation and temperature → SPI and SPEI at 1, 3, 6, 9 and 12 months, with AICc-selected candidate distributions |
| `03_scale_selection.ipynb` | The four-criterion scale-selection framework and composite ranking |
| `04_pet_sensitivity_yield_panel.ipynb` | Oudin-PET sensitivity arm and the district × year yield panel |
| `05_vhi_drought_maps.ipynb`, `05b_…_standalone.ipynb` | Spatial VHI drought maps |
| `06_publication_figures.ipynb` | Publication figures |
| `07_statistical_robustness.ipynb` | Two-way block bootstrap, clustered standard errors, categorical skill scores, distribution-fit diagnostics |
| `08_resolution_sensitivity.ipynb` | 30 m vs 900 m, masked vs unmasked aggregation arms (Figure 9) |
| `revision_analysis.py` | Consolidated analysis driver used to produce the tables |
| `regenerate_selected_figures.py` | Re-renders selected figures from the derived CSVs |
| `verify_headline_numbers.py` | Standalone reproduction of the headline numbers above |

### `data/`

| File | Contents |
|---|---|
| `trakya_district_vhi_vci_tci_monthly.csv` | Monthly district-level cropland NDVI, LST, VCI, TCI and **VHI (α = 0.5)**, 7 districts × 1985–2024 |
| `trakya_spi_spei_flexible_1965_2024.csv` | SPI and SPEI at five accumulation scales, AICc-flexible distribution choice; `SPEIe*` columns are the Oudin-PET sensitivity arm |
| `scale_selection_multicriteria.csv` | The four-criterion ranking table (Table 2) |
| `sensitivity_summary.csv` | Scale selection repeated under seven alternative specifications |
| `bootstrap_selection_frequency.csv` | Two-way block bootstrap selection frequencies |
| `categorical_skill_summary.csv` | POD, FAR, CSI, bias and HSS with confidence intervals |
| `distribution_fit_diagnostics.csv` | Per-station, per-scale candidate-distribution fit diagnostics |
| `trakya_resolution_sensitivity.csv` | 30 m / 900 m, masked / unmasked aggregation arms |
| `yield_panel_wheat.csv`, `yield_panel_sunflower.csv` | District × year yield panels joined to SPEI |
| `spatial/Trakya_Merged.*` | District boundaries used for zonal aggregation |
| `spatial/station_info.xlsx` | Station names, coordinates and elevations |

**Note on the meteorological inputs.** The monthly station precipitation and temperature series
supplied by the Turkish State Meteorological Service (MGM) are **not** redistributed here: MGM
retains all rights in those records and they were provided to the authors for research use only.
What is provided instead are the standardized indices computed from them
(`trakya_spi_spei_flexible_1965_2024.csv`), which are dimensionless anomalies and are all that the
published analyses use. Re-deriving SPI and SPEI from the raw series therefore requires a data
request to MGM (https://www.mgm.gov.tr). Everything reported in the paper reproduces from the files
included here.

**Note on α.** The Vegetation Health Index is VHI = α·VCI + (1 − α)·TCI. This paper uses a fixed,
unweighted α = 0.5 throughout; the `VHI` column in
`trakya_district_vhi_vci_tci_monthly.csv` is computed with that value. Earlier exploratory work
used station-calibrated α values and does **not** reproduce the published table.

---

## Reproducing the results

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python code/verify_headline_numbers.py
```

The notebooks assume the working directory is the repository root.

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
| Copernicus Land Monitoring Service / EEA | CORINE Land Cover 2018, class 211 | Public |
| Google Earth Engine | Processing platform | Account required |
