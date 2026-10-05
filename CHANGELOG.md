# Changelog

## v1.1.0 — revised manuscript (October 2026)

Release accompanying the revised version of the manuscript (International Journal of Climatology,
manuscript 2684634) after major revision. The title changed from "Purpose-dependent …" to
"Season-dependent drought accumulation scales for cropland monitoring and yield modelling in rainfed
Turkish Thrace: a Landsat Vegetation Health Index evaluation".

**Primary vegetation series changed.** `data/trakya_district_vhi_vci_tci_monthly.csv` is now the
monthly **median** composite of the per-scene district means (NDVI and LST taken separately). The
maximum-NDVI / mean-LST composite used in v1.0.x is kept as
`data/trakya_district_vhi_vci_tci_monthly_maxcomposite.csv`. Reason: over İpsala's small,
fragmented cropland the maximum operator selects the greenest partial or SLC-off scene and
produced an apparent decoupling (r = 0.22) that a median composite removes (r = 0.45); the land-use
interpretation of that contrast in the submitted version has been withdrawn (paper, Sections 3.5
and 4.2). The primary scale selection is unchanged: SPEI-3, 40/40, r = 0.521 (was 0.520); the
runner-up order is now SPEI-9 (34) then SPI-3 (33) (was SPI-3 34, SPEI-9 32); bootstrap selection
frequency 67.7 % (was 63.7 %); HSS 0.253 (was 0.297 with the maximum composite; the index is now
also re-selected inside every leave-one-year-out fold, which selects SPEI-3 in all 40 folds).

**Added.**
- `code/09_gee_revision_extractions.ipynb` and `data/rev_scene_means_1985_2024.csv.gz`: per-scene
  district table under ten masks with valid-pixel counts, sensor and SLC-off flags.
- `code/revision_gee/`: compositing operators, mask vintages (CORINE 1990–2018, stable intersection,
  time-matched, ESA WorldCover), scene and coverage filters, SLC-off / Landsat-7 ablations, crop
  phenology masks (Tables S2–S3, S6; Figures 5–6).
- `code/revision_offline/`: wild cluster bootstrap-t with test-inversion intervals under four trend
  specifications, year-block yield-scale bootstrap, fold-internal detrending LOYO, nested
  leave-one-year-out categorical agreement, rank-aggregation and MI-k sensitivity, percentile
  VCI/TCI normalization, seasonal-window selection with 1,000-replicate paired bootstrap, district
  correlation intervals (Tables 3–4, S1, S3, S5–S7).
- `code/revision_homogeneity/`: exact re-implementation of the index pipeline, gap inventory,
  relative/absolute SNHT and Pettitt screen, step-adjusted / restricted-reference / station-subset /
  alternative gap-fill scenarios (Table S4). Require the raw MGM series (not redistributed).
- `data/trakya_spei_oudin_flexible_1965_2024.csv` (Oudin-PET arm), `data/rev_district_landcover_fractions.csv`,
  `data/rev_phenology_fractions.csv`, `data/rev_september_green.csv`, `data/revision_results/`.
- `figures/`: revised Figures 1–8, Figures S1–S7 (`figures/SI/`) and a new graphical abstract.
- `code/reproduce_from_derived.py`: regenerates the primary tables and figures from the deposited derived
  files without the raw station series (the driver `revision_analysis.py` needs them for its index-fitting stage).

**Changed.** `README.md`, `CITATION.cff`, `.zenodo.json`, `LICENSE-DATA.md` (title);
`Supporting_Information.docx` (revised SI with Tables S1–S7); `code/revision_analysis.py`
(north arrow and scale bar on Figure 1; İpsala legend label); `code/verify_headline_numbers.py`
(expected values for the median series); `data/scale_selection_multicriteria.csv`,
`sensitivity_summary.csv`, `bootstrap_selection_frequency.csv`, `categorical_skill_summary.csv`,
`distribution_fit_diagnostics.csv` (recomputed on the median series). The v1.0.x README described
the `SPEIe*` columns as the Oudin-PET arm; they are the empirical-standardization arm (Oudin PET
is the separate file above).

**Removed.** `data/ipsala_coupling_subperiods.csv` (the sub-period analysis is superseded by
`data/revision_results/H3_ipsala_subperiods.csv`); the submitted-version figure files
`figures/Fig_*.png` (replaced by the numbered revised figures).

## v1.0.1 — submission version (August 2026)
README/CITATION corrections; Supporting Information merged into one file; raw station file removed.

## v1.0.0 — submission version (August 2026)
Initial release accompanying the submitted manuscript.
