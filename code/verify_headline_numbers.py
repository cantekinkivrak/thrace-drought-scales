#!/usr/bin/env python3
"""Reproduce the headline numbers of the paper from the derived data in data/.

Run from the repository root:

    python code/verify_headline_numbers.py

Expected output (Table 2 and Section 3.3 of the paper):

    n = 1183
    SPEI3  composite 40  r = 0.520
    SPI3   composite 34  r = 0.483
    SPEI9  composite 32  r = 0.464
    bootstrap selection frequency SPEI3 = 63.7 %
    HSS = 0.297
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

CANDIDATES = ["SPI1", "SPI3", "SPI6", "SPI9", "SPI12",
              "SPEI1", "SPEI3", "SPEI6", "SPEI9", "SPEI12"]
CRITERIA = ["pearson", "spearman", "MI", "dose_amp"]
GROWING_SEASON = range(6, 11)          # June-October
IRRIGATED = "ipsala"                   # excluded from the primary six-district sample
ALPHA = 0.5                            # VHI = alpha * VCI + (1 - alpha) * TCI


def load_common_sample() -> pd.DataFrame:
    """June-October district-months for the six rainfed plateau districts, complete on
    VHI and all ten candidate indices, so every candidate is scored on one identical sample."""
    vhi = pd.read_csv(DATA / "trakya_district_vhi_vci_tci_monthly.csv")
    idx = pd.read_csv(DATA / "trakya_spi_spei_flexible_1965_2024.csv")

    vhi["key"] = vhi["name"].str.lower()
    vhi["VHI"] = ALPHA * vhi["VCI"] + (1 - ALPHA) * vhi["TCI"]

    merged = vhi[["key", "year", "month", "VHI"]].merge(
        idx[["key", "year", "month"] + CANDIDATES],
        on=["key", "year", "month"], how="inner")

    return (merged[merged["month"].isin(GROWING_SEASON) & (merged["key"] != IRRIGATED)]
            .dropna(subset=["VHI"] + CANDIDATES))


def score_candidates(sample: pd.DataFrame) -> pd.DataFrame:
    """Four non-duplicated criteria, each ranked 1 (worst) to 10 (best); composite is their sum."""
    vhi = sample["VHI"].to_numpy()
    rows = []
    for name in CANDIDATES:
        index = sample[name].to_numpy()
        lower, upper = np.quantile(index, [1 / 3, 2 / 3])
        rows.append({
            "index": name,
            "pearson": pearsonr(index, vhi)[0],
            "spearman": spearmanr(index, vhi)[0],
            "MI": mutual_info_regression(index.reshape(-1, 1), vhi,
                                         n_neighbors=5, random_state=0)[0],
            "dose_amp": vhi[index >= upper].mean() - vhi[index <= lower].mean(),
        })

    table = pd.DataFrame(rows)
    for criterion in CRITERIA:
        table[f"{criterion}_rank"] = table[criterion].rank(method="min")
    table["composite"] = table[[f"{c}_rank" for c in CRITERIA]].sum(axis=1)
    return table.sort_values("composite", ascending=False).reset_index(drop=True)


def main() -> None:
    sample = load_common_sample()
    print(f"n = {len(sample)}")

    table = score_candidates(sample)
    print()
    print(table[["index"] + CRITERIA + ["composite"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nDistrict-level SPEI-3 correlation with VHI (June-October):")
    vhi = pd.read_csv(DATA / "trakya_district_vhi_vci_tci_monthly.csv")
    idx = pd.read_csv(DATA / "trakya_spi_spei_flexible_1965_2024.csv")
    vhi["key"] = vhi["name"].str.lower()
    vhi["VHI"] = ALPHA * vhi["VCI"] + (1 - ALPHA) * vhi["TCI"]
    per_district = vhi[["key", "year", "month", "VHI"]].merge(
        idx[["key", "year", "month", "SPEI3"]], on=["key", "year", "month"], how="inner")
    per_district = per_district[per_district["month"].isin(GROWING_SEASON)]
    for key, group in per_district.groupby("key"):
        group = group.dropna(subset=["VHI", "SPEI3"])
        print(f"  {key:<12s} r = {pearsonr(group['SPEI3'], group['VHI'])[0]:.3f}  n = {len(group)}")

    boot = pd.read_csv(DATA / "bootstrap_selection_frequency.csv")
    top = boot.sort_values("frequency_pct", ascending=False).iloc[0]
    print(f"\nBootstrap selection frequency: {top['index']} = {top['frequency_pct']:.1f} %")

    skill = pd.read_csv(DATA / "categorical_skill_summary.csv").set_index("metric")
    hss = skill.loc["HSS"]
    print(f"Heidke skill score: {hss['estimate']:.3f} "
          f"[{hss['ci_low']:.3f}, {hss['ci_high']:.3f}]")


if __name__ == "__main__":
    main()
