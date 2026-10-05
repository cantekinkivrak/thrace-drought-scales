#!/usr/bin/env python3
"""Reproduce the primary tables and figures of the revised paper from the DEPOSITED derived files only
(no raw station data needed). This is the second half of revision_analysis.main(): it starts from the
archived standardized indices (data/trakya_spi_spei_flexible_1965_2024.csv, data/trakya_spei_oudin_flexible_1965_2024.csv)
and the median-composite VHI series, and writes revision_outputs/tables and revision_outputs/figures.

Run from the repository root:  python code/reproduce_from_derived.py      (about 10-15 minutes)

Covers: Table 2 and the lag profiles, the sensitivity scenarios of Section 2.8, the two-way block bootstrap (Figure 4),
leave-one-out rankings, the leave-one-year-out categorical agreement (Table 4; the nested-selection check is
code/revision_offline/B_nested_loyo.py), the yield panels (Table 3 uses code/revision_offline/A3_yield_ci_inversion.py
for the wild-cluster inference), Figures 1-4 and 8, and the text summary. The homogeneity screen of
revision_analysis.main() is skipped because it needs the raw station series.
"""
import shutil, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from revision_analysis import *   # noqa: F401,F403  (functions, constants, TABLES/FIGURES paths)
import revision_analysis as ra

DATA = ra.ROOT / "data"
TABLES, FIGURES = ra.TABLES, ra.FIGURES


def main():
    index_data = pd.read_csv(DATA / "trakya_spi_spei_flexible_1965_2024.csv")
    index_data = index_data[index_data["key"].isin(ALL7_KEYS)].copy()
    flexible_oudin = pd.read_csv(DATA / "trakya_spei_oudin_flexible_1965_2024.csv")
    vhi = pd.read_csv(DATA / "trakya_district_vhi_vci_tci_monthly.csv")
    vhi["key"] = vhi["name"].map(fold)
    vhi["VHI"] = 0.5 * (vhi["VCI"] + vhi["TCI"])
    vhi = add_leave_one_year_out_vhi(vhi)
    merged = vhi.merge(index_data.drop(columns="name"), on=["key", "year", "month"])

    core_monthly = merged[merged["key"].isin(CORE_KEYS)].copy()
    core = core_monthly[core_monthly["month"].isin(WIN)].dropna(
        subset=["VHI"] + IDX
    )
    all7 = merged[merged["month"].isin(WIN)].dropna(subset=["VHI"] + IDX)

    primary = metric_table(core)
    all7_table = metric_table(all7)
    primary.to_csv(TABLES / "scale_selection_primary_six_districts.csv", index=False)
    all7_table.to_csv(TABLES / "scale_selection_all_seven_districts.csv", index=False)
    lag = lag_table(core_monthly)
    lag.to_csv(TABLES / "lag_profiles_primary.csv", index=False)
    print("\nPrimary four-criterion ranking:\n", primary.round(4).to_string(index=False))

    sensitivity_rows = []
    scenarios = [
        ("Primary VHI, six districts", core, "VHI", IDX),
        ("All seven districts", all7, "VHI", IDX),
        ("VCI only", core, "VCI", IDX),
        ("TCI only", core, "TCI", IDX),
        ("Leave-one-year-out VHI", core.dropna(subset=["VHI_LOO"]), "VHI_LOO", IDX),
    ]
    for label, frame, reference, indices in scenarios:
        table = metric_table(frame, reference, indices)
        top = table.iloc[0]
        sensitivity_rows.append(
            {
                "scenario": label,
                "top_index": top["index"],
                "top_composite": top["composite"],
                "SPEI3_r": table.loc[table["index"] == "SPEI3", "pearson"].iloc[0],
                "SPI3_r": table.loc[table["index"] == "SPI3", "pearson"].iloc[0],
                "n": int(table["n"].min()),
            }
        )

    empirical_columns = [f"SPEIe{k}" for k in SCALES]
    empirical_frame = core_monthly[["key", "year", "month", "VHI"] + [f"SPI{k}" for k in SCALES] + empirical_columns].copy()
    empirical_frame = empirical_frame[empirical_frame["month"].isin(WIN)].dropna(
        subset=["VHI"] + [f"SPI{k}" for k in SCALES] + empirical_columns
    )
    renamed_empirical = empirical_frame.rename(
        columns={f"SPEIe{k}": f"SPEI{k}" for k in SCALES}
    )
    table = metric_table(renamed_empirical)
    sensitivity_rows.append(
        {
            "scenario": "Empirical SPEI standardisation",
            "top_index": table.iloc[0]["index"],
            "top_composite": table.iloc[0]["composite"],
            "SPEI3_r": table.loc[table["index"] == "SPEI3", "pearson"].iloc[0],
            "SPI3_r": table.loc[table["index"] == "SPI3", "pearson"].iloc[0],
            "n": int(table["n"].min()),
        }
    )

    oudin_index = flexible_oudin.copy()
    oudin_index["key"] = oudin_index["name"].map(fold)
    oudin_frame = vhi[vhi["key"].isin(CORE_KEYS)].merge(
        index_data[["key", "year", "month"] + [f"SPI{k}" for k in SCALES]],
        on=["key", "year", "month"],
    ).merge(
        oudin_index[["key", "year", "month"] + [f"SPEIo{k}" for k in SCALES]],
        on=["key", "year", "month"],
    )
    oudin_frame = oudin_frame[oudin_frame["month"].isin(WIN)].dropna()
    oudin_frame = oudin_frame[["key", "year", "month", "VHI"] + [f"SPI{k}" for k in SCALES] + [f"SPEIo{k}" for k in SCALES]].rename(columns={f"SPEIo{k}": f"SPEI{k}" for k in SCALES})
    table = metric_table(oudin_frame)
    sensitivity_rows.append(
        {
            "scenario": "Oudin PET, flexible SPEI",
            "top_index": table.iloc[0]["index"],
            "top_composite": table.iloc[0]["composite"],
            "SPEI3_r": table.loc[table["index"] == "SPEI3", "pearson"].iloc[0],
            "SPI3_r": table.loc[table["index"] == "SPI3", "pearson"].iloc[0],
            "n": int(table["n"].min()),
        }
    )
    pd.DataFrame(sensitivity_rows).to_csv(TABLES / "sensitivity_summary.csv", index=False)

    print("\nRunning two-way moving-block bootstrap...", flush=True)
    rng = np.random.default_rng(20260717)
    winners = []
    boot_rows = []
    for replicate in range(1000):
        sample = resample_two_way(core, rng, block_length=3)
        result = metric_table(sample)
        winners.append(result.iloc[0]["index"])
        row = {"replicate": replicate, "winner": result.iloc[0]["index"]}
        for index_name in ["SPEI3", "SPI3", "SPEI6"]:
            row[f"r_{index_name}"] = result.loc[
                result["index"] == index_name, "pearson"
            ].iloc[0]
        boot_rows.append(row)
    boot = pd.DataFrame(boot_rows)
    boot["delta_SPEI3_SPI3"] = boot["r_SPEI3"] - boot["r_SPI3"]
    boot["delta_SPEI3_SPEI6"] = boot["r_SPEI3"] - boot["r_SPEI6"]
    boot.to_csv(TABLES / "bootstrap_two_way_results.csv", index=False)
    frequencies = (
        pd.Series(winners).value_counts().rename_axis("index").reset_index(name="count")
    )
    frequencies["frequency_pct"] = 100 * frequencies["count"] / len(winners)
    frequencies.to_csv(TABLES / "bootstrap_selection_frequency.csv", index=False)

    cv_rows = []
    for year in sorted(core["year"].unique()):
        result = metric_table(core[core["year"] != year])
        cv_rows.append({"type": "leave_year_out", "omitted": year, "winner": result.iloc[0]["index"]})
    for district in sorted(core["key"].unique()):
        result = metric_table(core[core["key"] != district])
        cv_rows.append({"type": "leave_district_out", "omitted": district, "winner": result.iloc[0]["index"]})
    pd.DataFrame(cv_rows).to_csv(TABLES / "cross_validation_ranking.csv", index=False)

    threshold_rows = []
    for year in sorted(core["year"].unique()):
        train = core[core["year"] != year]
        test = core[core["year"] == year].copy()
        drought_rate = (train["SPEI3"] < -1).mean()
        threshold = train["VHI"].quantile(drought_rate)
        test["drought"] = test["SPEI3"] < -1
        test["detected"] = test["VHI"] < threshold
        test["threshold"] = threshold
        threshold_rows.append(test)
    threshold_predictions = pd.concat(threshold_rows, ignore_index=True)
    threshold_predictions.to_csv(TABLES / "categorical_leave_year_out_predictions.csv", index=False)
    categorical_point = contingency(
        threshold_predictions["drought"], threshold_predictions["detected"]
    )
    categorical_boot = []
    rng_cat = np.random.default_rng(811)
    years = sorted(threshold_predictions["year"].unique())
    by_year = {year: threshold_predictions[threshold_predictions["year"] == year] for year in years}
    for _ in range(2000):
        sampled_years = moving_year_sample(rng_cat, years, block_length=3)
        sample = pd.concat([by_year[year] for year in sampled_years], ignore_index=True)
        categorical_boot.append(contingency(sample["drought"], sample["detected"]))
    categorical_boot = pd.DataFrame(categorical_boot)
    summary = []
    for metric in ["POD", "FAR", "CSI", "bias", "HSS"]:
        summary.append(
            {
                "metric": metric,
                "estimate": categorical_point[metric],
                "ci_low": categorical_boot[metric].quantile(0.025),
                "ci_high": categorical_boot[metric].quantile(0.975),
            }
        )
    pd.DataFrame(summary).to_csv(TABLES / "categorical_skill_summary.csv", index=False)

    district_skill = []
    for district, district_data in core.groupby("key"):
        predictions = []
        for year in sorted(district_data["year"].unique()):
            train = district_data[district_data["year"] != year]
            test = district_data[district_data["year"] == year].copy()
            drought_rate = (train["SPEI3"] < -1).mean()
            threshold = train["VHI"].quantile(drought_rate)
            test["drought"] = test["SPEI3"] < -1
            test["detected"] = test["VHI"] < threshold
            predictions.append(test)
        pred = pd.concat(predictions)
        district_skill.append({"district": district, **contingency(pred["drought"], pred["detected"])})
    pd.DataFrame(district_skill).to_csv(TABLES / "categorical_skill_by_district.csv", index=False)

    make_figures(primary, lag, frequencies, boot, core_monthly, threshold_predictions, pd.DataFrame(district_skill), index_data, vhi)
    for crop in ["sunflower", "wheat"]:   # deposited panels (district x year yields joined to SPEI)
        shutil.copy(DATA / f"yield_panel_{crop}.csv", TABLES / f"yield_panel_{crop}.csv")
    yield_analysis(index_data)
    write_summary(primary, all7_table, frequencies, boot, categorical_point, categorical_boot, sensitivity_rows)
    print("done:", TABLES, FIGURES)

if __name__ == "__main__":
    main()
