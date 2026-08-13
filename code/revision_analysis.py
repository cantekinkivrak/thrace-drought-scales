from __future__ import annotations

import json
import math
import os
import shutil
import struct
import unicodedata
import warnings
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression


warnings.filterwarnings("ignore")
mpl.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    }
)

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "data" / "derived"
SPATIAL = ROOT / "data" / "spatial"
OUT = ROOT / "revision_outputs"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
YIELD_FILE = Path(os.environ.get("YIELD_FILE", ROOT / "Yield.xls"))
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

SCALES = [1, 3, 6, 9, 12]
IDX = [f"{family}{scale}" for family in ["SPI", "SPEI"] for scale in SCALES]
WIN = [6, 7, 8, 9, 10]
CORE_KEYS = ["corlu", "edirne", "kirklareli", "luleburgaz", "tekirdag", "uzunkopru"]
ALL7_KEYS = CORE_KEYS + ["ipsala"]
TR = str.maketrans("ıİşŞçÇğĞöÖüÜ", "iissccggoouu")


def fold(value: object) -> str:
    value = str(value).translate(TR)
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    ).lower().strip()


def clean_pair(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


DISTROS = {
    "normal": stats.norm,
    "logistic": stats.logistic,
    "pearson3": stats.pearson3,
    "gev": stats.genextreme,
    "genlogistic": stats.genlogistic,
    "loglogistic": stats.fisk,
}


def fit_flexible_standardization(values: np.ndarray):
    """Fit candidate distributions and choose the minimum-AICc model."""
    values = np.asarray(values, dtype=float)
    output = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    x = values[valid]
    if len(x) < 20:
        return output, {"distribution": "insufficient", "n": len(x)}

    candidates = []
    for name, distribution in DISTROS.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parameters = distribution.fit(x)
                logpdf = distribution.logpdf(x, *parameters)
                cdf = distribution.cdf(x, *parameters)
            if not np.all(np.isfinite(logpdf)) or not np.all(np.isfinite(cdf)):
                continue
            parameter_count = len(parameters)
            log_likelihood = float(logpdf.sum())
            aic = 2 * parameter_count - 2 * log_likelihood
            correction = (
                2 * parameter_count * (parameter_count + 1)
                / (len(x) - parameter_count - 1)
            )
            aicc = aic + correction
            ks = stats.kstest(x, distribution.cdf, args=parameters)
            candidates.append(
                {
                    "distribution": name,
                    "object": distribution,
                    "parameters": parameters,
                    "aicc": float(aicc),
                    "ks_stat": float(ks.statistic),
                    "ks_p": float(ks.pvalue),
                }
            )
        except Exception:
            continue

    if not candidates:
        ranks = stats.rankdata(x, method="average")
        probabilities = (ranks - 0.44) / (len(x) + 0.12)
        output[valid] = stats.norm.ppf(np.clip(probabilities, 1e-6, 1 - 1e-6))
        return output, {"distribution": "empirical_fallback", "n": len(x)}

    best = min(candidates, key=lambda row: row["aicc"])
    probabilities = best["object"].cdf(x, *best["parameters"])
    output[valid] = stats.norm.ppf(np.clip(probabilities, 1e-6, 1 - 1e-6))
    diagnostic = {
        "distribution": best["distribution"],
        "n": len(x),
        "aicc": best["aicc"],
        "ks_stat": best["ks_stat"],
        "ks_p": best["ks_p"],
        "parameters": json.dumps([float(v) for v in best["parameters"]]),
    }
    return output, diagnostic


def empirical_standardization(values: np.ndarray):
    values = np.asarray(values, dtype=float)
    output = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    x = values[valid]
    if len(x) < 20:
        return output
    ranks = stats.rankdata(x, method="average")
    probabilities = (ranks - 0.44) / (len(x) + 0.12)
    output[valid] = stats.norm.ppf(np.clip(probabilities, 1e-6, 1 - 1e-6))
    return output


def drought_indices_from_balance(balance: pd.DataFrame, value_col: str, prefix: str):
    outputs = []
    diagnostics = []
    for station, group in balance.groupby("name"):
        group = group.sort_values(["year", "month"]).copy()
        base = group[["name", "year", "month"]].copy()
        for scale in SCALES:
            accumulated = group[value_col].rolling(scale, min_periods=scale).sum()
            flexible = np.full(len(group), np.nan)
            empirical = np.full(len(group), np.nan)
            for month in range(1, 13):
                mask = group["month"].to_numpy() == month
                flex_values, diagnostic = fit_flexible_standardization(
                    accumulated.to_numpy()[mask]
                )
                flexible[mask] = flex_values
                empirical[mask] = empirical_standardization(accumulated.to_numpy()[mask])
                diagnostics.append(
                    {
                        "station": station,
                        "scale": scale,
                        "month": month,
                        "balance": value_col,
                        **diagnostic,
                    }
                )
            base[f"{prefix}{scale}"] = flexible
            base[f"{prefix}e{scale}"] = empirical
        outputs.append(base)
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(diagnostics)


def oudin_pet(temperature: float, latitude_degrees: float, month: int) -> float:
    latitude = np.radians(latitude_degrees)
    julian = np.array([15, 45, 74, 105, 135, 162, 198, 228, 258, 288, 318, 344])[
        month - 1
    ]
    inverse_distance = 1 + 0.033 * np.cos(2 * np.pi / 365 * julian)
    declination = 0.409 * np.sin(2 * np.pi / 365 * julian - 1.39)
    sunset_angle = np.arccos(
        np.clip(-np.tan(latitude) * np.tan(declination), -1, 1)
    )
    extraterrestrial = (
        (24 * 60 / np.pi)
        * 0.0820
        * inverse_distance
        * (
            sunset_angle * np.sin(latitude) * np.sin(declination)
            + np.cos(latitude) * np.cos(declination) * np.sin(sunset_angle)
        )
    )
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
    return max(extraterrestrial * (temperature + 5) / 245, 0) * days if temperature > -5 else 0


def add_leave_one_year_out_vhi(vhi: pd.DataFrame) -> pd.DataFrame:
    vhi = vhi.copy()
    vhi["VCI_LOO"] = np.nan
    vhi["TCI_LOO"] = np.nan
    for (_, _), index in vhi.groupby(["key", "month"]).groups.items():
        group = vhi.loc[index]
        for row_index, row in group.iterrows():
            reference = group[group["year"] != row["year"]]
            ndvi = reference["NDVI"].dropna()
            lst = reference["LST"].dropna()
            if np.isfinite(row["NDVI"]) and len(ndvi) > 10 and ndvi.max() > ndvi.min():
                vhi.loc[row_index, "VCI_LOO"] = np.clip(
                    100 * (row["NDVI"] - ndvi.min()) / (ndvi.max() - ndvi.min()),
                    0,
                    100,
                )
            if np.isfinite(row["LST"]) and len(lst) > 10 and lst.max() > lst.min():
                vhi.loc[row_index, "TCI_LOO"] = np.clip(
                    100 * (lst.max() - row["LST"]) / (lst.max() - lst.min()),
                    0,
                    100,
                )
    vhi["VHI_LOO"] = 0.5 * (vhi["VCI_LOO"] + vhi["TCI_LOO"])
    return vhi


def metric_table(frame: pd.DataFrame, reference: str = "VHI", indices=IDX):
    rows = []
    for index_name in indices:
        x, y = clean_pair(frame[reference], frame[index_name])
        if len(x) < 30:
            continue
        q1, q3 = np.quantile(y, [1 / 3, 2 / 3])
        rows.append(
            {
                "index": index_name,
                "n": len(x),
                "pearson": stats.pearsonr(x, y).statistic,
                "spearman": stats.spearmanr(x, y).statistic,
                "MI": mutual_info_regression(
                    y.reshape(-1, 1), x, random_state=0, n_neighbors=5
                )[0],
                "tercile_delta": x[y >= q3].mean() - x[y <= q1].mean(),
            }
        )
    result = pd.DataFrame(rows)
    metric_names = ["pearson", "spearman", "MI", "tercile_delta"]
    for metric in metric_names:
        result[f"{metric}_score"] = result[metric].rank(method="average")
        result[f"{metric}_rank"] = result[metric].rank(
            method="min", ascending=False
        )
    result["composite"] = result[
        [f"{metric}_score" for metric in metric_names]
    ].sum(axis=1)
    result["overall_rank"] = result["composite"].rank(
        method="min", ascending=False
    )
    return result.sort_values(
        ["composite", "pearson"], ascending=False
    ).reset_index(drop=True)


def lag_table(monthly_frame: pd.DataFrame, reference: str = "VHI"):
    rows = []
    base = monthly_frame.sort_values(["key", "year", "month"]).copy()
    for index_name in IDX:
        profile = {}
        for lag in range(-3, 4):
            shifted = base.groupby("key")[index_name].shift(lag)
            subset = base[base["month"].isin(WIN)]
            x, y = clean_pair(subset[reference], shifted.loc[subset.index])
            profile[lag] = stats.pearsonr(x, y).statistic
            rows.append({"index": index_name, "lag": lag, "r": profile[lag]})
        best_lag = max(profile, key=profile.get)
        r0 = profile[0]
        rmax = profile[best_lag]
        rows.append(
            {
                "index": index_name,
                "lag": "summary",
                "r": r0,
                "best_lag": best_lag,
                "r_max": rmax,
                "r0_over_rmax": r0 / rmax if rmax > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def moving_year_sample(rng: np.random.Generator, years, block_length=3):
    years = list(sorted(years))
    sampled = []
    while len(sampled) < len(years):
        start = int(rng.integers(0, len(years)))
        sampled.extend(
            years[(start + offset) % len(years)] for offset in range(block_length)
        )
    return sampled[: len(years)]


def resample_two_way(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    block_length: int = 3,
):
    years = sorted(frame["year"].unique())
    districts = sorted(frame["key"].unique())
    sampled_years = moving_year_sample(rng, years, block_length)
    sampled_districts = rng.choice(districts, len(districts), replace=True)
    pieces = []
    for district_number, district in enumerate(sampled_districts):
        district_data = frame[frame["key"] == district]
        for year_number, year in enumerate(sampled_years):
            part = district_data[district_data["year"] == year].copy()
            if len(part):
                part["bootstrap_key"] = f"d{district_number}"
                part["bootstrap_year"] = year_number
                pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def contingency(reference_flag, detection_flag):
    reference_flag = np.asarray(reference_flag, dtype=bool)
    detection_flag = np.asarray(detection_flag, dtype=bool)
    hit = int((reference_flag & detection_flag).sum())
    false_alarm = int((~reference_flag & detection_flag).sum())
    miss = int((reference_flag & ~detection_flag).sum())
    correct_negative = int((~reference_flag & ~detection_flag).sum())
    total = hit + false_alarm + miss + correct_negative
    pod = hit / (hit + miss) if hit + miss else np.nan
    far = false_alarm / (hit + false_alarm) if hit + false_alarm else np.nan
    csi = hit / (hit + false_alarm + miss) if hit + false_alarm + miss else np.nan
    bias = (hit + false_alarm) / (hit + miss) if hit + miss else np.nan
    expected = (
        (hit + false_alarm) * (hit + miss)
        + (miss + correct_negative) * (false_alarm + correct_negative)
    ) / total
    hss = (hit + correct_negative - expected) / (total - expected)
    return {
        "hits": hit,
        "false_alarms": false_alarm,
        "misses": miss,
        "correct_negatives": correct_negative,
        "n": total,
        "POD": pod,
        "FAR": far,
        "CSI": csi,
        "bias": bias,
        "HSS": hss,
    }


def _pretty_index(index_name: str) -> str:
    return index_name.replace("SPEI", "SPEI-").replace("SPI", "SPI-")


def _save_figure(fig: plt.Figure, filename: str):
    fig.savefig(FIGURES / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _dms_to_decimal(value: str) -> float:
    number = str(value).replace("°", " ").replace("'", " ").replace('"', " ")
    direction = number[-1]
    degrees, minutes, seconds = [float(v) for v in number[:-1].split()]
    result = degrees + minutes / 60 + seconds / 3600
    return -result if direction in {"S", "W"} else result


def make_study_area_figure():
    """Draw the study boundary and the seven district-matched stations."""
    try:
        polygons = []
        with open(SPATIAL / "Trakya_Merged.shp", "rb") as stream:
            stream.seek(100)
            while True:
                record_header = stream.read(8)
                if len(record_header) < 8:
                    break
                _, length_words = struct.unpack(">2i", record_header)
                content = stream.read(length_words * 2)
                shape_type = struct.unpack("<i", content[:4])[0]
                if shape_type not in {5, 15, 25}:
                    continue
                n_parts, n_points = struct.unpack("<2i", content[36:44])
                parts = list(struct.unpack(f"<{n_parts}i", content[44:44 + 4 * n_parts]))
                point_start = 44 + 4 * n_parts
                points = np.asarray(struct.unpack(f"<{2 * n_points}d", content[point_start:point_start + 16 * n_points])).reshape(-1, 2)
                ends = parts[1:] + [n_points]
                polygons.extend(points[start:end] for start, end in zip(parts, ends))
        fig, ax = plt.subplots(figsize=(9.0, 6.3))
        for polygon in polygons:
            ax.fill(polygon[:, 0], polygon[:, 1], color="#e9f2df", zorder=0)
            ax.plot(polygon[:, 0], polygon[:, 1], color="#638d56", lw=1.4, zorder=1)

        stations = pd.read_excel(SPATIAL / "station_info.xlsx")
        stations["key"] = stations["İstasyon ismi"].map(fold)
        stations = stations[stations["key"].isin(ALL7_KEYS)].copy()
        stations["lat"] = stations["Enlem"].map(_dms_to_decimal)
        stations["lon"] = stations["Boylam"].map(_dms_to_decimal)
        core = stations[stations["key"].isin(CORE_KEYS)]
        contrast = stations[stations["key"] == "ipsala"]
        ax.scatter(
            core["lon"], core["lat"], s=80, color="#264f92", edgecolor="white",
            linewidth=1.0, label="Primary rainfed district", zorder=3,
        )
        ax.scatter(
            contrast["lon"], contrast["lat"], s=100, marker="^", color="#d4483b",
            edgecolor="white", linewidth=1.0, label="İpsala contrast district", zorder=3,
        )
        offsets = {
            "corlu": (0.05, 0.06), "edirne": (-0.33, 0.06),
            "ipsala": (-0.30, -0.12), "kirklareli": (0.04, 0.06),
            "luleburgaz": (0.05, -0.10), "tekirdag": (0.05, 0.06),
            "uzunkopru": (-0.38, -0.10),
        }
        labels = {
            "corlu": "Çorlu", "edirne": "Edirne", "ipsala": "İpsala",
            "kirklareli": "Kırklareli", "luleburgaz": "Lüleburgaz",
            "tekirdag": "Tekirdağ", "uzunkopru": "Uzunköprü",
        }
        for row in stations.itertuples():
            dx, dy = offsets[row.key]
            ax.text(row.lon + dx, row.lat + dy, labels[row.key], fontsize=9, weight="bold")
        ax.set(xlabel="Longitude (°E)", ylabel="Latitude (°N)")
        ax.legend(loc="lower left", frameon=True)
        ax.grid(alpha=0.25, linestyle=":")
        ax.set_aspect("equal", adjustable="box")
        _save_figure(fig, "Fig1_study_area_revised.png")
    except Exception as exc:
        print(f"Study-area figure could not be redrawn: {exc}", flush=True)


def make_figures(
    ranking: pd.DataFrame,
    lag: pd.DataFrame,
    frequencies: pd.DataFrame,
    bootstrap: pd.DataFrame,
    core_monthly: pd.DataFrame,
    predictions: pd.DataFrame,
    district_skill: pd.DataFrame,
    index_data: pd.DataFrame,
    vhi: pd.DataFrame,
):
    make_study_area_figure()

    annual = (
        core_monthly[core_monthly["month"].isin(WIN)]
        .groupby(["key", "year"], as_index=False)["VHI"].mean()
    )
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for district, group in annual.groupby("key"):
        ax.plot(group["year"], group["VHI"], lw=1.0, alpha=0.65, label=district.title())
    regional = annual.groupby("year")["VHI"].mean()
    ax.plot(regional.index, regional.values, color="black", lw=2.4, label="Six-district mean")
    ax.axhline(40, color="#b33f3f", ls="--", lw=1, label="VHI = 40")
    ax.set(xlabel="Year", ylabel="June–October mean VHI")
    ax.legend(ncol=4, fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.2)
    _save_figure(fig, "Fig2_vhi_series_revised.png")

    ordered = ranking.sort_values("composite")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.7), gridspec_kw={"width_ratios": [1.1, 1]})
    axes[0].barh(ordered["index"].map(_pretty_index), ordered["composite"], color="#476f9e")
    axes[0].set(xlabel="Four-criterion rank score (maximum 40)", ylabel="")
    axes[0].axvline(36, color="0.7", lw=0.8, ls=":")
    summary_lag = lag[lag["lag"].astype(str) == "summary"].copy().sort_values("r_max")
    positions = np.arange(len(summary_lag))
    axes[1].scatter(summary_lag["r_max"], positions, s=45, color="#9a4d3f")
    for position, row in zip(positions, summary_lag.itertuples()):
        axes[1].annotate(f"lag {int(row.best_lag):+d}", (row.r_max, position),
                         xytext=(5, 0), va="center", textcoords="offset points", fontsize=7)
    axes[1].set(xlabel="Maximum Pearson r over lags −3 to +3", ylabel="",
                yticks=positions, yticklabels=summary_lag["index"].map(_pretty_index))
    axes[1].grid(axis="x", alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, "Fig4_scale_selection_revised.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.1))
    frequency_plot = frequencies.sort_values("frequency_pct")
    axes[0].barh(frequency_plot["index"].map(_pretty_index),
                 frequency_plot["frequency_pct"], color="#4a7c59")
    axes[0].set(xlabel="Winner frequency (%)", ylabel="")
    axes[1].hist(bootstrap["delta_SPEI3_SPI3"], bins=30, alpha=0.72,
                 color="#476f9e", label="SPEI-3 minus SPI-3")
    axes[1].hist(bootstrap["delta_SPEI3_SPEI6"], bins=30, alpha=0.62,
                 color="#d08a37", label="SPEI-3 minus SPEI-6")
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set(xlabel="Bootstrap difference in Pearson r", ylabel="Replicates")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, "Fig5_bootstrap_revised.png")

    annual_pair = (
        core_monthly[core_monthly["month"].isin(WIN)]
        .groupby(["key", "year"], as_index=False)[["VHI", "SPEI3"]].mean()
        .dropna()
    )
    district_rows = []
    for district, group in annual_pair.groupby("key"):
        district_rows.append(
            {"district": district, "r": stats.pearsonr(group["VHI"], group["SPEI3"]).statistic,
             "n": len(group)}
        )
    district_corr = pd.DataFrame(district_rows).sort_values("r")
    district_corr.to_csv(TABLES / "annual_correlation_by_district.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.1))
    axes[0].barh(district_corr["district"].str.title(), district_corr["r"], color="#567ea4")
    axes[0].set(xlabel="Pearson r: annual VHI vs SPEI-3", xlim=(0, 1), ylabel="")
    axes[1].scatter(annual_pair["SPEI3"], annual_pair["VHI"], s=12, alpha=0.45, color="#355c7d")
    slope, intercept = np.polyfit(annual_pair["SPEI3"], annual_pair["VHI"], 1)
    xx = np.linspace(annual_pair["SPEI3"].min(), annual_pair["SPEI3"].max(), 100)
    axes[1].plot(xx, intercept + slope * xx, color="#b7473f", lw=2)
    axes[1].set(xlabel="June–October mean SPEI-3", ylabel="June–October mean VHI")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, "Fig6_interannual_revised.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    metric_names = ["POD", "FAR", "CSI", "HSS"]
    aggregate = contingency(predictions["drought"], predictions["detected"])
    axes[0].bar(metric_names, [aggregate[m] for m in metric_names], color=["#4773a8", "#b85c4b", "#708e57", "#7663a8"])
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set(ylabel="Skill value", ylim=(min(-0.05, aggregate["HSS"] - 0.1), 1))
    axes[1].barh(district_skill["district"].str.title(), district_skill["HSS"], color="#7663a8")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set(xlabel="District HSS", ylabel="")
    fig.tight_layout()
    _save_figure(fig, "Fig7_detection_revised.png")

    case = core_monthly[core_monthly["year"] == 2024].copy()
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.0), sharex=True)
    for district, group in case.groupby("key"):
        axes[0].plot(group["month"], group["VHI"], marker="o", ms=3, lw=1, label=district.title())
        axes[1].plot(group["month"], group["SPEI3"], marker="o", ms=3, lw=1)
    axes[0].axhline(40, color="#b7473f", ls="--", lw=1)
    axes[1].axhline(-1, color="#b7473f", ls="--", lw=1)
    axes[0].set(ylabel="VHI")
    axes[1].set(xlabel="Month in 2024", ylabel="SPEI-3", xticks=range(1, 13))
    axes[0].legend(ncol=3, frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, "Fig9_case2024_revised.png")

    existing_map = ROOT / "figures" / "Fig_VHI_SPEI.png"
    if existing_map.exists():
        shutil.copyfile(existing_map, FIGURES / "Fig3_spatial_patterns_existing.png")


def fixed_effects_design(data: pd.DataFrame, index_name: str, levels: list[str]):
    columns = [np.ones(len(data)), data[index_name].to_numpy(float), data["year_c"].to_numpy(float)]
    names = ["intercept", index_name, "year_c"]
    for level in levels[1:]:
        columns.append((data["key"] == level).to_numpy(float))
        names.append(f"key_{level}")
    return np.column_stack(columns), names


def fit_fixed_effects(data: pd.DataFrame, index_name: str, levels: list[str]):
    design, names = fixed_effects_design(data, index_name, levels)
    response = data["yield"].to_numpy(float)
    beta, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    fitted = design @ beta
    residual = response - fitted
    n, k = design.shape
    rss = float(residual @ residual)
    tss = float(np.square(response - response.mean()).sum())
    r2 = 1 - rss / tss
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - k)
    aic = n * (np.log(2 * np.pi) + 1 + np.log(rss / n)) + 2 * k
    return {
        "beta": beta,
        "names": names,
        "design": design,
        "residual": residual,
        "rss": rss,
        "adjusted_R2": adjusted_r2,
        "AIC": aic,
    }


def two_way_cluster_covariance(fit: dict, district, year):
    design = fit["design"]
    residual = fit["residual"]
    n, k = design.shape
    bread = np.linalg.pinv(design.T @ design)
    scores = design * residual[:, None]

    def meat(labels):
        labels = np.asarray(labels)
        unique = np.unique(labels)
        result = np.zeros((k, k))
        for label in unique:
            score_sum = scores[labels == label].sum(axis=0)
            result += np.outer(score_sum, score_sum)
        correction = (len(unique) / (len(unique) - 1)) * ((n - 1) / (n - k)) if len(unique) > 1 else 1
        return correction * result

    intersection = np.array([f"{a}::{b}" for a, b in zip(district, year)])
    covariance = bread @ (meat(district) + meat(year) - meat(intersection)) @ bread
    return (covariance + covariance.T) / 2


def yield_analysis(index_data: pd.DataFrame):
    """Fixed-effects yield models with two-way clustered uncertainty and LOYO RMSE."""
    model_rows = []
    plot_rows = []
    for crop, month in [("Sunflower", 8), ("Wheat", 6)]:
        panel_file = TABLES / f"yield_panel_{crop.lower()}.csv"
        if YIELD_FILE.exists():
            yield_wide = pd.read_excel(YIELD_FILE, sheet_name=crop)
            yield_long = yield_wide.melt(
                id_vars="Year", var_name="name", value_name="yield"
            ).rename(columns={"Year": "year"})
            yield_long["key"] = yield_long["name"].map(fold)
            yield_long = yield_long[yield_long["key"].isin(CORE_KEYS)]
            drought = index_data[index_data["month"] == month][
                ["key", "year"] + [f"SPEI{k}" for k in SCALES]
            ]
            panel = yield_long.merge(drought, on=["key", "year"]).dropna(subset=["yield"])
            panel["year_c"] = panel["year"] - panel["year"].mean()
            panel.to_csv(panel_file, index=False)
        else:
            panel = pd.read_csv(panel_file)
        levels = sorted(panel["key"].unique())
        for scale in SCALES:
            index_name = f"SPEI{scale}"
            data = panel.dropna(subset=[index_name]).copy()
            model = fit_fixed_effects(data, index_name, levels)
            covariance = two_way_cluster_covariance(model, data["key"], data["year"])
            coefficient_position = model["names"].index(index_name)
            se = float(np.sqrt(max(covariance[coefficient_position, coefficient_position], 0)))
            coefficient = float(model["beta"][coefficient_position])
            errors = []
            for omitted_year in sorted(data["year"].unique()):
                train = data[data["year"] != omitted_year]
                test = data[data["year"] == omitted_year]
                fitted = fit_fixed_effects(train, index_name, levels)
                test_design, _ = fixed_effects_design(test, index_name, levels)
                errors.extend((test["yield"].to_numpy() - test_design @ fitted["beta"]).tolist())
            rmse = float(np.sqrt(np.mean(np.square(errors))))
            model_rows.append(
                {
                    "crop": crop, "month": month, "index": index_name, "n": len(data),
                    "coefficient_kg_da": coefficient, "cluster_se": se,
                    "ci_low": coefficient - 1.96 * se, "ci_high": coefficient + 1.96 * se,
                    "p_cluster_approx": 2 * stats.norm.sf(abs(coefficient / se)) if se > 0 else np.nan,
                    "AIC": model["AIC"], "adjusted_R2": model["adjusted_R2"], "LOYO_RMSE": rmse,
                }
            )
            plot_rows.append(
                {"crop": crop, "scale": scale, "coef": coefficient, "low": coefficient - 1.96 * se,
                 "high": coefficient + 1.96 * se}
            )
    results = pd.DataFrame(model_rows)
    results.to_csv(TABLES / "yield_fixed_effects_models.csv", index=False)
    plotting = pd.DataFrame(plot_rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=False)
    for ax, crop in zip(axes, ["Sunflower", "Wheat"]):
        subset = plotting[plotting["crop"] == crop]
        ax.errorbar(subset["scale"], subset["coef"],
                    yerr=[subset["coef"] - subset["low"], subset["high"] - subset["coef"]],
                    fmt="o-", color="#4773a8", capsize=3)
        ax.axhline(0, color="black", lw=0.8)
        ax.set(title=crop, xlabel="SPEI accumulation scale (months)", ylabel="Yield coefficient (kg da⁻¹ per SPEI unit)", xticks=SCALES)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, "Fig8_yield_revised.png")


def homogeneity_screen(raw_balance: pd.DataFrame):
    """Relative SNHT screen; diagnostic only, not an undocumented homogenisation."""
    annual_temp = raw_balance.pivot_table(index="year", columns="key", values="temp", aggfunc="mean")
    annual_precip = raw_balance.pivot_table(index="year", columns="key", values="precip", aggfunc="sum")
    rng = np.random.default_rng(437)
    rows = []
    for variable, annual in [("temperature", annual_temp), ("precipitation", annual_precip)]:
        standardized = (annual - annual.mean()) / annual.std(ddof=1)
        for station in standardized.columns:
            relative = (standardized[station] - standardized.drop(columns=station).mean(axis=1)).dropna()
            z = ((relative - relative.mean()) / relative.std(ddof=1)).to_numpy()
            n = len(z)
            def maximum_stat(values):
                candidates = []
                for split in range(10, n - 9):
                    candidates.append(split * values[:split].mean() ** 2 + (n - split) * values[split:].mean() ** 2)
                best = int(np.argmax(candidates))
                return float(candidates[best]), best + 10
            observed, split = maximum_stat(z)
            permutation = np.array([maximum_stat(rng.permutation(z))[0] for _ in range(999)])
            p_value = (1 + np.sum(permutation >= observed)) / 1000
            rows.append({"variable": variable, "station": station, "n_years": n,
                         "SNHT_max": observed, "candidate_break_year": int(relative.index[split]),
                         "permutation_p": p_value})
    pd.DataFrame(rows).to_csv(TABLES / "relative_snht_screen.csv", index=False)


def write_summary(
    primary: pd.DataFrame,
    all7: pd.DataFrame,
    frequencies: pd.DataFrame,
    bootstrap: pd.DataFrame,
    categorical_point: dict,
    categorical_boot: pd.DataFrame,
    sensitivities: list[dict],
):
    winner = primary.iloc[0]
    all7_winner = all7.iloc[0]
    frequency = frequencies.set_index("index")["frequency_pct"].to_dict()
    lines = [
        "# Revised analysis summary",
        "",
        f"- Primary sample: six rainfed plateau districts; June–October, 1985–2024.",
        f"- Primary winner: {winner['index']} (four-criterion score {winner['composite']:.1f}/40; Pearson r={winner['pearson']:.3f}; n={int(winner['n'])}).",
        f"- Seven-district sensitivity winner: {all7_winner['index']} (r={all7_winner['pearson']:.3f}).",
        f"- Two-way moving-block bootstrap winner frequency for SPEI3: {frequency.get('SPEI3', 0):.1f}%.",
        f"- Bootstrap difference SPEI3-SPI3: median {bootstrap['delta_SPEI3_SPI3'].median():.3f}, 95% interval [{bootstrap['delta_SPEI3_SPI3'].quantile(.025):.3f}, {bootstrap['delta_SPEI3_SPI3'].quantile(.975):.3f}].",
        f"- Cross-validated categorical HSS: {categorical_point['HSS']:.3f}; 95% block-bootstrap interval [{categorical_boot['HSS'].quantile(.025):.3f}, {categorical_boot['HSS'].quantile(.975):.3f}].",
        "",
        "## Sensitivity scenarios",
        "",
    ]
    for row in sensitivities:
        lines.append(f"- {row['scenario']}: {row['top_index']} (SPEI3 r={row['SPEI3_r']:.3f}, SPI3 r={row['SPI3_r']:.3f}, n={row['n']}).")
    (OUT / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    raw_balance = pd.read_csv(DERIVED / "trakya_TP_PET_1965_2024.csv")
    raw_balance["key"] = raw_balance["name"].map(fold)
    raw_balance = raw_balance[raw_balance["key"].isin(ALL7_KEYS)].copy()

    print("Fitting flexible SPEI distributions...", flush=True)
    flexible, diagnostics = drought_indices_from_balance(raw_balance, "D", "SPEI")
    flexible["key"] = flexible["name"].map(fold)
    original = pd.read_csv(DERIVED / "trakya_spi_spei_1965_2024.csv")
    original["key"] = original["name"].map(fold)
    original = original[original["key"].isin(ALL7_KEYS)]
    index_data = original[["key", "name", "year", "month"] + [f"SPI{k}" for k in SCALES]].merge(
        flexible.drop(columns="name"), on=["year", "month", "key"], how="left"
    )

    oudin = raw_balance.copy()
    oudin["PET_oudin"] = [
        oudin_pet(row.temp, row.lat, int(row.month)) for row in oudin.itertuples()
    ]
    oudin["D_oudin"] = oudin["precip"] - oudin["PET_oudin"]
    flexible_oudin, diagnostics_oudin = drought_indices_from_balance(
        oudin, "D_oudin", "SPEIo"
    )

    index_data.to_csv(DERIVED / "trakya_spi_spei_flexible_1965_2024.csv", index=False)
    flexible_oudin.to_csv(DERIVED / "trakya_spei_oudin_flexible_1965_2024.csv", index=False)
    pd.concat([diagnostics, diagnostics_oudin], ignore_index=True).to_csv(
        TABLES / "distribution_fit_diagnostics.csv", index=False
    )

    vhi = pd.read_csv(DERIVED / "trakya_district_vhi_vci_tci_monthly.csv")
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
        original[["key", "year", "month"] + [f"SPI{k}" for k in SCALES]],
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
    if YIELD_FILE.exists() or all((TABLES / f"yield_panel_{crop}.csv").exists() for crop in ["sunflower", "wheat"]):
        yield_analysis(index_data)
    homogeneity_screen(raw_balance)
    write_summary(primary, all7_table, frequencies, boot, categorical_point, categorical_boot, sensitivity_rows)


if __name__ == "__main__":
    main()
