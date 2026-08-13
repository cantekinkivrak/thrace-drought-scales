from pathlib import Path

import pandas as pd

import revision_analysis as ra


def main():
    ra.make_study_area_figure()

    ranking = pd.read_csv(ra.TABLES / "scale_selection_primary_six_districts.csv")
    lag = pd.read_csv(ra.TABLES / "lag_profiles_primary.csv")
    ordered = ranking.sort_values("composite")
    fig, axes = ra.plt.subplots(
        1, 2, figsize=(9.0, 4.7), gridspec_kw={"width_ratios": [1.1, 1]}
    )
    axes[0].barh(
        ordered["index"].map(ra._pretty_index),
        ordered["composite"],
        color="#476f9e",
    )
    axes[0].set(xlabel="Four-criterion rank score (maximum 40)", ylabel="")
    axes[0].axvline(36, color="0.7", lw=0.8, ls=":")

    summary = lag[lag["lag"].astype(str) == "summary"].copy().sort_values("r_max")
    positions = ra.np.arange(len(summary))
    axes[1].scatter(summary["r_max"], positions, s=45, color="#9a4d3f")
    for position, row in zip(positions, summary.itertuples()):
        axes[1].annotate(
            f"lag {int(row.best_lag):+d}",
            (row.r_max, position),
            xytext=(5, 0),
            va="center",
            textcoords="offset points",
            fontsize=7,
        )
    axes[1].set(
        xlabel="Maximum Pearson r over lags −3 to +3",
        ylabel="",
        yticks=positions,
        yticklabels=summary["index"].map(ra._pretty_index),
    )
    axes[1].grid(axis="x", alpha=0.2)
    fig.tight_layout()
    ra._save_figure(fig, "Fig3_scale_selection_revised.png")


if __name__ == "__main__":
    main()
