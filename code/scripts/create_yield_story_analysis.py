from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FRAMES = PROJECT_ROOT / "data" / "processed" / "model_frames"
MODEL_OUT = PROJECT_ROOT / "outputs" / "decision_support_models"
STORY_OUT = MODEL_OUT / "yield_story"
FIG_OUT = PROJECT_ROOT / "figures" / "yield_story"

COMBINED_FRAME = MODEL_FRAMES / "aus_us_overlap_crops_with_extreme_weather_1989_2021.csv"
PREDICTIONS = MODEL_OUT / "predictions_all_best_models.csv"
WEATHER_CHOICE = MODEL_OUT / "weather_condition_crop_choice_by_region_year.csv"
REGION_OBSERVED_CHOICE_TOP = MODEL_OUT / "region_observed_crop_choice_advantage_top_region_cases.csv"

KEYS = ["country", "region", "crop", "year_start"]
COUNTRY_ORDER = ["Australia", "United States"]
CROP_ORDER = ["Wheat", "Barley", "Canola", "Oats"]
COUNTRY_SHORT = {"Australia": "AUS", "United States": "US"}
COUNTRY_COLORS = {"Australia": "#2A9D8F", "United States": "#E76F51"}
CROP_COLORS = {
    "Wheat": "#D8A31A",
    "Barley": "#5E9F6E",
    "Canola": "#E9C46A",
    "Oats": "#8AB6D6",
}
REGION_SHORT = {
    "New South Wales": "NSW",
    "Queensland": "QLD",
    "South Australia": "SA",
    "Tasmania": "TAS",
    "Victoria": "VIC",
    "Western Australia": "WA",
    "Colorado": "CO",
    "Illinois": "IL",
    "Iowa": "IA",
    "Kansas": "KS",
    "Minnesota": "MN",
    "Montana": "MT",
    "Nebraska": "NE",
    "North Dakota": "ND",
    "Oklahoma": "OK",
    "South Dakota": "SD",
    "Texas": "TX",
    "Washington": "WA",
}

DRIVERS = [
    "season_rain_sum",
    "season_rain_mean",
    "season_tmax_mean",
    "season_tmin_mean",
    "growing_degree_days_base5",
    "season_radiation_sum",
    "radiation_sum_mid_y",
    "radiation_sum_late_y",
    "heat_days_35",
    "max_consecutive_heat_days_35",
    "hot_dry_days_30_1mm",
    "max_dry_spell_1mm",
    "dry_spell_events_14d",
    "max_3day_rain",
    "max_7day_rain",
    "rain_last_14d_before_harvest",
    "storm_before_harvest_flag",
    "frost_days_0",
]

DRIVER_LABELS = {
    "season_rain_sum": "season rain",
    "season_rain_mean": "daily rain",
    "season_tmax_mean": "mean max temp",
    "season_tmin_mean": "mean min temp",
    "growing_degree_days_base5": "growing degree days",
    "season_radiation_sum": "season radiation",
    "radiation_sum_mid_y": "mid-season radiation",
    "radiation_sum_late_y": "late-season radiation",
    "heat_days_35": "very hot days",
    "max_consecutive_heat_days_35": "longest heat spell",
    "hot_dry_days_30_1mm": "hot-dry days",
    "max_dry_spell_1mm": "longest dry spell",
    "dry_spell_events_14d": "long dry-spell events",
    "max_3day_rain": "max 3-day rain",
    "max_7day_rain": "max 7-day rain",
    "rain_last_14d_before_harvest": "late-season rain",
    "storm_before_harvest_flag": "storm-before-harvest flag",
    "frost_days_0": "frost days",
}

DRIVER_UNITS = {
    "season_rain_sum": "mm",
    "season_rain_mean": "mm/day",
    "season_tmax_mean": "C",
    "season_tmin_mean": "C",
    "growing_degree_days_base5": "degree-days",
    "season_radiation_sum": "MJ/m2",
    "radiation_sum_mid_y": "MJ/m2",
    "radiation_sum_late_y": "MJ/m2",
    "heat_days_35": "days",
    "max_consecutive_heat_days_35": "days",
    "hot_dry_days_30_1mm": "days",
    "max_dry_spell_1mm": "days",
    "dry_spell_events_14d": "events",
    "max_3day_rain": "mm",
    "max_7day_rain": "mm",
    "rain_last_14d_before_harvest": "mm",
    "storm_before_harvest_flag": "flag",
    "frost_days_0": "days",
}

STRESS_DRIVERS = {
    "heat_days_35",
    "max_consecutive_heat_days_35",
    "hot_dry_days_30_1mm",
    "max_dry_spell_1mm",
    "dry_spell_events_14d",
    "max_3day_rain",
    "max_7day_rain",
    "rain_last_14d_before_harvest",
    "storm_before_harvest_flag",
    "frost_days_0",
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.dpi": 120,
            "savefig.dpi": 320,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT / f"{stem}.png", bbox_inches="tight", dpi=320)
    fig.savefig(FIG_OUT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def load_data() -> pd.DataFrame:
    require_file(COMBINED_FRAME)
    require_file(PREDICTIONS)
    frame = pd.read_csv(COMBINED_FRAME, low_memory=False)
    pred = pd.read_csv(PREDICTIONS, low_memory=False)
    pred = pred[pred["dataset"].eq("aus_us_overlap")].copy()
    pred = pred[KEYS + ["split", "predicted_yield_t_ha"]].copy()
    df = frame.merge(pred, on=KEYS, how="left", validate="one_to_one")
    if df["predicted_yield_t_ha"].isna().any():
        raise ValueError("Missing predictions after merge.")
    for col in DRIVERS + ["yield_t_ha", "predicted_yield_t_ha", "year_start"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def status_from_delta(delta: float, threshold: float = 0.20) -> str:
    if pd.isna(delta):
        return "unknown"
    if delta >= threshold:
        return "increase"
    if delta <= -threshold:
        return "decrease"
    return "stable"


def build_region_crop_change(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(["country", "region", "crop"], dropna=False):
        early = group[group["year_start"].between(1989, 2000)]
        late = group[group["year_start"].between(2010, 2021)]
        if early.empty or late.empty:
            continue
        x = group["year_start"].to_numpy(dtype=float)
        obs_y = group["yield_t_ha"].to_numpy(dtype=float)
        pred_y = group["predicted_yield_t_ha"].to_numpy(dtype=float)
        obs_slope = float(np.polyfit(x, obs_y, 1)[0]) if np.isfinite(obs_y).all() and len(group) >= 3 else np.nan
        pred_slope = float(np.polyfit(x, pred_y, 1)[0]) if np.isfinite(pred_y).all() and len(group) >= 3 else np.nan
        early_obs = float(early["yield_t_ha"].mean())
        late_obs = float(late["yield_t_ha"].mean())
        early_pred = float(early["predicted_yield_t_ha"].mean())
        late_pred = float(late["predicted_yield_t_ha"].mean())
        rows.append(
            {
                "country": keys[0],
                "region": keys[1],
                "crop": keys[2],
                "early_observed_yield_1989_2000": early_obs,
                "late_observed_yield_2010_2021": late_obs,
                "delta_observed_yield_t_ha": late_obs - early_obs,
                "early_predicted_yield_1989_2000": early_pred,
                "late_predicted_yield_2010_2021": late_pred,
                "delta_predicted_yield_t_ha": late_pred - early_pred,
                "observed_slope_t_ha_per_year": obs_slope,
                "predicted_slope_t_ha_per_year": pred_slope,
                "observed_change_status": status_from_delta(late_obs - early_obs),
                "predicted_change_status": status_from_delta(late_pred - early_pred),
                "rows": len(group),
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values("delta_observed_yield_t_ha", ascending=False)
    out.to_csv(STORY_OUT / "region_crop_yield_change_early_late.csv", index=False)
    out.sort_values("delta_observed_yield_t_ha", ascending=False).to_csv(
        STORY_OUT / "crop_region_gain_loss_ranking.csv", index=False
    )
    return out


def format_delta(value: float, unit: str) -> str:
    if pd.isna(value):
        return "NA"
    sign = "+" if value >= 0 else ""
    if unit in {"days", "events", "flag"}:
        return f"{sign}{value:.1f} {unit}"
    return f"{sign}{value:.1f} {unit}"


def driver_phrase(feature: str, delta: float) -> str:
    label = DRIVER_LABELS.get(feature, feature)
    unit = DRIVER_UNITS.get(feature, "")
    if feature in STRESS_DRIVERS:
        direction = "more" if delta > 0 else "less"
    else:
        direction = "higher" if delta > 0 else "lower"
    return f"{direction} {label} ({format_delta(delta, unit)})"


def build_high_low_driver_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    driver_cols = [col for col in DRIVERS if col in df.columns]
    rows = []
    detail_rows = []
    for keys, group in df.groupby(["country", "region", "crop"], dropna=False):
        if len(group) < 12:
            continue
        low_cut = group["predicted_yield_t_ha"].quantile(0.25)
        high_cut = group["predicted_yield_t_ha"].quantile(0.75)
        low = group[group["predicted_yield_t_ha"] <= low_cut]
        high = group[group["predicted_yield_t_ha"] >= high_cut]
        if len(low) < 3 or len(high) < 3:
            continue
        story_items = []
        for feature in driver_cols:
            std = group[feature].std()
            if pd.isna(std) or math.isclose(float(std), 0.0):
                continue
            low_mean = float(low[feature].mean())
            high_mean = float(high[feature].mean())
            delta = high_mean - low_mean
            z_delta = delta / float(std)
            detail_rows.append(
                {
                    "country": keys[0],
                    "region": keys[1],
                    "crop": keys[2],
                    "feature": feature,
                    "feature_label": DRIVER_LABELS.get(feature, feature),
                    "low_predicted_yield_group_mean": low_mean,
                    "high_predicted_yield_group_mean": high_mean,
                    "high_minus_low_feature_value": delta,
                    "standardized_delta": z_delta,
                    "unit": DRIVER_UNITS.get(feature, ""),
                }
            )
            story_items.append((feature, delta, abs(z_delta)))
        top_items = sorted(story_items, key=lambda item: item[2], reverse=True)[:4]
        rows.append(
            {
                "country": keys[0],
                "region": keys[1],
                "crop": keys[2],
                "low_group_mean_observed_yield_t_ha": float(low["yield_t_ha"].mean()),
                "high_group_mean_observed_yield_t_ha": float(high["yield_t_ha"].mean()),
                "observed_yield_gap_high_minus_low_t_ha": float(high["yield_t_ha"].mean() - low["yield_t_ha"].mean()),
                "low_group_mean_predicted_yield_t_ha": float(low["predicted_yield_t_ha"].mean()),
                "high_group_mean_predicted_yield_t_ha": float(high["predicted_yield_t_ha"].mean()),
                "predicted_yield_gap_high_minus_low_t_ha": float(
                    high["predicted_yield_t_ha"].mean() - low["predicted_yield_t_ha"].mean()
                ),
                "top_weather_driver_story": "; ".join(driver_phrase(feature, delta) for feature, delta, _ in top_items),
                "low_group_years": ", ".join(str(int(year)) for year in sorted(low["year_start"].unique())),
                "high_group_years": ", ".join(str(int(year)) for year in sorted(high["year_start"].unique())),
            }
        )
    summary = pd.DataFrame(rows).sort_values("predicted_yield_gap_high_minus_low_t_ha", ascending=False)
    detail = pd.DataFrame(detail_rows)
    summary.to_csv(STORY_OUT / "high_low_yield_story_by_region_crop.csv", index=False)
    detail.to_csv(STORY_OUT / "high_low_yield_weather_driver_detail.csv", index=False)
    return summary, detail


def build_country_crop_driver_matrix(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in detail.groupby(["country", "crop", "feature"], dropna=False):
        rows.append(
            {
                "country": keys[0],
                "crop": keys[1],
                "feature": keys[2],
                "feature_label": DRIVER_LABELS.get(keys[2], keys[2]),
                "mean_standardized_delta_high_minus_low": float(group["standardized_delta"].mean()),
                "mean_raw_delta_high_minus_low": float(group["high_minus_low_feature_value"].mean()),
                "unit": DRIVER_UNITS.get(keys[2], ""),
                "region_count": group["region"].nunique(),
            }
        )
    matrix = pd.DataFrame(rows)
    matrix.to_csv(STORY_OUT / "country_crop_high_low_driver_matrix.csv", index=False)
    return matrix


def build_weather_choice_gain_stories() -> pd.DataFrame:
    require_file(WEATHER_CHOICE)
    choice = pd.read_csv(WEATHER_CHOICE, low_memory=False)
    rows = []
    for keys, group in choice.groupby(["country", "region"], dropna=False):
        best_crop_counts = group["best_crop"].value_counts()
        most_common_crop = str(best_crop_counts.index[0])
        largest = group.sort_values("best_crop_yield_advantage_t_ha", ascending=False).iloc[0]
        rows.append(
            {
                "country": keys[0],
                "region": keys[1],
                "most_common_best_crop": most_common_crop,
                "most_common_best_crop_share": float(best_crop_counts.iloc[0] / len(group)),
                "largest_advantage_year": int(largest["year_start"]),
                "largest_advantage_weather_regime": largest["weather_regime"],
                "largest_advantage_best_crop": largest["best_crop"],
                "largest_advantage_runner_up_crop": largest["runner_up_crop"],
                "largest_advantage_t_ha": float(largest["best_crop_yield_advantage_t_ha"]),
                "largest_advantage_best_predicted_yield_t_ha": float(largest["best_predicted_yield_t_ha"]),
            }
        )
    out = pd.DataFrame(rows).sort_values("largest_advantage_t_ha", ascending=False)
    out.to_csv(STORY_OUT / "weather_condition_crop_choice_gain_stories.csv", index=False)
    return out


def figure_17_observed_yield_change_heatmap(change: pd.DataFrame) -> None:
    countries = COUNTRY_ORDER
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [0.82, 1.18]})
    vmax = max(abs(change["delta_observed_yield_t_ha"].min()), abs(change["delta_observed_yield_t_ha"].max()))
    for ax, country in zip(axes, countries):
        part = change[change["country"].eq(country)].copy()
        regions = sorted(part["region"].unique(), key=lambda r: REGION_SHORT.get(r, r))
        pivot = part.pivot(index="region", columns="crop", values="delta_observed_yield_t_ha").reindex(regions)[CROP_ORDER]
        im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(country)
        ax.set_xticks(np.arange(len(CROP_ORDER)))
        ax.set_xticklabels(CROP_ORDER)
        ax.set_yticks(np.arange(len(regions)))
        ax.set_yticklabels([REGION_SHORT.get(r, r) for r in regions])
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.values[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=7.8)
    fig.subplots_adjust(left=0.08, right=0.88, top=0.84, bottom=0.12, wspace=0.22)
    cax = fig.add_axes([0.91, 0.2, 0.018, 0.55])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Observed yield change, late minus early (t/ha)")
    fig.suptitle("Which crop-region yields increased or decreased?", fontsize=15, weight="bold", y=0.96)
    fig.text(
        0.5,
        0.035,
        "Change compares 2010-2021 mean yield with 1989-2000 mean yield. Green means higher recent yield; red means lower.",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    save_figure(fig, "fig17_observed_yield_change_region_crop")


def figure_18_top_gain_loss_bars(change: pd.DataFrame) -> None:
    top_gain = change.sort_values("delta_observed_yield_t_ha", ascending=False).head(12)
    top_loss = change.sort_values("delta_observed_yield_t_ha", ascending=True).head(12)
    plot_df = pd.concat([top_loss, top_gain], ignore_index=True)
    plot_df["label"] = (
        plot_df["country"].map(COUNTRY_SHORT)
        + " "
        + plot_df["region"].map(lambda r: REGION_SHORT.get(r, r))
        + " "
        + plot_df["crop"]
    )
    colors = [CROP_COLORS.get(crop, "#999999") for crop in plot_df["crop"]]
    fig, ax = plt.subplots(figsize=(11.5, 8.2))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["delta_observed_yield_t_ha"], color=colors, alpha=0.88)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Observed yield change, 2010-2021 minus 1989-2000 (t/ha)")
    ax.set_title("Largest crop-region yield gains and losses")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CROP_COLORS[crop], alpha=0.88) for crop in CROP_ORDER]
    ax.legend(handles, CROP_ORDER, frameon=False, ncol=4, loc="lower right")
    fig.tight_layout()
    save_figure(fig, "fig18_top_yield_gain_loss_region_crop")


def figure_19_driver_matrix(matrix: pd.DataFrame) -> None:
    selected_features = [
        "season_rain_sum",
        "season_radiation_sum",
        "heat_days_35",
        "hot_dry_days_30_1mm",
        "max_dry_spell_1mm",
        "max_3day_rain",
        "rain_last_14d_before_harvest",
        "frost_days_0",
    ]
    matrix = matrix[matrix["feature"].isin(selected_features)].copy()
    matrix["column"] = matrix["country"].map(COUNTRY_SHORT) + " " + matrix["crop"]
    columns = [f"{COUNTRY_SHORT[country]} {crop}" for country in COUNTRY_ORDER for crop in CROP_ORDER]
    rows = selected_features
    pivot = matrix.pivot(index="feature", columns="column", values="mean_standardized_delta_high_minus_low").reindex(rows)[
        columns
    ]
    vmax = np.nanmax(np.abs(pivot.values))
    fig, ax = plt.subplots(figsize=(14, 6.8))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("Why high-yield years differ from low-yield years")
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([DRIVER_LABELS[f] for f in rows])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            if not np.isnan(value):
                ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=7.6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    cbar.set_label("High-yield minus low-yield years (standardized)")
    fig.text(
        0.5,
        0.02,
        "Positive values mean the feature is higher in high-predicted-yield years; negative values mean lower.",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save_figure(fig, "fig19_high_vs_low_yield_weather_driver_matrix")


def figure_20_story_cards(summary: pd.DataFrame, change: pd.DataFrame) -> None:
    selected = summary.sort_values("predicted_yield_gap_high_minus_low_t_ha", ascending=False).head(6).copy()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle("Example yield stories: where yield rises and what changes in high-yield years", fontsize=15, weight="bold")
    for ax, row in zip(axes.ravel(), selected.itertuples(index=False)):
        ax.axis("off")
        change_row = change[
            change["country"].eq(row.country) & change["region"].eq(row.region) & change["crop"].eq(row.crop)
        ].iloc[0]
        title = f"{COUNTRY_SHORT[row.country]} {REGION_SHORT.get(row.region, row.region)} {row.crop}"
        body = (
            f"Early-late observed change: {change_row.delta_observed_yield_t_ha:+.2f} t/ha\n"
            f"High vs low predicted-yield gap: {row.predicted_yield_gap_high_minus_low_t_ha:+.2f} t/ha\n\n"
            f"High-yield years tend to have:\n{textwrap.fill(row.top_weather_driver_story, width=42)}"
        )
        ax.text(0.02, 0.96, title, ha="left", va="top", fontsize=12, weight="bold", color="#222222")
        ax.text(0.02, 0.78, body, ha="left", va="top", fontsize=9.5, color="#333333", linespacing=1.35)
        rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, edgecolor="#CCCCCC", linewidth=1.0)
        ax.add_patch(rect)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, "fig20_region_crop_yield_story_cards")


def figure_21_weather_choice_advantage(choice_story: pd.DataFrame) -> None:
    # Keep Figure 8 aligned with the manuscript crop-choice table. The table is
    # filtered to crops historically observed in the same region, with at most
    # one case per region, so the figure should use the same filtered source.
    if REGION_OBSERVED_CHOICE_TOP.exists():
        plot_df = pd.read_csv(REGION_OBSERVED_CHOICE_TOP).rename(
            columns={
                "year_start": "largest_advantage_year",
                "weather_regime": "largest_advantage_weather_regime",
                "best_crop": "largest_advantage_best_crop",
                "runner_up_crop": "largest_advantage_runner_up_crop",
                "advantage_over_runner_up_t_ha": "largest_advantage_t_ha",
            }
        )
    else:
        plot_df = choice_story.copy()
    plot_df = plot_df.sort_values("largest_advantage_t_ha", ascending=False).head(10).iloc[::-1].copy()
    plot_df["label"] = (
        plot_df["country"].map(COUNTRY_SHORT)
        + " "
        + plot_df["region"].map(lambda r: REGION_SHORT.get(r, r))
        + " "
        + plot_df["largest_advantage_year"].astype(int).astype(str)
        + "\n"
        + plot_df["largest_advantage_best_crop"]
        + " vs "
        + plot_df["largest_advantage_runner_up_crop"]
    )
    colors = [CROP_COLORS.get(crop, "#999999") for crop in plot_df["largest_advantage_best_crop"]]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["largest_advantage_t_ha"], color=colors, alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Best-crop advantage over runner-up (t/ha)")
    ax.set_title("Largest weather-conditioned crop-choice advantages")
    for idx, row in enumerate(plot_df.itertuples(index=False)):
        ax.text(
            row.largest_advantage_t_ha + 0.03,
            idx,
            f"{row.largest_advantage_t_ha:.3f} t/ha; {row.largest_advantage_weather_regime}",
            va="center",
            fontsize=8.2,
        )
    ax.set_xlim(0, plot_df["largest_advantage_t_ha"].max() + 0.9)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "fig21_weather_condition_crop_choice_advantage")


def write_story_markdown(change: pd.DataFrame, summary: pd.DataFrame, choice_story: pd.DataFrame) -> None:
    def md_table(df: pd.DataFrame, columns: list[str], limit: int = 8) -> list[str]:
        rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for _, row in df.head(limit).iterrows():
            vals = []
            for col in columns:
                value = row[col]
                if isinstance(value, float):
                    vals.append(f"{value:.3f}")
                else:
                    vals.append(str(value))
            rows.append("| " + " | ".join(vals) + " |")
        return rows

    top_gain = change.sort_values("delta_observed_yield_t_ha", ascending=False)
    top_loss = change.sort_values("delta_observed_yield_t_ha", ascending=True)
    top_stories = summary.sort_values("predicted_yield_gap_high_minus_low_t_ha", ascending=False)

    lines = [
        "# Yield Story Analysis",
        "",
        "This layer answers more detailed practical questions:",
        "",
        "- Which crop-region combinations increased or decreased?",
        "- Where are high-yield years most different from low-yield years?",
        "- Which weather signals help explain the difference?",
        "- In which region-year does crop choice matter most?",
        "",
        "## Top observed yield gains",
        "",
    ]
    lines.extend(
        md_table(
            top_gain,
            ["country", "region", "crop", "delta_observed_yield_t_ha", "observed_change_status"],
        )
    )
    lines.extend(["", "## Top observed yield losses", ""])
    lines.extend(
        md_table(
            top_loss,
            ["country", "region", "crop", "delta_observed_yield_t_ha", "observed_change_status"],
        )
    )
    lines.extend(["", "## Strongest high-vs-low yield stories", ""])
    lines.extend(
        md_table(
            top_stories,
            ["country", "region", "crop", "predicted_yield_gap_high_minus_low_t_ha", "top_weather_driver_story"],
            limit=10,
        )
    )
    lines.extend(["", "## Largest weather-conditioned crop-choice advantages", ""])
    lines.extend(
        md_table(
            choice_story,
            [
                "country",
                "region",
                "largest_advantage_year",
                "largest_advantage_weather_regime",
                "largest_advantage_best_crop",
                "largest_advantage_runner_up_crop",
                "largest_advantage_t_ha",
            ],
            limit=10,
        )
    )
    lines.extend(
        [
            "",
            "## Paper wording",
            "",
            "Use wording such as:",
            "",
            "```text",
            "In high-predicted-yield years for this crop-region pair, the model tends to see more/less of these weather indicators.",
            "```",
            "",
            "Avoid causal wording unless a causal design is added.",
        ]
    )
    (STORY_OUT / "YIELD_STORY_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_index() -> None:
    lines = [
        "# Yield Story Figure Index",
        "",
        "| Figure | Meaning |",
        "|---|---|",
        "| `fig17_observed_yield_change_region_crop.png` | Which crop-region yields increased or decreased from early to recent years. |",
        "| `fig18_top_yield_gain_loss_region_crop.png` | Largest crop-region yield gains and losses. |",
        "| `fig19_high_vs_low_yield_weather_driver_matrix.png` | Which weather signals distinguish high-yield years from low-yield years. |",
        "| `fig20_region_crop_yield_story_cards.png` | Example region-crop stories with observed change and high-yield drivers. |",
        "| `fig21_weather_condition_crop_choice_advantage.png` | Where crop choice has the largest predicted advantage under a specific weather year. |",
        "",
        "All figures are model-supported descriptive analyses. They should be described as associations and planning signals, not causal proof.",
    ]
    (FIG_OUT / "YIELD_STORY_FIGURE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    STORY_OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    df = load_data()
    change = build_region_crop_change(df)
    summary, detail = build_high_low_driver_summary(df)
    matrix = build_country_crop_driver_matrix(detail)
    choice_story = build_weather_choice_gain_stories()
    figure_17_observed_yield_change_heatmap(change)
    figure_18_top_gain_loss_bars(change)
    figure_19_driver_matrix(matrix)
    figure_20_story_cards(summary, change)
    figure_21_weather_choice_advantage(choice_story)
    write_story_markdown(change, summary, choice_story)
    write_index()
    print(f"Saved yield story tables to: {STORY_OUT}")
    print(f"Saved yield story figures to: {FIG_OUT}")


if __name__ == "__main__":
    main()
