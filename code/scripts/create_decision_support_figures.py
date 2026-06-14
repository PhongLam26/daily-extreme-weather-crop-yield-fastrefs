from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.collections import PatchCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Polygon
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
OUT_DIR = PROJECT_ROOT / "figures" / "decision_support"
MODEL_OUT = PROJECT_ROOT / "outputs" / "decision_support_models"
METHOD_OUT = PROJECT_ROOT / "outputs" / "method_reproducibility"
MODEL_FRAMES = PROJECT_ROOT / "data" / "processed" / "model_frames"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "extreme_weather_features_aus_us_region_year.csv"
GEO_CACHE = REPO_ROOT / "output" / "paper_yield_study" / "08_geo_maps" / "geojson_cache"

COMBINED_FRAME = MODEL_FRAMES / "aus_us_overlap_crops_with_extreme_weather_1989_2021.csv"
RESULTS = MODEL_OUT / "model_results.csv"
PREDICTIONS = MODEL_OUT / "predictions_all_best_models.csv"
IMPORTANCE_COMBINED = MODEL_OUT / "feature_importance_aus_us_overlap.csv"
IMPORTANCE_WHEAT = MODEL_OUT / "feature_importance_aus_us_wheat.csv"
RISK = MODEL_OUT / "risk_threshold_summary.csv"
CROP_RANK = MODEL_OUT / "scenario_crop_ranking_by_region_aus_us_overlap.csv"
CROP_RANK_YEAR = MODEL_OUT / "scenario_crop_ranking_by_region_year_aus_us_overlap.csv"
ADVANTAGE = MODEL_OUT / "country_crop_advantage_matrix.csv"
LOW_CASES = MODEL_OUT / "low_yield_explanation_cases.csv"
WEATHER_CHOICE = MODEL_OUT / "weather_condition_crop_choice_by_region_year.csv"
WEATHER_REGIME_SUMMARY = MODEL_OUT / "weather_regime_crop_choice_summary.csv"

COUNTRY_COLORS = {"Australia": "#2A9D8F", "United States": "#E76F51"}
CROP_COLORS = {
    "Wheat": "#D8A31A",
    "Barley": "#5E9F6E",
    "Canola": "#E9C46A",
    "Oats": "#8AB6D6",
}
CROP_MARKERS = {"Wheat": "^", "Barley": "s", "Canola": "P", "Oats": "D"}
CROP_CODES = {"Wheat": "Wh", "Barley": "Ba", "Canola": "Ca", "Oats": "Oa"}
WEATHER_REGIME_ORDER = [
    "Hot-dry stress",
    "Dry-spell stress",
    "Wet/storm risk",
    "Cold/frost stress",
    "Moderate season",
]
WEATHER_REGIME_COLORS = {
    "Hot-dry stress": "#D55E00",
    "Dry-spell stress": "#B07D2B",
    "Wet/storm risk": "#0072B2",
    "Cold/frost stress": "#56B4E9",
    "Moderate season": "#4E9F3D",
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

MANUAL_LABEL_POSITIONS = {
    "Western Australia": (122.7, -25.4),
    "Queensland": (145.2, -22.4),
    "New South Wales": (147.0, -32.0),
    "South Australia": (136.1, -30.5),
    "Victoria": (144.6, -36.8),
    "Tasmania": (147.0, -42.1),
    "Colorado": (-105.5, 39.1),
    "Illinois": (-89.1, 40.2),
    "Iowa": (-93.4, 42.0),
    "Kansas": (-98.4, 38.5),
    "Minnesota": (-94.3, 46.0),
    "Montana": (-109.4, 47.0),
    "Nebraska": (-99.7, 41.5),
    "North Dakota": (-100.3, 47.6),
    "Oklahoma": (-97.5, 35.4),
    "South Dakota": (-100.2, 44.4),
    "Texas": (-99.0, 31.1),
    "Washington": (-120.7, 47.3),
}

CONTIGUOUS_EXCLUDE = {"Alaska", "Hawaii", "Puerto Rico"}


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight", pad_inches=0.08, dpi=320)
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def clean_label(value: str, max_len: int = 34) -> str:
    text = str(value)
    text = text.replace("_", " ")
    text = text.replace("aus us", "AUS-US")
    return "\n".join(textwrap.wrap(text, max_len))


def feature_short_name(value: str) -> str:
    replacements = {
        "importance_rmse_increase_mean": "RMSE increase",
        "rainfall_concentration_index": "rain concentration",
        "max_consecutive_heat_days_35": "max heat spell 35C",
        "max_consecutive_heat_days_30": "max heat spell 30C",
        "heatwave_events_3d_35": "heatwave events 35C",
        "hot_dry_days_30_1mm": "hot-dry days",
        "rain_last_14d_before_harvest": "rain before harvest",
        "heavy_rain_days_last_14d_before_harvest": "heavy rain before harvest",
        "max_3day_rain_last_21d_before_harvest": "max 3-day pre-harvest rain",
        "growing_degree_days_base5": "growing degree days",
    }
    text = replacements.get(value, value)
    text = text.replace("_y", "").replace("_x", "")
    text = text.replace("_", " ")
    text = text.replace("soil ", "soil ")
    return text


def load_csv(path: Path) -> pd.DataFrame:
    require_file(path)
    return pd.read_csv(path, low_memory=False)


def best_models(results: pd.DataFrame) -> pd.DataFrame:
    soil = results[results["feature_set"].eq("extreme_weather_soil")].copy()
    idx = soil.groupby("dataset")["test_rmse"].idxmin()
    return soil.loc[idx].sort_values("dataset").reset_index(drop=True)


def figure_01_dataset_overview() -> None:
    frame = load_csv(COMBINED_FRAME)
    features = load_csv(FEATURES_PATH)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.2))
    fig.suptitle("Dataset scope for the daily extreme-weather yield study", fontsize=15, weight="bold", y=0.98)

    counts = frame.groupby(["country", "crop"]).size().unstack(0).fillna(0)
    counts = counts.reindex(["Wheat", "Barley", "Canola", "Oats"])
    x = np.arange(len(counts.index))
    width = 0.36
    for i, country in enumerate(["Australia", "United States"]):
        axes[0, 0].bar(
            x + (i - 0.5) * width,
            counts.get(country, pd.Series(0, index=counts.index)),
            width,
            label=country,
            color=COUNTRY_COLORS[country],
        )
    axes[0, 0].set_title("Annual yield records by crop")
    axes[0, 0].set_ylabel("Records")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(counts.index)
    axes[0, 0].legend(frameon=False)

    region_counts = frame.groupby("country")["region"].nunique()
    axes[0, 1].bar(region_counts.index, region_counts.values, color=[COUNTRY_COLORS[c] for c in region_counts.index])
    axes[0, 1].set_title("Spatial units included")
    axes[0, 1].set_ylabel("Regions or states")
    for idx, value in enumerate(region_counts.values):
        axes[0, 1].text(idx, value + 0.25, str(int(value)), ha="center", va="bottom", weight="bold")

    yield_data = []
    yield_labels = []
    colors = []
    for country in ["Australia", "United States"]:
        for crop in ["Wheat", "Barley", "Canola", "Oats"]:
            part = frame[(frame["country"].eq(country)) & (frame["crop"].eq(crop))]["yield_t_ha"].dropna()
            if not part.empty:
                yield_data.append(part.values)
                yield_labels.append(f"{country[:3]}\n{crop}")
                colors.append(CROP_COLORS[crop])
    box = axes[1, 0].boxplot(yield_data, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    axes[1, 0].set_title("Observed yield distributions")
    axes[1, 0].set_ylabel("Yield (t/ha)")
    axes[1, 0].set_xticklabels(yield_labels, rotation=0)

    weather_counts = features.groupby("country").size()
    axes[1, 1].bar(weather_counts.index, weather_counts.values, color=[COUNTRY_COLORS[c] for c in weather_counts.index])
    axes[1, 1].set_title("Region-year extreme-weather feature rows")
    axes[1, 1].set_ylabel("Region-year rows")
    for idx, value in enumerate(weather_counts.values):
        axes[1, 1].text(idx, value + 5, str(int(value)), ha="center", va="bottom", weight="bold")

    for ax in axes.ravel():
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, "fig01_dataset_scope")


def figure_02_feature_design_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.7))
    ax.axis("off")
    fig.suptitle("From daily weather to practical crop-yield signals", fontsize=16, weight="bold", y=0.92)

    boxes = [
        ("Daily weather records", "rainfall, maximum and\nminimum temperature,\nradiation by day", (0.18, 0.74), 0.235, "#F4F1DE"),
        ("Extreme-weather indicators", "heat exposure, dry spells,\nheavy rain, frost, and\npre-harvest rain", (0.50, 0.74), 0.255, "#E9F5DB"),
        ("Annual model frame", "country-region-crop-year\n+ soil context\n+ observed yield", (0.82, 0.74), 0.235, "#DDEAF6"),
        ("Yield prediction", "predicted yield\nin tonnes per hectare", (0.82, 0.35), 0.285, "#FFF1E6"),
        ("Decision-support signals", "why yield changes,\nwhich crop fits which region,\nand where risk is higher", (0.38, 0.35), 0.325, "#E6F2EE"),
    ]
    box_height = 0.22

    for title, body, (x, y), box_width, color in boxes:
        half_width = box_width / 2
        patch = FancyBboxPatch(
            (x - half_width, y - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            facecolor=color,
            edgecolor="#333333",
            linewidth=1.1,
            clip_on=False,
        )
        ax.add_patch(patch)
        ax.text(x, y + 0.055, title, ha="center", va="center", fontsize=11.8, weight="bold", color="#222222")
        ax.text(x, y - 0.055, body, ha="center", va="center", fontsize=10.2, color="#333333", linespacing=1.12)

    arrows = [
        ((0.18 + 0.1175 + 0.016, 0.74), (0.50 - 0.1275 - 0.016, 0.74)),
        ((0.50 + 0.1275 + 0.016, 0.74), (0.82 - 0.1175 - 0.016, 0.74)),
        ((0.82, 0.74 - box_height / 2 - 0.018), (0.82, 0.35 + box_height / 2 + 0.018)),
        ((0.82 - 0.1425 - 0.02, 0.35), (0.38 + 0.1625 + 0.02, 0.35)),
    ]
    for start, end in arrows:
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.6,
            color="#555555",
            clip_on=False,
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(arrow)

    ax.text(
        0.5,
        0.08,
        "Daily rows are summarized before modelling, so the model unit remains interpretable: country-region-crop-year.",
        ha="center",
        fontsize=10.8,
        color="#444444",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.86, bottom=0.12)
    save_figure(fig, "fig02_daily_extreme_feature_design")


def figure_03_model_performance() -> None:
    selected = load_csv(METHOD_OUT / "table02_validation_selected_model_performance.csv")
    ablation = load_csv(METHOD_OUT / "table03_feature_group_ablation_aus_us_overlap.csv")

    datasets = ["aus_overlap", "aus_us_overlap", "aus_us_wheat", "us_overlap"]
    dataset_labels = {
        "aus_overlap": "AUS overlap",
        "aus_us_overlap": "AUS-US overlap",
        "aus_us_wheat": "AUS-US wheat",
        "us_overlap": "US overlap",
    }
    selected = selected.set_index("dataset").loc[datasets].reset_index()

    ablation_order = [
        "historical crop-region mean",
        "previous-year yield",
        "crop + region + year only",
        "rainfall only",
        "temperature only",
        "radiation only",
        "extreme events only",
        "weather all",
        "weather + soil",
    ]
    ablation = ablation.set_index("feature_set").loc[ablation_order].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [1.0, 1.25]})
    fig.suptitle("Prediction performance under forward-time evaluation", fontsize=17, weight="bold", y=0.98)

    x = np.arange(len(selected))
    width = 0.36
    axes[0].bar(x - width / 2, selected["validation_r2"], width, label="Validation R2", color="#7EA8DA")
    axes[0].bar(x + width / 2, selected["test_r2"], width, label="Test R2", color="#2A9D8F")
    for xpos, val in zip(x - width / 2, selected["validation_r2"]):
        axes[0].text(xpos, val + 0.015, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    for xpos, val in zip(x + width / 2, selected["test_r2"]):
        axes[0].text(xpos, val + 0.015, f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_title("Pre-specified task performance", fontsize=13)
    axes[0].set_ylabel("R2")
    axes[0].set_ylim(0, 0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([dataset_labels[d] for d in selected["dataset"]], rotation=20, ha="right")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].grid(axis="y", color="#D0D0D0", linewidth=0.6, alpha=0.5)

    colors = ["#BDBDBD", "#9E9E9E", "#8D99A6", "#56B1E1", "#E9785E", "#DDAA2A", "#9684C3", "#2A9D8F", "#315866"]
    y = np.arange(len(ablation))
    axes[1].barh(y, ablation["test_r2"], color=colors, alpha=0.95)
    for ypos, val in zip(y, ablation["test_r2"]):
        axes[1].text(val + 0.01, ypos, f"{val:.2f}", va="center", fontsize=9)
    axes[1].set_title("Feature-group ablation on AUS-US overlap crops", fontsize=13)
    axes[1].set_xlabel("Test R2")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(
        [s.replace(" + ", " +\n") if s == "crop + region + year only" else s for s in ablation["feature_set"]]
    )
    axes[1].set_xlim(0, 0.85)
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", color="#D0D0D0", linewidth=0.6, alpha=0.5)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, "fig03_model_performance")


def figure_04_actual_vs_predicted() -> None:
    pred = load_csv(PREDICTIONS)
    pred = pred[pred["split"].eq("test")].copy()
    datasets = ["aus_overlap", "aus_us_overlap", "aus_us_wheat", "us_overlap"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=False, sharey=False)
    fig.suptitle("Actual vs predicted yield on the 2016-2021 test period", fontsize=15, weight="bold", y=0.98)

    for ax, dataset in zip(axes.ravel(), datasets):
        part = pred[pred["dataset"].eq(dataset)].copy()
        for country, cpart in part.groupby("country"):
            ax.scatter(
                cpart["yield_t_ha"],
                cpart["predicted_yield_t_ha"],
                s=28,
                alpha=0.72,
                label=country,
                color=COUNTRY_COLORS.get(country, "#555555"),
                edgecolor="white",
                linewidth=0.4,
            )
        low = min(part["yield_t_ha"].min(), part["predicted_yield_t_ha"].min())
        high = max(part["yield_t_ha"].max(), part["predicted_yield_t_ha"].max())
        ax.plot([low, high], [low, high], color="#333333", linewidth=1, linestyle="--")
        ax.set_title(clean_label(dataset, 24))
        ax.set_xlabel("Observed yield (t/ha)")
        ax.set_ylabel("Predicted yield (t/ha)")
        ax.grid(color="#E3E3E3", linewidth=0.6)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, "fig04_actual_vs_predicted_test")


def plot_importance(ax: plt.Axes, path: Path, title: str, top_n: int = 14) -> None:
    df = load_csv(path).head(top_n).iloc[::-1]
    colors = ["#2A9D8F" if "soil" in f else "#E76F51" if "heat" in f or "dry" in f else "#457B9D" for f in df["feature"]]
    ax.barh(np.arange(len(df)), df["importance_rmse_increase_mean"], color=colors, alpha=0.88)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels([feature_short_name(f) for f in df["feature"]])
    ax.set_xlabel("Permutation importance\n(RMSE increase)")
    ax.set_title(title)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def figure_05_feature_importance() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    plot_importance(axes[0], IMPORTANCE_COMBINED, "AUS-US overlap crops")
    plot_importance(axes[1], IMPORTANCE_WHEAT, "AUS-US Wheat-only")
    fig.suptitle("Top explanation signals from the selected models", fontsize=15, weight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, "fig05_top_feature_importance")


def figure_06_risk_thresholds() -> None:
    risk = load_csv(RISK)
    sub = risk[risk["dataset"].eq("aus_us_overlap")].copy()
    sub["abs_delta"] = sub["high_minus_low_yield_t_ha"].abs()
    sub = sub.sort_values("abs_delta", ascending=False).head(18).iloc[::-1]
    labels = [
        f"{row.country} {row.crop}\n{feature_short_name(row.feature)}"
        for row in sub.itertuples(index=False)
    ]
    values = sub["high_minus_low_yield_t_ha"].values
    colors = ["#2A9D8F" if value >= 0 else "#E76F51" for value in values]

    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    ax.barh(np.arange(len(values)), values, color=colors, alpha=0.88)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean yield difference: high-feature group minus low-feature group (t/ha)")
    ax.set_title("Risk-threshold associations in the combined overlap-crop dataset")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.01,
        0.01,
        "Positive bars mean years with high feature values had higher observed mean yield; negative bars mean lower observed mean yield. These are associations, not causal effects.",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    save_figure(fig, "fig06_risk_threshold_associations")


def feature_name(feature: dict[str, Any]) -> str:
    props = feature.get("properties", {})
    for key in ["name", "NAME", "STATE_NAME", "STATE", "State", "NAME_1", "state"]:
        value = props.get(key)
        if value:
            return str(value).strip()
    return "Unknown"


def iter_exterior_rings(geometry: dict[str, Any]):
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "Polygon":
        if coords:
            yield coords[0]
    elif gtype == "MultiPolygon":
        for polygon in coords:
            if polygon:
                yield polygon[0]


def load_geojson(filename: str) -> dict[str, Any]:
    path = GEO_CACHE / filename
    require_file(path)
    return json.loads(path.read_text(encoding="utf-8"))


def add_crop_map(
    ax: plt.Axes,
    geojson: dict[str, Any],
    selected: pd.DataFrame,
    title: str,
    exclude: set[str] | None = None,
    pad_scale: float = 1.0,
) -> None:
    exclude = exclude or set()
    crop_by_region = dict(zip(selected["region"], selected["crop"]))
    yield_by_region = dict(zip(selected["region"], selected["mean_predicted_yield_t_ha"]))
    selected_names = set(crop_by_region)
    base_patches: list[Polygon] = []
    selected_patches: list[Polygon] = []
    selected_colors: list[str] = []
    label_points: dict[str, tuple[float, float]] = {}
    bounds = [180.0, 90.0, -180.0, -90.0]

    for feature in geojson.get("features", []):
        name = feature_name(feature)
        if name in exclude:
            continue
        rings = list(iter_exterior_rings(feature.get("geometry", {})))
        xs_all: list[float] = []
        ys_all: list[float] = []
        for ring in rings:
            pts = [(float(x), float(y)) for x, y, *rest in ring]
            if len(pts) < 3:
                continue
            patch = Polygon(pts, closed=True)
            if name in selected_names:
                selected_patches.append(patch)
                selected_colors.append(CROP_COLORS.get(crop_by_region[name], "#CCCCCC"))
            else:
                base_patches.append(patch)
            xs, ys = zip(*pts)
            xs_all.extend(xs)
            ys_all.extend(ys)
            bounds[0] = min(bounds[0], min(xs))
            bounds[1] = min(bounds[1], min(ys))
            bounds[2] = max(bounds[2], max(xs))
            bounds[3] = max(bounds[3], max(ys))
        if name in selected_names and xs_all and ys_all:
            label_points[name] = MANUAL_LABEL_POSITIONS.get(name, (sum(xs_all) / len(xs_all), sum(ys_all) / len(ys_all)))

    ax.add_collection(PatchCollection(base_patches, facecolor="#E6E8EB", edgecolor="white", linewidth=0.6, zorder=1))
    ax.add_collection(
        PatchCollection(
            selected_patches,
            facecolor=selected_colors,
            edgecolor="#2B2B2B",
            linewidth=0.9,
            alpha=0.78,
            zorder=2,
        )
    )

    for name, (x, y) in label_points.items():
        crop = crop_by_region[name]
        marker = CROP_MARKERS.get(crop, "o")
        color = CROP_COLORS.get(crop, "#AAAAAA")
        ax.scatter([x], [y], s=580, marker=marker, color=color, edgecolor="#111111", linewidth=1.4, zorder=4)
        ax.text(
            x,
            y,
            CROP_CODES.get(crop, crop[:2]),
            ha="center",
            va="center",
            fontsize=7.5,
            weight="bold",
            color="#111111",
            zorder=5,
        )
        ax.text(
            x,
            y - (1.8 if title.startswith("Australia") else 1.1),
            f"{REGION_SHORT.get(name, name)}\n{yield_by_region[name]:.1f}",
            ha="center",
            va="top",
            fontsize=6.9,
            color="#111111",
            zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.5),
        )

    pad_x = (bounds[2] - bounds[0]) * 0.05 * pad_scale
    pad_y = (bounds[3] - bounds[1]) * 0.05 * pad_scale
    ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
    ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.axis("off")


def figure_07_crop_suitability_map() -> None:
    ranking = load_csv(CROP_RANK)
    best = ranking[ranking["rank_within_region"].eq(1)].copy()
    aus = best[best["country"].eq("Australia")]
    us = best[best["country"].eq("United States")]
    aus_geo = load_geojson("australian-states.json")
    us_geo = load_geojson("us-states.json")

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7), gridspec_kw={"width_ratios": [0.82, 1.18], "wspace": 0.03})
    add_crop_map(axes[0], aus_geo, aus, "Australia: best predicted crop by region", pad_scale=1.4)
    add_crop_map(axes[1], us_geo, us, "United States: best predicted crop by state", exclude=CONTIGUOUS_EXCLUDE, pad_scale=0.7)

    handles = []
    labels = []
    for crop, color in CROP_COLORS.items():
        handle = axes[1].scatter([], [], s=120, marker=CROP_MARKERS[crop], color=color, edgecolor="#111111", linewidth=1.0)
        handles.append(handle)
        labels.append(crop)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
        title="Best predicted crop marker",
    )
    fig.suptitle("Model-based crop suitability map with crop markers", fontsize=15, weight="bold", y=0.96)
    fig.text(
        0.5,
        0.105,
        "Marker text shows crop code; label below marker shows region/state abbreviation and mean predicted yield (t/ha).",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.02, right=0.99, top=0.88, bottom=0.24)
    save_figure(fig, "fig07_crop_suitability_map")


def draw_heatmap(ax: plt.Axes, data: pd.DataFrame, title: str) -> None:
    crops = ["Wheat", "Barley", "Canola", "Oats"]
    regions = list(data["region"].drop_duplicates())
    pivot = data.pivot(index="region", columns="crop", values="mean_predicted_yield_t_ha").reindex(regions)[crops]
    im = ax.imshow(pivot.values, cmap="YlGn", aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(crops)))
    ax.set_xticklabels(crops)
    ax.set_yticks(np.arange(len(regions)))
    ax.set_yticklabels([REGION_SHORT.get(r, r) for r in regions])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7.5, color="#111111")
    return im


def figure_08_crop_region_heatmap() -> None:
    ranking = load_csv(CROP_RANK)
    aus = ranking[ranking["country"].eq("Australia")].sort_values(["region", "crop"])
    us = ranking[ranking["country"].eq("United States")].sort_values(["region", "crop"])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2), gridspec_kw={"width_ratios": [0.82, 1.18]})
    im1 = draw_heatmap(axes[0], aus, "Australia predicted yield by crop")
    im2 = draw_heatmap(axes[1], us, "United States predicted yield by crop")
    fig.subplots_adjust(left=0.06, right=0.88, top=0.87, bottom=0.12, wspace=0.12)
    cax = fig.add_axes([0.91, 0.18, 0.018, 0.62])
    cbar = fig.colorbar(im2, cax=cax)
    cbar.set_label("Mean predicted yield (t/ha)")
    fig.suptitle("Crop-region suitability heatmap", fontsize=15, weight="bold", y=0.98)
    save_figure(fig, "fig08_crop_region_suitability_heatmap")


def figure_09_country_crop_advantage() -> None:
    adv = load_csv(ADVANTAGE)
    adv = adv[adv["dataset"].eq("aus_us_overlap")].copy()
    crops = ["Wheat", "Barley", "Canola", "Oats"]
    adv = adv.set_index("crop").reindex(crops).reset_index()
    x = np.arange(len(adv))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.1, 0.9]})
    axes[0].bar(x - width / 2, adv["Australia"], width, label="Australia", color=COUNTRY_COLORS["Australia"])
    axes[0].bar(x + width / 2, adv["United States"], width, label="United States", color=COUNTRY_COLORS["United States"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(crops)
    axes[0].set_ylabel("Mean predicted yield (t/ha)")
    axes[0].set_title("Predicted yield by country and crop")
    axes[0].legend(frameon=False)

    delta = adv["us_minus_australia_predicted_yield_t_ha"]
    axes[1].barh(x, delta, color=["#E76F51" if v > 0 else "#2A9D8F" for v in delta], alpha=0.88)
    axes[1].axvline(0, color="#333333", linewidth=1)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(crops)
    axes[1].set_xlabel("U.S. minus Australia (t/ha)")
    axes[1].set_title("Predicted advantage signal")
    for ax in axes:
        ax.grid(axis="y" if ax is axes[0] else "x", color="#DDDDDD", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Country-crop predicted yield comparison", fontsize=15, weight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, "fig09_country_crop_advantage")


def figure_10_low_yield_fingerprint() -> None:
    frame = load_csv(COMBINED_FRAME)
    pred = load_csv(PREDICTIONS)
    pred = pred[(pred["dataset"].eq("aus_us_overlap")) & (pred["split"].eq("test"))].copy()
    keys = ["country", "region", "crop", "year_start"]
    cols = [
        "heat_days_35",
        "max_dry_spell_1mm",
        "hot_dry_days_30_1mm",
        "max_3day_rain",
        "rain_last_14d_before_harvest",
        "frost_days_0",
        "season_rain_mean",
        "season_radiation_mean",
    ]
    data = pred[keys + ["predicted_yield_t_ha"]].merge(frame[keys + cols], on=keys, how="left")
    low_cut = data["predicted_yield_t_ha"].quantile(0.25)
    high_cut = data["predicted_yield_t_ha"].quantile(0.75)
    low = data[data["predicted_yield_t_ha"] <= low_cut]
    high = data[data["predicted_yield_t_ha"] >= high_cut]
    rows = []
    for col in cols:
        std = data[col].std()
        if std == 0 or np.isnan(std):
            continue
        rows.append(
            {
                "feature": col,
                "low_minus_high_std": (low[col].mean() - high[col].mean()) / std,
                "low_mean": low[col].mean(),
                "high_mean": high[col].mean(),
            }
        )
    out = pd.DataFrame(rows).sort_values("low_minus_high_std")
    colors = ["#2A9D8F" if v < 0 else "#E76F51" for v in out["low_minus_high_std"]]
    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    ax.barh(np.arange(len(out)), out["low_minus_high_std"], color=colors, alpha=0.88)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(np.arange(len(out)))
    ax.set_yticklabels([feature_short_name(f) for f in out["feature"]])
    ax.set_xlabel("Low-yield group minus high-yield group, in standard deviations")
    ax.set_title("Weather fingerprint of low predicted yield years")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "fig10_low_yield_weather_fingerprint")


def figure_11_stress_distributions() -> None:
    frame = load_csv(COMBINED_FRAME)
    features = [
        "heat_days_35",
        "max_dry_spell_1mm",
        "max_3day_rain",
        "rain_last_14d_before_harvest",
        "frost_days_0",
    ]
    fig, axes = plt.subplots(1, len(features), figsize=(16, 4.8))
    fig.suptitle("Extreme-weather stress distributions by country", fontsize=15, weight="bold", y=0.98)
    for ax, feature in zip(axes, features):
        data = [frame[frame["country"].eq(country)][feature].dropna().values for country in ["Australia", "United States"]]
        box = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, country in zip(box["boxes"], ["Australia", "United States"]):
            patch.set_facecolor(COUNTRY_COLORS[country])
            patch.set_alpha(0.72)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["AUS", "US"])
        ax.set_title(feature_short_name(feature), fontsize=10)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, "fig11_weather_stress_distributions")


def figure_12_best_crop_share() -> None:
    ranking = load_csv(CROP_RANK)
    best = ranking[ranking["rank_within_region"].eq(1)].copy()
    best["region_label"] = best["country"].map({"Australia": "AUS", "United States": "US"}) + " " + best["region"].map(
        lambda x: REGION_SHORT.get(x, x)
    )
    best = best.sort_values(["country", "mean_predicted_yield_t_ha"])
    colors = [CROP_COLORS.get(crop, "#999999") for crop in best["crop"]]

    fig, ax = plt.subplots(figsize=(11.2, 7.4))
    y = np.arange(len(best))
    ax.barh(y, best["best_rank_share"], color=colors, alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(best["region_label"])
    ax.set_xlabel("Share of years where selected crop ranks first")
    ax.set_xlim(0, 1.05)
    ax.set_title("Stability of the best predicted crop by region")
    for idx, row in enumerate(best.itertuples(index=False)):
        ax.text(
            row.best_rank_share + 0.02,
            idx,
            f"{row.crop} ({row.mean_predicted_yield_t_ha:.1f} t/ha)",
            va="center",
            fontsize=8.3,
        )
    handles = [Patch(facecolor=color, label=crop, alpha=0.88) for crop, color in CROP_COLORS.items()]
    ax.legend(handles=handles, frameon=False, loc="lower right", title="Best crop")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "fig12_best_crop_share_by_region")


def add_weather_regime(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    threshold_features = [
        "hot_dry_days_30_1mm",
        "heat_days_35",
        "max_dry_spell_1mm",
        "max_3day_rain",
        "rain_last_14d_before_harvest",
        "frost_days_0",
        "season_rain_mean",
    ]
    thresholds: dict[str, dict[str, float]] = {}
    for country, part in features.groupby("country"):
        thresholds[str(country)] = {}
        for feature in threshold_features:
            thresholds[str(country)][f"{feature}_q25"] = float(part[feature].quantile(0.25))
            thresholds[str(country)][f"{feature}_q75"] = float(part[feature].quantile(0.75))

    def classify(row: pd.Series) -> str:
        country_thresholds = thresholds[str(row["country"])]
        hot = row["heat_days_35"] >= country_thresholds["heat_days_35_q75"]
        hot_dry = row["hot_dry_days_30_1mm"] >= country_thresholds["hot_dry_days_30_1mm_q75"]
        dry_spell = row["max_dry_spell_1mm"] >= country_thresholds["max_dry_spell_1mm_q75"]
        low_rain = row["season_rain_mean"] <= country_thresholds["season_rain_mean_q25"]
        wet = (
            row["max_3day_rain"] >= country_thresholds["max_3day_rain_q75"]
            or row["rain_last_14d_before_harvest"] >= country_thresholds["rain_last_14d_before_harvest_q75"]
            or row.get("storm_before_harvest_flag", 0) == 1
        )
        frost = row["frost_days_0"] >= country_thresholds["frost_days_0_q75"]
        if hot and hot_dry:
            return "Hot-dry stress"
        if dry_spell and low_rain:
            return "Dry-spell stress"
        if wet:
            return "Wet/storm risk"
        if frost:
            return "Cold/frost stress"
        return "Moderate season"

    features["weather_regime"] = features.apply(classify, axis=1)
    return features


def build_weather_condition_choice_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranking = load_csv(CROP_RANK_YEAR)
    features = add_weather_regime(load_csv(FEATURES_PATH))
    feature_cols = [
        "country",
        "region",
        "year_start",
        "weather_regime",
        "season_rain_sum",
        "season_rain_mean",
        "heat_days_35",
        "hot_dry_days_30_1mm",
        "max_dry_spell_1mm",
        "max_3day_rain",
        "rain_last_14d_before_harvest",
        "storm_before_harvest_flag",
        "frost_days_0",
    ]
    weather = features[feature_cols].copy()
    ranking_with_weather = ranking.merge(
        weather,
        on=["country", "region", "year_start"],
        how="left",
        validate="many_to_one",
    )

    ordered = ranking_with_weather.sort_values(["country", "region", "year_start", "rank_within_region_year"])
    best = ordered[ordered["rank_within_region_year"].eq(1)].copy()
    second = ordered[ordered["rank_within_region_year"].eq(2)].copy()

    choice = best[
        [
            "country",
            "region",
            "year_start",
            "weather_regime",
            "crop",
            "predicted_yield_t_ha",
            "season_rain_sum",
            "season_rain_mean",
            "heat_days_35",
            "hot_dry_days_30_1mm",
            "max_dry_spell_1mm",
            "max_3day_rain",
            "rain_last_14d_before_harvest",
            "storm_before_harvest_flag",
            "frost_days_0",
        ]
    ].rename(columns={"crop": "best_crop", "predicted_yield_t_ha": "best_predicted_yield_t_ha"})
    second_small = second[["country", "region", "year_start", "crop", "predicted_yield_t_ha"]].rename(
        columns={"crop": "runner_up_crop", "predicted_yield_t_ha": "runner_up_predicted_yield_t_ha"}
    )
    choice = choice.merge(second_small, on=["country", "region", "year_start"], how="left", validate="one_to_one")
    choice["best_crop_yield_advantage_t_ha"] = (
        choice["best_predicted_yield_t_ha"] - choice["runner_up_predicted_yield_t_ha"]
    )
    choice.to_csv(WEATHER_CHOICE, index=False)

    best_counts = (
        choice.groupby(["country", "weather_regime", "best_crop"], as_index=False)
        .size()
        .rename(columns={"best_crop": "crop", "size": "best_crop_count"})
    )
    total_counts = (
        choice.groupby(["country", "weather_regime"], as_index=False)
        .size()
        .rename(columns={"size": "region_year_rows"})
    )
    predicted = (
        ranking_with_weather.groupby(["country", "weather_regime", "crop"], as_index=False)
        .agg(
            mean_predicted_yield_t_ha=("predicted_yield_t_ha", "mean"),
            median_predicted_yield_t_ha=("predicted_yield_t_ha", "median"),
        )
    )
    summary = predicted.merge(best_counts, on=["country", "weather_regime", "crop"], how="left")
    summary = summary.merge(total_counts, on=["country", "weather_regime"], how="left")
    summary["best_crop_count"] = summary["best_crop_count"].fillna(0).astype(int)
    summary["best_crop_share"] = summary["best_crop_count"] / summary["region_year_rows"]
    summary["rank_by_regime_mean_yield"] = summary.groupby(["country", "weather_regime"])[
        "mean_predicted_yield_t_ha"
    ].rank(method="first", ascending=False)
    summary = summary.sort_values(["country", "weather_regime", "rank_by_regime_mean_yield"])
    summary.to_csv(WEATHER_REGIME_SUMMARY, index=False)
    return choice, ranking_with_weather, summary


def figure_13_weather_conditioned_winner_timeline(choice: pd.DataFrame) -> None:
    choice = choice.copy()
    choice["region_label"] = choice["country"].map({"Australia": "AUS", "United States": "US"}) + " " + choice[
        "region"
    ].map(lambda value: REGION_SHORT.get(value, value))
    region_order = (
        choice.groupby(["country", "region_label"])["best_predicted_yield_t_ha"]
        .mean()
        .reset_index()
        .sort_values(["country", "region_label"])
    )
    regions = region_order["region_label"].tolist()
    years = sorted(choice["year_start"].unique())
    crop_order = ["Wheat", "Barley", "Canola", "Oats"]
    crop_to_idx = {crop: idx for idx, crop in enumerate(crop_order)}
    matrix = np.full((len(regions), len(years)), np.nan)
    for _, row in choice.iterrows():
        i = regions.index(row["region_label"])
        j = years.index(row["year_start"])
        matrix[i, j] = crop_to_idx.get(row["best_crop"], np.nan)

    cmap = ListedColormap([CROP_COLORS[crop] for crop in crop_order])
    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=-0.5, vmax=len(crop_order) - 0.5)
    ax.set_title("Best predicted crop for each region-year weather condition")
    ax.set_xlabel("Year")
    ax.set_ylabel("Region/state")
    tick_years = [year for year in years if (year - min(years)) % 4 == 0 or year == max(years)]
    ax.set_xticks([years.index(year) for year in tick_years])
    ax.set_xticklabels(tick_years, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(regions)))
    ax.set_yticklabels(regions)
    handles = [Patch(facecolor=CROP_COLORS[crop], label=crop) for crop in crop_order]
    ax.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.set_xticks(np.arange(-0.5, len(years), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(regions), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.text(
        0.5,
        0.01,
        "Each cell asks: under that region-year soil and weather profile, which crop has the highest predicted t/ha?",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    save_figure(fig, "fig13_weather_conditioned_crop_winner_timeline")


def draw_regime_yield_heatmap(ax: plt.Axes, summary: pd.DataFrame, country: str) -> None:
    crops = ["Wheat", "Barley", "Canola", "Oats"]
    part = summary[summary["country"].eq(country)].copy()
    regimes = [regime for regime in WEATHER_REGIME_ORDER if regime in set(part["weather_regime"])]
    pivot = part.pivot(index="weather_regime", columns="crop", values="mean_predicted_yield_t_ha").reindex(regimes)[crops]
    im = ax.imshow(pivot.values, cmap="YlGn", aspect="auto")
    ax.set_title(country)
    ax.set_xticks(np.arange(len(crops)))
    ax.set_xticklabels(crops)
    ax.set_yticks(np.arange(len(regimes)))
    ax.set_yticklabels(regimes)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            if not np.isnan(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8.5)
    return im


def figure_14_weather_regime_crop_yield_matrix(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw={"width_ratios": [1, 1]})
    im1 = draw_regime_yield_heatmap(axes[0], summary, "Australia")
    im2 = draw_regime_yield_heatmap(axes[1], summary, "United States")
    fig.subplots_adjust(left=0.12, right=0.88, top=0.82, bottom=0.16, wspace=0.35)
    cax = fig.add_axes([0.91, 0.22, 0.018, 0.52])
    cbar = fig.colorbar(im2, cax=cax)
    cbar.set_label("Mean predicted yield (t/ha)")
    fig.suptitle("Which crop is predicted to yield more under each weather regime?", fontsize=15, weight="bold", y=0.96)
    fig.text(
        0.5,
        0.04,
        "Weather regimes are assigned by country-specific thresholds from daily extreme-weather indicators.",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    save_figure(fig, "fig14_weather_regime_crop_yield_matrix")


def figure_15_weather_regime_best_crop_share(summary: pd.DataFrame) -> None:
    countries = ["Australia", "United States"]
    crops = ["Wheat", "Barley", "Canola", "Oats"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharex=True)
    for ax, country in zip(axes, countries):
        part = summary[summary["country"].eq(country)].copy()
        regimes = [regime for regime in WEATHER_REGIME_ORDER if regime in set(part["weather_regime"])]
        left = np.zeros(len(regimes))
        for crop in crops:
            values = (
                part[part["crop"].eq(crop)]
                .set_index("weather_regime")
                .reindex(regimes)["best_crop_share"]
                .fillna(0)
                .values
            )
            ax.barh(regimes, values, left=left, color=CROP_COLORS[crop], label=crop, alpha=0.88)
            left += values
        ax.set_title(country)
        ax.set_xlabel("Share of region-years where crop ranks first")
        ax.set_xlim(0, 1.0)
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles = [Patch(facecolor=CROP_COLORS[crop], label=crop, alpha=0.88) for crop in crops]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        title="Winning crop",
    )
    fig.suptitle("Best predicted crop by weather regime", fontsize=15, weight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    save_figure(fig, "fig15_weather_regime_best_crop_share")


def figure_16_latest_year_weather_choice_map(choice: pd.DataFrame) -> None:
    year = int(choice["year_start"].max())
    selected = choice[choice["year_start"].eq(year)].copy()
    selected = selected.rename(
        columns={"best_crop": "crop", "best_predicted_yield_t_ha": "mean_predicted_yield_t_ha"}
    )
    aus = selected[selected["country"].eq("Australia")]
    us = selected[selected["country"].eq("United States")]
    aus_geo = load_geojson("australian-states.json")
    us_geo = load_geojson("us-states.json")

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7), gridspec_kw={"width_ratios": [0.82, 1.18], "wspace": 0.03})
    add_crop_map(axes[0], aus_geo, aus, f"Australia: best crop under {year} weather", pad_scale=1.4)
    add_crop_map(axes[1], us_geo, us, f"United States: best crop under {year} weather", exclude=CONTIGUOUS_EXCLUDE, pad_scale=0.7)

    handles = []
    labels = []
    for crop, color in CROP_COLORS.items():
        handle = axes[1].scatter([], [], s=120, marker=CROP_MARKERS[crop], color=color, edgecolor="#111111", linewidth=1.0)
        handles.append(handle)
        labels.append(crop)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
        title="Best predicted crop marker",
    )
    fig.suptitle(f"Weather-conditioned crop choice map for {year}", fontsize=15, weight="bold", y=0.96)
    fig.text(
        0.5,
        0.105,
        "This map uses the actual daily-weather summary for the selected year and predicts each crop under the same region-year conditions.",
        ha="center",
        fontsize=9.5,
        color="#444444",
    )
    fig.subplots_adjust(left=0.02, right=0.99, top=0.88, bottom=0.24)
    save_figure(fig, "fig16_latest_year_weather_conditioned_crop_map")


def write_index() -> None:
    captions = {
        "fig01_dataset_scope": "Dataset coverage, crop counts, yield distributions, and region-year weather feature rows.",
        "fig02_daily_extreme_feature_design": "Pipeline from daily weather to extreme-event indicators, yield prediction, and planning signals.",
        "fig03_model_performance": "Time-split model performance and selected best model per task.",
        "fig04_actual_vs_predicted_test": "Actual vs predicted yield on the 2016-2021 test period.",
        "fig05_top_feature_importance": "Top permutation-importance signals for combined overlap and Wheat-only models.",
        "fig06_risk_threshold_associations": "Observed yield differences between high-feature and low-feature risk groups.",
        "fig07_crop_suitability_map": "Australia and U.S. map with crop markers for best predicted crop by region/state.",
        "fig08_crop_region_suitability_heatmap": "Predicted t/ha by crop and region/state.",
        "fig09_country_crop_advantage": "Country-crop predicted yield comparison and U.S. minus Australia signal.",
        "fig10_low_yield_weather_fingerprint": "Weather fingerprint comparing low predicted yield years with high predicted yield years.",
        "fig11_weather_stress_distributions": "Distribution of selected stress indicators by country.",
        "fig12_best_crop_share_by_region": "How often the selected crop ranks first across years in each region/state.",
        "fig13_weather_conditioned_crop_winner_timeline": "For each region-year weather condition, which crop is predicted to have the highest t/ha.",
        "fig14_weather_regime_crop_yield_matrix": "Predicted crop yield under hot-dry, dry-spell, wet/storm, cold/frost, and moderate seasons.",
        "fig15_weather_regime_best_crop_share": "How often each crop wins under each weather regime.",
        "fig16_latest_year_weather_conditioned_crop_map": "Map of best predicted crop under the latest modeled year's actual weather conditions.",
    }
    lines = [
        "# Decision-Support Figure Index",
        "",
        "Generated from the first daily extreme-weather decision-support run.",
        "",
        "| Figure | Suggested use |",
        "|---|---|",
    ]
    for stem, caption in captions.items():
        lines.append(f"| `{stem}.png` | {caption} |")
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- Crop suitability maps and rankings are model-based planning signals, not causal proof.",
            "- Weather-conditioned crop choice tries each candidate crop under the same region-year weather and soil profile.",
            "- Risk-threshold bars are observed associations within grouped records.",
            "- The current scope remains winter/overlap crops with May-October as the first harmonized season proxy.",
        ]
    )
    (OUT_DIR / "FIGURE_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_01_dataset_overview()
    figure_02_feature_design_pipeline()
    figure_03_model_performance()
    figure_04_actual_vs_predicted()
    figure_05_feature_importance()
    figure_06_risk_thresholds()
    figure_07_crop_suitability_map()
    figure_08_crop_region_heatmap()
    figure_09_country_crop_advantage()
    figure_10_low_yield_fingerprint()
    figure_11_stress_distributions()
    figure_12_best_crop_share()
    choice, _, summary = build_weather_condition_choice_outputs()
    figure_13_weather_conditioned_winner_timeline(choice)
    figure_14_weather_regime_crop_yield_matrix(summary)
    figure_15_weather_regime_best_crop_share(summary)
    figure_16_latest_year_weather_choice_map(choice)
    write_index()
    print(f"Saved figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
