from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.utils import resample

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))

import train_decision_support_models as model_script  # noqa: E402
from create_decision_support_figures import add_weather_regime  # noqa: E402


RANDOM_STATE = 42
BOOTSTRAP_REPEATS = 120
TRAIN_VALIDATION_END = 2015
TEST_START = 2016
DATASET_NAME = "aus_us_overlap"
FEATURE_SET_NAME = "extreme_weather_only"
MODEL_NAME = "ExtraTrees"

METHOD_OUT = PROJECT_ROOT / "outputs" / "method_reproducibility"
MODEL_OUT = PROJECT_ROOT / "outputs" / "decision_support_models"
FIG_OUT = PROJECT_ROOT / "figures" / "decision_support"
OVERLEAF_TABLE_DIRS = [
    PROJECT_ROOT / "overleaf_upload_fastrefs" / "tables",
    PROJECT_ROOT / "overleaf_upload_extreme_weather" / "tables",
    PROJECT_ROOT / "overleaf_upload_minimal" / "tables",
    PROJECT_ROOT / "paper_assets_selected" / "latex",
]
OVERLEAF_FIG_DIRS = [
    PROJECT_ROOT / "overleaf_upload_fastrefs" / "figures" / "main",
    PROJECT_ROOT / "overleaf_upload_extreme_weather" / "figures" / "main",
    PROJECT_ROOT / "overleaf_upload_minimal" / "figures" / "main",
    PROJECT_ROOT / "paper_assets_selected" / "figures" / "main",
]

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "extreme_weather_features_aus_us_region_year.csv"

CROP_COLORS = {
    "Wheat": "#D8A31A",
    "Barley": "#5E9F6E",
    "Canola": "#E9C46A",
    "Oats": "#8AB6D6",
}
WEATHER_EXCLUDE = {"country", "region", "crop", "year_start", "lat_mean", "lon_mean", "season_days_observed"}


def clean_latex(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def feature_label(name: str) -> str:
    replacements = {
        "season_rain_sum": "Season rainfall",
        "season_rain_mean": "Mean daily rainfall",
        "season_tmax_mean": "Mean Tmax",
        "season_tmin_mean": "Mean Tmin",
        "season_tmean_mean": "Mean temperature",
        "season_radiation_sum": "Season radiation",
        "season_radiation_mean": "Mean radiation",
        "growing_degree_days_base5": "Growing degree days",
        "max_3day_rain": "Max 3-day rainfall",
        "max_7day_rain": "Max 7-day rainfall",
        "max_dry_spell_1mm": "Max dry spell",
        "dry_days_1mm": "Dry days",
        "hot_dry_days_30_1mm": "Hot-dry days",
        "heat_days_35": "Heat days >=35C",
        "heat_degree_days_30": "Heat degree days >=30C",
        "frost_days_0": "Frost days",
        "cold_days_5": "Cold days <=5C",
        "rainfall_concentration_index": "Rain concentration",
        "rain_last_14d_before_harvest": "Pre-harvest rain",
        "max_3day_rain_last_21d_before_harvest": "Pre-harvest max 3-day rain",
    }
    if name in replacements:
        return replacements[name]
    text = name.replace("_y", "").replace("_", " ")
    text = text.replace("tmax", "Tmax").replace("tmin", "Tmin")
    return text[:1].upper() + text[1:]


def load_selected_frame() -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    frame = model_script.load_frame(model_script.DATASETS[DATASET_NAME])
    features = model_script.feature_sets(frame)[FEATURE_SET_NAME]
    numeric = features["numeric"]
    categorical = features["categorical"]
    feature_cols = numeric + categorical
    return frame, numeric, categorical, feature_cols


def fit_selected_pipeline(train_frame: pd.DataFrame, numeric: list[str], categorical: list[str], random_state: int):
    model = model_script.build_models(random_state)[MODEL_NAME]
    pipeline = model_script.build_pipeline(model, numeric, categorical)
    feature_cols = numeric + categorical
    pipeline.fit(train_frame[feature_cols], train_frame[model_script.TARGET])
    return pipeline


def build_region_observed_scenarios(frame: pd.DataFrame, feature_cols: list[str], pipeline) -> pd.DataFrame:
    region_crops = {
        key: sorted(part["crop"].dropna().astype(str).unique())
        for key, part in frame.groupby(["country", "region"], dropna=False)
    }
    base_keys = ["country", "region", "year_start"]
    scenario_base = frame.sort_values(base_keys + ["crop"]).drop_duplicates(base_keys).copy()
    rows = []
    for _, row in scenario_base.iterrows():
        key = (row["country"], row["region"])
        for crop in region_crops.get(key, []):
            scenario = row.copy()
            scenario["crop"] = crop
            rows.append(scenario)
    scenarios = pd.DataFrame(rows)
    scenarios["predicted_yield_t_ha"] = pipeline.predict(scenarios[feature_cols])
    scenarios["rank_within_region_year"] = scenarios.groupby(["country", "region", "year_start"])[
        "predicted_yield_t_ha"
    ].rank(method="first", ascending=False)
    return scenarios


def top_crop_choice_cases(scenarios: pd.DataFrame) -> pd.DataFrame:
    features = add_weather_regime(pd.read_csv(FEATURES_PATH, low_memory=False))
    weather = features[["country", "region", "year_start", "weather_regime"]].copy()
    ordered = scenarios.sort_values(["country", "region", "year_start", "rank_within_region_year"])
    best = ordered[ordered["rank_within_region_year"].eq(1)].copy()
    runner = ordered[ordered["rank_within_region_year"].eq(2)].copy()
    choice = best[
        ["country", "region", "year_start", "crop", "predicted_yield_t_ha"]
    ].rename(columns={"crop": "best_crop", "predicted_yield_t_ha": "best_predicted_yield_t_ha"})
    runner = runner[
        ["country", "region", "year_start", "crop", "predicted_yield_t_ha"]
    ].rename(columns={"crop": "runner_up_crop", "predicted_yield_t_ha": "runner_up_predicted_yield_t_ha"})
    choice = choice.merge(runner, on=["country", "region", "year_start"], how="inner", validate="one_to_one")
    choice["advantage_over_runner_up_t_ha"] = (
        choice["best_predicted_yield_t_ha"] - choice["runner_up_predicted_yield_t_ha"]
    )
    candidates = (
        scenarios.groupby(["country", "region", "year_start"], as_index=False)["crop"]
        .nunique()
        .rename(columns={"crop": "candidate_crops_in_region"})
    )
    choice = choice.merge(candidates, on=["country", "region", "year_start"], how="left", validate="one_to_one")
    choice = choice.merge(weather, on=["country", "region", "year_start"], how="left", validate="many_to_one")

    choice = choice.sort_values("advantage_over_runner_up_t_ha", ascending=False)
    top = (
        choice.groupby(["country", "region"], as_index=False, group_keys=False)
        .head(1)
        .sort_values("advantage_over_runner_up_t_ha", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    return top


def scenario_pair_rows(frame: pd.DataFrame, top_cases: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    rows = []
    pairs = []
    for idx, case in top_cases.iterrows():
        base = frame[
            frame["country"].eq(case["country"])
            & frame["region"].eq(case["region"])
            & frame["year_start"].eq(case["year_start"])
        ].iloc[0]
        best = base.copy()
        best["crop"] = case["best_crop"]
        runner = base.copy()
        runner["crop"] = case["runner_up_crop"]
        pairs.append((len(rows), len(rows) + 1))
        rows.extend([best, runner])
    return pd.DataFrame(rows), pairs


def bootstrap_crop_choice_ci(
    frame: pd.DataFrame,
    top_cases: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    feature_cols: list[str],
) -> pd.DataFrame:
    train_frame = frame[frame["year_start"] <= TRAIN_VALIDATION_END].copy()
    scenario_pairs, pairs = scenario_pair_rows(frame, top_cases)
    rng = np.random.default_rng(RANDOM_STATE)
    samples = {i: [] for i in range(len(top_cases))}
    for repeat in range(BOOTSTRAP_REPEATS):
        boot = resample(
            train_frame,
            replace=True,
            n_samples=len(train_frame),
            random_state=int(rng.integers(0, 2_000_000_000)),
            stratify=train_frame["country"],
        )
        pipeline = fit_selected_pipeline(boot, numeric, categorical, RANDOM_STATE + repeat + 1)
        preds = pipeline.predict(scenario_pairs[feature_cols])
        for case_idx, (best_i, runner_i) in enumerate(pairs):
            samples[case_idx].append(float(preds[best_i] - preds[runner_i]))
    out = top_cases.copy()
    lows = []
    highs = []
    means = []
    for idx in range(len(out)):
        arr = np.asarray(samples[idx], dtype=float)
        lows.append(float(np.quantile(arr, 0.025)))
        highs.append(float(np.quantile(arr, 0.975)))
        means.append(float(np.mean(arr)))
    out["advantage_bootstrap_mean_t_ha"] = means
    out["advantage_ci95_low_t_ha"] = lows
    out["advantage_ci95_high_t_ha"] = highs
    out["advantage_ci95_t_ha"] = [
        f"[{lo:.2f}, {hi:.2f}]" for lo, hi in zip(out["advantage_ci95_low_t_ha"], out["advantage_ci95_high_t_ha"])
    ]
    return out


def crop_choice_latex(top_cases: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{llllllllll}",
        r"\hline",
        r"country & region & year & weather regime & best crop & runner-up crop & advantage, t/ha & 95\% CI, t/ha & best yield, t/ha & candidates \\",
        r"\hline",
    ]
    for _, row in top_cases.iterrows():
        lines.append(
            f"{clean_latex(row['country'])} & {clean_latex(row['region'])} & {int(row['year_start'])} & "
            f"{clean_latex(row['weather_regime'])} & {clean_latex(row['best_crop'])} & "
            f"{clean_latex(row['runner_up_crop'])} & {row['advantage_over_runner_up_t_ha']:.3f} & "
            f"{clean_latex(row['advantage_ci95_t_ha'])} & {row['best_predicted_yield_t_ha']:.3f} & "
            f"{int(row['candidate_crops_in_region'])} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    return "\n".join(lines)


def compute_permutation_importance(frame: pd.DataFrame, pipeline, feature_cols: list[str]) -> pd.DataFrame:
    test = frame[frame["year_start"] >= TEST_START].copy()
    result = permutation_importance(
        pipeline,
        test[feature_cols],
        test[model_script.TARGET],
        n_repeats=20,
        random_state=RANDOM_STATE,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    out = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_rmse_increase_mean": result.importances_mean,
            "importance_rmse_increase_std": result.importances_std,
        }
    ).sort_values("importance_rmse_increase_mean", ascending=False)
    return out.reset_index(drop=True)


def plot_top_weather_importance(importance: pd.DataFrame) -> None:
    weather = importance[
        ~importance["feature"].isin(WEATHER_EXCLUDE)
        & ~importance["feature"].str.startswith("soil_")
        & importance["importance_rmse_increase_mean"].gt(0)
    ].copy()
    top = weather.head(10).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    y = np.arange(len(top))
    ax.barh(
        y,
        top["importance_rmse_increase_mean"],
        xerr=top["importance_rmse_increase_std"],
        color="#2A9D8F",
        alpha=0.88,
        ecolor="#333333",
        capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([feature_label(f) for f in top["feature"]])
    ax.set_xlabel("Permutation importance (test RMSE increase, t/ha)")
    ax.set_title("Top weather features for the validation-selected AUS-US overlap model")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT / "fig05_top_feature_importance.png", dpi=260, bbox_inches="tight")
    fig.savefig(FIG_OUT / "fig05_top_feature_importance.svg", bbox_inches="tight")
    plt.close(fig)
    for fig_dir in OVERLEAF_FIG_DIRS:
        fig_dir.mkdir(parents=True, exist_ok=True)
        (fig_dir / "fig05_top_feature_importance.png").write_bytes(
            (FIG_OUT / "fig05_top_feature_importance.png").read_bytes()
        )
        (fig_dir / "fig05_top_feature_importance.svg").write_bytes(
            (FIG_OUT / "fig05_top_feature_importance.svg").read_bytes()
        )


def write_outputs() -> None:
    METHOD_OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    frame, numeric, categorical, feature_cols = load_selected_frame()
    train_validation = frame[frame["year_start"] <= TRAIN_VALIDATION_END].copy()
    final_pipeline = fit_selected_pipeline(train_validation, numeric, categorical, RANDOM_STATE)

    scenarios = build_region_observed_scenarios(frame, feature_cols, final_pipeline)
    scenario_path = MODEL_OUT / "region_observed_crop_choice_advantage_validation_selected_all.csv"
    scenarios[["country", "region", "year_start", "crop", "predicted_yield_t_ha", "rank_within_region_year"]].to_csv(
        scenario_path, index=False
    )
    top_cases = top_crop_choice_cases(scenarios)
    top_cases_with_ci = bootstrap_crop_choice_ci(frame, top_cases, numeric, categorical, feature_cols)
    top_cases_path = MODEL_OUT / "region_observed_crop_choice_advantage_top_region_cases.csv"
    top_cases_with_ci.to_csv(top_cases_path, index=False)
    latex = crop_choice_latex(top_cases_with_ci)
    for table_dir in OVERLEAF_TABLE_DIRS:
        table_dir.mkdir(parents=True, exist_ok=True)
        (table_dir / "table06_weather_conditioned_crop_choice.tex").write_text(latex, encoding="utf-8")

    importance = compute_permutation_importance(frame, final_pipeline, feature_cols)
    importance_path = MODEL_OUT / "permutation_importance_validation_selected_aus_us_overlap.csv"
    importance.to_csv(importance_path, index=False)
    plot_top_weather_importance(importance)

    print(f"Wrote crop-choice cases with CI: {top_cases_path}")
    print(f"Wrote permutation importance: {importance_path}")


if __name__ == "__main__":
    write_outputs()
