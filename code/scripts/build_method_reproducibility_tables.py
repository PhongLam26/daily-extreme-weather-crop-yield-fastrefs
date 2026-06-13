from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]
sys.path.insert(0, str(SCRIPT_DIR))

import train_decision_support_models as model_script  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "method_reproducibility"
OVERLEAF_TABLE_DIR = PROJECT_ROOT / "overleaf_upload_fastrefs" / "tables"
PAPER_TABLE_DIR = PROJECT_ROOT / "paper_assets_selected" / "latex"

TRAIN_END_FOR_SELECTION = 2010
VALIDATION_START = 2011
VALIDATION_END = 2015
TEST_START = 2016

MAIN_DATASET = "aus_us_overlap"
MAIN_DATASET_LABEL = "Australia + U.S. overlap crops"


FEATURE_DEFINITIONS = [
    (
        "Seasonal rainfall",
        r"Sum and mean of daily rainfall over May--Oct.",
        r"\texttt{rain\_sum}; \texttt{rain\_mean}",
    ),
    (
        "Radiation",
        r"Sum and mean of daily solar radiation over May--Oct.",
        r"\texttt{radiation\_sum}; \texttt{radiation\_mean}",
    ),
    (
        "Growing degree days",
        r"Sum of max($Tmean_d - 5^\circ$C, 0). If missing, $Tmean_d=(Tmax_d+Tmin_d)/2$.",
        "degree-days",
    ),
    (
        "Seasonal windows",
        r"Repeat selected rainfall, dry, heat, frost, heavy-rain, and radiation features for early, mid, and late season.",
        "early = May--Jun; mid = Jul--Aug; late = Sep--Oct",
    ),
    (
        "Dry exposure",
        r"Dry days use rain $<$ 1 mm and rain $<$ 2 mm. Dry spells are the longest consecutive dry run. Long dry-spell events count runs $\geq$ 7 or $\geq$ 14 days using rain $<$ 1 mm.",
        "days; event count",
    ),
    (
        "Heavy-rain exposure",
        r"Count days with rain $\geq$ 10, 20, 25, or 50 mm/day. Also compute maximum rolling 1-, 3-, and 7-day rainfall.",
        "days; mm",
    ),
    (
        "Heat exposure",
        r"Heat days use $Tmax_d \geq$ 30, 35, or 40$^\circ$C. Heatwave events count runs of at least 3 days at 30 or 35$^\circ$C.",
        "days; event count",
    ),
    (
        "Heat degree days",
        r"Sum of max($Tmax_d - 30^\circ$C, 0) and max($Tmax_d - 35^\circ$C, 0).",
        r"$^\circ$C-days",
    ),
    (
        "Hot-dry days",
        r"Count days where $Tmax_d \geq$ 30$^\circ$C and rain $<$ 1 mm.",
        "days",
    ),
    (
        "Frost and cold exposure",
        r"Frost days use $Tmin_d \leq$ 0$^\circ$C; cold days use $Tmin_d \leq$ 5$^\circ$C. Frost events require at least 2 consecutive frost days.",
        "days; event count",
    ),
    (
        "Pre-harvest rain risk",
        r"Use the last 7, 14, and 21 days before Oct 31. Storm flag = 1 if last-21-day max 3-day rain $\geq$ 25 mm or last-14-day heavy-rain count $\geq$ 1.",
        "mm; count; binary",
    ),
]


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


def metric_frame(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return model_script.metrics(y_true, y_pred)


def split_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame[frame["year_start"] <= TRAIN_END_FOR_SELECTION].copy()
    validation = frame[
        frame["year_start"].between(VALIDATION_START, VALIDATION_END)
    ].copy()
    train_validation = frame[frame["year_start"] < TEST_START].copy()
    test = frame[frame["year_start"] >= TEST_START].copy()
    if train.empty or validation.empty or train_validation.empty or test.empty:
        raise ValueError("The train/validation/test split produced an empty frame.")
    return train, validation, train_validation, test


def fit_predict_model(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    model_name: str,
    random_state: int,
) -> dict[str, Any]:
    train, validation, train_validation, test = split_frame(frame)
    feature_cols = numeric_features + categorical_features
    model = model_script.build_models(random_state)[model_name]
    pipeline = model_script.build_pipeline(model, numeric_features, categorical_features)
    pipeline.fit(train[feature_cols], train[model_script.TARGET])
    validation_pred = pipeline.predict(validation[feature_cols])
    validation_metrics = metric_frame(validation[model_script.TARGET], validation_pred)

    final_model = model_script.build_models(random_state)[model_name]
    final_pipeline = model_script.build_pipeline(final_model, numeric_features, categorical_features)
    final_pipeline.fit(train_validation[feature_cols], train_validation[model_script.TARGET])
    test_pred = final_pipeline.predict(test[feature_cols])
    test_metrics = metric_frame(test[model_script.TARGET], test_pred)

    return {
        "validation_mae": validation_metrics["mae"],
        "validation_rmse": validation_metrics["rmse"],
        "validation_r2": validation_metrics["r2"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_validation_rows": len(train_validation),
        "test_rows": len(test),
    }


def evaluate_feature_set(
    dataset_name: str,
    frame: pd.DataFrame,
    feature_set_name: str,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
    candidate_models: list[str] | None = None,
) -> list[dict[str, Any]]:
    candidate_models = candidate_models or list(model_script.build_models(random_state).keys())
    rows = []
    for model_name in candidate_models:
        metrics = fit_predict_model(frame, numeric_features, categorical_features, model_name, random_state)
        rows.append(
            {
                "dataset": dataset_name,
                "feature_set": feature_set_name,
                "model": model_name,
                "numeric_features": len(numeric_features),
                "categorical_features": len(categorical_features),
                **metrics,
            }
        )
    return rows


def select_by_validation(rows: pd.DataFrame) -> pd.Series:
    return rows.sort_values(["validation_rmse", "validation_mae", "model"], ascending=True).iloc[0]


def model_selection_tables(random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    selected_rows = []
    for dataset_name, path in model_script.DATASETS.items():
        frame = model_script.load_frame(path)
        for feature_set_name, features in model_script.feature_sets(frame).items():
            rows = evaluate_feature_set(
                dataset_name,
                frame,
                feature_set_name,
                features["numeric"],
                features["categorical"],
                random_state,
            )
            all_rows.extend(rows)
        selected_rows.append(select_by_validation(pd.DataFrame([row for row in all_rows if row["dataset"] == dataset_name])))
    return pd.DataFrame(all_rows), pd.DataFrame(selected_rows)


def present(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in frame.columns]


def feature_group_columns(frame: pd.DataFrame) -> dict[str, list[str]]:
    weather_all = present(frame, model_script.CORE_EXTREME_FEATURES + model_script.WINDOW_EXTREME_FEATURES)
    soil = model_script.soil_features(frame)
    rainfall = [
        col
        for col in weather_all
        if any(token in col.lower() for token in ["rain", "dry", "wet", "storm"])
    ]
    temperature = [
        col
        for col in weather_all
        if any(
            token in col.lower()
            for token in ["tmax", "tmin", "tmean", "heat", "cold", "frost", "degree", "hot_dry"]
        )
    ]
    radiation = [col for col in weather_all if "radiation" in col.lower()]
    extreme_events = [
        col
        for col in weather_all
        if any(
            token in col.lower()
            for token in [
                "dry_spell",
                "max_dry",
                "heavy_rain",
                "max_1day",
                "max_3day",
                "max_7day",
                "heatwave",
                "max_consecutive_heat",
                "hot_dry",
                "storm",
                "frost_events",
                "last_",
            ]
        )
    ]
    return {
        "crop + region + year only": [],
        "rainfall only": rainfall,
        "temperature only": temperature,
        "radiation only": radiation,
        "extreme events only": extreme_events,
        "weather all": weather_all,
        "weather + soil": weather_all + soil,
    }


def group_mean_predict(
    train: pd.DataFrame,
    target: pd.DataFrame,
    group_cols: list[str],
    target_col: str = model_script.TARGET,
) -> np.ndarray:
    means = train.groupby(group_cols, dropna=False)[target_col].mean()
    fallback_crop = train.groupby("crop", dropna=False)[target_col].mean() if "crop" in train.columns else pd.Series(dtype=float)
    global_mean = float(train[target_col].mean())
    preds = []
    for _, row in target.iterrows():
        key = tuple(row[col] for col in group_cols)
        if key in means.index:
            preds.append(float(means.loc[key]))
        elif "crop" in row and row["crop"] in fallback_crop.index:
            preds.append(float(fallback_crop.loc[row["crop"]]))
        else:
            preds.append(global_mean)
    return np.asarray(preds)


def previous_year_predict(frame: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    index_cols = ["country", "region", "crop", "year_start"]
    lookup = frame[index_cols + [model_script.TARGET]].copy()
    lookup["next_year"] = lookup["year_start"] + 1
    lookup = lookup.rename(columns={model_script.TARGET: "previous_yield_t_ha"})
    merged = target.merge(
        lookup[["country", "region", "crop", "next_year", "previous_yield_t_ha"]],
        left_on=["country", "region", "crop", "year_start"],
        right_on=["country", "region", "crop", "next_year"],
        how="left",
    )
    missing = merged["previous_yield_t_ha"].isna()
    if missing.any():
        train_like = frame[frame["year_start"] < int(target["year_start"].min())]
        fallback = group_mean_predict(train_like, target.iloc[np.flatnonzero(missing.to_numpy())], ["country", "region", "crop"])
        merged.loc[missing, "previous_yield_t_ha"] = fallback
    return merged["previous_yield_t_ha"].to_numpy(dtype=float)


def baseline_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    train, validation, train_validation, test = split_frame(frame)
    rows = []

    val_pred = group_mean_predict(train, validation, ["country", "region", "crop"])
    test_pred = group_mean_predict(train_validation, test, ["country", "region", "crop"])
    val_metrics = metric_frame(validation[model_script.TARGET], val_pred)
    test_metrics = metric_frame(test[model_script.TARGET], test_pred)
    rows.append(
        {
            "feature_set": "historical crop-region mean",
            "model": "Mean baseline",
            "numeric_features": 0,
            "categorical_features": 3,
            "validation_rmse": val_metrics["rmse"],
            "validation_r2": val_metrics["r2"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
        }
    )

    val_pred = previous_year_predict(frame, validation)
    test_pred = previous_year_predict(frame, test)
    val_metrics = metric_frame(validation[model_script.TARGET], val_pred)
    test_metrics = metric_frame(test[model_script.TARGET], test_pred)
    rows.append(
        {
            "feature_set": "previous-year yield",
            "model": "Lag-1 baseline",
            "numeric_features": 1,
            "categorical_features": 3,
            "validation_rmse": val_metrics["rmse"],
            "validation_r2": val_metrics["r2"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
        }
    )
    return rows


def ablation_table(random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = model_script.load_frame(model_script.DATASETS[MAIN_DATASET])
    categorical = present(frame, model_script.CATEGORICAL_FEATURES)
    year = ["year_start"] if "year_start" in frame.columns else []
    groups = feature_group_columns(frame)

    all_rows = []
    selected = []
    for feature_set_name, group_features in groups.items():
        numeric = year + group_features
        rows = evaluate_feature_set(
            MAIN_DATASET,
            frame,
            feature_set_name,
            numeric,
            categorical,
            random_state,
        )
        all_rows.extend(rows)
        selected.append(select_by_validation(pd.DataFrame(rows)))
    selected_frame = pd.DataFrame(selected)
    baseline_frame = pd.DataFrame(baseline_rows(frame))
    baseline_frame.insert(0, "dataset", MAIN_DATASET)
    selected_frame = pd.concat([baseline_frame, selected_frame], ignore_index=True, sort=False)
    return pd.DataFrame(all_rows), selected_frame


def feature_definition_latex() -> str:
    lines = [
        r"\begingroup",
        r"\arrayrulecolor{black!55}",
        r"\begin{tabularx}{\textwidth}{p{0.22\textwidth}Yp{0.24\textwidth}}",
        r"\toprule",
        r"Indicator group & Computation / threshold & Output \\",
        r"\midrule",
        r"\arrayrulecolor{black!18}",
    ]
    for feature, formula, unit in FEATURE_DEFINITIONS:
        lines.append(f"{feature} & {formula} & {unit} \\\\")
        lines.append(r"\hline")
    if lines[-1] == r"\hline":
        lines.pop()
    lines.extend([r"\arrayrulecolor{black!55}", r"\bottomrule", r"\end{tabularx}", r"\arrayrulecolor{black}", r"\endgroup", ""])
    return "\n".join(lines)


def selected_model_latex(selected: pd.DataFrame) -> str:
    labels = {
        "aus_overlap": "Australia overlap crops",
        "aus_us_overlap": "Australia + U.S. overlap crops",
        "aus_us_wheat": "Australia + U.S. wheat",
        "us_overlap": "U.S. overlap crops",
    }
    feature_labels = {
        "extreme_weather_only": "Extreme-weather only",
        "extreme_weather_soil": "Extreme-weather + soil",
    }
    rows = selected.copy()
    rows["dataset_label"] = rows["dataset"].map(labels).fillna(rows["dataset"])
    rows["feature_label"] = rows["feature_set"].map(feature_labels).fillna(rows["feature_set"])
    rows = rows.sort_values("dataset_label")
    lines = [
        r"\begin{tabular}{llllrrr}",
        r"\toprule",
        r"Dataset & Feature set & Selected model & Validation R$^2$ & Test R$^2$ & Test RMSE & Test rows \\",
        r"\midrule",
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"{clean_latex(row['dataset_label'])} & {clean_latex(row['feature_label'])} & {clean_latex(row['model'])} & "
            f"{row['validation_r2']:.3f} & {row['test_r2']:.3f} & {row['test_rmse']:.3f} & "
            f"{int(row['test_rows'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def ablation_latex(selected: pd.DataFrame) -> str:
    order = [
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
    labels = {
        "historical crop-region mean": "Historical crop-region mean",
        "previous-year yield": "Previous-year yield",
        "crop + region + year only": "Crop + region + year only",
        "rainfall only": "Rainfall only",
        "temperature only": "Temperature only",
        "radiation only": "Radiation only",
        "extreme events only": "Extreme events only",
        "weather all": "Weather all",
        "weather + soil": "Weather + soil",
    }
    rows = selected.copy()
    rows["feature_set"] = pd.Categorical(rows["feature_set"], categories=order, ordered=True)
    rows = rows.sort_values("feature_set")
    rows["feature_label"] = rows["feature_set"].astype(str).map(labels)
    lines = [
        r"\begin{tabularx}{\textwidth}{Ylrr}",
        r"\toprule",
        r"Feature / baseline & Selected model & Test R$^2$ & Test RMSE \\",
        r"\midrule",
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"{clean_latex(row['feature_label'])} & {clean_latex(row['model'])} & "
            f"{row['test_r2']:.3f} & {row['test_rmse']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}", ""])
    return "\n".join(lines)


def write_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLEAF_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    all_selection, selected = model_selection_tables(model_script.parse_args().random_state)
    all_ablation, selected_ablation = ablation_table(model_script.parse_args().random_state)

    all_selection.to_csv(OUTPUT_DIR / "model_selection_validation_all_models.csv", index=False)
    selected.to_csv(OUTPUT_DIR / "table02_validation_selected_model_performance.csv", index=False)
    all_ablation.to_csv(OUTPUT_DIR / "feature_group_ablation_all_models_aus_us_overlap.csv", index=False)
    selected_ablation.to_csv(OUTPUT_DIR / "table03_feature_group_ablation_aus_us_overlap.csv", index=False)

    feature_tex = feature_definition_latex()
    model_tex = selected_model_latex(selected)
    ablation_tex = ablation_latex(selected_ablation)

    for table_dir in [OVERLEAF_TABLE_DIR, PAPER_TABLE_DIR]:
        (table_dir / "table00_extreme_weather_feature_definitions.tex").write_text(feature_tex, encoding="utf-8")
        (table_dir / "table02_best_model_performance.tex").write_text(model_tex, encoding="utf-8")
        (table_dir / "table03_feature_group_ablation.tex").write_text(ablation_tex, encoding="utf-8")

    # Keep the old filename used by main.tex, but with richer ablation content.
    (OVERLEAF_TABLE_DIR / "table03_weather_vs_weather_soil.tex").write_text(ablation_tex, encoding="utf-8")
    (PAPER_TABLE_DIR / "table03_weather_vs_weather_soil.tex").write_text(ablation_tex, encoding="utf-8")

    print("Wrote reproducibility tables to:")
    print(f"  {OUTPUT_DIR}")
    print(f"  {OVERLEAF_TABLE_DIR}")


if __name__ == "__main__":
    write_outputs()
