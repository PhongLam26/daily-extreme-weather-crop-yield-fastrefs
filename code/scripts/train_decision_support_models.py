from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing modelling dependency. Install it with:\n"
        "  python -m pip install -r newProject_extreme_weather\\requirements-modeling.txt"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FRAME_DIR = PROJECT_ROOT / "data" / "processed" / "model_frames"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "decision_support_models"

DATASETS = {
    "aus_us_overlap": MODEL_FRAME_DIR / "aus_us_overlap_crops_with_extreme_weather_1989_2021.csv",
    "aus_us_wheat": MODEL_FRAME_DIR / "aus_us_wheat_with_extreme_weather_1989_2021.csv",
    "aus_overlap": MODEL_FRAME_DIR / "aus_overlap_crops_with_extreme_weather_1989_2021.csv",
    "us_overlap": MODEL_FRAME_DIR / "us_overlap_crops_with_extreme_weather_1989_2021.csv",
}

CATEGORICAL_FEATURES = ["country", "region", "crop"]
TARGET = "yield_t_ha"

CORE_EXTREME_FEATURES = [
    "lat_mean",
    "lon_mean",
    "season_days_observed",
    "season_rain_sum",
    "season_rain_mean",
    "season_tmax_mean",
    "season_tmin_mean",
    "season_tmean_mean",
    "season_radiation_sum",
    "season_radiation_mean",
    "growing_degree_days_base5",
    "heat_days_30",
    "heat_days_35",
    "heat_days_40",
    "max_consecutive_heat_days_30",
    "max_consecutive_heat_days_35",
    "heatwave_events_3d_30",
    "heatwave_events_3d_35",
    "heat_degree_days_30",
    "heat_degree_days_35",
    "dry_days_1mm",
    "dry_days_2mm",
    "max_dry_spell_1mm",
    "max_dry_spell_2mm",
    "dry_spell_events_7d",
    "dry_spell_events_14d",
    "hot_dry_days_30_1mm",
    "heavy_rain_days_10",
    "heavy_rain_days_20",
    "heavy_rain_days_25",
    "heavy_rain_days_50",
    "max_1day_rain",
    "max_3day_rain",
    "max_7day_rain",
    "wet_days_1mm",
    "max_wet_spell_1mm",
    "rainfall_concentration_index",
    "rain_last_7d_before_harvest",
    "rain_last_14d_before_harvest",
    "rain_last_21d_before_harvest",
    "wet_days_last_14d_before_harvest",
    "heavy_rain_days_last_14d_before_harvest",
    "max_3day_rain_last_21d_before_harvest",
    "storm_before_harvest_flag",
    "frost_days_0",
    "cold_days_5",
    "min_tmin",
    "frost_events_2d",
]

WINDOW_EXTREME_FEATURES = [
    "rain_sum_early_y",
    "rain_sum_mid_y",
    "rain_sum_late_y",
    "dry_days_1mm_early",
    "dry_days_1mm_mid",
    "dry_days_1mm_late",
    "max_dry_spell_1mm_early",
    "max_dry_spell_1mm_mid",
    "max_dry_spell_1mm_late",
    "heavy_rain_days_20_early",
    "heavy_rain_days_20_mid",
    "heavy_rain_days_20_late",
    "max_3day_rain_early",
    "max_3day_rain_mid",
    "max_3day_rain_late",
    "heat_days_35_early_y",
    "heat_days_35_mid_y",
    "heat_days_35_late_y",
    "max_consecutive_heat_days_35_early",
    "max_consecutive_heat_days_35_mid",
    "max_consecutive_heat_days_35_late",
    "frost_days_0_early_y",
    "frost_days_0_mid_y",
    "frost_days_0_late_y",
    "radiation_sum_early_y",
    "radiation_sum_mid_y",
    "radiation_sum_late_y",
]

RISK_FEATURES = [
    "heat_days_35",
    "max_consecutive_heat_days_35",
    "heatwave_events_3d_35",
    "max_dry_spell_1mm",
    "dry_spell_events_14d",
    "hot_dry_days_30_1mm",
    "heavy_rain_days_20",
    "heavy_rain_days_50",
    "max_3day_rain",
    "max_7day_rain",
    "rainfall_concentration_index",
    "rain_last_14d_before_harvest",
    "heavy_rain_days_last_14d_before_harvest",
    "storm_before_harvest_flag",
    "frost_days_0",
]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_models(random_state: int) -> dict[str, Any]:
    return {
        "Ridge": Ridge(alpha=1.0),
        "ElasticNet": ElasticNet(alpha=0.003, l1_ratio=0.25, max_iter=10000, random_state=random_state),
        "RandomForest": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=350,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=350,
            learning_rate=0.045,
            l2_regularization=0.05,
            random_state=random_state,
        ),
        "ANN_MLP": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=900,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=random_state,
        ),
    }


def build_pipeline(model: Any, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric, numeric_features),
            ("cat", categorical, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    observed = y_true.to_numpy(dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    error = observed - predicted
    mae = float(np.mean(np.abs(error)))
    rmse = float(math.sqrt(np.mean(np.square(error))))
    denominator = float(np.sum(np.square(observed - observed.mean())))
    r2 = float(1 - np.sum(np.square(error)) / denominator) if denominator > 0 else np.nan
    return {"mae": mae, "rmse": rmse, "r2": r2}


def present(columns: list[str], df: pd.DataFrame) -> list[str]:
    return [col for col in columns if col in df.columns]


def soil_features(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("soil_")]


def feature_sets(df: pd.DataFrame) -> dict[str, dict[str, list[str]]]:
    extreme = present(CORE_EXTREME_FEATURES + WINDOW_EXTREME_FEATURES, df)
    soil = soil_features(df)
    categorical = present(CATEGORICAL_FEATURES, df)
    year = ["year_start"] if "year_start" in df.columns else []
    return {
        "extreme_weather_only": {
            "numeric": year + extreme,
            "categorical": categorical,
        },
        "extreme_weather_soil": {
            "numeric": year + extreme + soil,
            "categorical": categorical,
        },
    }


def train_test_split_by_year(df: pd.DataFrame, train_end: int, test_start: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["year_start"] <= train_end].copy()
    test = df[df["year_start"] >= test_start].copy()
    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty frame.")
    return train, test


def load_frame(path: Path) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path, low_memory=False)
    df = df[df[TARGET].notna()].copy()
    df["year_start"] = pd.to_numeric(df["year_start"], errors="coerce")
    df = df[df["year_start"].notna()].copy()
    df["year_start"] = df["year_start"].astype(int)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def fit_and_score(
    dataset_name: str,
    feature_set_name: str,
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
    train_end: int,
    test_start: int,
) -> tuple[list[dict[str, Any]], dict[str, Pipeline]]:
    train, test = train_test_split_by_year(frame, train_end, test_start)
    feature_cols = numeric_features + categorical_features
    X_train = train[feature_cols]
    y_train = train[TARGET]
    X_test = test[feature_cols]
    y_test = test[TARGET]

    rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    for model_name, model in build_models(random_state).items():
        pipeline = build_pipeline(model, numeric_features, categorical_features)
        pipeline.fit(X_train, y_train)
        pred_train = pipeline.predict(X_train)
        pred_test = pipeline.predict(X_test)
        train_metrics = metrics(y_train, pred_train)
        test_metrics = metrics(y_test, pred_test)
        rows.append(
            {
                "dataset": dataset_name,
                "feature_set": feature_set_name,
                "model": model_name,
                "train_rows": len(train),
                "test_rows": len(test),
                "numeric_features": len(numeric_features),
                "categorical_features": len(categorical_features),
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
            }
        )
        fitted[model_name] = pipeline
    return rows, fitted


def select_best_model(results: pd.DataFrame, dataset_name: str) -> pd.Series:
    subset = results[results["dataset"].eq(dataset_name)].copy()
    preferred = subset[subset["feature_set"].eq("extreme_weather_soil")]
    if not preferred.empty:
        subset = preferred
    return subset.sort_values(["test_rmse", "test_mae"], ascending=True).iloc[0]


def prediction_frame(
    dataset_name: str,
    frame: pd.DataFrame,
    pipeline: Pipeline,
    feature_cols: list[str],
    train_end: int,
    test_start: int,
) -> pd.DataFrame:
    train, test = train_test_split_by_year(frame, train_end, test_start)
    keep_cols = present(["country", "region", "crop", "season", "year_start", TARGET], frame)
    output_parts = []
    for split_name, part in [("train", train), ("test", test)]:
        pred = pipeline.predict(part[feature_cols])
        out = part[keep_cols].copy()
        out.insert(0, "dataset", dataset_name)
        out["split"] = split_name
        out["predicted_yield_t_ha"] = pred
        out["prediction_error_t_ha"] = out[TARGET] - out["predicted_yield_t_ha"]
        output_parts.append(out)
    return pd.concat(output_parts, ignore_index=True)


def permutation_feature_importance(
    dataset_name: str,
    feature_set_name: str,
    model_name: str,
    frame: pd.DataFrame,
    pipeline: Pipeline,
    feature_cols: list[str],
    train_end: int,
    test_start: int,
    random_state: int,
) -> pd.DataFrame:
    _, test = train_test_split_by_year(frame, train_end, test_start)
    result = permutation_importance(
        pipeline,
        test[feature_cols],
        test[TARGET],
        n_repeats=8,
        random_state=random_state,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    out = pd.DataFrame(
        {
            "dataset": dataset_name,
            "feature_set": feature_set_name,
            "model": model_name,
            "feature": feature_cols,
            "importance_rmse_increase_mean": result.importances_mean,
            "importance_rmse_increase_std": result.importances_std,
        }
    )
    return out.sort_values("importance_rmse_increase_mean", ascending=False).reset_index(drop=True)


def risk_threshold_summary(dataset_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = present(["country", "crop"], frame)
    for group_keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        group_id = dict(zip(group_cols, group_keys))
        for feature in present(RISK_FEATURES, group):
            values = pd.to_numeric(group[feature], errors="coerce")
            if values.notna().sum() < 10 or values.nunique(dropna=True) < 2:
                continue
            low_threshold = float(values.quantile(0.25))
            high_threshold = float(values.quantile(0.75))
            if math.isclose(low_threshold, high_threshold):
                continue
            low = group[values <= low_threshold]
            high = group[values >= high_threshold]
            if len(low) < 3 or len(high) < 3:
                continue
            low_yield = float(low[TARGET].mean())
            high_yield = float(high[TARGET].mean())
            rows.append(
                {
                    "dataset": dataset_name,
                    **group_id,
                    "feature": feature,
                    "low_threshold_q25": low_threshold,
                    "high_threshold_q75": high_threshold,
                    "mean_yield_low_feature": low_yield,
                    "mean_yield_high_feature": high_yield,
                    "high_minus_low_yield_t_ha": high_yield - low_yield,
                    "low_group_rows": len(low),
                    "high_group_rows": len(high),
                }
            )
    return pd.DataFrame(rows)


def crop_scenario_rankings(
    dataset_name: str,
    frame: pd.DataFrame,
    pipeline: Pipeline,
    feature_cols: list[str],
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "crop" not in frame.columns or frame["crop"].nunique(dropna=True) < 2:
        return pd.DataFrame(), pd.DataFrame()

    candidate_crops_by_country = {
        country: sorted(part["crop"].dropna().astype(str).unique())
        for country, part in frame.groupby("country", dropna=False)
    }
    base_keys = present(["country", "region", "year_start"], frame)
    scenario_base = frame.sort_values(base_keys + ["crop"]).drop_duplicates(base_keys).copy()
    scenario_rows = []
    for _, row in scenario_base.iterrows():
        country = str(row["country"])
        for crop in candidate_crops_by_country.get(country, []):
            scenario = row.copy()
            scenario["crop"] = crop
            scenario_rows.append(scenario)
    scenarios = pd.DataFrame(scenario_rows)
    scenarios["predicted_yield_t_ha"] = pipeline.predict(scenarios[feature_cols])

    detail_cols = present(["country", "region", "year_start", "crop"], scenarios) + ["predicted_yield_t_ha"]
    detail = scenarios[detail_cols].copy()
    detail["rank_within_region_year"] = detail.groupby(["country", "region", "year_start"])[
        "predicted_yield_t_ha"
    ].rank(method="first", ascending=False)

    summary = (
        detail.groupby(["country", "region", "crop"], as_index=False)
        .agg(
            mean_predicted_yield_t_ha=("predicted_yield_t_ha", "mean"),
            median_predicted_yield_t_ha=("predicted_yield_t_ha", "median"),
            years=("year_start", "nunique"),
            best_rank_share=("rank_within_region_year", lambda s: float((s == 1).mean())),
        )
        .sort_values(["country", "region", "mean_predicted_yield_t_ha"], ascending=[True, True, False])
    )
    summary["rank_within_region"] = summary.groupby(["country", "region"])["mean_predicted_yield_t_ha"].rank(
        method="first", ascending=False
    )

    detail.to_csv(output_dir / f"scenario_crop_ranking_by_region_year_{dataset_name}.csv", index=False)
    summary.to_csv(output_dir / f"scenario_crop_ranking_by_region_{dataset_name}.csv", index=False)
    return detail, summary


def country_crop_advantage(dataset_name: str, crop_summary: pd.DataFrame) -> pd.DataFrame:
    if crop_summary.empty:
        return pd.DataFrame()
    country_summary = (
        crop_summary.groupby(["country", "crop"], as_index=False)
        .agg(mean_predicted_yield_t_ha=("mean_predicted_yield_t_ha", "mean"))
        .pivot(index="crop", columns="country", values="mean_predicted_yield_t_ha")
        .reset_index()
    )
    country_summary.insert(0, "dataset", dataset_name)
    if {"Australia", "United States"}.issubset(country_summary.columns):
        country_summary["us_minus_australia_predicted_yield_t_ha"] = (
            country_summary["United States"] - country_summary["Australia"]
        )
    return country_summary


def low_yield_cases(dataset_name: str, predictions: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    risk_cols = present(RISK_FEATURES, frame)
    id_cols = present(["country", "region", "crop", "year_start"], frame)
    enriched = predictions.merge(frame[id_cols + risk_cols], on=id_cols, how="left")
    test = enriched[enriched["split"].eq("test")].copy()
    if test.empty:
        return pd.DataFrame()
    return test.sort_values("predicted_yield_t_ha", ascending=True).head(30)


def write_markdown_summary(results: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# Decision-Support Model Summary",
        "",
        "Goal: predict crop yield in tonnes per hectare and support practical interpretation of why yield may rise or fall.",
        "",
        "Evaluation uses time split: train 1989-2015, test 2016-2021.",
        "",
        "## Best Models",
        "",
        "| Dataset | Feature set | Model | Test MAE | Test RMSE | Test R2 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for dataset in sorted(results["dataset"].unique()):
        best = select_best_model(results, dataset)
        lines.append(
            f"| {dataset} | {best['feature_set']} | {best['model']} | "
            f"{best['test_mae']:.3f} | {best['test_rmse']:.3f} | {best['test_r2']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Linear models are included for transparent baselines.",
            "- Tree and boosting models are included for stronger tabular prediction.",
            "- ANN_MLP is included as an artificial neural network baseline, but it should only be emphasized if it wins under the time split.",
            "- Feature importance is estimated by permutation importance on the test period.",
            "- Crop ranking outputs are model-based suitability signals, not causal proof or farm-level planting advice.",
        ]
    )
    (output_dir / "decision_support_model_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train prediction and planning models for extreme-weather yield study.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--train-end", type=int, default=2015)
    parser.add_argument("--test-start", type=int, default=2016)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    fitted_by_dataset: dict[tuple[str, str, str], Pipeline] = {}
    frames: dict[str, pd.DataFrame] = {}
    feature_columns: dict[tuple[str, str], list[str]] = {}
    risk_outputs: list[pd.DataFrame] = []
    advantage_outputs: list[pd.DataFrame] = []
    low_case_outputs: list[pd.DataFrame] = []

    for dataset_name, path in DATASETS.items():
        frame = load_frame(path)
        frames[dataset_name] = frame
        risk_outputs.append(risk_threshold_summary(dataset_name, frame))
        for feature_set_name, features in feature_sets(frame).items():
            numeric_features = features["numeric"]
            categorical_features = features["categorical"]
            if not numeric_features or not categorical_features:
                continue
            rows, fitted = fit_and_score(
                dataset_name,
                feature_set_name,
                frame,
                numeric_features,
                categorical_features,
                args.random_state,
                args.train_end,
                args.test_start,
            )
            all_results.extend(rows)
            feature_cols = numeric_features + categorical_features
            feature_columns[(dataset_name, feature_set_name)] = feature_cols
            for model_name, pipeline in fitted.items():
                fitted_by_dataset[(dataset_name, feature_set_name, model_name)] = pipeline

    results = pd.DataFrame(all_results).sort_values(["dataset", "feature_set", "test_rmse"])
    results.to_csv(args.output_dir / "model_results.csv", index=False)

    importance_outputs = []
    prediction_outputs = []
    for dataset_name in sorted(frames):
        best = select_best_model(results, dataset_name)
        key = (dataset_name, str(best["feature_set"]), str(best["model"]))
        pipeline = fitted_by_dataset[key]
        feature_cols = feature_columns[(dataset_name, str(best["feature_set"]))]
        frame = frames[dataset_name]

        predictions = prediction_frame(dataset_name, frame, pipeline, feature_cols, args.train_end, args.test_start)
        predictions.to_csv(args.output_dir / f"predictions_{dataset_name}.csv", index=False)
        prediction_outputs.append(predictions)

        importance = permutation_feature_importance(
            dataset_name,
            str(best["feature_set"]),
            str(best["model"]),
            frame,
            pipeline,
            feature_cols,
            args.train_end,
            args.test_start,
            args.random_state,
        )
        importance.to_csv(args.output_dir / f"feature_importance_{dataset_name}.csv", index=False)
        importance_outputs.append(importance)

        _, crop_summary = crop_scenario_rankings(dataset_name, frame, pipeline, feature_cols, args.output_dir)
        advantage = country_crop_advantage(dataset_name, crop_summary)
        if not advantage.empty:
            advantage_outputs.append(advantage)

        low_cases = low_yield_cases(dataset_name, predictions, frame)
        if not low_cases.empty:
            low_case_outputs.append(low_cases)

    pd.concat(importance_outputs, ignore_index=True).to_csv(
        args.output_dir / "feature_importance_all_best_models.csv", index=False
    )
    pd.concat(prediction_outputs, ignore_index=True).to_csv(args.output_dir / "predictions_all_best_models.csv", index=False)
    pd.concat([df for df in risk_outputs if not df.empty], ignore_index=True).to_csv(
        args.output_dir / "risk_threshold_summary.csv", index=False
    )
    if advantage_outputs:
        pd.concat(advantage_outputs, ignore_index=True).to_csv(
            args.output_dir / "country_crop_advantage_matrix.csv", index=False
        )
    if low_case_outputs:
        pd.concat(low_case_outputs, ignore_index=True).to_csv(args.output_dir / "low_yield_explanation_cases.csv", index=False)

    write_markdown_summary(results, args.output_dir)

    selected = pd.DataFrame([select_best_model(results, dataset) for dataset in sorted(frames)])
    print(f"Saved model results and decision-support outputs to: {args.output_dir}")
    print(selected[["dataset", "feature_set", "model", "test_rmse", "test_r2"]])


if __name__ == "__main__":
    main()
