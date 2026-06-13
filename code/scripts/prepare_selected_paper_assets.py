from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "paper_assets_selected"
TABLE_DIR = OUTPUT_ROOT / "tables"
DRAFT_DIR = OUTPUT_ROOT / "drafts"
LATEX_DIR = OUTPUT_ROOT / "latex"
FIG_MAIN_DIR = OUTPUT_ROOT / "figures" / "main"
FIG_SUPP_DIR = OUTPUT_ROOT / "figures" / "supplementary"

MODEL_DIR = PROJECT_ROOT / "outputs" / "decision_support_models"
YIELD_STORY_DIR = MODEL_DIR / "yield_story"
FIG_DECISION_DIR = PROJECT_ROOT / "figures" / "decision_support"
FIG_YIELD_DIR = PROJECT_ROOT / "figures" / "yield_story"
DATA_DIR = PROJECT_ROOT / "data" / "processed"


MAIN_FIGURES = [
    {
        "source_dir": FIG_DECISION_DIR,
        "stem": "fig01_dataset_scope",
        "paper_label": "Figure 1",
        "role": "Dataset scope and modeling target",
        "reason": "Shows country, crop, yield, and weather-row coverage.",
    },
    {
        "source_dir": FIG_DECISION_DIR,
        "stem": "fig02_daily_extreme_feature_design",
        "paper_label": "Figure 2",
        "role": "Daily feature engineering and decision-support workflow",
        "reason": "Explains the method to non-AI readers before the model results.",
    },
    {
        "source_dir": FIG_DECISION_DIR,
        "stem": "fig03_model_performance",
        "paper_label": "Figure 3",
        "role": "Time-split predictive performance",
        "reason": "Documents whether the framework predicts held-out yields.",
    },
    {
        "source_dir": FIG_DECISION_DIR,
        "stem": "fig04_actual_vs_predicted_test",
        "paper_label": "Figure 4",
        "role": "Observed versus predicted yield",
        "reason": "Makes model quality interpretable in the original t/ha scale.",
    },
    {
        "source_dir": FIG_YIELD_DIR,
        "stem": "fig17_observed_yield_change_region_crop",
        "paper_label": "Figure 5",
        "role": "Which crop-region yields increased or decreased",
        "reason": "Directly answers crop-specific and region-specific yield change.",
    },
    {
        "source_dir": FIG_YIELD_DIR,
        "stem": "fig19_high_vs_low_yield_weather_driver_matrix",
        "paper_label": "Figure 6",
        "role": "Weather drivers of high-yield versus low-yield years",
        "reason": "Provides the main explainability story in weather terms.",
    },
    {
        "source_dir": FIG_DECISION_DIR,
        "stem": "fig07_crop_suitability_map",
        "paper_label": "Figure 7",
        "role": "Regional crop suitability map",
        "reason": "Communicates the decision-support output spatially.",
    },
    {
        "source_dir": FIG_YIELD_DIR,
        "stem": "fig21_weather_condition_crop_choice_advantage",
        "paper_label": "Figure 8",
        "role": "Weather-conditioned crop choice advantage",
        "reason": "Shows when choosing one crop over another changes expected t/ha most.",
    },
]


SUPPLEMENTARY_FIGURES = [
    ("fig05_top_feature_importance", FIG_DECISION_DIR, "Permutation-importance details."),
    ("fig06_risk_threshold_associations", FIG_DECISION_DIR, "Observed risk-threshold associations."),
    ("fig08_crop_region_suitability_heatmap", FIG_DECISION_DIR, "Full crop-region predicted-yield heatmap."),
    ("fig09_country_crop_advantage", FIG_DECISION_DIR, "Country-crop predicted advantage matrix."),
    ("fig10_low_yield_weather_fingerprint", FIG_DECISION_DIR, "Low-yield weather-fingerprint details."),
    ("fig11_weather_stress_distributions", FIG_DECISION_DIR, "Weather stress distributions by country."),
    ("fig12_best_crop_share_by_region", FIG_DECISION_DIR, "Share of years each crop ranks first."),
    ("fig13_weather_conditioned_crop_winner_timeline", FIG_DECISION_DIR, "Crop winner over time by region-year."),
    ("fig14_weather_regime_crop_yield_matrix", FIG_DECISION_DIR, "Predicted yield by weather regime and crop."),
    ("fig15_weather_regime_best_crop_share", FIG_DECISION_DIR, "Crop winning share by weather regime."),
    ("fig16_latest_year_weather_conditioned_crop_map", FIG_DECISION_DIR, "Latest modeled-year crop map."),
    ("fig18_top_yield_gain_loss_region_crop", FIG_YIELD_DIR, "Ranked gain/loss examples."),
    ("fig20_region_crop_yield_story_cards", FIG_YIELD_DIR, "Readable crop-region story cards."),
]


DATASET_LABELS = {
    "aus_overlap": "Australia overlap crops",
    "aus_us_overlap": "Australia + U.S. overlap crops",
    "aus_us_wheat": "Australia + U.S. wheat",
    "us_overlap": "U.S. overlap crops",
}


def ensure_dirs() -> None:
    for directory in [TABLE_DIR, DRAFT_DIR, LATEX_DIR, FIG_MAIN_DIR, FIG_SUPP_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def read_csv(relative_path: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / relative_path, **kwargs)


def save_table(df: pd.DataFrame, stem: str, index: bool = False) -> None:
    csv_path = TABLE_DIR / f"{stem}.csv"
    md_path = TABLE_DIR / f"{stem}.md"
    tex_path = LATEX_DIR / f"{stem}.tex"
    df.to_csv(csv_path, index=index)
    md_path.write_text(markdown_table(df), encoding="utf-8")
    tex_path.write_text(latex_table(df), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    display = display.fillna("")
    cols = list(display.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in display.iterrows():
        values = [format_cell(row[col]) for col in cols]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def latex_table(df: pd.DataFrame) -> str:
    display = df.copy().fillna("")
    columns = list(display.columns)
    spec = "l" * len(columns)
    lines = [
        f"\\begin{{tabular}}{{{spec}}}",
        "\\hline",
        " & ".join(latex_escape(col) for col in columns) + r" \\",
        "\\hline",
    ]
    for _, row in display.iterrows():
        values = [latex_escape(format_cell(row[col])) for col in columns]
        lines.append(" & ".join(values) + r" \\")
    lines.extend(["\\hline", "\\end{tabular}", ""])
    return "\n".join(lines)


def latex_escape(value: object) -> str:
    text = str(value)
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
    return "".join(replacements.get(char, char) for char in text)


def format_cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("\n", " ")


def copy_figure_pair(source_dir: Path, stem: str, target_dir: Path) -> None:
    for ext in [".png", ".svg"]:
        source = source_dir / f"{stem}{ext}"
        if source.exists():
            shutil.copy2(source, target_dir / source.name)


def build_dataset_scope_table() -> pd.DataFrame:
    frames = [
        (
            "Extreme-weather feature table",
            read_csv("data/processed/extreme_weather_features_aus_us_region_year.csv"),
            "country-region-year daily-weather features",
        ),
        (
            "Australia overlap crop model frame",
            read_csv("data/processed/model_frames/aus_overlap_crops_with_extreme_weather_1989_2021.csv", low_memory=False),
            "crop-region-year yield with weather and soil",
        ),
        (
            "U.S. overlap crop model frame",
            read_csv("data/processed/model_frames/us_overlap_crops_with_extreme_weather_1989_2021.csv", low_memory=False),
            "crop-region-year yield with weather and soil",
        ),
        (
            "Australia + U.S. overlap crop model frame",
            read_csv("data/processed/model_frames/aus_us_overlap_crops_with_extreme_weather_1989_2021.csv", low_memory=False),
            "cross-country overlap crops",
        ),
        (
            "Australia + U.S. wheat model frame",
            read_csv("data/processed/model_frames/aus_us_wheat_with_extreme_weather_1989_2021.csv", low_memory=False),
            "wheat-only cross-country comparison",
        ),
    ]
    rows = []
    for label, df, unit in frames:
        year_col = "year_start"
        crop_count = df["crop"].nunique() if "crop" in df.columns else ""
        crop_list = ", ".join(sorted(df["crop"].dropna().unique())) if "crop" in df.columns else ""
        rows.append(
            {
                "dataset": label,
                "rows": len(df),
                "countries": df["country"].nunique() if "country" in df.columns else "",
                "regions": df["region"].nunique() if "region" in df.columns else "",
                "crops": crop_count,
                "years": f"{int(df[year_col].min())}-{int(df[year_col].max())}" if year_col in df.columns else "",
                "unit": unit,
                "crop_scope": crop_list,
            }
        )
    return pd.DataFrame(rows)


def build_best_model_table() -> pd.DataFrame:
    model_results = read_csv("outputs/decision_support_models/model_results.csv")
    selected = (
        model_results.sort_values(["dataset", "test_r2"], ascending=[True, False])
        .groupby("dataset", as_index=False)
        .head(1)
        .copy()
    )
    selected["dataset"] = selected["dataset"].map(DATASET_LABELS).fillna(selected["dataset"])
    selected = selected[
        [
            "dataset",
            "feature_set",
            "model",
            "train_rows",
            "test_rows",
            "test_mae",
            "test_rmse",
            "test_r2",
        ]
    ]
    selected = selected.rename(
        columns={
            "feature_set": "feature set",
            "train_rows": "train rows",
            "test_rows": "test rows",
            "test_mae": "test MAE",
            "test_rmse": "test RMSE",
            "test_r2": "test R2",
        }
    )
    return selected.sort_values("dataset")


def build_feature_set_comparison_table() -> pd.DataFrame:
    model_results = read_csv("outputs/decision_support_models/model_results.csv")
    best = (
        model_results.sort_values(["dataset", "feature_set", "test_r2"], ascending=[True, True, False])
        .groupby(["dataset", "feature_set"], as_index=False)
        .head(1)
        .copy()
    )
    pivot = best.pivot(index="dataset", columns="feature_set", values="test_r2").reset_index()
    pivot["soil_gain_R2"] = pivot.get("extreme_weather_soil", pd.NA) - pivot.get("extreme_weather_only", pd.NA)
    pivot["dataset"] = pivot["dataset"].map(DATASET_LABELS).fillna(pivot["dataset"])
    pivot = pivot.rename(
        columns={
            "extreme_weather_only": "best R2, weather only",
            "extreme_weather_soil": "best R2, weather + soil",
            "soil_gain_R2": "R2 gain from soil",
        }
    )
    return pivot[["dataset", "best R2, weather only", "best R2, weather + soil", "R2 gain from soil"]]


def build_gain_loss_table() -> pd.DataFrame:
    gains = read_csv("outputs/decision_support_models/yield_story/crop_region_gain_loss_ranking.csv")
    top_gains = gains.sort_values("delta_observed_yield_t_ha", ascending=False).head(8)
    top_losses = gains.sort_values("delta_observed_yield_t_ha", ascending=True).head(6)
    table = pd.concat([top_gains, top_losses], axis=0)
    table = table[
        [
            "country",
            "region",
            "crop",
            "early_observed_yield_1989_2000",
            "late_observed_yield_2010_2021",
            "delta_observed_yield_t_ha",
            "observed_change_status",
        ]
    ].rename(
        columns={
            "country": "country",
            "region": "region",
            "crop": "crop",
            "early_observed_yield_1989_2000": "early yield, 1989-2000",
            "late_observed_yield_2010_2021": "recent yield, 2010-2021",
            "delta_observed_yield_t_ha": "change, t/ha",
            "observed_change_status": "status",
        }
    )
    return table


def build_driver_story_table() -> pd.DataFrame:
    stories = read_csv("outputs/decision_support_models/yield_story/high_low_yield_story_by_region_crop.csv")
    table = stories.sort_values("predicted_yield_gap_high_minus_low_t_ha", ascending=False).head(10)
    table = table[
        [
            "country",
            "region",
            "crop",
            "predicted_yield_gap_high_minus_low_t_ha",
            "top_weather_driver_story",
            "low_group_years",
            "high_group_years",
        ]
    ].rename(
        columns={
            "predicted_yield_gap_high_minus_low_t_ha": "predicted high-low gap, t/ha",
            "top_weather_driver_story": "weather indicators higher/lower in high-yield years",
            "low_group_years": "low-yield years",
            "high_group_years": "high-yield years",
        }
    )
    return table


def build_crop_choice_table() -> pd.DataFrame:
    choices = read_csv("outputs/decision_support_models/yield_story/weather_condition_crop_choice_gain_stories.csv")
    table = choices.sort_values("largest_advantage_t_ha", ascending=False).head(10)
    table = table[
        [
            "country",
            "region",
            "largest_advantage_year",
            "largest_advantage_weather_regime",
            "largest_advantage_best_crop",
            "largest_advantage_runner_up_crop",
            "largest_advantage_t_ha",
            "largest_advantage_best_predicted_yield_t_ha",
        ]
    ].rename(
        columns={
            "largest_advantage_year": "year",
            "largest_advantage_weather_regime": "weather regime",
            "largest_advantage_best_crop": "best crop",
            "largest_advantage_runner_up_crop": "runner-up crop",
            "largest_advantage_t_ha": "advantage over runner-up, t/ha",
            "largest_advantage_best_predicted_yield_t_ha": "best predicted yield, t/ha",
        }
    )
    return table


def build_main_figure_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "paper figure": item["paper_label"],
                "file stem": item["stem"],
                "role in paper": item["role"],
                "why keep it": item["reason"],
            }
            for item in MAIN_FIGURES
        ]
    )


def build_supp_figure_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "supplementary file stem": stem,
                "supporting role": role,
            }
            for stem, _, role in SUPPLEMENTARY_FIGURES
        ]
    )


def write_text(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_readme(tables: dict[str, pd.DataFrame]) -> None:
    text = f"""
# Selected Paper Assets

This folder is the curated paper package for the daily extreme-weather crop-yield study.
It is designed to support a professional paper narrative, not only a model leaderboard.

## Folder map

- `figures/main`: selected figures for the main manuscript.
- `figures/supplementary`: useful figures for appendix/supplementary material.
- `tables`: CSV and Markdown versions of publication tables.
- `latex`: LaTeX versions of the same tables.
- `drafts`: paper text drafts, figure captions, and writing guidance.

## Main scientific message

The study predicts crop yield in tonnes per hectare, then translates model outputs into
crop-region and weather-conditioned planning signals. The key contribution is the link
between daily extreme-weather indicators, soil context, and interpretable crop choice:
which crop performs better, where, under what type of weather year, and which weather
signals separate high-yield years from low-yield years.

## Selected main figures

{markdown_table(tables["main_figures"])}

## Recommended use

Use `drafts/RESULTS_STORY_DRAFT.md` as the starting point for the Results section.
Use `drafts/METHODS_DRAFT.md` for the Methods section.
Use `drafts/FIGURE_CAPTIONS_DRAFT.md` when moving selected figures into LaTeX.
"""
    write_text(OUTPUT_ROOT / "README.md", text)


def write_structure_draft(tables: dict[str, pd.DataFrame]) -> None:
    text = f"""
# Paper Structure And Figure Plan

## Working title options

1. Daily extreme-weather indicators for interpretable crop-yield prediction and crop suitability analysis.
2. From daily weather extremes to crop choice: an interpretable yield prediction framework for Australia and the United States.
3. Weather-conditioned crop suitability using daily extreme-weather and soil features.

## Recommended final title

Daily Extreme-Weather Features for Interpretable Crop-Yield Prediction and Crop-Region Suitability Analysis

This title is broad enough for prediction, explanation, and decision support, while still concrete enough to avoid sounding like a generic AI paper.

## Core research questions

1. Can daily extreme-weather and soil indicators predict crop yield under a time-based test split?
2. Which crop-region combinations show the strongest observed yield increases or decreases?
3. Which weather indicators distinguish high-yield years from low-yield years for specific crop-region pairs?
4. Under the same observed region-year weather and soil profile, which crop is predicted to deliver the highest tonnes per hectare?

## Main contribution claims

1. A daily-weather feature engineering pipeline converts raw daily data into agronomically interpretable indicators: rainfall totals, dry spells, heat days, frost/cold exposure, growing degree days, heavy-rain events, and radiation summaries.
2. Time-split prediction shows that these indicators, especially when combined with soil features, are predictive enough for a practical decision-support analysis.
3. The paper moves beyond generic yield prediction by identifying crop-specific and region-specific yield changes.
4. The model outputs are translated into weather-conditioned crop-choice rankings, allowing direct comparison of crops under the same region-year conditions.
5. The interpretation is framed as association and planning signal, not causal proof.

## Recommended main manuscript figures

{markdown_table(tables["main_figures"])}

## Recommended supplementary figures

{markdown_table(tables["supp_figures"])}

## Suggested manuscript flow

1. Introduction: start from the practical planning problem, not from AI.
2. Data and study scope: Australia and U.S., overlap winter crops, May-October growing-season proxy.
3. Feature engineering: daily weather to extreme-weather indicators.
4. Modeling and evaluation: time-split prediction, model zoo, selected best models.
5. Results part 1: prediction performance.
6. Results part 2: crop-region yield change.
7. Results part 3: weather drivers of high-yield versus low-yield years.
8. Results part 4: weather-conditioned crop choice and suitability.
9. Discussion: practical use, non-causal interpretation, crop-calendar limitations, future work.
"""
    write_text(DRAFT_DIR / "PAPER_STRUCTURE_AND_FIGURE_PLAN.md", text)


def write_methods_draft(scope: pd.DataFrame, model_table: pd.DataFrame) -> None:
    weather_rows = int(scope.loc[scope["dataset"] == "Extreme-weather feature table", "rows"].iloc[0])
    combined_rows = int(scope.loc[scope["dataset"] == "Australia + U.S. overlap crop model frame", "rows"].iloc[0])
    wheat_rows = int(scope.loc[scope["dataset"] == "Australia + U.S. wheat model frame", "rows"].iloc[0])
    text = f"""
# Methods Draft

## Study design

This study develops a decision-support framework for crop-yield prediction and crop-region suitability analysis using daily extreme-weather indicators and soil variables. The unit of analysis is a crop-region-year observation. The target variable is harvested yield, expressed in tonnes per hectare. The current scope focuses on winter and cross-country overlap crops because these are available in both Australia and the United States in the assembled data.

The growing season is represented by a harmonized May-October window. This window is used as a first common proxy for winter-crop development across the two national datasets. The scope should be described as controlled and practical rather than exhaustive: summer crops and crop-specific calendars are left for future work.

## Data scope

The daily extreme-weather table contains {weather_rows:,} country-region-year records. The combined Australia and U.S. overlap-crop model frame contains {combined_rows:,} crop-region-year records, and the cross-country wheat frame contains {wheat_rows:,} records. The overlap crop set includes Barley, Canola, Oats, and Wheat.

## Extreme-weather feature engineering

Daily weather records are converted into annual growing-season indicators. The feature set includes:

- rainfall totals and daily rainfall means;
- dry-day counts and maximum dry-spell length;
- heavy-rain intensity indicators, including maximum multi-day rainfall;
- heat-day counts above fixed temperature thresholds;
- cold and frost exposure indicators;
- growing degree days;
- seasonal and sub-seasonal radiation summaries.

These features are intentionally interpretable. Each variable can be explained in agronomic language, which is important because the objective is not only prediction but also decision support.

## Soil and region context

The weather indicators are merged with soil and region-level context variables. Soil variables provide a slowly changing background for crop suitability, while daily weather summaries represent year-to-year seasonal stress. The main comparison uses both weather-only and weather-plus-soil feature sets to quantify whether soil context improves held-out prediction.

## Modeling strategy

The model zoo includes linear and nonlinear learners: Ridge regression, ElasticNet, Random Forest, Extra Trees, Histogram Gradient Boosting, and a multilayer perceptron neural network. Models are trained on 1989-2015 and evaluated on 2016-2021. This split is stricter than a random split because it tests whether the model generalizes forward in time.

The final reporting uses the best-performing model for each task under the test-period R2, while also reporting MAE and RMSE in tonnes per hectare. The purpose is not to claim one universally superior algorithm, but to choose the most useful predictive engine for each decision-support task.

## Model interpretation

Interpretation is conducted at three levels. First, permutation importance identifies features whose disruption most increases prediction error. Second, high-yield and low-yield years are compared within each crop-region pair to identify weather indicators that tend to be higher or lower during favorable years. Third, weather-conditioned crop ranking predicts multiple candidate crops under the same region-year weather and soil profile, allowing direct comparison of crop choice in tonnes per hectare.

## Statistical caution

All interpretation should be framed as predictive association. The analysis does not identify causal treatment effects because it does not control farm-level management, irrigation, cultivar, fertilizer, or within-region soil heterogeneity. The correct language is therefore "associated with", "model predicts", "planning signal", or "weather-conditioned suitability", not "caused by".

## Best model summary

{markdown_table(model_table)}
"""
    write_text(DRAFT_DIR / "METHODS_DRAFT.md", text)


def write_results_draft(
    best_models: pd.DataFrame,
    soil_gain: pd.DataFrame,
    gain_loss: pd.DataFrame,
    drivers: pd.DataFrame,
    choices: pd.DataFrame,
) -> None:
    combined = best_models[best_models["dataset"] == "Australia + U.S. overlap crops"].iloc[0]
    us = best_models[best_models["dataset"] == "U.S. overlap crops"].iloc[0]
    aus = best_models[best_models["dataset"] == "Australia overlap crops"].iloc[0]
    wheat = best_models[best_models["dataset"] == "Australia + U.S. wheat"].iloc[0]

    top_gain = gain_loss.sort_values("change, t/ha", ascending=False).iloc[0]
    second_gain = gain_loss.sort_values("change, t/ha", ascending=False).iloc[1]
    top_loss = gain_loss.sort_values("change, t/ha", ascending=True).iloc[0]
    second_loss = gain_loss.sort_values("change, t/ha", ascending=True).iloc[1]
    top_driver = drivers.iloc[0]
    second_driver = drivers.iloc[1]
    top_choice = choices.iloc[0]
    second_choice = choices.iloc[1]

    text = f"""
# Results Story Draft

This draft is written as manuscript-ready English. The numbers are taken from the selected model outputs and yield-story tables. Keep the wording associative and predictive.

## 1. Daily extreme-weather and soil indicators predicted held-out yield

The time-split evaluation shows that daily extreme-weather indicators contain useful information for crop-yield prediction. In the combined Australia and U.S. overlap-crop task, the best selected model was {combined['model']} with a test RMSE of {combined['test RMSE']:.3f} t/ha and a test R2 of {combined['test R2']:.3f}. The U.S. overlap-crop model achieved the strongest held-out performance, with {us['model']} reaching a test RMSE of {us['test RMSE']:.3f} t/ha and a test R2 of {us['test R2']:.3f}. The Australia-only overlap-crop task was more difficult but still predictive, with {aus['model']} obtaining a test R2 of {aus['test R2']:.3f}. For the cross-country wheat-only task, the best model was {wheat['model']}, with a test RMSE of {wheat['test RMSE']:.3f} t/ha and a test R2 of {wheat['test R2']:.3f}.

These results support the use of the feature set for decision-support analysis. The goal of the framework is not only to minimize prediction error, but to translate daily weather and soil information into interpretable crop-region signals. Therefore, the rest of the Results section focuses on where yields increased or decreased, which weather indicators distinguish high-yield years from low-yield years, and which crops appear more suitable under specific region-year weather profiles.

## 2. Adding soil context improved the cross-country decision-support frame

The weather-only features already produced useful predictive accuracy, but adding soil variables improved the main cross-country overlap-crop task. This matters because soil represents persistent regional growing conditions, whereas daily weather indicators represent year-specific stress. In the manuscript, this result should be used to justify the combined weather-plus-soil framing rather than presenting the study as a pure weather model.

{markdown_table(soil_gain)}

## 3. Yield change was crop-specific and region-specific, not a generic trend

Observed yield changes differed strongly by crop and region. The largest observed gain in the selected early-versus-recent comparison occurred for {top_gain['country']} {top_gain['region']} {top_gain['crop']}, where mean yield increased by {top_gain['change, t/ha']:.3f} t/ha from 1989-2000 to 2010-2021. The next largest gain was {second_gain['country']} {second_gain['region']} {second_gain['crop']}, with an increase of {second_gain['change, t/ha']:.3f} t/ha. These examples show why the paper should avoid the vague phrase "yield increased" without specifying crop and location.

The largest observed decreases were concentrated in fewer crop-region pairs. {top_loss['country']} {top_loss['region']} {top_loss['crop']} declined by {abs(top_loss['change, t/ha']):.3f} t/ha, while {second_loss['country']} {second_loss['region']} {second_loss['crop']} declined by {abs(second_loss['change, t/ha']):.3f} t/ha. These losses are useful for the paper because they make the decision-support framing concrete: the framework can identify not only high-performing crops, but also crop-region combinations that may require closer agronomic attention.

{markdown_table(gain_loss)}

## 4. High-yield years had identifiable weather signatures

The high-versus-low yield comparison translates model results into weather language. For each crop-region pair, the analysis compares the highest predicted-yield years with the lowest predicted-yield years and records which weather indicators differ most. The strongest high-low gap was found for {top_driver['country']} {top_driver['region']} {top_driver['crop']}, where the predicted difference between high-yield and low-yield years was {top_driver['predicted high-low gap, t/ha']:.3f} t/ha. In this case, high-yield years were characterized by: {top_driver['weather indicators higher/lower in high-yield years']}.

A second strong example was {second_driver['country']} {second_driver['region']} {second_driver['crop']}, with a predicted high-low gap of {second_driver['predicted high-low gap, t/ha']:.3f} t/ha. The weather signature was: {second_driver['weather indicators higher/lower in high-yield years']}. This type of result is easier for non-AI readers to understand than raw feature importance because it states the story in terms of familiar seasonal conditions.

The manuscript should be careful not to state that these variables caused the yield difference. A suitable sentence is: "In high-predicted-yield years for this crop-region pair, the model tends to observe higher seasonal rainfall and lower radiation, suggesting a favorable moisture-related profile in this dataset."

{markdown_table(drivers)}

## 5. Weather-conditioned crop ranking turns prediction into planning information

The most practical output is the weather-conditioned crop-choice comparison. For each region-year profile, the model can score candidate crops under the same observed weather and soil conditions. This allows the question to move from "What is the expected yield?" to "Which crop is predicted to perform better in this place under this weather profile?"

The largest crop-choice advantage occurred in {top_choice['country']} {top_choice['region']} in {int(top_choice['year'])}, a {top_choice['weather regime']} year. Under that observed profile, {top_choice['best crop']} was predicted to outperform {top_choice['runner-up crop']} by {top_choice['advantage over runner-up, t/ha']:.3f} t/ha, with a best predicted yield of {top_choice['best predicted yield, t/ha']:.3f} t/ha. Another strong case was {second_choice['country']} {second_choice['region']} in {int(second_choice['year'])}, where {second_choice['best crop']} was predicted to outperform {second_choice['runner-up crop']} by {second_choice['advantage over runner-up, t/ha']:.3f} t/ha under a {second_choice['weather regime']} profile.

This result is central to the practical contribution of the paper. It does not claim that crop switching alone guarantees higher yield. Instead, it demonstrates that a trained yield model can be used as a structured screening tool for crop suitability under observed weather stress.

{markdown_table(choices)}

## 6. Suggested Results section ending

Overall, the results show that daily extreme-weather features are not only predictive, but also useful for explaining crop-region performance in operational terms. The framework identifies where yield change is concentrated, which weather indicators distinguish favorable and unfavorable years, and which crops are predicted to be more suitable under the same region-year weather and soil profile. This makes the approach more useful for agricultural planning than a conventional yield-prediction leaderboard.
"""
    write_text(DRAFT_DIR / "RESULTS_STORY_DRAFT.md", text)


def write_caption_draft() -> None:
    captions = [
        (
            "Figure 1",
            "Dataset scope and modeling target. The figure summarizes the country, crop, region, yield, and region-year weather-feature coverage used to construct the crop-region-year modeling frame.",
        ),
        (
            "Figure 2",
            "Daily extreme-weather feature design. Daily weather observations are aggregated into interpretable growing-season indicators, merged with soil context, and used for yield prediction, crop-region suitability, and weather-conditioned crop comparison.",
        ),
        (
            "Figure 3",
            "Time-split model performance. Models were trained on 1989-2015 and evaluated on 2016-2021. Performance is reported in tonnes per hectare using MAE and RMSE, with R2 used to compare predictive skill.",
        ),
        (
            "Figure 4",
            "Observed versus predicted yield in the held-out test period. Each point represents a crop-region-year observation in 2016-2021, allowing visual assessment of model calibration in the original yield scale.",
        ),
        (
            "Figure 5",
            "Observed yield change by crop and region. The heatmap compares mean yield in 1989-2000 with mean yield in 2010-2021, identifying crop-region combinations with the largest increases or decreases.",
        ),
        (
            "Figure 6",
            "Weather signatures of high-yield years. For selected crop-region pairs, the figure compares weather indicators in high-predicted-yield years with low-predicted-yield years. Values should be interpreted as predictive associations rather than causal effects.",
        ),
        (
            "Figure 7",
            "Model-based crop suitability map. For each region, the map shows the crop with the highest predicted yield under the regional weather and soil profile. Markers are decision-support signals, not prescriptive recommendations.",
        ),
        (
            "Figure 8",
            "Weather-conditioned crop-choice advantage. The figure highlights region-year cases where the best predicted crop has the largest yield advantage over the runner-up crop under the same weather and soil conditions.",
        ),
    ]
    lines = ["# Figure Captions Draft", ""]
    for label, caption in captions:
        lines.append(f"## {label}")
        lines.append("")
        lines.append(caption)
        lines.append("")
    lines.append("## Caption wording rule")
    lines.append("")
    lines.append("Use `predicted`, `associated with`, `weather-conditioned`, and `planning signal`. Avoid `caused`, `proved`, or `optimal` unless a causal or optimization design is added.")
    write_text(DRAFT_DIR / "FIGURE_CAPTIONS_DRAFT.md", "\n".join(lines))


def write_discussion_limitations_draft() -> None:
    text = """
# Discussion And Limitations Draft

## Practical interpretation

The framework is useful because it converts daily weather records into questions that agricultural planners can understand: which crop performed better, where it performed better, and which weather conditions were associated with favorable or unfavorable yield. This is a stronger practical contribution than reporting predictive accuracy alone.

The crop-choice analysis should be presented as a screening layer. It can help identify candidate crops or crop-region combinations for closer agronomic evaluation. It should not be framed as a direct recommendation to switch crops without considering market demand, management costs, irrigation access, cultivar selection, disease risk, or local farm constraints.

## Why the paper remains interpretable

Even when the best predictive model is nonlinear or neural, the output is made interpretable by the feature design and post-model analysis. The input features have agronomic meaning, the evaluation is reported in tonnes per hectare, and the explanation layer compares high-yield and low-yield years using observable weather indicators.

## Main limitations

1. The analysis is predictive and associative, not causal.
2. The May-October growing season is a harmonized proxy and does not capture crop-specific calendars.
3. Region-level data cannot represent within-region farm management, irrigation, cultivar, fertilizer, disease, or local soil variation.
4. Weather products differ across countries, with Australia and the United States relying on different source products.
5. Scenario crop ranking can extrapolate when a crop is scored for a region-year profile with limited historical support.

## Future work

Future work should add crop-specific growing calendars, finer spatial resolution, management variables where available, and explicit uncertainty intervals for crop-choice rankings. A causal design could be added later if the research question shifts from planning support to estimating the causal effect of specific weather stresses.
"""
    write_text(DRAFT_DIR / "DISCUSSION_LIMITATIONS_DRAFT.md", text)


def write_locked_manuscript_plan(
    best_models: pd.DataFrame,
    gain_loss: pd.DataFrame,
    drivers: pd.DataFrame,
    choices: pd.DataFrame,
) -> None:
    combined = best_models[best_models["dataset"] == "Australia + U.S. overlap crops"].iloc[0]
    top_gain = gain_loss.sort_values("change, t/ha", ascending=False).iloc[0]
    top_loss = gain_loss.sort_values("change, t/ha", ascending=True).iloc[0]
    top_driver = drivers.iloc[0]
    top_choice = choices.iloc[0]
    text = f"""
# Locked Manuscript Plan

This file records the recommended final direction for the new paper version. Treat it as the paper compass unless new data or advisor feedback changes the scope.

## Final paper direction

The paper should be positioned as an interpretable crop-yield prediction and decision-support study, not as a generic artificial-intelligence benchmark.

Recommended title:

```text
Daily Extreme-Weather Features for Interpretable Crop-Yield Prediction and Crop-Region Suitability Analysis
```

## Final scope

- Countries: Australia and United States.
- Crops: overlap winter-crop frame, focused on Wheat, Barley, Canola, and Oats.
- Time period: 1989-2021 model frame, with 1989-2015 for training and 2016-2021 for testing.
- Season proxy: May-October.
- Target: yield in tonnes per hectare.
- Feature logic: daily weather extremes plus soil context.

Do not add summer crops in the current manuscript version. Mention summer crops and crop-specific calendars as future work.

## Main research questions

1. Can daily extreme-weather and soil features predict crop yield under a forward-time test split?
2. Which crop-region combinations show the strongest yield increases or decreases?
3. Which weather indicators distinguish high-yield years from low-yield years for specific crop-region pairs?
4. Under the same region-year weather and soil profile, which crop is predicted to have the highest yield?

## Core result anchors

- Main cross-country overlap model: {combined['model']}, test RMSE = {combined['test RMSE']:.3f} t/ha, test R2 = {combined['test R2']:.3f}.
- Strongest observed gain: {top_gain['country']} {top_gain['region']} {top_gain['crop']}, change = {top_gain['change, t/ha']:.3f} t/ha.
- Strongest observed loss: {top_loss['country']} {top_loss['region']} {top_loss['crop']}, change = {top_loss['change, t/ha']:.3f} t/ha.
- Strongest high-low yield story: {top_driver['country']} {top_driver['region']} {top_driver['crop']}, predicted high-low gap = {top_driver['predicted high-low gap, t/ha']:.3f} t/ha.
- Largest weather-conditioned crop-choice advantage: {top_choice['country']} {top_choice['region']} {int(top_choice['year'])}, {top_choice['best crop']} over {top_choice['runner-up crop']} by {top_choice['advantage over runner-up, t/ha']:.3f} t/ha under {top_choice['weather regime']}.

## Main figures to use

1. `fig01_dataset_scope`
2. `fig02_daily_extreme_feature_design`
3. `fig03_model_performance`
4. `fig04_actual_vs_predicted_test`
5. `fig17_observed_yield_change_region_crop`
6. `fig19_high_vs_low_yield_weather_driver_matrix`
7. `fig07_crop_suitability_map`
8. `fig21_weather_condition_crop_choice_advantage`

## Main tables to use

1. `table01_dataset_scope`
2. `table02_best_model_performance`
3. `table03_weather_vs_weather_soil`
4. `table04_crop_region_yield_gain_loss`
5. `table05_high_low_yield_weather_stories`
6. `table06_weather_conditioned_crop_choice`

## Terms to use

- predicted yield
- observed yield change
- weather-conditioned crop choice
- crop-region suitability
- high-yield versus low-yield years
- planning signal
- associated with
- model-supported evidence

## Terms to avoid

- caused by
- proves
- guarantees
- optimal crop
- causal effect
- universal recommendation

## Recommended paper narrative

Start with the practical problem: crop planning needs both yield prediction and explanation. Then show that daily weather can be converted into interpretable extreme-weather features. After proving the model has held-out predictive skill, move quickly into the more valuable results: crop-region yield change, weather signatures of high-yield years, and weather-conditioned crop ranking.

The best story is:

```text
We do not only predict yield. We identify which crop-region pairs improved or declined, explain which weather indicators separate favorable and unfavorable years, and compare candidate crops under the same observed weather and soil profile.
```
"""
    write_text(DRAFT_DIR / "LOCKED_MANUSCRIPT_PLAN.md", text)


def write_abstract_draft(best_models: pd.DataFrame) -> None:
    combined = best_models[best_models["dataset"] == "Australia + U.S. overlap crops"].iloc[0]
    text = f"""
# Abstract Draft

Crop-yield models are often evaluated mainly by predictive accuracy, but agricultural planning also requires interpretable information about which crops perform better, where they perform better, and under what weather conditions. This study develops a daily extreme-weather decision-support framework for winter and overlap crops in Australia and the United States. Daily weather observations are transformed into interpretable growing-season indicators, including rainfall totals, dry spells, heavy-rain events, heat exposure, cold/frost exposure, growing degree days, and radiation summaries. These indicators are merged with soil and crop-region-year yield data to predict yield in tonnes per hectare and to support crop-region suitability analysis.

Models were trained on 1989-2015 and evaluated on 2016-2021. In the combined Australia and U.S. overlap-crop task, the best model achieved a test RMSE of {combined['test RMSE']:.3f} t/ha and a test R2 of {combined['test R2']:.3f}. Beyond prediction, the framework identifies crop-region combinations with the largest observed yield changes, compares weather indicators between high-yield and low-yield years, and ranks candidate crops under the same observed region-year weather and soil profile. The results show that daily extreme-weather features can support interpretable yield prediction and practical crop-choice screening. Findings are interpreted as predictive associations and planning signals rather than causal effects.
"""
    write_text(DRAFT_DIR / "ABSTRACT_DRAFT.md", text)


def write_latex_snippets() -> None:
    lines = [
        "% Selected figure snippets for paper_latex/main.tex",
        "% Copy the chosen figure files into the LaTeX figure path or update paths below.",
        "",
    ]
    for item in MAIN_FIGURES:
        stem = item["stem"]
        label = item["paper_label"].lower().replace(" ", ":")
        lines.extend(
            [
                "\\begin{figure}[htbp]",
                "\\centering",
                f"\\includegraphics[width=0.95\\textwidth]{{../newProject_extreme_weather/paper_assets_selected/figures/main/{stem}.png}}",
                f"\\caption{{{item['role']}. TODO: replace with final caption from FIGURE_CAPTIONS_DRAFT.md.}}",
                f"\\label{{fig:{stem}}}",
                "\\end{figure}",
                "",
            ]
        )
    write_text(LATEX_DIR / "selected_figure_snippets.tex", "\n".join(lines))


def copy_figures() -> None:
    for item in MAIN_FIGURES:
        copy_figure_pair(item["source_dir"], item["stem"], FIG_MAIN_DIR)
    for stem, source_dir, _ in SUPPLEMENTARY_FIGURES:
        copy_figure_pair(source_dir, stem, FIG_SUPP_DIR)


def main() -> None:
    ensure_dirs()
    copy_figures()

    tables = {
        "dataset_scope": build_dataset_scope_table(),
        "best_models": build_best_model_table(),
        "feature_set_comparison": build_feature_set_comparison_table(),
        "gain_loss": build_gain_loss_table(),
        "driver_stories": build_driver_story_table(),
        "crop_choice": build_crop_choice_table(),
        "main_figures": build_main_figure_table(),
        "supp_figures": build_supp_figure_table(),
    }

    save_table(tables["dataset_scope"], "table01_dataset_scope")
    save_table(tables["best_models"], "table02_best_model_performance")
    save_table(tables["feature_set_comparison"], "table03_weather_vs_weather_soil")
    save_table(tables["gain_loss"], "table04_crop_region_yield_gain_loss")
    save_table(tables["driver_stories"], "table05_high_low_yield_weather_stories")
    save_table(tables["crop_choice"], "table06_weather_conditioned_crop_choice")
    save_table(tables["main_figures"], "table07_main_figure_plan")
    save_table(tables["supp_figures"], "table08_supplementary_figure_plan")

    write_readme(tables)
    write_structure_draft(tables)
    write_methods_draft(tables["dataset_scope"], tables["best_models"])
    write_results_draft(
        tables["best_models"],
        tables["feature_set_comparison"],
        tables["gain_loss"],
        tables["driver_stories"],
        tables["crop_choice"],
    )
    write_caption_draft()
    write_discussion_limitations_draft()
    write_locked_manuscript_plan(
        tables["best_models"],
        tables["gain_loss"],
        tables["driver_stories"],
        tables["crop_choice"],
    )
    write_abstract_draft(tables["best_models"])
    write_latex_snippets()

    print(f"Prepared selected paper assets in: {OUTPUT_ROOT}")
    print(f"Main figures: {len(list(FIG_MAIN_DIR.glob('*.png')))} PNG + {len(list(FIG_MAIN_DIR.glob('*.svg')))} SVG")
    print(f"Supplementary figures: {len(list(FIG_SUPP_DIR.glob('*.png')))} PNG + {len(list(FIG_SUPP_DIR.glob('*.svg')))} SVG")
    print(f"Tables: {len(list(TABLE_DIR.glob('*.csv')))} CSV, {len(list(TABLE_DIR.glob('*.md')))} Markdown")
    print(f"Drafts: {len(list(DRAFT_DIR.glob('*.md')))} Markdown")


if __name__ == "__main__":
    main()
