# Daily Extreme-Weather Crop-Yield Decision Support

This repository contains the `fastrefs` submission package for:

**Daily Extreme-Weather Features for Interpretable Crop-Yield Prediction and Crop-Region Suitability Analysis**

The paper studies winter/overlap crops in Australia and the United States using daily extreme-weather indicators, soil context, forward-time model validation, feature-group ablation, crop-region yield stories, and weather-conditioned crop-choice analysis.

## Repository Contents

```text
paper/
  latex_zip/
    overleaf_extreme_weather_fastrefs.zip
  pdf/
    DAP2.pdf
  latex_source/
    main.tex
    references.bib
    tables/
    figures/

code/
  requirements-modeling.txt
  scripts/

dataset/
  processed/

results/
  csv/
    decision_support_models/
    method_reproducibility/
```

## Main Paper Files

- `paper/pdf/DAP2.pdf`: compiled paper PDF.
- `paper/latex_zip/overleaf_extreme_weather_fastrefs.zip`: Overleaf-ready LaTeX package.
- `paper/latex_source/`: extracted LaTeX source for browsing in GitHub.

## Code

The `code/scripts/` folder contains the main scripts used to build the extreme-weather features, model frames, validation-period learning-algorithm selection results, paper tables, and decision-support figures.

Key scripts:

- `build_extreme_weather_features.py`
- `build_extreme_weather_model_frames.py`
- `train_decision_support_models.py`
- `build_method_reproducibility_tables.py`
- `build_conference_enhancements.py`
- `create_decision_support_figures.py`
- `create_yield_story_analysis.py`
- `prepare_selected_paper_assets.py`

The modeling environment used during development was Python 3.12 with scikit-learn. Install the lightweight modeling requirements with:

```powershell
pip install -r code/requirements-modeling.txt
```

## Dataset

The `dataset/` folder contains the processed data used for the fastrefs package:

- `dataset/processed/`: generated extreme-weather features and crop-region-year model frames used by the modeling scripts.

Raw public-source datasets are not included in this repository. The paper cites the original data services, including ABARES, USDA NASS Quick Stats, SILO, and NASA POWER.

## Results CSVs

The `results/csv/` folder stores generated result tables used by the paper, including validation-period algorithm-selection results, feature-group ablation, held-out predictions, permutation importance, crop-choice advantages, and yield-story summaries.

## Notes

The analysis should be interpreted as predictive and associative. Crop-choice outputs are decision-support screening signals, not direct planting recommendations.
