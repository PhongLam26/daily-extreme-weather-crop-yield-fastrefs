# Practical Findings From First Decision-Support Run

This is the first modelling run for the daily extreme-weather direction.
It combines prediction, explanation, and crop-region planning signals.

## Main Evaluation Setup

- Train period: 1989-2015.
- Test period: 2016-2021.
- Target: `yield_t_ha`.
- Main feature set: daily extreme-weather indicators + soil + `country`, `region`, `crop`, `year_start`.
- Models tested: Ridge, ElasticNet, RandomForest, ExtraTrees, HistGradientBoosting, ANN_MLP.

## Best Models

| Dataset | Best model | Test MAE | Test RMSE | Test R2 |
|---|---:|---:|---:|---:|
| Australia overlap crops | ANN_MLP | 0.595 | 0.784 | 0.604 |
| Australia + U.S. overlap crops | ANN_MLP | 0.463 | 0.610 | 0.761 |
| Australia + U.S. Wheat | ElasticNet | 0.497 | 0.674 | 0.709 |
| U.S. overlap crops | ExtraTrees | 0.437 | 0.533 | 0.802 |

Interpretation:

- ANN is useful in the combined and Australia overlap setting, so it should stay in the benchmark.
- ElasticNet winning for Wheat is useful because it is easier to explain.
- ExtraTrees winning for U.S. overlap crops is consistent with tabular crop-yield data.
- The paper should say the model choice is empirical: the selected model is the one that performs best under time-split evaluation.

## Explanation Signals

Top permutation-importance signals in the combined overlap model:

1. `crop`
2. `soil_SILT_5_15`
3. `soil_SOC_5_15`
4. `season_rain_mean`
5. `radiation_sum_mid_y`
6. `heat_days_35`

Plain-language read:

- Crop choice matters strongly.
- Soil texture and organic carbon are important background conditions.
- Rainfall distribution and mid-season radiation affect yield prediction.
- Very hot days remain a practical yield-risk signal.

For Wheat-only, top signals include:

- `region`
- `soil_cec_100_200`
- `growing_degree_days_base5`
- `radiation_sum_mid_y`
- `radiation_sum_late_y`
- `dry_days_2mm`

Plain-language read:

- Wheat yield differences are strongly regional.
- Soil nutrient-exchange capacity, temperature accumulation, radiation, and dry-day exposure are relevant.

## Example Crop-Region Planning Signals

In the combined overlap-crop scenario ranking, the model predicts these crops as the best average t/ha option by region:

| Country | Region | Best predicted crop |
|---|---|---|
| Australia | New South Wales | Wheat |
| Australia | Queensland | Barley |
| Australia | South Australia | Barley |
| Australia | Tasmania | Wheat |
| Australia | Victoria | Wheat |
| Australia | Western Australia | Barley |
| United States | Colorado | Barley |
| United States | Illinois | Wheat |
| United States | Iowa | Barley |
| United States | Kansas | Wheat |
| United States | Minnesota | Barley |
| United States | Montana | Barley |
| United States | Nebraska | Wheat |
| United States | North Dakota | Barley |
| United States | Oklahoma | Wheat |
| United States | South Dakota | Wheat |
| United States | Texas | Barley |
| United States | Washington | Wheat |

This is a model-based suitability signal, not a direct recommendation to farmers.
It should be used as a planning discussion: which crop appears more productive under the observed regional weather and soil profile?

## Country-Crop Yield Advantage Signal

In the combined overlap scenario summary:

| Crop | Australia predicted t/ha | U.S. predicted t/ha | U.S. minus Australia |
|---|---:|---:|---:|
| Barley | 2.062 | 3.100 | 1.038 |
| Canola | 1.231 | 1.800 | 0.568 |
| Oats | 1.556 | 2.120 | 0.563 |
| Wheat | 2.175 | 2.785 | 0.610 |

This compares model-predicted yield across the observed country-region profiles.
It does not prove that one country is inherently better; climate, soil, region mix, and reporting scale all contribute.

## Risk-Threshold Examples

Examples from the threshold table:

- U.S. Barley: high `hot_dry_days_30_1mm` is associated with much lower average yield.
- Australia Wheat: high `hot_dry_days_30_1mm`, high `max_dry_spell_1mm`, and high `heat_days_35` are associated with lower average yield.
- Rainfall concentration can be harmful when rain arrives in fewer intense events instead of being spread across the season.

Some signals, such as frost appearing positive in a few groups, likely reflect regional confounding rather than a direct benefit of frost.
The paper should describe these as associations and use model explanations plus agronomic reasoning together.

## Recommended Paper Framing

The paper can now be framed as:

> We build a crop-yield decision-support model that predicts yield in t/ha, explains weather and soil stress signals associated with yield gains or losses, and ranks crop suitability across regions in Australia and the United States.

This supports all three goals:

1. Prediction accuracy.
2. Explanation of yield increase/decrease.
3. Practical planning: crop-region suitability and country-crop comparison.
