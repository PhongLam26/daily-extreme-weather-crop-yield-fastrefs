# Decision-Support Model Summary

Goal: predict crop yield in tonnes per hectare and support practical interpretation of why yield may rise or fall.

Evaluation uses time split: train 1989-2015, test 2016-2021.

## Best Models

| Dataset | Feature set | Model | Test MAE | Test RMSE | Test R2 |
|---|---|---|---:|---:|---:|
| aus_overlap | extreme_weather_soil | ANN_MLP | 0.595 | 0.784 | 0.604 |
| aus_us_overlap | extreme_weather_soil | ANN_MLP | 0.463 | 0.610 | 0.761 |
| aus_us_wheat | extreme_weather_soil | ElasticNet | 0.497 | 0.674 | 0.709 |
| us_overlap | extreme_weather_soil | ExtraTrees | 0.437 | 0.533 | 0.802 |

## Interpretation Notes

- Linear models are included for transparent baselines.
- Tree and boosting models are included for stronger tabular prediction.
- ANN_MLP is included as an artificial neural network baseline, but it should only be emphasized if it wins under the time split.
- Feature importance is estimated by permutation importance on the test period.
- Crop ranking outputs are model-based suitability signals, not causal proof or farm-level planting advice.
