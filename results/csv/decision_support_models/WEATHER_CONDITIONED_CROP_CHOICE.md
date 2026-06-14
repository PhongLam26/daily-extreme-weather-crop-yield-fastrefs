# Weather-Conditioned Crop Choice Outputs

These files answer the practical question:

```text
Under this region-year weather and soil profile, which crop is predicted to have the highest yield?
```

The model keeps the same country, region, year, daily extreme-weather features, and soil features.
It then tries each candidate overlap crop:

```text
Wheat, Barley, Canola, Oats
```

and ranks them by predicted `yield_t_ha`.

## Main CSV

```text
weather_condition_crop_choice_by_region_year.csv
```

Each row is one region-year.

Important columns:

| Column | Meaning |
|---|---|
| `country` | Australia or United States. |
| `region` | Region/state. |
| `year_start` | Year of the May-October weather window. |
| `weather_regime` | Interpretable weather type assigned from daily extreme indicators. |
| `best_crop` | Crop with the highest model-predicted yield under that weather/soil profile. |
| `best_predicted_yield_t_ha` | Predicted yield of the best crop. |
| `runner_up_crop` | Second-best crop. |
| `runner_up_predicted_yield_t_ha` | Predicted yield of the second-best crop. |
| `best_crop_yield_advantage_t_ha` | Best crop minus runner-up in t/ha. |
| `heat_days_35` | Number of very hot days. |
| `max_dry_spell_1mm` | Longest dry spell. |
| `max_3day_rain` | Largest 3-day rainfall total. |
| `rain_last_14d_before_harvest` | Late-season rain proxy. |
| `frost_days_0` | Frost days. |

## Weather Regime Summary

```text
weather_regime_crop_choice_summary.csv
```

This aggregates predictions by country, weather regime, and crop.
Use it to discuss questions like:

- Under hot-dry years, which crop is predicted to yield more?
- Under wet/storm-risk years, which crop is most often ranked first?
- Does the best crop under a weather regime differ between Australia and the United States?

## New Figures

```text
fig13_weather_conditioned_crop_winner_timeline.png
fig14_weather_regime_crop_yield_matrix.png
fig15_weather_regime_best_crop_share.png
fig16_latest_year_weather_conditioned_crop_map.png
```

Recommended use:

- Figure 13: shows the winning crop for every region-year weather condition.
- Figure 14: shows predicted t/ha by crop under each weather regime.
- Figure 15: shows which crop most often wins under each weather regime.
- Figure 16: shows a map for the latest modeled weather year.

## Caveat For Paper

This is a model-based planning signal, not causal proof.
It should be written as:

```text
Under similar observed weather and soil conditions, the model predicts crop X
to have a higher yield than crop Y.
```

Do not write:

```text
Crop X will definitely outperform crop Y.
```
