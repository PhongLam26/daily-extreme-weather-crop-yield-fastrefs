# Yield Story Analysis

This layer answers more detailed practical questions:

- Which crop-region combinations increased or decreased?
- Where are high-yield years most different from low-yield years?
- Which weather signals help explain the difference?
- In which region-year does crop choice matter most?

## Top observed yield gains

| country | region | crop | delta_observed_yield_t_ha | observed_change_status |
| --- | --- | --- | --- | --- |
| Australia | Tasmania | Wheat | 1.986 | increase |
| United States | Colorado | Barley | 1.923 | increase |
| Australia | Tasmania | Barley | 1.507 | increase |
| United States | Minnesota | Wheat | 1.179 | increase |
| United States | Illinois | Wheat | 1.149 | increase |
| Australia | Tasmania | Canola | 1.049 | increase |
| United States | Iowa | Wheat | 0.902 | increase |
| United States | North Dakota | Wheat | 0.807 | increase |

## Top observed yield losses

| country | region | crop | delta_observed_yield_t_ha | observed_change_status |
| --- | --- | --- | --- | --- |
| Australia | New South Wales | Oats | -0.350 | decrease |
| United States | Washington | Oats | -0.270 | decrease |
| United States | South Dakota | Barley | -0.163 | stable |
| Australia | New South Wales | Canola | -0.079 | stable |
| Australia | Queensland | Oats | -0.045 | stable |
| United States | Montana | Oats | -0.030 | stable |
| Australia | Victoria | Oats | -0.025 | stable |
| United States | Kansas | Oats | 0.081 | stable |

## Strongest high-vs-low yield stories

| country | region | crop | predicted_yield_gap_high_minus_low_t_ha | top_weather_driver_story |
| --- | --- | --- | --- | --- |
| United States | Colorado | Barley | 1.667 | higher mean min temp (+0.6 C); higher growing degree days (+114.3 degree-days); higher mean max temp (+0.7 C); more longest dry spell (+4.1 days) |
| Australia | New South Wales | Wheat | 1.624 | higher season rain (+255.1 mm); higher daily rain (+1.4 mm/day); more max 3-day rain (+27.8 mm); lower season radiation (-185.1 MJ/m2) |
| Australia | New South Wales | Barley | 1.492 | higher daily rain (+1.1 mm/day); higher season rain (+194.7 mm); lower late-season radiation (-113.4 MJ/m2); lower season radiation (-167.7 MJ/m2) |
| Australia | Tasmania | Wheat | 1.420 | lower late-season radiation (-56.8 MJ/m2); lower season radiation (-80.7 MJ/m2); higher growing degree days (+70.3 degree-days); higher mean max temp (+0.4 C) |
| Australia | Victoria | Wheat | 1.389 | higher season rain (+163.6 mm); higher daily rain (+0.9 mm/day); more late-season rain (+29.0 mm); more max 3-day rain (+24.7 mm) |
| Australia | Victoria | Barley | 1.334 | lower late-season radiation (-97.9 MJ/m2); less hot-dry days (-3.7 days); higher season rain (+122.7 mm); higher daily rain (+0.7 mm/day) |
| United States | Illinois | Wheat | 1.280 | higher growing degree days (+150.1 degree-days); higher mean min temp (+0.8 C); higher mean max temp (+0.9 C); higher season radiation (+103.3 MJ/m2) |
| United States | Iowa | Wheat | 1.255 | more hot-dry days (+16.4 days); higher growing degree days (+174.5 degree-days); higher mean max temp (+1.3 C); lower daily rain (-0.8 mm/day) |
| Australia | New South Wales | Oats | 1.247 | higher season rain (+243.9 mm); higher daily rain (+1.3 mm/day); more max 3-day rain (+30.2 mm); lower season radiation (-205.6 MJ/m2) |
| Australia | Queensland | Barley | 1.239 | higher daily rain (+0.6 mm/day); higher season rain (+106.1 mm); more max 7-day rain (+44.7 mm); more max 3-day rain (+36.2 mm) |

## Largest weather-conditioned crop-choice advantages

| country | region | largest_advantage_year | largest_advantage_weather_regime | largest_advantage_best_crop | largest_advantage_runner_up_crop | largest_advantage_t_ha |
| --- | --- | --- | --- | --- | --- | --- |
| United States | Colorado | 2010 | Dry-spell stress | Barley | Wheat | 3.778 |
| Australia | Tasmania | 2008 | Cold/frost stress | Wheat | Barley | 1.196 |
| United States | Washington | 2003 | Dry-spell stress | Wheat | Barley | 0.958 |
| United States | North Dakota | 2018 | Cold/frost stress | Barley | Wheat | 0.900 |
| United States | Iowa | 2001 | Wet/storm risk | Barley | Wheat | 0.827 |
| United States | Montana | 1991 | Cold/frost stress | Barley | Wheat | 0.791 |
| Australia | New South Wales | 1991 | Wet/storm risk | Wheat | Oats | 0.764 |
| United States | Illinois | 2017 | Wet/storm risk | Barley | Wheat | 0.630 |
| Australia | Victoria | 2016 | Wet/storm risk | Wheat | Barley | 0.624 |
| Australia | Western Australia | 2019 | Hot-dry stress | Barley | Oats | 0.623 |

## Paper wording

Use wording such as:

```text
In high-predicted-yield years for this crop-region pair, the model tends to see more/less of these weather indicators.
```

Avoid causal wording unless a causal design is added.
