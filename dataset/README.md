# Dataset

This folder contains the dataset files used by the fastrefs paper package.

```text
raw/
  australia/
  united_states/

processed/
  extreme_weather_features_aus_us_region_year.csv
  model_frames/
```

## Raw Data

The `raw/` folder contains copied daily weather tables for the Australia and United States regions used in the paper. The paper cites the original public data services, including SILO and NASA POWER.

## Processed Data

The `processed/` folder contains generated features and model frames:

- country-region-year extreme-weather features
- Australia overlap-crop model frames
- U.S. overlap-crop model frames
- combined Australia + U.S. overlap-crop and wheat model frames

These processed files are the direct inputs for the modeling scripts in `../code/scripts/`.
