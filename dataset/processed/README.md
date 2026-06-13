# Processed Data

Folder này chuẩn bị chứa dữ liệu đã xử lý cho nhánh daily extreme-weather features.

Expected files sau khi chạy script:

```text
extreme_weather_features_aus_us_region_year.csv
extreme_weather_features_aus_region_year.csv
extreme_weather_features_us_region_year.csv
model_frames/
```

Các file này là weather features ở mức:

```text
country-region-year_start
```

Bước tiếp theo là merge chúng vào annual yield/soil datasets để tạo modelling frame ở mức:

```text
country-region-crop-year_start
```

Script `scripts/build_extreme_weather_model_frames.py` sẽ tạo các modelling frames trong:

```text
data/processed/model_frames/
```
