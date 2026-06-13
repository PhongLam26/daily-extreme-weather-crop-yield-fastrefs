from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_AUS = PROJECT_ROOT / "data" / "raw" / "australia" / "australia_silo_daily_all_states_1989_present.csv"
RAW_US = PROJECT_ROOT / "data" / "raw" / "united_states" / "us_nasa_power_daily_all_states_1989_present.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"

GROWING_MONTHS = [5, 6, 7, 8, 9, 10]
WINDOW_BY_MONTH = {
    5: "early",
    6: "early",
    7: "mid",
    8: "mid",
    9: "late",
    10: "late",
}


def season_label(year_start: int) -> str:
    return f"{int(year_start)}-{str((int(year_start) + 1) % 100).zfill(2)}"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def max_consecutive(values: pd.Series) -> int:
    max_run = 0
    current = 0
    for value in values.fillna(False).astype(bool):
        if value:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return int(max_run)


def count_run_events(values: pd.Series, min_length: int) -> int:
    events = 0
    current = 0
    for value in values.fillna(False).astype(bool):
        if value:
            current += 1
        else:
            if current >= min_length:
                events += 1
            current = 0
    if current >= min_length:
        events += 1
    return int(events)


def rolling_sum_max(values: pd.Series, window: int) -> float:
    if values.empty:
        return np.nan
    return float(values.fillna(0).rolling(window=window, min_periods=1).sum().max())


def safe_sum(values: pd.Series) -> float:
    if values.dropna().empty:
        return np.nan
    return float(values.sum(skipna=True))


def rainfall_concentration(rain: pd.Series) -> float:
    rain = rain.fillna(0)
    total = float(rain.sum())
    if total <= 0:
        return 0.0
    return float((rain.pow(2).sum()) / (total**2))


def load_australia(path: Path) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path, parse_dates=["date"])
    df["country"] = "Australia"
    return df


def load_united_states(path: Path) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path, parse_dates=["date"])
    if "country" not in df.columns:
        df["country"] = "United States"
    return df


def prepare_common_daily(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["country", "region", "date"]).copy()

    for col in ["lat", "lon", "rain_mm", "tmax_c", "tmin_c", "tmean_c", "radiation_mj_m2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "tmean_c" not in df.columns:
        df["tmean_c"] = np.nan
    missing_tmean = df["tmean_c"].isna() & df["tmax_c"].notna() & df["tmin_c"].notna()
    df.loc[missing_tmean, "tmean_c"] = (df.loc[missing_tmean, "tmax_c"] + df.loc[missing_tmean, "tmin_c"]) / 2

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["year_start"] = df["year"]
    df["season"] = df["year_start"].map(season_label)
    df["window"] = df["month"].map(WINDOW_BY_MONTH)

    df = df[df["month"].isin(GROWING_MONTHS)].copy()
    df = df[df["year_start"].between(start_year, end_year)].copy()
    df = df.sort_values(["country", "region", "date"]).reset_index(drop=True)
    return df


def summarize_window(group: pd.DataFrame) -> dict[str, float]:
    if group.empty:
        return {
            "rain_sum": np.nan,
            "dry_days_1mm": np.nan,
            "max_dry_spell_1mm": np.nan,
            "heavy_rain_days_20": np.nan,
            "max_3day_rain": np.nan,
            "heat_days_35": np.nan,
            "max_consecutive_heat_days_35": np.nan,
            "frost_days_0": np.nan,
            "radiation_sum": np.nan,
        }

    rain = group["rain_mm"]
    tmax = group["tmax_c"]
    tmin = group["tmin_c"]
    radiation = group["radiation_mj_m2"] if "radiation_mj_m2" in group.columns else pd.Series(dtype=float)

    return {
        "rain_sum": safe_sum(rain),
        "dry_days_1mm": int((rain < 1).sum()),
        "max_dry_spell_1mm": max_consecutive(rain < 1),
        "heavy_rain_days_20": int((rain >= 20).sum()),
        "max_3day_rain": rolling_sum_max(rain, 3),
        "heat_days_35": int((tmax >= 35).sum()),
        "max_consecutive_heat_days_35": max_consecutive(tmax >= 35),
        "frost_days_0": int((tmin <= 0).sum()),
        "radiation_sum": safe_sum(radiation),
    }


def summarize_region_year(group: pd.DataFrame) -> dict[str, float | int | str]:
    group = group.sort_values("date").copy()
    rain = group["rain_mm"]
    tmax = group["tmax_c"]
    tmin = group["tmin_c"]
    tmean = group["tmean_c"]
    radiation = group["radiation_mj_m2"] if "radiation_mj_m2" in group.columns else pd.Series(dtype=float)

    year_start = int(group["year_start"].iloc[0])
    harvest_end = pd.Timestamp(year=year_start, month=10, day=31)
    last_7 = group[(group["date"] >= harvest_end - pd.Timedelta(days=6)) & (group["date"] <= harvest_end)]
    last_14 = group[(group["date"] >= harvest_end - pd.Timedelta(days=13)) & (group["date"] <= harvest_end)]
    last_21 = group[(group["date"] >= harvest_end - pd.Timedelta(days=20)) & (group["date"] <= harvest_end)]

    last_14_rain = last_14["rain_mm"] if not last_14.empty else pd.Series(dtype=float)
    last_21_rain = last_21["rain_mm"] if not last_21.empty else pd.Series(dtype=float)
    max_3day_last_21 = rolling_sum_max(last_21_rain, 3)
    heavy_last_14 = int((last_14_rain >= 20).sum()) if not last_14_rain.empty else 0

    out: dict[str, float | int | str] = {
        "lat_mean": float(group["lat"].mean()) if "lat" in group.columns else np.nan,
        "lon_mean": float(group["lon"].mean()) if "lon" in group.columns else np.nan,
        "season_days_observed": int(len(group)),
        "season_start_date": group["date"].min().date().isoformat(),
        "season_end_date": group["date"].max().date().isoformat(),
        "season_rain_sum": safe_sum(rain),
        "season_rain_mean": float(rain.mean()),
        "season_tmax_mean": float(tmax.mean()),
        "season_tmin_mean": float(tmin.mean()),
        "season_tmean_mean": float(tmean.mean()),
        "season_radiation_sum": safe_sum(radiation),
        "season_radiation_mean": float(radiation.mean()),
        "growing_degree_days_base5": float(np.maximum(tmean - 5, 0).sum()),
        "heat_days_30": int((tmax >= 30).sum()),
        "heat_days_35": int((tmax >= 35).sum()),
        "heat_days_40": int((tmax >= 40).sum()),
        "max_consecutive_heat_days_30": max_consecutive(tmax >= 30),
        "max_consecutive_heat_days_35": max_consecutive(tmax >= 35),
        "heatwave_events_3d_30": count_run_events(tmax >= 30, 3),
        "heatwave_events_3d_35": count_run_events(tmax >= 35, 3),
        "heat_degree_days_30": float(np.maximum(tmax - 30, 0).sum()),
        "heat_degree_days_35": float(np.maximum(tmax - 35, 0).sum()),
        "dry_days_1mm": int((rain < 1).sum()),
        "dry_days_2mm": int((rain < 2).sum()),
        "max_dry_spell_1mm": max_consecutive(rain < 1),
        "max_dry_spell_2mm": max_consecutive(rain < 2),
        "dry_spell_events_7d": count_run_events(rain < 1, 7),
        "dry_spell_events_14d": count_run_events(rain < 1, 14),
        "hot_dry_days_30_1mm": int(((rain < 1) & (tmax >= 30)).sum()),
        "heavy_rain_days_10": int((rain >= 10).sum()),
        "heavy_rain_days_20": int((rain >= 20).sum()),
        "heavy_rain_days_25": int((rain >= 25).sum()),
        "heavy_rain_days_50": int((rain >= 50).sum()),
        "max_1day_rain": float(rain.max()),
        "max_3day_rain": rolling_sum_max(rain, 3),
        "max_7day_rain": rolling_sum_max(rain, 7),
        "wet_days_1mm": int((rain >= 1).sum()),
        "max_wet_spell_1mm": max_consecutive(rain >= 1),
        "rainfall_concentration_index": rainfall_concentration(rain),
        "rain_last_7d_before_harvest": safe_sum(last_7["rain_mm"]) if not last_7.empty else np.nan,
        "rain_last_14d_before_harvest": safe_sum(last_14_rain),
        "rain_last_21d_before_harvest": safe_sum(last_21_rain),
        "wet_days_last_14d_before_harvest": int((last_14_rain >= 1).sum()) if not last_14_rain.empty else 0,
        "heavy_rain_days_last_14d_before_harvest": heavy_last_14,
        "max_3day_rain_last_21d_before_harvest": max_3day_last_21,
        "storm_before_harvest_flag": int((max_3day_last_21 >= 25) or (heavy_last_14 >= 1)),
        "frost_days_0": int((tmin <= 0).sum()),
        "cold_days_5": int((tmin <= 5).sum()),
        "min_tmin": float(tmin.min()),
        "frost_events_2d": count_run_events(tmin <= 0, 2),
    }

    for window in ["early", "mid", "late"]:
        window_values = summarize_window(group[group["window"] == window])
        for feature_name, value in window_values.items():
            out[f"{feature_name}_{window}"] = value

    return out


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["country", "region", "year_start", "season"]
    for keys, group in daily.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys))
        row.update(summarize_region_year(group))
        rows.append(row)
    features = pd.DataFrame(rows)
    return features.sort_values(group_cols).reset_index(drop=True)


def describe_feature(name: str) -> str:
    exact = {
        "country": "Country name.",
        "region": "State or region name.",
        "year_start": "Start year of the May-October growing-season proxy.",
        "season": "Season label used for matching annual yield records.",
        "lat_mean": "Average latitude of the regional weather point.",
        "lon_mean": "Average longitude of the regional weather point.",
        "season_days_observed": "Number of daily records available in May-October.",
        "season_start_date": "First daily weather date used in the season.",
        "season_end_date": "Last daily weather date used in the season.",
        "season_rain_sum": "Total rainfall during May-October.",
        "season_rain_mean": "Average daily rainfall during May-October.",
        "season_tmax_mean": "Average daily maximum temperature during May-October.",
        "season_tmin_mean": "Average daily minimum temperature during May-October.",
        "season_tmean_mean": "Average daily mean temperature during May-October.",
        "season_radiation_sum": "Total solar radiation during May-October.",
        "season_radiation_mean": "Average daily solar radiation during May-October.",
        "growing_degree_days_base5": "Warmth accumulated above 5 degrees C during May-October.",
        "rainfall_concentration_index": "How concentrated rainfall was; higher values mean rain came in fewer days.",
        "storm_before_harvest_flag": "1 when late-season rain suggests possible storm risk before harvest.",
        "min_tmin": "Lowest daily minimum temperature during May-October.",
    }
    if name in exact:
        return exact[name]
    if name.startswith("heat_days_"):
        return "Number of days above a heat threshold."
    if name.startswith("max_consecutive_heat_days_"):
        return "Longest continuous sequence of hot days."
    if name.startswith("heatwave_events_"):
        return "Number of heatwave events lasting at least three days."
    if name.startswith("heat_degree_days_"):
        return "Total heat intensity above the named threshold."
    if name.startswith("dry_days_"):
        return "Number of days with rainfall below the named dry threshold."
    if name.startswith("max_dry_spell_"):
        return "Longest continuous sequence of dry days."
    if name.startswith("dry_spell_events_"):
        return "Number of long dry-spell events."
    if name.startswith("hot_dry_days_"):
        return "Number of days that were both hot and dry."
    if name.startswith("heavy_rain_days_"):
        return "Number of days above a heavy-rain threshold."
    if name.startswith("max_1day_rain"):
        return "Largest one-day rainfall amount."
    if name.startswith("max_3day_rain"):
        return "Largest rainfall total over any three-day window."
    if name.startswith("max_7day_rain"):
        return "Largest rainfall total over any seven-day window."
    if name.startswith("wet_days_"):
        return "Number of days with at least 1 mm rainfall."
    if name.startswith("max_wet_spell_"):
        return "Longest continuous sequence of wet days."
    if "before_harvest" in name:
        return "Rain or wet-day indicator in the late-October proxy harvest-risk window."
    if name.startswith("frost_days_"):
        return "Number of days with minimum temperature at or below 0 degrees C."
    if name.startswith("cold_days_"):
        return "Number of days with minimum temperature at or below 5 degrees C."
    if name.startswith("frost_events_"):
        return "Number of frost events lasting at least two days."
    if name.endswith("_early"):
        return "Early-season May-June version of this feature."
    if name.endswith("_mid"):
        return "Mid-season July-August version of this feature."
    if name.endswith("_late"):
        return "Late-season September-October version of this feature."
    return "Derived daily-weather feature."


def write_feature_dictionary(features: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Extreme Weather Feature Dictionary",
        "",
        "Each row in the CSV files is one country-region-year growing-season record.",
        "The first version uses May-October as a common proxy season so Australia and the United States can be compared with the existing paper period.",
        "The indicators are predictive and associative, not causal claims.",
        "",
        "| Column | Plain-language meaning |",
        "|---|---|",
    ]
    for col in features.columns:
        lines.append(f"| `{col}` | {describe_feature(col)} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build interpretable daily extreme-weather features.")
    parser.add_argument("--start-year", type=int, default=1989)
    parser.add_argument("--end-year", type=int, default=2021)
    parser.add_argument("--aus-file", type=Path, default=RAW_AUS)
    parser.add_argument("--us-file", type=Path, default=RAW_US)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)

    australia = prepare_common_daily(load_australia(args.aus_file), args.start_year, args.end_year)
    united_states = prepare_common_daily(load_united_states(args.us_file), args.start_year, args.end_year)
    daily = pd.concat([australia, united_states], ignore_index=True)

    features = build_features(daily)
    aus_features = features[features["country"] == "Australia"].copy()
    us_features = features[features["country"] == "United States"].copy()

    combined_path = args.processed_dir / "extreme_weather_features_aus_us_region_year.csv"
    aus_path = args.processed_dir / "extreme_weather_features_aus_region_year.csv"
    us_path = args.processed_dir / "extreme_weather_features_us_region_year.csv"
    dictionary_path = args.docs_dir / "extreme_weather_feature_dictionary.md"

    features.to_csv(combined_path, index=False)
    aus_features.to_csv(aus_path, index=False)
    us_features.to_csv(us_path, index=False)
    write_feature_dictionary(features, dictionary_path)

    print(f"Combined features: {features.shape} -> {combined_path}")
    print(f"Australia features: {aus_features.shape} -> {aus_path}")
    print(f"United States features: {us_features.shape} -> {us_path}")
    print(f"Feature dictionary: {dictionary_path}")


if __name__ == "__main__":
    main()
