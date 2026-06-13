from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_FRAME_DIR = PROCESSED_DIR / "model_frames"

EXTREME_FEATURES = PROCESSED_DIR / "extreme_weather_features_aus_us_region_year.csv"
AUS_FINAL = REPO_ROOT / "datanew" / "Aus" / "outputs" / "daily_weather_ml" / "final_ml_dataset.csv"
US_OVERLAP = REPO_ROOT / "datanew" / "US" / "harmonized" / "us_harmonized_all_overlap_crops.csv"
US_WHEAT = REPO_ROOT / "datanew" / "US" / "harmonized" / "us_harmonized_wheat_only.csv"

OVERLAP_CROPS = ["Barley", "Canola", "Oats", "Wheat"]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def load_model_frame(path: Path, country: str | None = None) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path)
    if country is not None and "country" not in df.columns:
        df.insert(0, "country", country)
    return df


def clean_for_main_period(df: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    df = df.copy()
    df = df[df["year_start"].between(start_year, end_year)].copy()
    if "yield_t_ha" in df.columns:
        df = df[df["yield_t_ha"].notna()].copy()
    return df


def merge_extreme_features(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    merge_keys = ["country", "region", "year_start"]
    missing = [col for col in merge_keys if col not in df.columns]
    if missing:
        raise ValueError(f"Input frame is missing merge keys: {missing}")

    feature_cols = [col for col in features.columns if col not in ["season"]]
    merged = df.merge(
        features[feature_cols],
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )
    missing_features = merged["season_days_observed"].isna().sum()
    if missing_features:
        print(f"WARNING: {missing_features} rows did not match extreme-weather features.")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge extreme-weather features into annual yield/soil frames.")
    parser.add_argument("--start-year", type=int, default=1989)
    parser.add_argument("--end-year", type=int, default=2021)
    parser.add_argument("--features", type=Path, default=EXTREME_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=MODEL_FRAME_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    require_file(args.features)

    features = pd.read_csv(args.features)
    australia = clean_for_main_period(load_model_frame(AUS_FINAL, country="Australia"), args.start_year, args.end_year)
    us_overlap = clean_for_main_period(load_model_frame(US_OVERLAP), args.start_year, args.end_year)
    us_wheat = clean_for_main_period(load_model_frame(US_WHEAT), args.start_year, args.end_year)

    aus_with_extreme = merge_extreme_features(australia, features)
    us_overlap_with_extreme = merge_extreme_features(us_overlap, features)
    us_wheat_with_extreme = merge_extreme_features(us_wheat, features)

    aus_overlap_with_extreme = aus_with_extreme[aus_with_extreme["crop"].isin(OVERLAP_CROPS)].copy()
    aus_wheat_with_extreme = aus_with_extreme[aus_with_extreme["crop"].eq("Wheat")].copy()

    combined_overlap = pd.concat([aus_overlap_with_extreme, us_overlap_with_extreme], ignore_index=True, sort=False)
    combined_wheat = pd.concat([aus_wheat_with_extreme, us_wheat_with_extreme], ignore_index=True, sort=False)

    outputs = {
        "aus_all_crops_with_extreme_weather_1989_2021.csv": aus_with_extreme,
        "aus_overlap_crops_with_extreme_weather_1989_2021.csv": aus_overlap_with_extreme,
        "us_overlap_crops_with_extreme_weather_1989_2021.csv": us_overlap_with_extreme,
        "us_wheat_with_extreme_weather_1989_2021.csv": us_wheat_with_extreme,
        "aus_us_overlap_crops_with_extreme_weather_1989_2021.csv": combined_overlap,
        "aus_us_wheat_with_extreme_weather_1989_2021.csv": combined_wheat,
    }

    for filename, frame in outputs.items():
        path = args.output_dir / filename
        frame.to_csv(path, index=False)
        print(f"{filename}: {frame.shape} -> {path}")


if __name__ == "__main__":
    main()
