"""Validate the generated electricity dataset with Dask."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import dask
import dask.dataframe as dd


PROFILE_COLUMNS = [
    "household_id",
    "meter_id",
    "region",
    "city",
    "building_type",
    "heating_type",
    "household_size",
    "has_solar_panels",
    "solar_capacity_kw",
]


def validate_dataset(path: Path, expected_records: int) -> dict[str, object]:
    started = perf_counter()
    frame = dd.read_csv(path, blocksize="32MB")

    grid_error = (
        frame["grid_consumption_kwh"]
        - (frame["consumption_kwh"] - frame["solar_generation_kwh"]).clip(lower=0)
    ).abs()
    cost_error = (
        frame["energy_cost_bgn"]
        - frame["grid_consumption_kwh"] * frame["price_per_kwh"]
    ).abs()

    computations = dask.compute(
        frame.map_partitions(len).sum(),
        frame["measurement_id"].min(),
        frame["measurement_id"].max(),
        frame["timestamp"].min(),
        frame["timestamp"].max(),
        frame["household_id"].nunique(),
        frame["meter_id"].nunique(),
        frame["region"].nunique(),
        frame["city"].nunique(),
        frame.isna().sum().sum(),
        (frame[["consumption_kwh", "solar_generation_kwh", "grid_consumption_kwh", "energy_cost_bgn"]] < 0)
        .sum()
        .sum(),
        grid_error.max(),
        cost_error.max(),
        frame["tariff_type"].value_counts(),
        frame.groupby("household_id").size(),
    )

    (
        record_count,
        minimum_id,
        maximum_id,
        first_timestamp,
        last_timestamp,
        household_count,
        meter_count,
        region_count,
        city_count,
        null_count,
        negative_count,
        maximum_grid_error,
        maximum_cost_error,
        tariff_counts,
        records_per_household,
    ) = computations

    profiles = frame[PROFILE_COLUMNS].drop_duplicates(subset=["household_id"]).compute()
    region_profiles = profiles.groupby("region").size().to_dict()
    city_profiles = profiles.groupby("city").size().to_dict()
    building_profiles = profiles["building_type"].value_counts().to_dict()
    heating_profiles = profiles["heating_type"].value_counts().to_dict()
    size_profiles = profiles["household_size"].value_counts().sort_index().to_dict()
    solar_households = int(profiles["has_solar_panels"].sum())

    checks = {
        "record_count": int(record_count) == expected_records,
        "measurement_ids": int(minimum_id) == 1 and int(maximum_id) == expected_records,
        "period": first_timestamp == "2025-01-01T00:00:00"
        and last_timestamp == "2025-12-31T23:00:00",
        "households": int(household_count) == 120 and int(meter_count) == 120,
        "locations": int(region_count) == 6 and int(city_count) == 12,
        "regional_balance": set(region_profiles.values()) == {20},
        "city_balance": set(city_profiles.values()) == {10},
        "building_distribution": building_profiles == {"apartment": 72, "house": 48},
        "heating_distribution": heating_profiles
        == {"electric": 48, "district": 30, "gas": 24, "other": 18},
        "household_size_distribution": size_profiles == {1: 24, 2: 36, 3: 30, 4: 24, 5: 6},
        "solar_households": solar_households == 24,
        "records_per_household": int(records_per_household.min()) == 8760
        and int(records_per_household.max()) == 8760,
        "no_missing_values": int(null_count) == 0,
        "no_negative_values": int(negative_count) == 0,
        "grid_formula": float(maximum_grid_error) < 0.0002,
        "cost_formula": float(maximum_cost_error) < 0.0002,
        "both_tariffs": set(tariff_counts.index) == {"day", "night"},
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise ValueError(f"Dataset validation failed: {', '.join(failed_checks)}")

    return {
        "status": "PASS",
        "records": int(record_count),
        "period": [first_timestamp, last_timestamp],
        "households": int(household_count),
        "regions": int(region_count),
        "cities": int(city_count),
        "tariff_records": {key: int(value) for key, value in tariff_counts.to_dict().items()},
        "solar_households": solar_households,
        "partitions": frame.npartitions,
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "validation_seconds": round(perf_counter() - started, 2),
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("data/raw/electricity_consumption_2025.csv"),
    )
    parser.add_argument("--expected-records", type=int, default=1_051_200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_dataset(args.path, args.expected_records)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
