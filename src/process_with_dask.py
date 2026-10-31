"""Clean and enrich the generated electricity dataset with Dask."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import dask
import dask.dataframe as dd
import pyarrow as pa


RAW_DTYPES = {
    "measurement_id": "int64",
    "timestamp": "string",
    "household_id": "string",
    "meter_id": "string",
    "region": "string",
    "city": "string",
    "building_type": "string",
    "heating_type": "string",
    "household_size": "int64",
    "has_solar_panels": "bool",
    "solar_capacity_kw": "float64",
    "temperature_c": "float64",
    "is_weekend": "bool",
    "tariff_type": "string",
    "price_per_kwh": "float64",
    "consumption_kwh": "float64",
    "solar_generation_kwh": "float64",
    "grid_consumption_kwh": "float64",
    "energy_cost_bgn": "float64",
}

REQUIRED_COLUMNS = tuple(RAW_DTYPES)
NON_NEGATIVE_COLUMNS = (
    "solar_capacity_kw",
    "consumption_kwh",
    "solar_generation_kwh",
    "grid_consumption_kwh",
    "energy_cost_bgn",
)

MONTH_NAMES_BG = {
    1: "януари",
    2: "февруари",
    3: "март",
    4: "април",
    5: "май",
    6: "юни",
    7: "юли",
    8: "август",
    9: "септември",
    10: "октомври",
    11: "ноември",
    12: "декември",
}

DAY_NAMES_BG = {
    0: "понеделник",
    1: "вторник",
    2: "сряда",
    3: "четвъртък",
    4: "петък",
    5: "събота",
    6: "неделя",
}

SEASONS_BG = {
    1: "зима",
    2: "зима",
    3: "пролет",
    4: "пролет",
    5: "пролет",
    6: "лято",
    7: "лято",
    8: "лято",
    9: "есен",
    10: "есен",
    11: "есен",
    12: "зима",
}

DERIVED_COLUMNS = (
    "date",
    "year",
    "month",
    "month_name_bg",
    "day_of_week",
    "day_name_bg",
    "hour",
    "season_bg",
    "is_peak_hour",
    "solar_used_kwh",
    "energy_cost_without_solar_bgn",
    "solar_savings_bgn",
)


def read_raw_data(input_path: Path) -> dd.DataFrame:
    return dd.read_csv(
        input_path,
        dtype=RAW_DTYPES,
        blocksize="32MB",
        assume_missing=False,
    )


def clean_data(frame: dd.DataFrame) -> dd.DataFrame:
    frame = frame.dropna(subset=list(REQUIRED_COLUMNS))
    frame = frame.drop_duplicates(subset=["measurement_id"])
    frame["timestamp"] = dd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"])

    valid = (
        (frame["measurement_id"] > 0)
        & frame["tariff_type"].isin(["day", "night"])
        & frame["building_type"].isin(["apartment", "house"])
        & frame["heating_type"].isin(["electric", "district", "gas", "other"])
        & frame["household_size"].between(1, 5)
        & (frame["price_per_kwh"] > 0)
    )
    for column in NON_NEGATIVE_COLUMNS:
        valid &= frame[column] >= 0
    return frame[valid]


def enrich_data(frame: dd.DataFrame) -> dd.DataFrame:
    frame = frame.assign(
        date=frame["timestamp"].dt.strftime("%Y-%m-%d"),
        year=frame["timestamp"].dt.year,
        month=frame["timestamp"].dt.month,
        day_of_week=frame["timestamp"].dt.dayofweek,
        hour=frame["timestamp"].dt.hour,
    )
    frame = frame.assign(
        month_name_bg=frame["month"].map(
            MONTH_NAMES_BG, meta=("month_name_bg", "string")
        ),
        day_name_bg=frame["day_of_week"].map(
            DAY_NAMES_BG, meta=("day_name_bg", "string")
        ),
        season_bg=frame["month"].map(SEASONS_BG, meta=("season_bg", "string")),
    )
    frame = frame.assign(
        is_peak_hour=frame["hour"].isin([6, 7, 8, 17, 18, 19, 20, 21]),
        solar_used_kwh=(
            frame["consumption_kwh"] - frame["grid_consumption_kwh"]
        ).clip(lower=0),
        energy_cost_without_solar_bgn=(
            frame["consumption_kwh"] * frame["price_per_kwh"]
        ),
    )
    frame = frame.assign(
        solar_savings_bgn=(
            frame["solar_used_kwh"] * frame["price_per_kwh"]
        )
    )
    return frame


def directory_size_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def process_dataset(
    input_path: Path,
    output_path: Path,
    metrics_path: Path,
    overwrite: bool,
) -> dict[str, object]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite.")
    if metrics_path.exists() and not overwrite:
        raise FileExistsError(f"Metrics already exist: {metrics_path}. Use --overwrite.")

    total_started = perf_counter()
    preparation_started = perf_counter()
    raw = read_raw_data(input_path)
    initial_partitions = raw.npartitions

    initial_records, unique_measurement_ids = dask.compute(
        raw.map_partitions(len).sum(),
        raw["measurement_id"].nunique(),
    )
    cleaned = clean_data(raw)
    enriched = enrich_data(cleaned).persist()
    cleaned_records = int(enriched.map_partitions(len).sum().compute())
    preparation_seconds = perf_counter() - preparation_started

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_started = perf_counter()
    enriched.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy",
        write_index=False,
        overwrite=overwrite,
        name_function=lambda index: f"part-{index:05d}.parquet",
        schema={
            "measurement_id": pa.int64(),
            "timestamp": pa.timestamp("us"),
            "household_size": pa.int64(),
            "has_solar_panels": pa.bool_(),
            "is_weekend": pa.bool_(),
        },
    )
    del enriched
    write_seconds = perf_counter() - write_started

    verification_started = perf_counter()
    processed = dd.read_parquet(output_path, engine="pyarrow")
    output_records, first_timestamp, last_timestamp, missing_values = dask.compute(
        processed.map_partitions(len).sum(),
        processed["timestamp"].min(),
        processed["timestamp"].max(),
        processed.isna().sum().sum(),
    )
    output_columns = list(processed.columns)
    missing_derived_columns = sorted(set(DERIVED_COLUMNS) - set(output_columns))
    verification_seconds = perf_counter() - verification_started

    if int(output_records) != cleaned_records:
        raise ValueError("Parquet record count does not match the cleaned data.")
    if int(missing_values) != 0:
        raise ValueError("The processed dataset contains missing values.")
    if missing_derived_columns:
        raise ValueError(f"Missing derived columns: {missing_derived_columns}")

    input_size = input_path.stat().st_size
    output_size = directory_size_bytes(output_path)
    parquet_files = sorted(output_path.glob("*.parquet"))
    metrics = {
        "status": "PASS",
        "input": input_path.as_posix(),
        "output": output_path.as_posix(),
        "records": {
            "initial": int(initial_records),
            "unique_measurement_ids": int(unique_measurement_ids),
            "cleaned": cleaned_records,
            "removed": int(initial_records) - cleaned_records,
            "written": int(output_records),
        },
        "period": {
            "start": first_timestamp.isoformat(),
            "end": last_timestamp.isoformat(),
        },
        "partitions": {
            "input": initial_partitions,
            "output": len(parquet_files),
        },
        "size_mb": {
            "csv": round(input_size / (1024 * 1024), 2),
            "parquet": round(output_size / (1024 * 1024), 2),
            "reduction_percent": round((1 - output_size / input_size) * 100, 2),
        },
        "timings_seconds": {
            "preparation": round(preparation_seconds, 2),
            "parquet_write": round(write_seconds, 2),
            "verification": round(verification_seconds, 2),
            "total": round(perf_counter() - total_started, 2),
        },
        "derived_columns": list(DERIVED_COLUMNS),
        "versions": {
            "dask": version("dask"),
            "pandas": version("pandas"),
            "pyarrow": version("pyarrow"),
        },
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/electricity_consumption_2025.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/electricity_consumption_2025"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("reports/dask_processing_metrics.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = process_dataset(
        input_path=args.input,
        output_path=args.output,
        metrics_path=args.metrics,
        overwrite=args.overwrite,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
