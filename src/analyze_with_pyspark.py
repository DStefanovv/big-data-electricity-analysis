"""Run the descriptive electricity analysis with PySpark."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from datetime import date, datetime
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any


def windows_short_path(value: str | Path) -> str:
    buffer = ctypes.create_unicode_buffer(32768)
    if not ctypes.windll.kernel32.GetShortPathNameW(str(value), buffer, len(buffer)):
        raise OSError(f"Cannot resolve Windows short path: {value}")
    return buffer.value


def prepare_spark_environment() -> None:
    if os.name != "nt":
        return

    import pyspark

    java_home = Path(os.environ.get("PROJECT_JAVA_HOME", r"C:\Program Files\Java\jdk-17"))
    if not (java_home / "bin" / "java.exe").is_file():
        raise RuntimeError("Java 17 was not found. Set PROJECT_JAVA_HOME before running.")

    os.environ["JAVA_HOME"] = str(java_home)
    os.environ["SPARK_HOME"] = windows_short_path(Path(pyspark.__file__).parent)
    os.environ["PYSPARK_PYTHON"] = windows_short_path(sys.executable)
    os.environ["PYSPARK_DRIVER_PYTHON"] = windows_short_path(sys.executable)
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"


def json_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return round(float(value), 4)
    if isinstance(value, float):
        return round(value, 4)
    return value


def rows_to_dicts(rows: list[Any]) -> list[dict[str, Any]]:
    return [
        {key: json_value(value) for key, value in row.asDict(recursive=True).items()}
        for row in rows
    ]


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_analysis(input_path: Path, output_dir: Path, overwrite: bool) -> dict[str, Any]:
    if not input_path.is_dir():
        raise FileNotFoundError(f"Processed dataset not found: {input_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Analysis output already exists: {output_dir}. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    prepare_spark_environment()
    from pyspark.sql import SparkSession, functions as F

    total_started = perf_counter()
    startup_started = perf_counter()
    spark = (
        SparkSession.builder.master("local[4]")
        .appName("ElectricityDescriptiveAnalysis")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    startup_seconds = perf_counter() - startup_started

    query_timings: dict[str, float] = {}
    try:
        load_started = perf_counter()
        parquet_files = sorted(input_path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No Parquet files found in: {input_path}")
        if os.name == "nt":
            spark_input_paths = [windows_short_path(path.resolve()) for path in parquet_files]
        else:
            spark_input_paths = [str(path.resolve()) for path in parquet_files]
        frame = spark.read.parquet(*spark_input_paths).cache()
        record_count = frame.count()
        load_seconds = perf_counter() - load_started

        def collect_query(name: str, query: Any) -> list[dict[str, Any]]:
            started = perf_counter()
            result = rows_to_dicts(query.collect())
            query_timings[name] = round(perf_counter() - started, 3)
            return result

        summary_rows = collect_query(
            "summary",
            frame.agg(
                F.countDistinct("household_id").alias("households"),
                F.countDistinct("region").alias("regions"),
                F.countDistinct("city").alias("cities"),
                F.min("timestamp").alias("period_start"),
                F.max("timestamp").alias("period_end"),
                F.sum("consumption_kwh").alias("total_consumption_kwh"),
                F.sum("grid_consumption_kwh").alias("total_grid_consumption_kwh"),
                F.sum("solar_generation_kwh").alias("total_solar_generation_kwh"),
                F.sum("energy_cost_bgn").alias("total_cost_bgn"),
                F.sum("solar_savings_bgn").alias("total_solar_savings_bgn"),
                F.avg("consumption_kwh").alias("average_hourly_consumption_kwh"),
                F.avg("temperature_c").alias("average_temperature_c"),
                F.max("grid_consumption_kwh").alias("maximum_grid_consumption_kwh"),
            ),
        )
        summary = {
            "data_type": "synthetic",
            "records": record_count,
            **summary_rows[0],
        }

        monthly = collect_query(
            "monthly",
            frame.groupBy("month", "month_name_bg")
            .agg(
                F.sum("consumption_kwh").alias("consumption_kwh"),
                F.sum("grid_consumption_kwh").alias("grid_consumption_kwh"),
                F.sum("solar_generation_kwh").alias("solar_generation_kwh"),
                F.sum("energy_cost_bgn").alias("cost_bgn"),
                F.sum("solar_savings_bgn").alias("solar_savings_bgn"),
                F.avg("temperature_c").alias("average_temperature_c"),
            )
            .orderBy("month"),
        )

        hourly = collect_query(
            "hourly",
            frame.groupBy("hour")
            .agg(
                F.avg("consumption_kwh").alias("average_consumption_kwh"),
                F.avg("grid_consumption_kwh").alias("average_grid_consumption_kwh"),
                F.sum("energy_cost_bgn").alias("cost_bgn"),
                F.avg("temperature_c").alias("average_temperature_c"),
            )
            .orderBy("hour"),
        )

        regions = collect_query(
            "regions",
            frame.groupBy("region")
            .agg(
                F.countDistinct("household_id").alias("households"),
                F.sum("consumption_kwh").alias("consumption_kwh"),
                F.sum("grid_consumption_kwh").alias("grid_consumption_kwh"),
                F.sum("solar_generation_kwh").alias("solar_generation_kwh"),
                F.sum("energy_cost_bgn").alias("cost_bgn"),
                F.sum("solar_savings_bgn").alias("solar_savings_bgn"),
                (
                    F.sum("grid_consumption_kwh") / F.countDistinct("household_id")
                ).alias("grid_kwh_per_household"),
            )
            .orderBy(F.desc("grid_consumption_kwh")),
        )

        building_types = collect_query(
            "building_types",
            frame.groupBy("building_type")
            .agg(
                F.countDistinct("household_id").alias("households"),
                F.sum("grid_consumption_kwh").alias("grid_consumption_kwh"),
                (
                    F.sum("grid_consumption_kwh") / F.countDistinct("household_id")
                ).alias("grid_kwh_per_household"),
                F.sum("energy_cost_bgn").alias("cost_bgn"),
            )
            .orderBy("building_type"),
        )

        heating_types = collect_query(
            "heating_types",
            frame.groupBy("heating_type")
            .agg(
                F.countDistinct("household_id").alias("households"),
                F.sum("grid_consumption_kwh").alias("grid_consumption_kwh"),
                (
                    F.sum("grid_consumption_kwh") / F.countDistinct("household_id")
                ).alias("grid_kwh_per_household"),
                F.sum("energy_cost_bgn").alias("cost_bgn"),
            )
            .orderBy(F.desc("grid_kwh_per_household")),
        )

        tariffs = collect_query(
            "tariffs",
            frame.groupBy("tariff_type")
            .agg(
                F.count("measurement_id").alias("records"),
                F.sum("grid_consumption_kwh").alias("grid_consumption_kwh"),
                F.sum("energy_cost_bgn").alias("cost_bgn"),
                F.avg("price_per_kwh").alias("price_per_kwh"),
            )
            .orderBy("tariff_type"),
        )

        solar_comparison = collect_query(
            "solar_comparison",
            frame.groupBy("has_solar_panels")
            .agg(
                F.countDistinct("household_id").alias("households"),
                F.sum("consumption_kwh").alias("consumption_kwh"),
                F.sum("grid_consumption_kwh").alias("grid_consumption_kwh"),
                F.sum("solar_generation_kwh").alias("solar_generation_kwh"),
                F.sum("solar_savings_bgn").alias("solar_savings_bgn"),
                F.sum("energy_cost_bgn").alias("cost_bgn"),
                (
                    F.sum("grid_consumption_kwh") / F.countDistinct("household_id")
                ).alias("grid_kwh_per_household"),
            )
            .orderBy("has_solar_panels"),
        )

        day_hour_heatmap = collect_query(
            "day_hour_heatmap",
            frame.groupBy("day_of_week", "day_name_bg", "hour")
            .agg(
                F.avg("grid_consumption_kwh").alias("average_grid_consumption_kwh"),
                F.avg("consumption_kwh").alias("average_consumption_kwh"),
            )
            .orderBy("day_of_week", "hour"),
        )

        files = {
            "summary.json": summary,
            "monthly.json": monthly,
            "hourly.json": hourly,
            "regions.json": regions,
            "building_types.json": building_types,
            "heating_types.json": heating_types,
            "tariffs.json": tariffs,
            "solar_comparison.json": solar_comparison,
            "day_hour_heatmap.json": day_hour_heatmap,
        }
        for filename, content in files.items():
            write_json(output_dir / filename, content)

        peak_month = max(monthly, key=lambda item: item["grid_consumption_kwh"])
        peak_hour = max(hourly, key=lambda item: item["average_grid_consumption_kwh"])
        highest_region = max(regions, key=lambda item: item["grid_kwh_per_household"])
        solar_group = next(item for item in solar_comparison if item["has_solar_panels"])
        non_solar_group = next(item for item in solar_comparison if not item["has_solar_panels"])
        solar_reduction_percent = (
            1
            - solar_group["grid_kwh_per_household"]
            / non_solar_group["grid_kwh_per_household"]
        ) * 100
        insights = [
            f"Най-високото потребление от мрежата е през {peak_month['month_name_bg']}.",
            f"Най-високото средно почасово потребление от мрежата е около {peak_hour['hour']:02d}:00 ч.",
            f"{highest_region['region']} има най-високо средно годишно потребление от мрежата на домакинство.",
            f"Домакинствата със соларни панели използват средно с {solar_reduction_percent:.1f}% по-малко електроенергия от мрежата.",
            f"Изчислените общи соларни спестявания са {summary['total_solar_savings_bgn']:.2f} лв.",
        ]
        write_json(output_dir / "insights.json", insights)

        performance = {
            "status": "PASS",
            "engine": "PySpark",
            "records": record_count,
            "spark_partitions": frame.rdd.getNumPartitions(),
            "timings_seconds": {
                "spark_startup": round(startup_seconds, 3),
                "load_and_cache": round(load_seconds, 3),
                "queries": query_timings,
                "total": round(perf_counter() - total_started, 3),
            },
            "versions": {
                "pyspark": version("pyspark"),
                "java": spark.sparkContext._jvm.java.lang.System.getProperty("java.version"),
            },
            "output_files": sorted([*files, "insights.json", "performance.json"]),
        }
        write_json(output_dir / "performance.json", performance)
        return performance
    finally:
        if "frame" in locals():
            frame.unpersist()
        spark.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/electricity_consumption_2025"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/analysis"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_analysis(args.input, args.output, args.overwrite)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
