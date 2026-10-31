"""Run six regional electricity analyses in parallel with Ray."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any


REGIONS = (
    "Северозападен",
    "Северен централен",
    "Североизточен",
    "Югоизточен",
    "Югозападен",
    "Южен централен",
)

REGIONAL_COLUMNS = (
    "measurement_id",
    "timestamp",
    "household_id",
    "has_solar_panels",
    "month",
    "month_name_bg",
    "tariff_type",
    "consumption_kwh",
    "grid_consumption_kwh",
    "solar_generation_kwh",
    "energy_cost_bgn",
    "solar_savings_bgn",
)


def windows_short_path(value: str | Path) -> str:
    buffer = ctypes.create_unicode_buffer(32768)
    if not ctypes.windll.kernel32.GetShortPathNameW(str(value), buffer, len(buffer)):
        raise OSError(f"Cannot resolve Windows short path: {value}")
    return buffer.value


def rounded(value: Any, digits: int = 4) -> float:
    return round(float(value), digits)


def analyze_region(dataset_path: str, region: str) -> dict[str, Any]:
    """Analyze one region. Ray executes this function in a worker process."""
    import pyarrow.dataset as ds

    started = perf_counter()
    dataset = ds.dataset(dataset_path, format="parquet")
    table = dataset.to_table(
        columns=list(REGIONAL_COLUMNS),
        filter=ds.field("region") == region,
    )
    frame = table.to_pandas()
    if frame.empty:
        raise ValueError(f"No records found for region: {region}")

    household_count = int(frame["household_id"].nunique())
    regional_hourly = (
        frame.groupby("timestamp", as_index=False)
        .agg(
            grid_consumption_kwh=("grid_consumption_kwh", "sum"),
            consumption_kwh=("consumption_kwh", "sum"),
            cost_bgn=("energy_cost_bgn", "sum"),
        )
        .sort_values("timestamp")
    )
    peak_row = regional_hourly.loc[regional_hourly["grid_consumption_kwh"].idxmax()]

    monthly_frame = (
        frame.groupby(["month", "month_name_bg"], as_index=False)
        .agg(
            consumption_kwh=("consumption_kwh", "sum"),
            grid_consumption_kwh=("grid_consumption_kwh", "sum"),
            solar_generation_kwh=("solar_generation_kwh", "sum"),
            cost_bgn=("energy_cost_bgn", "sum"),
        )
        .sort_values("month")
    )
    monthly = [
        {
            "month": int(row.month),
            "month_name_bg": str(row.month_name_bg),
            "consumption_kwh": rounded(row.consumption_kwh),
            "grid_consumption_kwh": rounded(row.grid_consumption_kwh),
            "solar_generation_kwh": rounded(row.solar_generation_kwh),
            "cost_bgn": rounded(row.cost_bgn),
        }
        for row in monthly_frame.itertuples(index=False)
    ]
    peak_month = max(monthly, key=lambda item: item["grid_consumption_kwh"])

    tariff_frame = frame.groupby("tariff_type", as_index=False).agg(
        records=("measurement_id", "count"),
        grid_consumption_kwh=("grid_consumption_kwh", "sum"),
        cost_bgn=("energy_cost_bgn", "sum"),
    )
    tariffs = {
        str(row.tariff_type): {
            "records": int(row.records),
            "grid_consumption_kwh": rounded(row.grid_consumption_kwh),
            "cost_bgn": rounded(row.cost_bgn),
        }
        for row in tariff_frame.itertuples(index=False)
    }

    household_grid = (
        frame.groupby(["household_id", "has_solar_panels"], as_index=False)[
            "grid_consumption_kwh"
        ]
        .sum()
    )
    solar_means = household_grid.groupby("has_solar_panels")[
        "grid_consumption_kwh"
    ].mean()
    solar_households = int(
        household_grid.loc[household_grid["has_solar_panels"], "household_id"].nunique()
    )

    return {
        "region": region,
        "records": int(len(frame)),
        "households": household_count,
        "consumption_kwh": rounded(frame["consumption_kwh"].sum()),
        "grid_consumption_kwh": rounded(frame["grid_consumption_kwh"].sum()),
        "grid_kwh_per_household": rounded(
            frame["grid_consumption_kwh"].sum() / household_count
        ),
        "solar_generation_kwh": rounded(frame["solar_generation_kwh"].sum()),
        "solar_savings_bgn": rounded(frame["solar_savings_bgn"].sum()),
        "cost_bgn": rounded(frame["energy_cost_bgn"].sum()),
        "solar_households": solar_households,
        "average_grid_kwh_solar_household": rounded(solar_means.get(True, 0.0)),
        "average_grid_kwh_non_solar_household": rounded(solar_means.get(False, 0.0)),
        "peak": {
            "timestamp": peak_row.timestamp.isoformat(),
            "grid_consumption_kwh": rounded(peak_row.grid_consumption_kwh),
            "consumption_kwh": rounded(peak_row.consumption_kwh),
            "cost_bgn": rounded(peak_row.cost_bgn),
        },
        "peak_month": {
            "month": peak_month["month"],
            "month_name_bg": peak_month["month_name_bg"],
            "grid_consumption_kwh": peak_month["grid_consumption_kwh"],
        },
        "tariffs": tariffs,
        "monthly": monthly,
        "task_seconds": round(perf_counter() - started, 3),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_against_pyspark(
    results: list[dict[str, Any]], pyspark_summary_path: Path
) -> dict[str, bool]:
    summary = json.loads(pyspark_summary_path.read_text(encoding="utf-8"))
    checks = {
        "records": sum(item["records"] for item in results) == summary["records"],
        "consumption": abs(
            sum(item["consumption_kwh"] for item in results)
            - summary["total_consumption_kwh"]
        )
        < 0.1,
        "grid_consumption": abs(
            sum(item["grid_consumption_kwh"] for item in results)
            - summary["total_grid_consumption_kwh"]
        )
        < 0.1,
        "solar_generation": abs(
            sum(item["solar_generation_kwh"] for item in results)
            - summary["total_solar_generation_kwh"]
        )
        < 0.1,
        "cost": abs(
            sum(item["cost_bgn"] for item in results) - summary["total_cost_bgn"]
        )
        < 0.1,
        "solar_savings": abs(
            sum(item["solar_savings_bgn"] for item in results)
            - summary["total_solar_savings_bgn"]
        )
        < 0.1,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Ray/PySpark validation failed: {', '.join(failed)}")
    return checks


def run_ray_analysis(
    input_path: Path,
    output_dir: Path,
    pyspark_summary_path: Path,
    overwrite: bool,
) -> dict[str, Any]:
    if not input_path.is_dir():
        raise FileNotFoundError(f"Processed dataset not found: {input_path}")
    if not pyspark_summary_path.is_file():
        raise FileNotFoundError(f"PySpark summary not found: {pyspark_summary_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Ray output already exists: {output_dir}. Use --overwrite.")

    os.environ["RAY_USAGE_STATS_ENABLED"] = "0"
    import ray

    total_started = perf_counter()
    startup_started = perf_counter()
    ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False)
    startup_seconds = perf_counter() - startup_started
    try:
        dataset_path = (
            windows_short_path(input_path.resolve())
            if os.name == "nt"
            else str(input_path.resolve())
        )
        remote_analysis = ray.remote(analyze_region)
        parallel_started = perf_counter()
        futures = [remote_analysis.remote(dataset_path, region) for region in REGIONS]
        results = ray.get(futures, timeout=300)
        parallel_seconds = perf_counter() - parallel_started
        available_cpus = int(ray.cluster_resources().get("CPU", 0))
    finally:
        ray.shutdown()

    results.sort(key=lambda item: item["grid_kwh_per_household"], reverse=True)
    validation = validate_against_pyspark(results, pyspark_summary_path)
    estimated_sequential_seconds = sum(item["task_seconds"] for item in results)
    speedup_estimate = (
        estimated_sequential_seconds / parallel_seconds if parallel_seconds else 0.0
    )

    regional_summaries = [
        {key: value for key, value in item.items() if key != "monthly"}
        for item in results
    ]
    regional_monthly = [
        {"region": item["region"], "months": item["monthly"]} for item in results
    ]
    highest_average = results[0]
    highest_peak = max(results, key=lambda item: item["peak"]["grid_consumption_kwh"])
    greatest_solar_savings = max(results, key=lambda item: item["solar_savings_bgn"])
    insights = [
        f"Регион „{highest_average['region']}“ има най-високо годишно потребление от мрежата на домакинство.",
        f"Най-високият регионален часов пик е в регион „{highest_peak['region']}“ на {highest_peak['peak']['timestamp']}.",
        f"Най-големи соларни спестявания има регион „{greatest_solar_savings['region']}“: {greatest_solar_savings['solar_savings_bgn']:.2f} лв.",
        f"Шестте регионални задачи приключиха за {parallel_seconds:.2f} сек. при приблизително {speedup_estimate:.2f}x паралелно ускорение.",
    ]

    performance = {
        "status": "PASS",
        "engine": "Ray",
        "tasks": len(REGIONS),
        "configured_cpus": 4,
        "available_cpus": available_cpus,
        "records": sum(item["records"] for item in results),
        "timings_seconds": {
            "ray_startup": round(startup_seconds, 3),
            "parallel_execution": round(parallel_seconds, 3),
            "estimated_sequential_tasks": round(estimated_sequential_seconds, 3),
            "total": round(perf_counter() - total_started, 3),
        },
        "parallel_speedup_estimate": round(speedup_estimate, 3),
        "task_timings": {
            item["region"]: item["task_seconds"] for item in results
        },
        "validation_against_pyspark": validation,
        "versions": {
            "ray": version("ray"),
            "pyarrow": version("pyarrow"),
            "pandas": version("pandas"),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "regions_parallel.json", regional_summaries)
    write_json(output_dir / "regional_monthly.json", regional_monthly)
    write_json(output_dir / "insights.json", insights)
    write_json(output_dir / "performance.json", performance)
    return performance


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
        default=Path("reports/ray"),
    )
    parser.add_argument(
        "--pyspark-summary",
        type=Path,
        default=Path("reports/analysis/summary.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_ray_analysis(
        input_path=args.input,
        output_dir=args.output,
        pyspark_summary_path=args.pyspark_summary,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if os.name == "nt":
        short_executable = windows_short_path(sys.executable)
        if os.path.normcase(sys.executable) != os.path.normcase(short_executable):
            command = [
                short_executable,
                "-B",
                "-X",
                "utf8",
                str(Path(__file__).resolve()),
                *sys.argv[1:],
            ]
            raise SystemExit(subprocess.call(command))
    main()
