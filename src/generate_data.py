"""Generate synthetic hourly electricity-consumption data for the project."""

from __future__ import annotations

import argparse
import calendar
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from faker import Faker


CITY_SLOTS = (
    ("Северозападен", "Видин"),
    ("Северен централен", "Русе"),
    ("Североизточен", "Варна"),
    ("Югоизточен", "Бургас"),
    ("Югозападен", "София"),
    ("Южен централен", "Пловдив"),
    ("Северозападен", "Плевен"),
    ("Северен централен", "Велико Търново"),
    ("Североизточен", "Добрич"),
    ("Югоизточен", "Стара Загора"),
    ("Югозападен", "Благоевград"),
    ("Южен централен", "Смолян"),
)

MONTHLY_TEMPERATURE_C = {
    1: 2.0,
    2: 4.0,
    3: 8.0,
    4: 13.0,
    5: 18.0,
    6: 22.0,
    7: 25.0,
    8: 25.0,
    9: 20.0,
    10: 14.0,
    11: 8.0,
    12: 3.0,
}

REGION_TEMPERATURE_OFFSET = {
    "Северозападен": -0.8,
    "Северен централен": -0.3,
    "Североизточен": 0.2,
    "Югоизточен": 1.4,
    "Югозападен": -0.7,
    "Южен централен": 0.6,
}

DAY_PRICE_BGN = 0.25
NIGHT_PRICE_BGN = 0.15

FIELDNAMES = (
    "measurement_id",
    "timestamp",
    "household_id",
    "meter_id",
    "region",
    "city",
    "building_type",
    "heating_type",
    "household_size",
    "has_solar_panels",
    "solar_capacity_kw",
    "temperature_c",
    "is_weekend",
    "tariff_type",
    "price_per_kwh",
    "consumption_kwh",
    "solar_generation_kwh",
    "grid_consumption_kwh",
    "energy_cost_bgn",
)


@dataclass(frozen=True)
class Household:
    household_id: str
    meter_id: str
    region: str
    city: str
    building_type: str
    heating_type: str
    household_size: int
    has_solar_panels: bool
    solar_capacity_kw: float
    base_load_kwh: float


def allocate_values(total: int, weighted_values: tuple[tuple[object, float], ...]) -> list[object]:
    """Create a list whose value counts follow the requested proportions."""
    raw_counts = [total * weight for _, weight in weighted_values]
    counts = [math.floor(value) for value in raw_counts]
    remaining = total - sum(counts)
    order = sorted(
        range(len(weighted_values)),
        key=lambda index: raw_counts[index] - counts[index],
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1

    result: list[object] = []
    for (value, _), count in zip(weighted_values, counts, strict=True):
        result.extend([value] * count)
    return result


def build_households(count: int, seed: int) -> list[Household]:
    """Build balanced, reproducible synthetic household profiles."""
    if count < 1:
        raise ValueError("households must be at least 1")

    rng = random.Random(seed)
    fake = Faker("bg_BG")
    Faker.seed(seed)

    building_types = allocate_values(count, (("apartment", 0.60), ("house", 0.40)))
    heating_types = allocate_values(
        count,
        (("electric", 0.40), ("district", 0.25), ("gas", 0.20), ("other", 0.15)),
    )
    household_sizes = allocate_values(
        count,
        ((1, 0.20), (2, 0.30), (3, 0.25), (4, 0.20), (5, 0.05)),
    )
    rng.shuffle(building_types)
    rng.shuffle(heating_types)
    rng.shuffle(household_sizes)

    house_indices = [index for index, value in enumerate(building_types) if value == "house"]
    apartment_indices = [index for index, value in enumerate(building_types) if value == "apartment"]
    solar_count = round(count * 0.20)
    solar_house_count = min(len(house_indices), round(solar_count * 0.75))
    solar_indices = set(rng.sample(house_indices, solar_house_count))
    remaining_solar = solar_count - len(solar_indices)
    if remaining_solar:
        solar_indices.update(rng.sample(apartment_indices, remaining_solar))

    households: list[Household] = []
    for index in range(count):
        region, city = CITY_SLOTS[index % len(CITY_SLOTS)]
        building_type = str(building_types[index])
        has_solar = index in solar_indices
        if building_type == "house":
            base_load = rng.uniform(0.24, 0.38)
        else:
            base_load = rng.uniform(0.18, 0.30)

        households.append(
            Household(
                household_id=f"HH-{index + 1:04d}",
                meter_id=fake.unique.bothify(text="BG-########"),
                region=region,
                city=city,
                building_type=building_type,
                heating_type=str(heating_types[index]),
                household_size=int(household_sizes[index]),
                has_solar_panels=has_solar,
                solar_capacity_kw=round(rng.uniform(2.5, 6.0), 2) if has_solar else 0.0,
                base_load_kwh=base_load,
            )
        )
    return households


def generate_dataset(
    output_path: Path,
    household_count: int,
    year: int,
    days: int,
    seed: int,
) -> dict[str, object]:
    max_days = 366 if calendar.isleap(year) else 365
    if not 1 <= days <= max_days:
        raise ValueError(f"days must be between 1 and {max_days} for {year}")

    households = build_households(household_count, seed)
    demand_rng = np.random.default_rng(seed + 1)
    weather_rng = np.random.default_rng(seed + 2)
    cloud_rng = np.random.default_rng(seed + 3)
    regions = tuple(REGION_TEMPERATURE_OFFSET)
    start = datetime(year, 1, 1)
    total_hours = days * 24
    expected_rows = household_count * total_hours
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    timestamps = pd.date_range(start=start, periods=total_hours, freq="h")
    timestamp_text = timestamps.strftime("%Y-%m-%dT%H:%M:%S")
    months = timestamps.month.to_numpy()
    hours = timestamps.hour.to_numpy()
    is_weekend = timestamps.dayofweek.to_numpy() >= 5

    winter_tariff = np.isin(months, [11, 12, 1, 2, 3])
    is_night = np.where(
        winter_tariff,
        (hours >= 22) | (hours < 6),
        (hours >= 23) | (hours < 7),
    )
    tariff_type = np.where(is_night, "night", "day")
    price_per_kwh = np.where(is_night, NIGHT_PRICE_BGN, DAY_PRICE_BGN)

    month_temperatures = np.array(
        [MONTHLY_TEMPERATURE_C[int(month)] for month in months]
    )
    daily_cycle = 3.0 * np.sin(2.0 * np.pi * (hours - 8) / 24.0)
    temperatures = {
        region: (
            month_temperatures
            + REGION_TEMPERATURE_OFFSET[region]
            + daily_cycle
            + weather_rng.normal(0.0, 1.8, total_hours)
        )
        for region in regions
    }
    cloud_factors = {
        region: cloud_rng.uniform(0.55, 1.0, total_hours) for region in regions
    }

    daytime_factor = np.where(is_weekend, 1.05, 0.80)
    hourly_factors = np.select(
        [hours < 6, hours < 9, hours < 17, hours < 22],
        [0.55, 1.35, daytime_factor, 1.60],
        default=0.85,
    )
    weekend_factors = np.where(is_weekend, 1.08, 1.0)
    season_factors = np.select(
        [
            np.isin(months, [6, 7, 8]),
            np.isin(months, [3, 4, 5]),
            np.isin(months, [9, 10, 11]),
        ],
        [1.0, 0.72, 0.58],
        default=0.34,
    )
    daylight_curve = np.where(
        (hours >= 7) & (hours <= 19),
        np.maximum(0.0, np.sin(np.pi * (hours - 7) / 12.0)),
        0.0,
    )

    for household_index, household in enumerate(households):
        temperature_c = temperatures[household.region]
        temperature_factor = np.ones(total_hours)
        if household.heating_type == "electric":
            temperature_factor += np.where(
                temperature_c < 15.0,
                np.minimum(1.20, (15.0 - temperature_c) * 0.06),
                0.0,
            )
        temperature_factor += np.where(
            temperature_c > 26.0,
            np.minimum(0.45, (temperature_c - 26.0) * 0.04),
            0.0,
        )

        residents_factor = 1.0 + 0.15 * (household.household_size - 1)
        building_factor = 1.10 if household.building_type == "house" else 0.92
        random_variation = demand_rng.uniform(0.85, 1.15, total_hours)
        consumption_kwh = np.maximum(
            0.05,
            household.base_load_kwh
            * residents_factor
            * building_factor
            * hourly_factors
            * weekend_factors
            * temperature_factor
            * random_variation,
        )

        if household.has_solar_panels:
            solar_generation_kwh = (
                household.solar_capacity_kw
                * daylight_curve
                * season_factors
                * cloud_factors[household.region]
                * 0.75
            )
        else:
            solar_generation_kwh = np.zeros(total_hours)

        grid_consumption_kwh = np.maximum(
            0.0, consumption_kwh - solar_generation_kwh
        )
        energy_cost_bgn = grid_consumption_kwh * price_per_kwh
        first_id = household_index * total_hours + 1

        chunk = pd.DataFrame(
            {
                "measurement_id": np.arange(first_id, first_id + total_hours),
                "timestamp": timestamp_text,
                "household_id": household.household_id,
                "meter_id": household.meter_id,
                "region": household.region,
                "city": household.city,
                "building_type": household.building_type,
                "heating_type": household.heating_type,
                "household_size": household.household_size,
                "has_solar_panels": str(household.has_solar_panels).lower(),
                "solar_capacity_kw": household.solar_capacity_kw,
                "temperature_c": temperature_c,
                "is_weekend": np.where(is_weekend, "true", "false"),
                "tariff_type": tariff_type,
                "price_per_kwh": price_per_kwh,
                "consumption_kwh": consumption_kwh,
                "solar_generation_kwh": solar_generation_kwh,
                "grid_consumption_kwh": grid_consumption_kwh,
                "energy_cost_bgn": energy_cost_bgn,
            },
            columns=FIELDNAMES,
        )
        chunk.to_csv(
            output_path,
            mode="w" if household_index == 0 else "a",
            header=household_index == 0,
            index=False,
            encoding="utf-8",
            float_format="%.4f",
            lineterminator="\n",
        )

    elapsed_seconds = perf_counter() - started
    return {
        "output": str(output_path.resolve()),
        "records": expected_rows,
        "expected_records": expected_rows,
        "households": household_count,
        "hours": total_hours,
        "elapsed_seconds": round(elapsed_seconds, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--households", type=int, default=120)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/electricity_consumption_2025.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_dataset(
        output_path=args.output,
        household_count=args.households,
        year=args.year,
        days=args.days,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
