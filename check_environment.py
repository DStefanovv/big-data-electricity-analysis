"""Small functional checks for the four project libraries."""
import os
import sys
from pathlib import Path
from time import perf_counter


def main():
    from faker import Faker
    import pandas as pd
    import dask.dataframe as dd
    from pyspark.sql import SparkSession, functions as F
    import ray

    os.environ['JAVA_HOME'] = os.environ.get('PROJECT_JAVA_HOME', r'C:\Program Files\Java\jdk-17')
    import ctypes
    import pyspark
    def short_path(value):
        buffer = ctypes.create_unicode_buffer(32768)
        if not ctypes.windll.kernel32.GetShortPathNameW(str(value), buffer, len(buffer)):
            raise OSError('Cannot resolve Windows short path: ' + str(value))
        return buffer.value
    os.environ['SPARK_HOME'] = short_path(Path(pyspark.__file__).parent)
    os.environ['PYSPARK_PYTHON'] = short_path(sys.executable)
    os.environ['PYSPARK_DRIVER_PYTHON'] = short_path(sys.executable)
    os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
    os.environ['RAY_USAGE_STATS_ENABLED'] = '0'
    if not (Path(os.environ['JAVA_HOME']) / 'bin' / 'java.exe').is_file():
        raise RuntimeError('Set PROJECT_JAVA_HOME to your Java 17 installation.')

    start = perf_counter()
    fake = Faker('bg_BG')
    fake.seed_instance(42)
    rows = [(i, i % 4, fake.random_int(min=100, max=3000)) for i in range(1000)]
    expected = {region: sum(row[2] for row in rows if row[1] == region) for region in range(4)}
    assert len(rows) == 1000 and all(100 <= row[2] <= 3000 for row in rows)
    print(f'Faker: PASS - {len(rows)} measurements ({perf_counter()-start:.2f}s)', flush=True)

    start = perf_counter()
    frame = pd.DataFrame(rows, columns=['id', 'region', 'energy_wh'])
    ddf = dd.from_pandas(frame, npartitions=4)
    result = ddf.groupby('region').energy_wh.sum().compute().to_dict()
    assert result == expected, (result, expected)
    print(f'Dask: PASS - regional totals match ({perf_counter()-start:.2f}s)', flush=True)

    start = perf_counter()
    spark = SparkSession.builder.master('local[2]').appName('EnergyEnvironmentCheck').config('spark.ui.enabled', 'false').config('spark.sql.shuffle.partitions', '2').getOrCreate()
    try:
        spark.sparkContext.setLogLevel('ERROR')
        sdf = spark.createDataFrame(rows, 'id long, region long, energy_wh long')
        assert sdf.count() == 1000
        result = {r['region']: r['total'] for r in sdf.groupBy('region').agg(F.sum('energy_wh').alias('total')).collect()}
        assert result == expected, (result, expected)
        print(f'PySpark: PASS - count and regional totals match ({perf_counter()-start:.2f}s)', flush=True)
    finally:
        spark.stop()

    start = perf_counter()
    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
    try:
        @ray.remote
        def total_energy(values):
            return sum(values)
        tasks = [total_energy.remote([row[2] for row in rows if row[1] == region]) for region in range(4)]
        result = dict(zip(range(4), ray.get(tasks, timeout=90)))
        assert result == expected, (result, expected)
        print(f'Ray: PASS - four parallel tasks match ({perf_counter()-start:.2f}s)', flush=True)
    finally:
        ray.shutdown()
    print('ALL CHECKS PASSED', flush=True)


if __name__ == '__main__':
    # Windows worker launchers need paths without spaces or Cyrillic characters.
    import ctypes
    import subprocess
    buffer = ctypes.create_unicode_buffer(32768)
    if not ctypes.windll.kernel32.GetShortPathNameW(sys.executable, buffer, len(buffer)):
        raise OSError('Cannot resolve Python executable path.')
    executable = buffer.value
    if sys.executable != executable:
        raise SystemExit(subprocess.call([executable, '-X', 'utf8', str(Path(__file__).resolve())]))
    main()


