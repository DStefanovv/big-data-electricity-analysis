import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const files = {
  summary: "/data/analysis/summary.json",
  monthly: "/data/analysis/monthly.json",
  hourly: "/data/analysis/hourly.json",
  regions: "/data/analysis/regions.json",
  tariffs: "/data/analysis/tariffs.json",
  insights: "/data/analysis/insights.json",
  faker: "/data/generation_metrics.json",
  dask: "/data/dask_processing_metrics.json",
  pyspark: "/data/analysis/performance.json",
  ray: "/data/ray/performance.json",
};

const integer = new Intl.NumberFormat("bg-BG", { maximumFractionDigits: 0 });
const BGN_PER_EUR = 1.95583;
const decimal = new Intl.NumberFormat("bg-BG", {
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});
const money = new Intl.NumberFormat("bg-BG", {
  style: "currency",
  currency: "EUR",
  maximumFractionDigits: 2,
});

function formatMWh(value) {
  return `${decimal.format(value / 1000)} MWh`;
}

function bgnToEur(value) {
  return value / BGN_PER_EUR;
}

function ChartBox({ title, description, children }) {
  return (
    <article className="chart-box">
      <h3>{title}</h3>
      <p>{description}</p>
      <div className="chart-area">{children}</div>
    </article>
  );
}

function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadData() {
      try {
        const entries = await Promise.all(
          Object.entries(files).map(async ([name, path]) => {
            const response = await fetch(path);
            if (!response.ok) throw new Error(`Проблем при зареждане на ${path}`);
            return [name, await response.json()];
          }),
        );
        setData(Object.fromEntries(entries));
      } catch (loadError) {
        setError(loadError.message);
      }
    }

    loadData();
  }, []);

  if (error) {
    return <main className="status-message">Грешка: {error}</main>;
  }

  if (!data) {
    return <main className="status-message">Зареждане на отчета...</main>;
  }

  const tariffData = data.tariffs.map((item) => ({
    ...item,
    name: item.tariff_type === "day" ? "Дневна" : "Нощна",
    cost_eur: bgnToEur(item.cost_bgn),
    price_eur_per_kwh: bgnToEur(item.price_per_kwh),
  }));

  const conclusions = data.insights.map((insight) => {
    if (insight.includes("соларни спестявания")) {
      return `Изчислените общи соларни спестявания са ${money.format(
        bgnToEur(data.summary.total_solar_savings_bgn),
      )}.`;
    }
    return insight;
  });

  const timings = [
    { operation: "Генериране на данни", library: "Faker", seconds: data.faker.elapsed_seconds },
    { operation: "Обработка и Parquet", library: "Dask", seconds: data.dask.timings_seconds.total },
    { operation: "Описателен анализ", library: "PySpark", seconds: data.pyspark.timings_seconds.total },
    { operation: "Регионален анализ", library: "Ray", seconds: data.ray.timings_seconds.total },
  ];

  return (
    <div className="page">
      <header className="page-header">
        <h1>Анализ на електропотреблението</h1>
        <p>Автоматичен HTML отчет за синтетични данни от 2025 г.</p>
      </header>

      <main>
        <section>
          <h2>Обща информация</h2>
          <div className="summary-grid">
            <article>
              <span>Обработени записи</span>
              <strong>{integer.format(data.summary.records)}</strong>
            </article>
            <article>
              <span>Домакинства</span>
              <strong>{integer.format(data.summary.households)}</strong>
            </article>
            <article>
              <span>Общо потребление</span>
              <strong>{formatMWh(data.summary.total_consumption_kwh)}</strong>
            </article>
            <article>
              <span>Обща цена</span>
              <strong>{money.format(bgnToEur(data.summary.total_cost_bgn))}</strong>
            </article>
          </div>
        </section>

        <section>
          <h2>Визуализации</h2>
          <div className="charts-grid">
            <ChartBox
              title="1. Месечно потребление"
              description="Общо потребление и енергия, използвана от мрежата."
            >
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data.monthly} margin={{ top: 10, right: 20, left: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month_name_bg" />
                  <YAxis tickFormatter={(value) => `${Math.round(value / 1000)}k`} />
                  <Tooltip formatter={(value) => `${decimal.format(value)} kWh`} />
                  <Legend />
                  <Line type="monotone" dataKey="consumption_kwh" name="Общо" stroke="#2563eb" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="grid_consumption_kwh" name="От мрежата" stroke="#64748b" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </ChartBox>

            <ChartBox
              title="2. Почасов профил"
              description="Средно потребление от мрежата за всеки час от денонощието."
            >
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data.hourly} margin={{ top: 10, right: 20, left: 5, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hour" tickFormatter={(value) => `${value}:00`} />
                  <YAxis />
                  <Tooltip formatter={(value) => `${decimal.format(value)} kWh`} labelFormatter={(value) => `${value}:00 ч.`} />
                  <Line type="monotone" dataKey="average_grid_consumption_kwh" name="Средно kWh" stroke="#2563eb" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </ChartBox>

            <ChartBox
              title="3. Потребление по региони"
              description="Годишно потребление от мрежата на едно домакинство."
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.regions} margin={{ top: 10, right: 15, left: 5, bottom: 55 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="region" angle={-30} textAnchor="end" interval={0} height={70} />
                  <YAxis tickFormatter={(value) => `${Math.round(value / 1000)}k`} />
                  <Tooltip formatter={(value) => `${decimal.format(value)} kWh`} />
                  <Bar dataKey="grid_kwh_per_household" name="kWh на домакинство" fill="#2563eb" />
                </BarChart>
              </ResponsiveContainer>
            </ChartBox>

            <ChartBox
              title="4. Дневна и нощна тарифа"
              description="Общ разход според типа тарифа."
            >
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={tariffData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis tickFormatter={(value) => `${Math.round(value / 1000)}k`} />
                  <Tooltip formatter={(value) => money.format(value)} />
                  <Bar dataKey="cost_eur" name="Разход" fill="#2563eb" />
                </BarChart>
              </ResponsiveContainer>
              <div className="tariff-prices">
                {tariffData.map((item) => (
                  <span key={item.tariff_type}>
                    {item.name}: {money.format(item.price_eur_per_kwh)}/kWh
                  </span>
                ))}
              </div>
            </ChartBox>
          </div>
        </section>

        <section>
          <h2>Време за изпълнение</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Основна операция</th>
                  <th>Библиотека</th>
                  <th>Време</th>
                </tr>
              </thead>
              <tbody>
                {timings.map((item) => (
                  <tr key={item.library}>
                    <td>{item.operation}</td>
                    <td>{item.library}</td>
                    <td>{decimal.format(item.seconds)} сек.</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="note">
            Ray: {decimal.format(data.ray.timings_seconds.parallel_execution)} сек. паралелно изчисление и {decimal.format(data.ray.timings_seconds.total)} сек. общо със стартирането; приблизително {decimal.format(data.ray.parallel_speedup_estimate)}× ускорение при 6 задачи.
          </p>
        </section>

        <section>
          <h2>Кратки изводи</h2>
          <ul className="conclusions">
            {conclusions.map((insight) => <li key={insight}>{insight}</li>)}
          </ul>
        </section>

        <section className="details">
          <h2>Данни за отчета</h2>
          <p>Период: 01.01.2025 – 31.12.2025</p>
          <p>Използвани библиотеки: Faker, Dask, PySpark и Ray</p>
          <p>Формат: автоматично генериран HTML отчет</p>
          <p>Цените са показани в евро при фиксиран курс 1 EUR = 1,95583 BGN.</p>
          <p>Данните и цените са синтетични и се използват само за учебни цели.</p>
        </section>
      </main>
    </div>
  );
}

export default App;
