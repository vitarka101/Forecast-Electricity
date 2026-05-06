const title = document.querySelector("#comparison-title");
const subtitle = document.querySelector("#comparison-subtitle");
const banner = document.querySelector("#accuracy-banner");
const metrics = document.querySelector("#comparison-metrics");
const chart = document.querySelector("#comparison-chart");
const tbody = document.querySelector("#evidence-table tbody");
const clientInput = document.querySelector("#client-id");
const loadButton = document.querySelector("#load-client");

loadButton.addEventListener("click", () => loadComparison(clientInput.value.trim()));
clientInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") loadComparison(clientInput.value.trim());
});

async function loadComparison(clientId = "MT_200") {
  try {
    const url = `/api/v1/improvements/chart?client_id=${encodeURIComponent(clientId || "MT_200")}`;
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");
    renderComparison(data);
  } catch (err) {
    banner.textContent = err.message;
    metrics.innerHTML = "";
    chart.innerHTML = "";
    tbody.innerHTML = "";
  }
}

function renderComparison(data) {
  const ours = data.metrics.ours;
  const other = data.metrics.other_team;
  const delta = data.metrics.accuracy_delta_pct;
  title.textContent = `${data.client_id} Forecast Comparison`;
  subtitle.textContent = `${data.our_label} vs ${data.other_team_label}`;
  banner.textContent = `Ours is more accurate by ${delta.toFixed(3)} percentage points on ${data.rows_compared} shared prediction rows.`;

  metrics.innerHTML = [
    metric("Our accuracy", formatPct(ours.forecast_accuracy_pct)),
    metric("Other team accuracy", formatPct(other.forecast_accuracy_pct)),
    metric("Our MAPE", formatPct(ours.mape_pct)),
    metric("Other team MAPE", formatPct(other.mape_pct)),
  ].join("");

  tbody.innerHTML = [
    row("Ours", data.our_model, ours),
    row("Other team", data.other_team_model, other),
  ].join("");

  chart.innerHTML = "";
  chart.appendChild(multiLineChart(data.chart_rows));
}

function metric(label, value) {
  return `<div class="metric"><span>${label}</span><strong>${value}</strong></div>`;
}

function row(team, model, m) {
  return `
    <tr>
      <td>${team}</td>
      <td>${model}</td>
      <td>${formatPct(m.forecast_accuracy_pct)}</td>
      <td>${formatPct(m.mape_pct)}</td>
      <td>${formatNumber(m.rmse)}</td>
      <td>${formatNumber(m.mae)}</td>
    </tr>
  `;
}

function multiLineChart(rows) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "chart large-chart");
  svg.setAttribute("viewBox", "0 0 920 360");
  if (!rows.length) {
    svg.innerHTML = `<text x="40" y="180" fill="#667085">No rows available for chart.</text>`;
    return svg;
  }

  const keys = ["actual_kwh", "our_predicted_kwh", "other_predicted_kwh"];
  const values = rows.flatMap((row) => keys.map((key) => Number(row[key] || 0)));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 1);

  const pathFor = (key) => rows.map((row, i) => {
    const x = 50 + (i * 820) / Math.max(rows.length - 1, 1);
    const y = 295 - ((Number(row[key] || 0) - min) * 235) / span;
    return `${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
  }).join(" ");

  svg.innerHTML = `
    <line x1="50" y1="295" x2="870" y2="295" stroke="#d8dee8" />
    <line x1="50" y1="60" x2="50" y2="295" stroke="#d8dee8" />
    <text x="50" y="35" font-size="13" fill="#667085">${max.toFixed(1)} kWh</text>
    <text x="50" y="330" font-size="12" fill="#667085">${rows[0].datetime}</text>
    <text x="625" y="330" font-size="12" fill="#667085">${rows[rows.length - 1].datetime}</text>
    <path d="${pathFor("actual_kwh")}" fill="none" stroke="#17202a" stroke-width="3" />
    <path d="${pathFor("our_predicted_kwh")}" fill="none" stroke="#177245" stroke-width="3" />
    <path d="${pathFor("other_predicted_kwh")}" fill="none" stroke="#9a5b00" stroke-width="3" stroke-dasharray="7 5" />
  `;
  return svg;
}

function formatPct(value) {
  return value === undefined || value === null ? "n/a" : `${Number(value).toFixed(2)}%`;
}

function formatNumber(value) {
  return value === undefined || value === null ? "n/a" : Number(value).toFixed(2);
}

loadComparison("MT_200");
