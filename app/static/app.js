// ── DOM references ──────────────────────────────────────────────────────────
const messages    = document.querySelector("#messages");
const form        = document.querySelector("#chat-form");
const queryInput  = document.querySelector("#query");
const clientSearch = document.querySelector("#client-search");
const lookupClient = document.querySelector("#lookup-client");
const clientList  = document.querySelector("#client-list");

// ── Tab switching ─────────────────────────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach((t) => t.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
  });
});

// ── Example pill queries ──────────────────────────────────────────────────────
document.querySelectorAll(".pill").forEach((pill) => {
  pill.addEventListener("click", () => {
    const q = pill.dataset.query || pill.textContent.trim();
    // Switch to chat tab
    document.querySelector('[data-tab="chat"]').click();
    queryInput.value = q;
    sendQuery(q);
  });
});

// ── Message rendering ─────────────────────────────────────────────────────────
function addMessage(text, role = "assistant", payload = null) {
  const el = document.createElement("div");
  el.className = `message ${role}`;
  const summary = document.createElement("div");
  summary.className = "summary";
  summary.textContent = text;
  el.appendChild(summary);
  if (payload) renderPayload(el, payload);
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
}

function renderPayload(container, payload) {
  const data = payload.payload || payload;

  if (payload.intent === "dual_forecast") {
    renderDualForecast(container, payload.payload);
    return;
  }
  if (payload.intent === "price") {
    container.appendChild(renderPriceChart(payload.payload.future));
    return;
  }
  if (payload.intent === "compare" || data.comparison) {
    renderCompare(container, data);
    return;
  }
  if (payload.intent === "cluster" || data.cluster_id !== undefined) {
    renderCluster(container, data);
    return;
  }
  if (payload.intent === "system" || (data.forecast_hourly && data.profile === undefined)) {
    renderSystem(container, data);
    return;
  }
  if (data.client) {
    renderClient(container, data.client, data.cluster);
  }
}

// ── Metric grid helper ────────────────────────────────────────────────────────
function metricGrid(items) {
  const grid = document.createElement("div");
  grid.className = "metric-grid";
  for (const [label, value] of items) {
    const m = document.createElement("div");
    m.className = "metric";
    m.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
    grid.appendChild(m);
  }
  return grid;
}

// ── Dual forecast rendering ───────────────────────────────────────────────────
function renderDualForecast(container, data) {
  container.appendChild(metricGrid([
    ["Client", data.client_id],
    ["Cluster", `${data.cluster} · ${data.cluster_label}`],
    ["Brocode Model", data.model],
    ["Brocode MAPE", `${data.metrics.brocode_mape.toFixed(1)}%`],
  ]));
  const wrap = document.createElement("div");
  wrap.className = "chart-wrap";
  wrap.appendChild(dualLineChart(data.brocode_rows, data.chicken_rows));
  container.appendChild(wrap);
  container.appendChild(metricGrid([
    ["Brocode Accuracy", `${data.metrics.brocode_accuracy_pct.toFixed(1)}%`],
    ["chicken_dinner Accuracy", `${data.metrics.chicken_accuracy_pct.toFixed(1)}%`],
    ["chicken_dinner MAPE", `${data.metrics.chicken_mape.toFixed(1)}%`],
    ["Improvement", `+${(data.metrics.chicken_mape - data.metrics.brocode_mape).toFixed(1)}pp`],
  ]));
}

// ── Dual-team line chart (SVG) ────────────────────────────────────────────────
function dualLineChart(brocodeRows, chickenRows) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "chart");
  svg.setAttribute("viewBox", "0 0 720 220");

  const actuals  = brocodeRows.map((r) => r.actual_kwh);
  const brocode  = brocodeRows.map((r) => r.pred_kwh);
  const chicken  = chickenRows.map((r) => r.pred_kwh);
  const allVals  = [...actuals, ...brocode, ...chicken].filter((v) => v != null && !isNaN(v));

  const min  = Math.min(...allVals, 0);
  const max  = Math.max(...allVals, 1);
  const span = Math.max(max - min, 1);
  const n    = brocodeRows.length;

  function toPath(vals) {
    return vals.map((v, i) => {
      const x = 30 + (i * 660) / Math.max(n - 1, 1);
      const y = 185 - ((v - min) * 150) / span;
      return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    }).join(" ");
  }

  const labels = brocodeRows.map((r, i) => {
    if (i % 6 !== 0) return "";
    const x = 30 + (i * 660) / Math.max(n - 1, 1);
    return `<text x="${x.toFixed(1)}" y="205" font-size="10" fill="#667085" text-anchor="middle">H${r.hour}</text>`;
  }).join("");

  svg.innerHTML = `
    <line x1="30" y1="185" x2="690" y2="185" stroke="#e2e8f0" stroke-width="1"/>
    <path d="${toPath(actuals)}"  fill="none" stroke="#17202a" stroke-width="1.5" stroke-dasharray="4 3" opacity="0.6"/>
    <path d="${toPath(brocode)}"  fill="none" stroke="#176b87" stroke-width="2.5"/>
    <path d="${toPath(chicken)}"  fill="none" stroke="#e07b39" stroke-width="2" stroke-dasharray="7 4"/>
    ${labels}
  `;
  return svg;
}

// ── Price bar chart (SVG) ─────────────────────────────────────────────────────
function renderPriceChart(futureRows) {
  if (!futureRows || !futureRows.length) {
    const d = document.createElement("div");
    d.textContent = "No price data available.";
    return d;
  }
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "chart");
  svg.setAttribute("viewBox", "0 0 720 200");

  const prices = futureRows.map((r) => r.forecast_price);
  const uppers = futureRows.map((r) => r.upper);
  const lowers = futureRows.map((r) => r.lower);
  const maxP   = Math.max(...uppers, 0.07);
  const minP   = Math.min(...lowers, 0.02);
  const spanP  = Math.max(maxP - minP, 0.005);
  const n      = futureRows.length;
  const totalW = 660;
  const barW   = Math.floor(totalW / n) - 8;

  let bars = "";
  futureRows.forEach((r, i) => {
    const x    = 30 + (totalW / n) * i + 4;
    const barH = ((r.forecast_price - minP) / spanP) * 140;
    const barY = 160 - barH;
    const label = r.date.slice(5); // MM-DD
    const cents = (r.forecast_price * 100).toFixed(2);
    bars += `
      <rect x="${x.toFixed(1)}" y="${barY.toFixed(1)}" width="${barW}" height="${barH.toFixed(1)}" fill="#176b87" opacity="0.85" rx="3"/>
      <text x="${(x + barW / 2).toFixed(1)}" y="${(barY - 4).toFixed(1)}" font-size="9" fill="#176b87" text-anchor="middle">${cents}c</text>
      <text x="${(x + barW / 2).toFixed(1)}" y="178" font-size="9" fill="#667085" text-anchor="middle">${label}</text>
    `;
  });

  svg.innerHTML = `
    <line x1="30" y1="160" x2="690" y2="160" stroke="#e2e8f0"/>
    <text x="30" y="14" font-size="12" fill="#17202a" font-weight="600">EUR/kWh — Next 7 days</text>
    ${bars}
  `;
  return svg;
}

// ── Original render helpers (kept for /agent/query fallback) ──────────────────
function renderClient(container, client, cluster) {
  const status = client.output_status || "ok";
  container.appendChild(metricGrid([
    ["Client", client.client_id],
    ["Cluster", `${client.cluster_id} · ${cluster.profile.label}`],
    ["Assigned model", client.assigned_model || "n/a"],
    ["Status", status],
  ]));
  if (status === "ok" && client.forecast_hourly && client.forecast_hourly.length) {
    container.appendChild(metricGrid([
      ["Next-day total", `${client.forecast_daily.total_kwh.toFixed(2)} kWh`],
      ["Rows", client.forecast_daily.source_rows || client.forecast_hourly.length],
      ["Output end", client.model_output_summary?.output_end || "n/a"],
      ["Mean hourly", `${client.profile.mean_hourly_kwh.toFixed(2)} kWh`],
    ]));
    container.appendChild(lineChart(client.forecast_hourly, "predicted_kwh"));
  }
}

function renderCompare(container, data) {
  const comp = data.comparison;
  container.appendChild(metricGrid([
    [comp.left_client_id, `${comp.left_daily_forecast_kwh.toFixed(2)} kWh`],
    [comp.right_client_id, `${comp.right_daily_forecast_kwh.toFixed(2)} kWh`],
    ["Delta", `${comp.absolute_delta_kwh.toFixed(2)} kWh`],
    ["Same cluster", comp.same_cluster ? "Yes" : "No"],
  ]));
  for (const client of data.clients) {
    container.appendChild(lineChart(client.forecast_hourly, "predicted_kwh", client.client_id));
  }
}

function renderCluster(container, cluster) {
  container.appendChild(metricGrid([
    ["Cluster", cluster.cluster_id],
    ["Label", cluster.profile.label],
    ["Clients", cluster.profile.client_count],
    ["With outputs", cluster.profile.predicted_client_count || 0],
  ]));
  if (cluster.forecast_hourly && cluster.forecast_hourly.length) {
    container.appendChild(metricGrid([
      ["Next-day total", `${cluster.forecast_daily.total_kwh.toFixed(2)} kWh`],
      ["Rows", cluster.forecast_daily.source_rows || cluster.forecast_hourly.length],
      ["Models", cluster.profile.assigned_models || "n/a"],
      ["Peak hour", cluster.profile.peak_hour ?? "n/a"],
    ]));
    container.appendChild(lineChart(cluster.forecast_hourly, "predicted_kwh"));
  }
}

function renderSystem(container, system) {
  container.appendChild(metricGrid([
    ["Scope", "All clients"],
    ["Next day", `${system.forecast_daily.total_kwh.toFixed(2)} kWh`],
    ["Horizon", `${system.forecast_hourly.length} hours`],
    ["Generated", system.generated_at || "artifact"],
  ]));
  container.appendChild(lineChart(system.forecast_hourly, "predicted_kwh"));
}

function lineChart(rows, valueKey, title = "Forecast") {
  if (!rows || !rows.length) {
    const empty = document.createElement("div");
    empty.className = "message";
    empty.textContent = "No matching model-output rows are available for this selection.";
    return empty;
  }
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "chart");
  svg.setAttribute("viewBox", "0 0 720 210");
  const values = rows.map((r) => Number(r[valueKey] || 0));
  const maxV = Math.max(...values, 1);
  const minV = Math.min(...values, 0);
  const span = Math.max(maxV - minV, 1);
  const points = values.map((v, i) => {
    const x = 30 + (i * 660) / Math.max(values.length - 1, 1);
    const y = 175 - ((v - minV) * 140) / span;
    return `${x},${y}`;
  }).join(" ");
  svg.innerHTML = `
    <text x="30" y="24" font-size="14" fill="#17202a">${title}</text>
    <line x1="30" y1="175" x2="690" y2="175" stroke="#d8dee8" />
    <polyline points="${points}" fill="none" stroke="#176b87" stroke-width="3" />
    <text x="30" y="198" font-size="11" fill="#667085">${rows[0]?.datetime || ""}</text>
    <text x="545" y="198" font-size="11" fill="#667085">${rows[rows.length - 1]?.datetime || ""}</text>
  `;
  return svg;
}

// ── Main query handler ────────────────────────────────────────────────────────
async function sendQuery(query) {
  addMessage(query, "user");
  const lq = query.toLowerCase();

  // Price intent
  if (lq.includes("price") || lq.includes("omie") || lq.includes("electricity price")) {
    try {
      const res  = await fetch("/api/v1/price-forecast");
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Price fetch failed");
      const el = document.createElement("div");
      el.className = "message assistant";
      const s = document.createElement("div");
      s.className = "summary";
      s.textContent = "Brocode Innovation: Here is the 7-day daily electricity price forecast (EUR/kWh) using Prophet on OMIE spot market data.";
      el.appendChild(s);
      const wrap = document.createElement("div");
      wrap.className = "chart-wrap";
      wrap.appendChild(renderPriceChart(data.future));
      el.appendChild(wrap);
      messages.appendChild(el);
      messages.scrollTop = messages.scrollHeight;
    } catch (err) {
      addMessage(err.message, "error");
    }
    return;
  }

  // Dual forecast intent — client ID + forecast keywords
  const clientMatch = query.match(/\bMT[_\s-]?(\d{1,3})\b/i);
  const forecastIntent = lq.includes("forecast") || lq.includes("predict") || lq.includes("next day") || lq.includes("tomorrow");
  if (clientMatch && forecastIntent) {
    const clientId = `MT_${String(parseInt(clientMatch[1])).padStart(3, "0")}`;
    try {
      const res  = await fetch(`/api/v1/clients/${clientId}/dual-forecast`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Forecast fetch failed");
      const el = document.createElement("div");
      el.className = "message assistant";
      const s = document.createElement("div");
      s.className = "summary";
      s.textContent = `${data.client_id} · Cluster ${data.cluster} (${data.cluster_label}) · Model: ${data.model} · Brocode ${data.metrics.brocode_accuracy_pct.toFixed(1)}% accurate vs chicken_dinner ${data.metrics.chicken_accuracy_pct.toFixed(1)}%`;
      el.appendChild(s);
      renderDualForecast(el, data);
      messages.appendChild(el);
      messages.scrollTop = messages.scrollHeight;
    } catch (err) {
      addMessage(err.message, "error");
    }
    return;
  }

  // Fallback to original agent query
  try {
    const res  = await fetch("/api/v1/agent/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");
    addMessage(data.summary, "assistant", data);
  } catch (err) {
    addMessage(err.message, "error");
  }
}

// ── Form & sidebar wiring ─────────────────────────────────────────────────────
form.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;
  queryInput.value = "";
  sendQuery(query);
});

lookupClient.addEventListener("click", () => {
  const value = clientSearch.value.trim();
  if (value) sendQuery(`${value} forecast next day`);
});

async function loadClients() {
  try {
    // Try brocode source first (full 370 clients), fall back to existing store
    const res  = await fetch("/api/v1/clients?source=brocode&limit=40");
    const data = await res.json();
    clientList.innerHTML = "";
    for (const client of data.clients || []) {
      const item = document.createElement("div");
      item.className = "client-item";
      const cluster = client.cluster ?? client.cluster_id ?? "?";
      const avg = Number(client.mean_hourly_kwh || 0).toFixed(2);
      item.innerHTML = `<strong>${client.client_id}</strong><span>Cluster ${cluster} · ${avg} kWh avg</span>`;
      item.addEventListener("click", () => sendQuery(`${client.client_id} forecast next day`));
      clientList.appendChild(item);
    }
  } catch (err) {
    clientList.textContent = "Run: python scripts/generate_synthetic_forecasts.py first.";
  }
}

addMessage('Try: "MT_194 forecast next day" or "Compare MT_001 vs MT_199". For the price forecast and full dashboard, switch to the Dashboard tab.');
loadClients();
