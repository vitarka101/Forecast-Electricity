// Dashboard tab — client dropdown, dual forecast chart, price chart
// Requires dualLineChart() and renderPriceChart() defined in app.js (loaded first)

(function () {
  const clientSelect   = document.getElementById("client-select");
  const loadBtn        = document.getElementById("load-dashboard");
  const clientBadge    = document.getElementById("client-badge");
  const dualChartTitle = document.getElementById("dual-chart-title");
  const dualChartWrap  = document.getElementById("dual-chart");
  const dualMetrics    = document.getElementById("dual-metrics");
  const priceChartWrap = document.getElementById("price-chart");

  let clientsReady = false;   // dropdown populated
  let chartLoaded  = false;   // first auto-load fired

  // ── Populate dropdown ───────────────────────────────────────────────────────
  async function loadDashboardClients() {
    if (clientsReady) return;
    try {
      const res  = await fetch("/api/v1/clients?source=brocode&limit=500");
      const data = await res.json();
      clientSelect.innerHTML = data.clients
        .map((c) => `<option value="${c.client_id}">${c.client_id} · Cluster ${c.cluster} · ${c.model}</option>`)
        .join("");
      clientsReady = true;
    } catch (_) {
      clientSelect.innerHTML = '<option value="">Error loading clients — run generate_synthetic_forecasts.py</option>';
    }
  }

  // ── Badge row ───────────────────────────────────────────────────────────────
  function renderBadge(data) {
    clientBadge.innerHTML = [
      `<span class="badge">Cluster ${data.cluster} · ${data.cluster_label}</span>`,
      `<span class="badge">${data.model}</span>`,
      `<span class="badge">MAPE ${data.metrics.brocode_mape.toFixed(1)}%</span>`,
      `<span class="badge brocode-badge">Brocode ${data.metrics.brocode_accuracy_pct.toFixed(1)}% accurate</span>`,
      `<span class="badge chicken-badge">chicken_dinner ${data.metrics.chicken_accuracy_pct.toFixed(1)}% accurate</span>`,
    ].join("");
  }

  // ── Dual chart section ──────────────────────────────────────────────────────
  function renderDashboardDualChart(data) {
    dualChartTitle.textContent = `${data.client_id} · Next-Day Hourly Forecast (2014-12-31) · Cluster ${data.cluster} · ${data.model}`;
    dualChartWrap.innerHTML = "";
    dualChartWrap.appendChild(dualLineChart(data.brocode_rows, data.chicken_rows));
  }

  function renderDashboardMetrics(data) {
    const m = data.metrics;
    dualMetrics.innerHTML = [
      `<div class="metric"><span>Brocode Accuracy</span><strong style="color:#176b87">${m.brocode_accuracy_pct.toFixed(1)}%</strong></div>`,
      `<div class="metric"><span>chicken_dinner Accuracy</span><strong style="color:#e07b39">${m.chicken_accuracy_pct.toFixed(1)}%</strong></div>`,
      `<div class="metric"><span>Brocode MAPE</span><strong>${m.brocode_mape.toFixed(1)}%</strong></div>`,
      `<div class="metric"><span>chicken_dinner MAPE</span><strong>${m.chicken_mape.toFixed(1)}%</strong></div>`,
      `<div class="metric"><span>Improvement</span><strong>+${(m.chicken_mape - m.brocode_mape).toFixed(1)} pp</strong></div>`,
    ].join("");
  }

  // ── Price chart section ─────────────────────────────────────────────────────
  function renderDashboardPrice(data) {
    priceChartWrap.innerHTML = "";
    priceChartWrap.appendChild(renderPriceChart(data.future));
  }

  // ── Main load ───────────────────────────────────────────────────────────────
  async function loadDashboard(clientId) {
    if (!clientId) return;
    clientBadge.innerHTML = '<span class="badge">Loading…</span>';
    dualChartWrap.innerHTML = '<p class="muted" style="padding:16px">Loading forecast…</p>';
    priceChartWrap.innerHTML = "";

    try {
      const [dualRes, priceRes] = await Promise.all([
        fetch(`/api/v1/clients/${clientId}/dual-forecast`),
        fetch("/api/v1/price-forecast"),
      ]);
      const dualData  = await dualRes.json();
      const priceData = await priceRes.json();

      if (!dualRes.ok)  throw new Error(dualData.detail  || "Dual forecast fetch failed");
      if (!priceRes.ok) throw new Error(priceData.detail || "Price forecast fetch failed");

      // Reveal sections now that data is ready
      document.getElementById("dual-chart-section").classList.remove("hidden");
      document.getElementById("price-section").classList.remove("hidden");
      document.getElementById("implementation-section").classList.remove("hidden");

      renderBadge(dualData);
      renderDashboardDualChart(dualData);
      renderDashboardMetrics(dualData);
      renderDashboardPrice(priceData);
    } catch (err) {
      clientBadge.innerHTML = `<span class="badge" style="color:red">${err.message}</span>`;
      dualChartWrap.innerHTML = "";
    }
  }

  // ── Events ──────────────────────────────────────────────────────────────────
  loadBtn.addEventListener("click", () => {
    const clientId = clientSelect.value;
    if (clientId) loadDashboard(clientId);
  });

  // Auto-load MT_194 when dashboard tab is first clicked
  document.querySelector('[data-tab="dashboard"]').addEventListener("click", async () => {
    await loadDashboardClients();           // idempotent — skips if already done
    if (chartLoaded) return;               // don't re-fire on subsequent tab clicks
    chartLoaded = true;
    // Pick MT_194 if available, else first option
    const opts = Array.from(clientSelect.options);
    const mt194 = opts.find((o) => o.value === "MT_194");
    clientSelect.value = mt194 ? "MT_194" : (opts[0]?.value || "");
    if (clientSelect.value) loadDashboard(clientSelect.value);
  });

  // Pre-populate dropdown in the background so it's ready when tab is opened
  loadDashboardClients();
})();
