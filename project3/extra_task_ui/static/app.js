const initialDataNode = document.getElementById("initialData");
const initialData = initialDataNode ? JSON.parse(initialDataNode.textContent || "{}") : {};

const featurePalette = {
  count: "#60a5fa",
  tfidf: "#34d399",
  pmi: "#f59e0b",
  word2vec: "#f472b6",
  glove: "#a78bfa",
};

const modelPalette = {
  rnn: "#38bdf8",
  birnn: "#f97316",
  lstm: "#a78bfa",
};

const appState = {
  allRows: initialData?.task5?.rows || [],
  trainingRows: initialData?.task5_training?.rows || [],
  featureFilter: "all",
  modelFilter: "all",
  sortBy: "feature",
  sortDir: "asc",
  activeTask4Tab: "neighbors",
  bubblePoints: [],
  raceAnimationId: null,
  seriesColorMap: {},
};

function prettyPercent(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function toFixed(value, digits = 4) {
  return Number(value || 0).toFixed(digits);
}

function buildSeriesColorMap() {
  const keys = Array.from(
    new Set(
      appState.allRows.map((item) => `${item.feature}:${item.model}`)
        .concat(appState.trainingRows.map((item) => `${item.feature}:${item.model}`)),
    ),
  ).sort((a, b) => a.localeCompare(b));

  const total = Math.max(keys.length, 1);
  const map = {};
  keys.forEach((key, index) => {
    const hue = Math.round((index * 360) / total);
    const saturation = 78;
    const lightness = 62;
    map[key] = `hsl(${hue} ${saturation}% ${lightness}%)`;
  });
  appState.seriesColorMap = map;
}

function getSeriesColor(feature, model) {
  const key = `${feature}:${model}`;
  return appState.seriesColorMap[key] || featurePalette[feature] || modelPalette[model] || "#60a5fa";
}

function prepareCanvas(canvas) {
  const ctx = canvas.getContext("2d");
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.max(window.devicePixelRatio || 1, 1);
  const displayWidth = Math.max(Math.round(rect.width || canvas.clientWidth || canvas.width), 1);
  const displayHeight = Math.max(Math.round(rect.height || canvas.clientHeight || canvas.height), 1);

  const targetWidth = Math.round(displayWidth * dpr);
  const targetHeight = Math.round(displayHeight * dpr);

  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth;
    canvas.height = targetHeight;
  }

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, displayWidth, displayHeight);
  return { ctx, width: displayWidth, height: displayHeight };
}

function renderTaskCards(cards) {
  const host = document.getElementById("taskCards");
  host.innerHTML = "";
  cards.forEach((card) => {
    const article = document.createElement("article");
    article.className = "task-card";
    article.style.background = card.gradient;
    article.innerHTML = `
      <div class="label">${card.task}</div>
      <h3>${card.title}</h3>
      <div class="metric">${card.metric}</div>
      <div class="sub">${card.metric_label} · ${card.sub}</div>
    `;
    host.appendChild(article);
  });
}

function renderHighlights(data) {
  const best = data?.task5?.best || {};
  const bestSetup = document.getElementById("bestSetup");
  const neighbor = document.getElementById("avgNeighborJaccard");
  const equation = document.getElementById("avgEquationJaccard");

  if (best.feature && best.model) {
    bestSetup.textContent = `${best.feature.toUpperCase()} + ${best.model.toUpperCase()} (${prettyPercent(best.accuracy)})`;
  } else {
    bestSetup.textContent = "No data";
  }

  neighbor.textContent = Number(data?.task4_overlap?.neighbor_avg_jaccard || 0).toFixed(4);
  equation.textContent = Number(data?.task4_overlap?.equation_avg_jaccard || 0).toFixed(4);
}

function renderBestByFeature(rows) {
  const host = document.getElementById("bestByFeature");
  host.innerHTML = "";
  rows.forEach((row) => {
    const div = document.createElement("div");
    div.className = "best-item";
    div.innerHTML = `
      <div><strong>${row.feature.toUpperCase()}</strong> → ${row.best_model.toUpperCase()}</div>
      <div>Accuracy: ${prettyPercent(row.accuracy)} · Macro-F1: ${prettyPercent(row.macro_f1)}</div>
    `;
    host.appendChild(div);
  });
}

function initFilters(rows) {
  const featureFilter = document.getElementById("featureFilter");
  const modelFilter = document.getElementById("modelFilter");
  const features = Array.from(new Set(rows.map((item) => item.feature))).sort();
  const models = Array.from(new Set(rows.map((item) => item.model))).sort();

  featureFilter.innerHTML = ["<option value='all'>All features</option>"]
    .concat(features.map((item) => `<option value='${item}'>${item.toUpperCase()}</option>`))
    .join("");

  modelFilter.innerHTML = ["<option value='all'>All models</option>"]
    .concat(models.map((item) => `<option value='${item}'>${item.toUpperCase()}</option>`))
    .join("");

  featureFilter.addEventListener("change", (event) => {
    appState.featureFilter = event.target.value;
    rerenderInteractive();
  });

  modelFilter.addEventListener("change", (event) => {
    appState.modelFilter = event.target.value;
    rerenderInteractive();
  });
}

function getFilteredRows() {
  return appState.allRows.filter((row) => {
    const featureMatch = appState.featureFilter === "all" || row.feature === appState.featureFilter;
    const modelMatch = appState.modelFilter === "all" || row.model === appState.modelFilter;
    return featureMatch && modelMatch;
  });
}

function getFilteredTrainingRows() {
  return appState.trainingRows.filter((row) => {
    const featureMatch = appState.featureFilter === "all" || row.feature === appState.featureFilter;
    const modelMatch = appState.modelFilter === "all" || row.model === appState.modelFilter;
    return featureMatch && modelMatch;
  });
}

function sortRows(rows) {
  const sorted = [...rows];
  const dir = appState.sortDir === "asc" ? 1 : -1;

  sorted.sort((a, b) => {
    if (appState.sortBy === "feature") {
      if (a.feature === b.feature) return a.model.localeCompare(b.model) * dir;
      return a.feature.localeCompare(b.feature) * dir;
    }
    if (appState.sortBy === "test_accuracy") {
      return (Number(a.test_accuracy) - Number(b.test_accuracy)) * dir;
    }
    if (appState.sortBy === "test_macro_f1") {
      return (Number(a.test_macro_f1) - Number(b.test_macro_f1)) * dir;
    }
    return 0;
  });

  return sorted;
}

function setSort(column) {
  if (appState.sortBy === column) {
    appState.sortDir = appState.sortDir === "asc" ? "desc" : "asc";
  } else {
    appState.sortBy = column;
    appState.sortDir = column === "feature" ? "asc" : "desc";
  }
  rerenderInteractive();
}

function renderPerformanceTable(rows) {
  const body = document.getElementById("perfBody");
  body.innerHTML = "";

  sortRows(rows).forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.feature}</td>
      <td>${row.model.toUpperCase()}</td>
      <td>${prettyPercent(row.test_accuracy)}</td>
      <td>${prettyPercent(row.test_macro_f1)}</td>
    `;
    body.appendChild(tr);
  });
}

function renderTask4Tab() {
  const tabNeighbors = document.getElementById("tabNeighbors");
  const tabEquations = document.getElementById("tabEquations");
  const body = document.getElementById("task4TabBody");

  tabNeighbors.classList.toggle("active", appState.activeTask4Tab === "neighbors");
  tabEquations.classList.toggle("active", appState.activeTask4Tab === "equations");

  const rows = appState.activeTask4Tab === "neighbors"
    ? (initialData?.task4_detail?.neighbors || [])
    : (initialData?.task4_detail?.equations || []);

  body.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.word2vec_model.toUpperCase()}</td>
      <td>${row.query_word}</td>
      <td>${toFixed(row.jaccard_similarity, 4)}</td>
      <td>${row.overlap_count}</td>
      <td>${row.overlap_words || "-"}</td>
    `;
    body.appendChild(tr);
  });
}

function drawBubble(ctx, x, y, radius, color, label) {
  const gradient = ctx.createRadialGradient(x - radius * 0.3, y - radius * 0.4, 2, x, y, radius);
  gradient.addColorStop(0, "rgba(255,255,255,0.9)");
  gradient.addColorStop(0.2, color);
  gradient.addColorStop(1, "rgba(255,255,255,0.08)");

  ctx.save();
  ctx.beginPath();
  ctx.fillStyle = gradient;
  ctx.shadowColor = "rgba(255,255,255,0.28)";
  ctx.shadowBlur = 20;
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  ctx.fillStyle = "#f8fafc";
  ctx.font = "600 12px Inter";
  ctx.fillText(label, x - radius * 0.7, y + radius + 16);
}

function renderBubbleChart(rows) {
  const canvas = document.getElementById("bubbleCanvas");
  const { ctx, width, height } = prepareCanvas(canvas);

  const padLeft = 72;
  const padBottom = 58;
  const plotWidth = width - padLeft - 32;
  const plotHeight = height - 30 - padBottom;

  ctx.strokeStyle = "rgba(203, 213, 225, 0.4)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padLeft, 20);
  ctx.lineTo(padLeft, 20 + plotHeight);
  ctx.lineTo(padLeft + plotWidth, 20 + plotHeight);
  ctx.stroke();

  ctx.fillStyle = "#dbeafe";
  ctx.font = "500 12px Inter";
  ctx.fillText("Accuracy", 16, 24 + plotHeight / 2);
  ctx.fillText("Macro-F1", padLeft + plotWidth / 2 - 26, height - 18);

  appState.bubblePoints = [];

  rows.forEach((row) => {
    const x = padLeft + Number(row.test_macro_f1) * plotWidth;
    const y = 20 + (1 - Number(row.test_accuracy)) * plotHeight;
    const radius = 10 + Math.sqrt(Math.max(Number(row.test_accuracy) * Number(row.test_macro_f1), 0)) * 28;
    const featureColor = featurePalette[row.feature] || "#60a5fa";
    drawBubble(ctx, x, y, radius, featureColor, `${row.feature}:${row.model}`);

    appState.bubblePoints.push({ x, y, radius, row });
  });
}

function installBubbleTooltip() {
  const canvas = document.getElementById("bubbleCanvas");
  const tooltip = document.getElementById("bubbleTooltip");

  canvas.addEventListener("mousemove", (event) => {
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    const hit = appState.bubblePoints.find((item) => {
      const dx = mouseX - item.x;
      const dy = mouseY - item.y;
      return dx * dx + dy * dy <= item.radius * item.radius;
    });

    if (!hit) {
      tooltip.style.display = "none";
      return;
    }

    const row = hit.row;
    tooltip.style.display = "block";
    tooltip.style.left = `${event.pageX + 16}px`;
    tooltip.style.top = `${event.pageY + 12}px`;
    tooltip.innerHTML = `
      <div><strong>${row.feature.toUpperCase()} + ${row.model.toUpperCase()}</strong></div>
      <div>Accuracy: ${prettyPercent(row.test_accuracy)}</div>
      <div>Macro-F1: ${prettyPercent(row.test_macro_f1)}</div>
      <div>Score: ${toFixed(Math.sqrt(Number(row.test_accuracy) * Number(row.test_macro_f1)), 4)}</div>
    `;
  });

  canvas.addEventListener("mouseleave", () => {
    tooltip.style.display = "none";
  });
}

function renderRaceFrame(epochProgress) {
  const canvas = document.getElementById("raceCanvas");
  const { ctx, width, height } = prepareCanvas(canvas);

  const filteredTraining = getFilteredTrainingRows();

  const groupedAll = {};
  filteredTraining.forEach((item) => {
    const key = `${item.feature}:${item.model}`;
    if (!groupedAll[key]) groupedAll[key] = [];
    groupedAll[key].push(item);
  });

  const legendLabels = Object.keys(groupedAll)
    .map((key) => key.replace(":", "/"))
    .sort((a, b) => a.localeCompare(b));
  const legendColumns = Math.max(1, Math.min(3, Math.floor((width - 40) / 220)));
  const legendRows = Math.max(1, Math.ceil(Math.max(legendLabels.length, 1) / legendColumns));
  const legendRowHeight = 16;
  const legendHeight = 16 + legendRows * legendRowHeight;

  const padL = 68;
  const padR = 24;
  const padT = 20;
  const padB = 56 + legendHeight;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  ctx.strokeStyle = "rgba(203, 213, 225, 0.35)";
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + plotH);
  ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();

  const grouped = groupedAll;

  let maxLoss = 1;
  Object.values(grouped).forEach((series) => {
    series.forEach((point) => {
      maxLoss = Math.max(maxLoss, Number(point.avg_loss));
    });
  });

  const maxEpoch = Math.max(1, ...filteredTraining.map((item) => Number(item.epoch) || 1));

  const legendItems = [];

  Object.entries(grouped).forEach(([key, series]) => {
    const [feature, model] = key.split(":");
    const color = getSeriesColor(feature, model);
    const orderedSeries = [...series].sort((a, b) => Number(a.epoch) - Number(b.epoch));
    const visibleSeries = orderedSeries.filter((point) => Number(point.epoch) <= epochProgress);
    const nextPoint = orderedSeries.find((point) => Number(point.epoch) > epochProgress);

    if (visibleSeries.length === 0) {
      return;
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();

    visibleSeries.forEach((point, index) => {
      const x = padL + (Number(point.epoch) / maxEpoch) * plotW;
      const y = padT + (Number(point.avg_loss) / maxLoss) * plotH;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    let markerEpoch = Number(visibleSeries[visibleSeries.length - 1].epoch);
    let markerLoss = Number(visibleSeries[visibleSeries.length - 1].avg_loss);

    if (nextPoint) {
      const previousPoint = visibleSeries[visibleSeries.length - 1];
      const previousEpoch = Number(previousPoint.epoch);
      const nextEpoch = Number(nextPoint.epoch);
      const span = Math.max(nextEpoch - previousEpoch, 1e-6);
      const t = Math.min(Math.max((epochProgress - previousEpoch) / span, 0), 1);
      markerEpoch = previousEpoch + (nextEpoch - previousEpoch) * t;
      markerLoss = Number(previousPoint.avg_loss) + (Number(nextPoint.avg_loss) - Number(previousPoint.avg_loss)) * t;

      const nextX = padL + (markerEpoch / maxEpoch) * plotW;
      const nextY = padT + (markerLoss / maxLoss) * plotH;
      const prevX = padL + (previousEpoch / maxEpoch) * plotW;
      const prevY = padT + (Number(previousPoint.avg_loss) / maxLoss) * plotH;
      if (Math.abs(nextX - prevX) > 0.001 || Math.abs(nextY - prevY) > 0.001) {
        ctx.lineTo(nextX, nextY);
      }
    }
    ctx.stroke();

    const lx = padL + (markerEpoch / maxEpoch) * plotW;
    const ly = padT + (markerLoss / maxLoss) * plotH;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(lx, ly, 3.5, 0, Math.PI * 2);
    ctx.fill();

    legendItems.push({ color, label: `${feature}/${model}` });
  });

  const epochLabel = Math.min(maxEpoch, Math.max(1, Math.floor(epochProgress)));

  ctx.fillStyle = "#dbeafe";
  ctx.font = "600 12px Inter";
  ctx.fillText(`Epoch ${epochLabel}`, width - 95, 18);
  ctx.fillText("Avg Loss", 14, 24 + plotH / 2);
  ctx.fillText("Epoch", padL + plotW / 2 - 18, padT + plotH + 24);

  const uniqueLegend = [];
  const seen = new Set();
  legendItems.forEach((item) => {
    if (!seen.has(item.label)) {
      seen.add(item.label);
      uniqueLegend.push(item);
    }
  });

  const legendTop = padT + plotH + 42;
  const legendLeft = padL;
  const legendColWidth = Math.max(180, Math.floor(plotW / legendColumns));

  uniqueLegend.forEach((item, index) => {
    const column = index % legendColumns;
    const row = Math.floor(index / legendColumns);
    const legendX = legendLeft + column * legendColWidth;
    const legendY = legendTop + row * legendRowHeight;
    ctx.fillStyle = item.color;
    ctx.fillRect(legendX, legendY - 8, 10, 10);
    ctx.fillStyle = "#e2e8f0";
    ctx.font = "500 10px Inter";
    ctx.fillText(item.label, legendX + 15, legendY);
  });
}

function playRaceAnimation() {
  if (appState.raceAnimationId !== null) {
    cancelAnimationFrame(appState.raceAnimationId);
    appState.raceAnimationId = null;
  }

  const filteredTraining = getFilteredTrainingRows();
  const maxEpoch = Math.max(1, ...filteredTraining.map((item) => Number(item.epoch) || 1));
  const startEpoch = 1;
  const durationMs = Math.max(1800, maxEpoch * 500);
  const startTime = performance.now();

  function step(now) {
    const elapsed = now - startTime;
    const t = Math.min(elapsed / durationMs, 1);
    const eased = 1 - (1 - t) * (1 - t);
    const currentEpoch = startEpoch + (maxEpoch - startEpoch) * eased;
    renderRaceFrame(currentEpoch);

    if (t < 1) {
      appState.raceAnimationId = requestAnimationFrame(step);
    } else {
      appState.raceAnimationId = null;
      renderRaceFrame(maxEpoch);
    }
  }

  appState.raceAnimationId = requestAnimationFrame(step);
}

function rerenderInteractive() {
  if (appState.raceAnimationId !== null) {
    cancelAnimationFrame(appState.raceAnimationId);
    appState.raceAnimationId = null;
  }
  const rows = getFilteredRows();
  renderPerformanceTable(rows);
  renderBubbleChart(rows);
  renderRaceFrame(1);
}

function initInteractions() {
  document.getElementById("sortAcc").addEventListener("click", () => setSort("test_accuracy"));
  document.getElementById("sortF1").addEventListener("click", () => setSort("test_macro_f1"));

  document.getElementById("tabNeighbors").addEventListener("click", () => {
    appState.activeTask4Tab = "neighbors";
    renderTask4Tab();
  });
  document.getElementById("tabEquations").addEventListener("click", () => {
    appState.activeTask4Tab = "equations";
    renderTask4Tab();
  });

  document.getElementById("playRace").addEventListener("click", playRaceAnimation);
}

function boot(data) {
  buildSeriesColorMap();
  renderTaskCards(data?.task_cards || []);
  renderHighlights(data);
  renderBestByFeature(data?.task5?.by_feature || []);
  initFilters(appState.allRows);
  initInteractions();
  renderTask4Tab();
  rerenderInteractive();
  renderRaceFrame(1);
  installBubbleTooltip();
}

boot(initialData);
