/* ── Project 4 Dashboard JS ── */
"use strict";

const D = JSON.parse(document.getElementById("initialData").textContent);
const t1 = D.task1 || {};
const t2 = D.task2 || {};

/* ── Navigation ── */
document.querySelectorAll(".nav-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    const view = btn.dataset.view;
    document.querySelectorAll(".dashboard-view").forEach(s => {
      s.classList.toggle("is-hidden", s.dataset.section !== view);
    });
  });
});

/* ── Hero badges ── */
const arch = t1.model_architecture || {};
document.getElementById("badgeModel").textContent = arch.base_model || "—";
const base = t2.baseline_bidaf || {};
const bert = t2.bert_bidaf || {};
document.getElementById("badgeBaseF1").textContent = base.best_val_f1 != null ? base.best_val_f1 + "%" : "—";
document.getElementById("badgeBertF1").textContent = bert.best_val_f1 != null ? bert.best_val_f1 + "%" : "—";

/* ── Overview Cards ── */
(function buildOverviewCards() {
  const cards = [
    { label: "Task 1", title: "Sentiment Classes", metric: arch.num_classes || "—", sub: "1–5 star ratings", bg: "linear-gradient(135deg,#6366f1,#8b5cf6)" },
    { label: "Task 1", title: "Max Input Length", metric: arch.max_sequence_length || "—", sub: "tokens", bg: "linear-gradient(135deg,#0ea5e9,#22d3ee)" },
    { label: "Task 1", title: "Model Params", metric: arch.total_parameters ? (arch.total_parameters / 1e6).toFixed(1) + "M" : "—", sub: arch.model_name || "", bg: "linear-gradient(135deg,#14b8a6,#34d399)" },
    { label: "Task 2", title: "Baseline EM / F1", metric: (base.final_val_em || "—") + " / " + (base.final_val_f1 || "—"), sub: "GloVe + Char CNN", bg: "linear-gradient(135deg,#f97316,#fb923c)" },
    { label: "Task 2", title: "BERT EM / F1", metric: (bert.final_val_em || "—") + " / " + (bert.final_val_f1 || "—"), sub: "Frozen mBERT", bg: "linear-gradient(135deg,#ec4899,#f472b6)" },
    { label: "Task 2", title: "Train Size", metric: t2.train_size || "—", sub: "SQuAD v1.1 subset", bg: "linear-gradient(135deg,#8b5cf6,#a78bfa)" },
  ];
  const wrap = document.getElementById("overviewCards");
  cards.forEach(c => {
    wrap.innerHTML += `<div class="task-card" style="background:${c.bg}">
      <div class="label">${c.label}</div><h3>${c.title}</h3>
      <div class="metric">${c.metric}</div><div class="sub">${c.sub}</div></div>`;
  });
})();


/* ── Task 1: Architecture Table ── */
(function buildArchTable() {
  const rows = [
    ["Model Name", arch.model_name],
    ["Base Model", arch.base_model],
    ["Classes", arch.num_classes],
    ["Max Sequence Length", arch.max_sequence_length],
    ["Vocab Size", arch.vocab_size?.toLocaleString()],
    ["Hidden Size", arch.hidden_size],
    ["Attention Heads", arch.num_attention_heads],
    ["Hidden Layers", arch.num_hidden_layers],
    ["Total Parameters", arch.total_parameters?.toLocaleString()],
  ];
  const body = document.getElementById("archBody");
  rows.forEach(([k, v]) => {
    body.innerHTML += `<tr><td>${k}</td><td>${v ?? "—"}</td></tr>`;
  });
})();

/* ── Task 1: Case Sensitivity ── */
(function buildCaseTable() {
  const tests = t1.case_sensitivity_analysis?.test_results || [];
  const body = document.getElementById("caseBody");
  tests.forEach(t => {
    body.innerHTML += `<tr>
      <td>${t.lower_text}</td>
      <td>${t.lower_prediction}</td>
      <td>${(t.lower_confidence * 100).toFixed(1)}%</td>
      <td>${t.predictions_match ? "✅" : "❌"}</td>
    </tr>`;
  });
})();

/* ── Task 1: Azerbaijani Comparison ── */
(function buildAzTable() {
  const items = t1.azerbaijani_analysis?.comparison_results || [];
  const body = document.getElementById("azBody");
  items.forEach(r => {
    const azConf = (r.azerbaijani_confidence * 100).toFixed(1);
    const enConf = (r.english_confidence * 100).toFixed(1);
    body.innerHTML += `<tr>
      <td>${r.azerbaijani_text}</td><td>${r.english_text}</td>
      <td>${r.expected_sentiment}</td>
      <td>${r.azerbaijani_prediction}</td><td>${r.english_prediction}</td>
      <td>${azConf}%</td><td>${enConf}%</td>
    </tr>`;
  });
})();

/* ── Task 1: Tokenization Cards ── */
(function buildTokenCards() {
  const items = t1.azerbaijani_analysis?.tokenization_analysis || [];
  const wrap = document.getElementById("tokenizationCards");
  items.forEach(t => {
    const pills = t.subword_tokens.map(tok => {
      const cls = tok.startsWith("##") ? "subword" : "";
      return `<span class="token-pill ${cls}">${tok}</span>`;
    }).join("");
    wrap.innerHTML += `<div class="panel glass" style="padding:1rem;">
      <h3 style="margin:0 0 0.3rem;font-size:1.1rem;">${t.azerbaijani_word}</h3>
      <p style="color:var(--muted);margin:0 0 0.6rem;font-size:0.85rem;">"${t.english_meaning}" → ${t.num_subwords} subwords</p>
      <div>${pills}</div>
    </div>`;
  });
})();


/* ── Task 2: Comparison Table ── */
(function buildCompTable() {
  const body = document.getElementById("compBody");
  const rows = [
    ["Exact Match (%)", base.final_val_em, bert.final_val_em],
    ["F1 Score (%)", base.final_val_f1, bert.final_val_f1],
    ["Best F1 (%)", base.best_val_f1, bert.best_val_f1],
    ["Final Val Loss", base.final_val_loss, bert.final_val_loss],
    ["Trainable Params", base.trainable_parameters, bert.trainable_parameters],
    ["Total Params", base.total_parameters, bert.total_parameters],
  ];
  rows.forEach(([label, bv, brtv]) => {
    let delta = "";
    if (typeof bv === "number" && typeof brtv === "number") {
      const d = brtv - bv;
      const cls = d >= 0 ? "delta-positive" : "delta-negative";
      const sign = d >= 0 ? "+" : "";
      const fmt = Number.isInteger(bv) ? d.toLocaleString() : d.toFixed(2);
      delta = `<span class="${cls}">${sign}${fmt}</span>`;
    }
    const fmtV = v => typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString() : v) : "—";
    body.innerHTML += `<tr><td>${label}</td><td>${fmtV(bv)}</td><td>${fmtV(brtv)}</td><td>${delta}</td></tr>`;
  });
})();

/* ── Task 2: Training Charts ── */
function drawLineChart(canvasId, datasets, yLabel) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const pad = { top: 20, right: 20, bottom: 40, left: 55 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  // Compute bounds
  let allVals = [];
  datasets.forEach(ds => allVals.push(...ds.values));
  let yMin = Math.min(...allVals);
  let yMax = Math.max(...allVals);
  const yPad = (yMax - yMin) * 0.1 || 1;
  yMin -= yPad; yMax += yPad;
  const xMax = Math.max(...datasets.map(ds => ds.values.length));

  const toX = i => pad.left + (i / (xMax - 1)) * plotW;
  const toY = v => pad.top + (1 - (v - yMin) / (yMax - yMin)) * plotH;

  // Clear
  ctx.clearRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (i / 4) * plotH;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    ctx.fillStyle = "#94a3b8"; ctx.font = "11px Inter,sans-serif"; ctx.textAlign = "right";
    const val = yMax - (i / 4) * (yMax - yMin);
    ctx.fillText(val.toFixed(2), pad.left - 8, y + 4);
  }

  // X labels
  ctx.fillStyle = "#94a3b8"; ctx.textAlign = "center";
  for (let i = 0; i < xMax; i++) {
    if (xMax <= 20 || i % Math.ceil(xMax / 10) === 0) {
      ctx.fillText(i + 1, toX(i), H - pad.bottom + 18);
    }
  }
  ctx.fillText("Epoch", pad.left + plotW / 2, H - 4);

  // Y label
  ctx.save(); ctx.translate(14, pad.top + plotH / 2);
  ctx.rotate(-Math.PI / 2); ctx.fillText(yLabel, 0, 0); ctx.restore();

  // Lines
  const colors = ["#22d3ee", "#fb7185", "#34d399", "#fbbf24"];
  datasets.forEach((ds, di) => {
    ctx.strokeStyle = colors[di % colors.length];
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ds.values.forEach((v, i) => {
      const x = toX(i), y = toY(v);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Dots
    ctx.fillStyle = colors[di % colors.length];
    ds.values.forEach((v, i) => {
      ctx.beginPath(); ctx.arc(toX(i), toY(v), 3, 0, Math.PI * 2); ctx.fill();
    });

    // Legend
    const lx = pad.left + 10 + di * 150;
    ctx.fillStyle = colors[di % colors.length];
    ctx.fillRect(lx, pad.top + 4, 14, 3);
    ctx.fillStyle = "#e2e8f0"; ctx.font = "12px Inter,sans-serif"; ctx.textAlign = "left";
    ctx.fillText(ds.label, lx + 20, pad.top + 10);
  });
}

const baseHist = base.training_history || [];
const bertHist = bert.training_history || [];

drawLineChart("lossCanvas", [
  { label: "Baseline", values: baseHist.map(h => h.val_loss) },
  { label: "BERT", values: bertHist.map(h => h.val_loss) },
], "Val Loss");

drawLineChart("f1Canvas", [
  { label: "Baseline", values: baseHist.map(h => h.val_f1) },
  { label: "BERT", values: bertHist.map(h => h.val_f1) },
], "F1 (%)");


/* ── Task 2: Epoch History Table ── */
function renderHistory(model) {
  const hist = model === "bert" ? bertHist : baseHist;
  const body = document.getElementById("historyBody");
  body.innerHTML = "";
  hist.forEach(h => {
    body.innerHTML += `<tr>
      <td>${h.epoch}</td><td>${h.train_loss}</td><td>${h.val_loss}</td>
      <td>${h.val_em}</td><td>${h.val_f1}</td><td>${h.time_sec}</td>
    </tr>`;
  });
}
renderHistory("baseline");
document.getElementById("historyModelSelect").addEventListener("change", e => {
  renderHistory(e.target.value);
});

/* ── Task 2: Hyperparameters ── */
(function buildHpTables() {
  const fill = (bodyId, hp) => {
    const body = document.getElementById(bodyId);
    if (!hp) return;
    Object.entries(hp).forEach(([k, v]) => {
      body.innerHTML += `<tr><td>${k}</td><td>${v}</td></tr>`;
    });
  };
  fill("hpBaseBody", base.hyperparameters);
  fill("hpBertBody", bert.hyperparameters);
})();
