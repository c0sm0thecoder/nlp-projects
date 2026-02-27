const TAB_META = {
  task1: [
    'Task 1: N-gram Language Models',
    'Train unigram / bigram / trigram models and inspect perplexity on train and test splits.',
  ],
  task2: [
    'Task 2: Smoothing Method Comparison',
    'Evaluate Laplace, Interpolation, Backoff, and Kneser-Ney with tuning grids.',
  ],
  task3: [
    'Task 3: Sentiment Classification',
    'Compare Naive Bayes variants and Logistic Regression with significance testing.',
  ],
  task4: [
    'Task 4: Sentence Boundary Detection',
    'Detect sentence boundaries and compare L1 vs L2 regularized logistic models.',
  ],
};

function setupTaskNavigation() {
  const buttons = document.querySelectorAll('.task-nav-btn');
  buttons.forEach((btn) => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      buttons.forEach((item) => item.classList.remove('active'));
      document.querySelectorAll('.task-panel').forEach((panel) => panel.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById(`panel-${tab}`).classList.add('active');
      document.getElementById('tabTitle').textContent = TAB_META[tab][0];
      document.getElementById('tabDesc').textContent = TAB_META[tab][1];
    });
  });
}

function setupInnerTabs() {
  const groups = document.querySelectorAll('.inner-tabs');
  groups.forEach((group) => {
    const task = group.dataset.task;
    const buttons = group.querySelectorAll('.inner-tab-btn');
    buttons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const inner = btn.dataset.inner;
        buttons.forEach((item) => item.classList.remove('active'));
        document.querySelectorAll(`#panel-${task} .inner-view`).forEach((view) => view.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(`${task}-${inner}`).classList.add('active');
      });
    });
  });
}

function v(id) {
  return document.getElementById(id).value;
}

function setOutputState(outputId, message, stateClass = '') {
  const out = document.getElementById(outputId);
  out.className = `result ${stateClass}`.trim();
  out.textContent = message;
}

function setAnswer(answerId, message, stateClass = '') {
  const box = document.getElementById(answerId);
  box.className = `result ${stateClass}`.trim();
  box.textContent = message;
}

function csvToFloatList(raw) {
  return raw
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
    .map(Number)
    .filter((item) => !Number.isNaN(item));
}

function formatValue(value) {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(4);
  return String(value);
}

function renderCards(container, items) {
  if (!items.length) return;
  const cards = document.createElement('div');
  cards.className = 'cards';
  items.forEach((item) => {
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `<span class="k">${item.label}</span><span class="v">${formatValue(item.value)}</span>`;
    cards.appendChild(card);
  });
  container.appendChild(cards);
}

function renderTable(container, title, rows) {
  if (!Array.isArray(rows) || !rows.length) return;
  const heading = document.createElement('h3');
  heading.textContent = title;
  container.appendChild(heading);

  const tableWrap = document.createElement('div');
  tableWrap.className = 'table-wrap';
  const table = document.createElement('table');

  const columns = Object.keys(rows[0]);
  const thead = document.createElement('thead');
  const trHead = document.createElement('tr');
  columns.forEach((col) => {
    const th = document.createElement('th');
    th.textContent = col;
    trHead.appendChild(th);
  });
  thead.appendChild(trHead);

  const tbody = document.createElement('tbody');
  rows.forEach((row) => {
    const tr = document.createElement('tr');
    columns.forEach((col) => {
      const td = document.createElement('td');
      td.textContent = formatValue(row[col]);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  table.appendChild(thead);
  table.appendChild(tbody);
  tableWrap.appendChild(table);
  container.appendChild(tableWrap);
}

function renderTask1(data) {
  const container = document.getElementById('task1_render');
  container.innerHTML = '';

  renderCards(container, [
    { label: 'Train Rows', value: data?.split?.train_rows },
    { label: 'Test Rows', value: data?.split?.test_rows },
    { label: 'Vocabulary Size', value: data?.vocab?.vocab_size },
    { label: 'UNK Threshold', value: data?.config?.min_freq },
  ]);

  const modelRows = Object.entries(data?.models || {}).map(([name, row]) => ({
    model: name,
    train_perplexity: row.train_perplexity,
    test_perplexity: row.test_perplexity,
    zero_prob_events_test: row.zero_prob_events_test,
    unseen_rate_test: row.unseen_rate_test,
  }));
  renderTable(container, 'Model Metrics', modelRows);
}

function renderTask2(data) {
  const container = document.getElementById('task2_render');
  container.innerHTML = '';

  renderCards(container, [
    { label: 'Best Default Method', value: data?.best_method_defaults },
    { label: 'Ranking Rows', value: data?.ranking_rows?.length || 0 },
    { label: 'Method Rows', value: data?.method_rows?.length || 0 },
  ]);

  renderTable(container, 'Ranking', data?.ranking_rows || []);
  renderTable(container, 'Method Comparison', data?.method_rows || []);
}

function renderTask3(data) {
  const container = document.getElementById('task3_render');
  container.innerHTML = '';

  renderCards(container, [
    { label: 'Best Classifier', value: data?.best_classifier },
    { label: 'Best Variant', value: data?.best?.algorithm_variant },
    { label: 'Train Rows', value: data?.split?.train_rows },
    { label: 'Test Rows', value: data?.split?.test_rows },
  ]);

  renderTable(container, 'Summary', data?.summary_rows || []);
  renderTable(container, 'Classifier Analysis', data?.classifier_analysis || []);
  renderTable(container, 'Significance', data?.significance_results || []);
}

function renderTask4(data) {
  const container = document.getElementById('task4_render');
  container.innerHTML = '';

  renderCards(container, [
    { label: 'Best Penalty', value: data?.best_penalty },
    { label: 'Sample Count', value: data?.dataset_stats?.samples },
    { label: 'L1 Best C', value: data?.results?.l1?.best_C },
    { label: 'L2 Best C', value: data?.results?.l2?.best_C },
  ]);

  const metricsRows = ['l1', 'l2'].map((penalty) => ({
    penalty,
    best_C: data?.results?.[penalty]?.best_C,
    test_f1: data?.results?.[penalty]?.test?.f1,
    test_accuracy: data?.results?.[penalty]?.test?.accuracy,
    n_nonzero_coefs: data?.results?.[penalty]?.n_nonzero_coefs,
  }));

  renderTable(container, 'Penalty Comparison', metricsRows);
  renderTable(container, 'Tuning', data?.tuning_rows || []);
}

const TASK_RENDERERS = {
  task1: renderTask1,
  task2: renderTask2,
  task3: renderTask3,
  task4: renderTask4,
};

async function callApi(taskKey, path, payload, outputId) {
  const render = TASK_RENDERERS[taskKey];
  setOutputState(outputId, 'Running...', 'state-running');

  try {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await res.json();
    if (!res.ok) {
      setOutputState(outputId, `Error: ${data.detail || 'Request failed'}`, 'state-error');
      return;
    }

    setOutputState(outputId, JSON.stringify(data, null, 2), 'state-ok');
    render(data);

    const outputTabBtn = document.querySelector(`#panel-${taskKey} .inner-tab-btn[data-inner="output"]`);
    if (outputTabBtn) {
      outputTabBtn.click();
    }
  } catch (err) {
    setOutputState(outputId, `Error: ${err.message}`, 'state-error');
  }
}

async function callPredict(path, payload, answerId, formatter) {
  setAnswer(answerId, 'Running...', 'state-running');
  try {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      setAnswer(answerId, `Error: ${data.detail || 'Request failed'}`, 'state-error');
      return;
    }
    setAnswer(answerId, formatter(data), 'state-ok');
  } catch (err) {
    setAnswer(answerId, `Error: ${err.message}`, 'state-error');
  }
}

function runTask1() {
  callApi(
    'task1',
    '/api/task1/run',
    {
      input_path: v('t1_input_path'),
      text_col: v('t1_text_col'),
      author_col: v('t1_author_col'),
      test_size: Number(v('t1_test_size')),
      seed: Number(v('t1_seed')),
      min_freq: Number(v('t1_min_freq')),
    },
    'task1_result'
  );
}

function runTask2() {
  callApi(
    'task2',
    '/api/task2/run',
    {
      input_path: v('t2_input_path'),
      text_col: v('t2_text_col'),
      author_col: v('t2_author_col'),
      test_size: Number(v('t2_test_size')),
      val_size: Number(v('t2_val_size')),
      seed: Number(v('t2_seed')),
      min_freq: Number(v('t2_min_freq')),
      interp_bigram_grid: csvToFloatList(v('t2_interp_bigram_grid')),
      interp_trigram_step: Number(v('t2_interp_trigram_step')),
      discount_grid: csvToFloatList(v('t2_discount_grid')),
    },
    'task2_result'
  );
}

function runTask3() {
  const dataset = v('t3_dataset');
  const maxRaw = v('t3_max_features').trim();

  callApi(
    'task3',
    '/api/task3/run',
    {
      dataset,
      input_path: dataset === 'parquet' ? v('t3_input_path') : null,
      text_col: v('t3_text_col'),
      label_col: v('t3_label_col'),
      test_size: Number(v('t3_test_size')),
      seed: Number(v('t3_seed')),
      max_features: maxRaw ? Number(maxRaw) : null,
    },
    'task3_result'
  );
}

function runTask4() {
  callApi(
    'task4',
    '/api/task4/run',
    {
      input_path: v('t4_input_path'),
      text_col: v('t4_text_col'),
      label_col: v('t4_label_col'),
      val_size: Number(v('t4_val_size')),
      test_size: Number(v('t4_test_size')),
      seed: Number(v('t4_seed')),
      c_grid: csvToFloatList(v('t4_c_grid')),
    },
    'task4_result'
  );
}

function predictTask1Next() {
  callPredict(
    '/api/task1/predict-next',
    {
      input_path: v('t1_input_path'),
      text_col: v('t1_text_col'),
      author_col: v('t1_author_col'),
      test_size: Number(v('t1_test_size')),
      seed: Number(v('t1_seed')),
      min_freq: Number(v('t1_min_freq')),
      model: v('t1_predict_model'),
      text: v('t1_prompt'),
      top_k: 5,
    },
    'task1_answer',
    (data) => {
      const top = (data.top_k || []).map((row, idx) => `${idx + 1}. ${row.word} (${Number(row.probability).toFixed(4)})`).join('\n');
      return `Prediction: ${data.prediction || 'N/A'}\n\nTop candidates:\n${top}`;
    }
  );
}

function predictTask2Next() {
  const methodRaw = v('t2_predict_method').trim();
  callPredict(
    '/api/task2/predict-next',
    {
      input_path: v('t2_input_path'),
      text_col: v('t2_text_col'),
      author_col: v('t2_author_col'),
      test_size: Number(v('t2_test_size')),
      val_size: Number(v('t2_val_size')),
      seed: Number(v('t2_seed')),
      min_freq: Number(v('t2_min_freq')),
      interp_bigram_grid: csvToFloatList(v('t2_interp_bigram_grid')),
      interp_trigram_step: Number(v('t2_interp_trigram_step')),
      discount_grid: csvToFloatList(v('t2_discount_grid')),
      method: methodRaw || null,
      order: Number(v('t2_predict_order')),
      text: v('t2_prompt'),
      top_k: 5,
    },
    'task2_answer',
    (data) => {
      const top = (data.top_k || []).map((row, idx) => `${idx + 1}. ${row.word} (${Number(row.probability).toFixed(4)})`).join('\n');
      return `Prediction: ${data.prediction || 'N/A'}\nModel: ${data.method} (${data.order}-gram)\n\nTop candidates:\n${top}`;
    }
  );
}

function predictTask3Sentiment() {
  callPredict(
    '/api/task3/predict-sentiment',
    {
      dataset: v('t3_dataset'),
      input_path: v('t3_dataset') === 'parquet' ? v('t3_input_path') : null,
      text_col: v('t3_text_col'),
      label_col: v('t3_label_col'),
      test_size: Number(v('t3_test_size')),
      seed: Number(v('t3_seed')),
      max_features: v('t3_max_features').trim() ? Number(v('t3_max_features')) : null,
      text: v('t3_prompt'),
    },
    'task3_answer',
    (data) => {
      const conf = data.confidence == null ? 'N/A' : Number(data.confidence).toFixed(4);
      return `Sentiment: ${data.predicted_label}\nConfidence: ${conf}\nModel: ${data.classifier} + ${data.feature_set}`;
    }
  );
}

function predictTask4Sentences() {
  const penaltyRaw = v('t4_predict_penalty').trim();
  callPredict(
    '/api/task4/predict-sentences',
    {
      input_path: v('t4_input_path'),
      text_col: v('t4_text_col'),
      label_col: v('t4_label_col'),
      val_size: Number(v('t4_val_size')),
      test_size: Number(v('t4_test_size')),
      seed: Number(v('t4_seed')),
      c_grid: csvToFloatList(v('t4_c_grid')),
      penalty: penaltyRaw || null,
      text: v('t4_prompt'),
    },
    'task4_answer',
    (data) => {
      const sentences = (data.sentences || []).map((s, i) => `${i + 1}. ${s}`).join('\n');
      return `Penalty: ${data.penalty}\n\nDetected sentences:\n${sentences || 'No sentences detected.'}`;
    }
  );
}

function toggleTask3Input() {
  const isParquet = v('t3_dataset') === 'parquet';
  document.getElementById('t3_input_path').disabled = !isParquet;
  document.getElementById('t3_text_col').value = isParquet ? 'modern_text' : 'text';
  document.getElementById('t3_label_col').value = isParquet ? 'author' : 'label';
}

async function checkHealth() {
  const box = document.getElementById('healthBox');
  try {
    const res = await fetch('/health');
    const data = await res.json();
    if (data.status === 'ok') {
      box.textContent = 'API Health: Online';
      box.classList.add('state-ok');
    } else {
      box.textContent = 'API Health: Unknown';
      box.classList.add('state-running');
    }
  } catch {
    box.textContent = 'API Health: Offline';
    box.classList.add('state-error');
  }
}

async function warmupModels() {
  const box = document.getElementById('warmupBox');
  box.textContent = 'Warmup running... this can take time on first run.';
  box.classList.remove('state-ok', 'state-error');
  box.classList.add('state-running');

  try {
    const res = await fetch('/api/warmup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task2: true, task3: true, task4: true }),
    });
    const data = await res.json();
    if (!res.ok) {
      box.textContent = `Warmup failed: ${data.detail || 'Request failed'}`;
      box.classList.remove('state-running');
      box.classList.add('state-error');
      return;
    }

    const t = data.timings || {};
    box.textContent = `Warmup done. T2=${t.task2_seconds ?? '-'}s, T3=${t.task3_seconds ?? '-'}s, T4=${t.task4_seconds ?? '-'}s`;
    box.classList.remove('state-running');
    box.classList.add('state-ok');
  } catch (err) {
    box.textContent = `Warmup failed: ${err.message}`;
    box.classList.remove('state-running');
    box.classList.add('state-error');
  }
}

setupTaskNavigation();
setupInnerTabs();
toggleTask3Input();
checkHealth();

window.runTask1 = runTask1;
window.runTask2 = runTask2;
window.runTask3 = runTask3;
window.runTask4 = runTask4;
window.toggleTask3Input = toggleTask3Input;
window.predictTask1Next = predictTask1Next;
window.predictTask2Next = predictTask2Next;
window.predictTask3Sentiment = predictTask3Sentiment;
window.predictTask4Sentences = predictTask4Sentences;
window.warmupModels = warmupModels;
