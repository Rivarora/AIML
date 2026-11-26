/* Student Performance Analytics — product UI.
 *
 * The file is organised as small services and renderers:
 *   1. formatting and DOM helpers
 *   2. API service
 *   3. navigation and shared feedback
 *   4. prediction schema, forms, and validation
 *   5. view loaders and renderers
 *
 * All business data comes from the Flask API. No metric, prediction,
 * confidence value, or chart is hardcoded in the client.
 */

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const state = {
  health: null,
  schema: null,
  summary: null,
  performance: null,
  importance: null,
  loaded: new Set(),
  loading: new Map(),
};

/* ---------------------------------------------------------------------------
 * 1. Formatting and DOM helpers
 * ------------------------------------------------------------------------- */

const ESCAPE_MAP = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ESCAPE_MAP[character]);
}

function icon(name, className = "icon") {
  return `<svg class="${className}" aria-hidden="true"><use href="#icon-${name}"></use></svg>`;
}

function isNumber(value) {
  return value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value));
}

const fmt = {
  int(value) {
    return isNumber(value) ? Math.round(Number(value)).toLocaleString() : "—";
  },
  dec(value, digits = 2) {
    return isNumber(value) ? Number(value).toFixed(digits) : "—";
  },
  pct(value, digits = 1) {
    return isNumber(value) ? `${(Number(value) * 100).toFixed(digits)}%` : "—";
  },
  signed(value, digits = 3) {
    if (!isNumber(value)) return "—";
    const number = Number(value);
    return `${number >= 0 ? "+" : ""}${number.toFixed(digits)}`;
  },
  meanStd(mean, std, digits = 3) {
    if (!isNumber(mean)) return "—";
    return `${fmt.dec(mean, digits)} ± ${fmt.dec(std, digits)}`;
  },
};

function statCard(label, value, sub = "") {
  return `<div class="stat-card">
    <div class="label">${esc(label)}</div>
    <div class="value">${esc(value)}</div>
    ${sub ? `<div class="sub">${esc(sub)}</div>` : ""}
  </div>`;
}

function metricChip(label, value) {
  return `<div class="metric-chip"><small>${esc(label)}</small><strong>${esc(value)}</strong></div>`;
}

function statusBadge(status) {
  const mapped = {
    ok: { className: "ok", label: "OK", iconName: "check" },
    review: { className: "review", label: "Review", iconName: "warning" },
    critical: { className: "critical", label: "Experimental / suspicious", iconName: "warning" },
    unknown: { className: "neutral", label: "Unknown", iconName: "info" },
  }[status] || { className: "neutral", label: "Unknown", iconName: "info" };
  return `<span class="badge badge-${mapped.className}">${icon(mapped.iconName, "icon icon-sm")}${esc(mapped.label)}</span>`;
}

function barRow(label, value, max, displayValue) {
  const safeMax = Number(max) > 0 ? Number(max) : 1;
  const width = Math.max(0, Math.min(100, (Math.max(0, Number(value) || 0) / safeMax) * 100));
  return `<div class="bar-row">
    <span class="bar-label">${esc(label)}</span>
    <div class="bar-track" role="img" aria-label="${esc(label)} importance ${esc(displayValue)}"><span style="width:${width.toFixed(1)}%"></span></div>
    <span class="bar-value">${esc(displayValue)}</span>
  </div>`;
}

/* ---------------------------------------------------------------------------
 * 2. API service
 * ------------------------------------------------------------------------- */

const API = {
  async request(path, { method = "GET", body, timeout = 15000 } = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    try {
      const options = {
        method,
        headers: { Accept: "application/json" },
        signal: controller.signal,
      };
      if (body !== undefined) {
        options.headers["Content-Type"] = "application/json";
        options.body = JSON.stringify(body);
      }
      const response = await fetch(path, options);
      const text = await response.text();
      let data = null;
      if (text) {
        try {
          data = JSON.parse(text);
        } catch {
          throw new Error(`Unexpected non-JSON response (${response.status}).`);
        }
      }
      if (!response.ok) {
        const error = new Error(
          (data && data.error) || `Request failed (${response.status}).`,
        );
        error.status = response.status;
        error.data = data;
        throw error;
      }
      return data;
    } catch (error) {
      if (error.name === "AbortError") {
        throw new Error("The request timed out. The server may still be starting.");
      }
      throw error;
    } finally {
      clearTimeout(timer);
    }
  },
  get(path) {
    return this.request(path);
  },
  post(path, body) {
    return this.request(path, { method: "POST", body });
  },
};

async function settled(promise) {
  try {
    return { ok: true, data: await promise };
  } catch (error) {
    return { ok: false, error };
  }
}

/* ---------------------------------------------------------------------------
 * 3. Navigation and shared feedback
 * ------------------------------------------------------------------------- */

const VIEWS = {
  overview: {
    title: "Overview",
    eyebrow: "Workspace",
    loader: loadOverview,
  },
  exam: {
    title: "Academic Performance",
    eyebrow: "Prediction track",
    loader: loadExamView,
  },
  placement: {
    title: "Placement Readiness",
    eyebrow: "Prediction track",
    loader: loadPlacementView,
  },
  insights: {
    title: "Student Insights",
    eyebrow: "Cohort context",
    loader: loadInsightsView,
  },
  recommendations: {
    title: "Recommendations",
    eyebrow: "Advisory support",
    loader: loadRecommendationsView,
  },
  models: {
    title: "Model Insights",
    eyebrow: "Evaluation evidence",
    loader: loadModelsView,
  },
};

let toastTimer = null;

function toast(message) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.add("is-visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => element.classList.remove("is-visible"), 4200);
}

function showGlobalAlert(message, type = "warning") {
  const alert = $("#global-alert");
  const iconName = type === "danger" ? "warning" : type === "info" ? "info" : "warning";
  alert.className = `callout callout-${type}`;
  alert.innerHTML = `${icon(iconName)}<div>${esc(message)}</div>`;
  alert.hidden = false;
}

function hideGlobalAlert() {
  $("#global-alert").hidden = true;
}

function setHealthPill(kind, text) {
  const pill = $("#health-pill");
  const dotClass = kind === "ok" ? "" : kind === "warn" ? "is-warn" : "is-bad";
  pill.innerHTML = `<span class="status-dot ${dotClass}" aria-hidden="true"></span><span>${esc(text)}</span>`;
  const sidebarDot = $("#sidebar-status-dot");
  sidebarDot.className = `status-dot ${dotClass}`;
  $("#sidebar-status-text").textContent = text;
}

function setHealthPillFromData(health) {
  const artifacts = health.artifacts || {};
  const missing = Object.entries(artifacts)
    .filter(([, ready]) => !ready)
    .map(([key]) => key);
  if (health.status === "ok" && missing.length === 0) {
    setHealthPill("ok", "Data ready");
    hideGlobalAlert();
    return;
  }
  if (health.core_ready) {
    setHealthPill("warn", "Limited data");
    showGlobalAlert(
      `Some optional artifacts are missing (${missing.slice(0, 5).join(", ")}${missing.length > 5 ? ", …" : ""}). Run python scripts/run_pipeline.py to regenerate them.`,
      "warning",
    );
    return;
  }
  setHealthPill("bad", "Data unavailable");
  showGlobalAlert(
    "Core datasets or saved models are missing. Run python scripts/run_pipeline.py, then reload this page.",
    "danger",
  );
}

async function loadHealth() {
  const result = await settled(API.get("/api/health"));
  if (!result.ok) {
    setHealthPill("bad", "Health check failed");
    showGlobalAlert(result.error.message, "danger");
    return;
  }
  state.health = result.data;
  setHealthPillFromData(result.data);
}

function showView(view, { updateHash = true, scroll = true, focus = true } = {}) {
  const meta = VIEWS[view] ? view : "overview";
  $$("[data-view-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.viewPanel !== meta;
  });
  $$(".nav-item").forEach((button) => {
    const active = button.dataset.view === meta;
    button.classList.toggle("is-active", active);
    if (active) {
      button.setAttribute("aria-current", "page");
    } else {
      button.removeAttribute("aria-current");
    }
  });
  $("#page-title").textContent = VIEWS[meta].title;
  $("#page-eyebrow").textContent = VIEWS[meta].eyebrow;
  document.title = `${VIEWS[meta].title} · Student Performance Analytics`;
  if (updateHash && location.hash !== `#${meta}`) {
    history.replaceState(null, "", `#${meta}`);
  }
  if (scroll) {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
  if (focus) {
    $("#main-content").focus({ preventScroll: true });
  }
  ensureView(meta);
}

function ensureView(view) {
  if (state.loaded.has(view)) return;
  if (state.loading.has(view)) return;
  const promise = Promise.resolve()
    .then(() => VIEWS[view].loader())
    .catch((error) => {
      toast(error.message || "Could not load this view.");
    })
    .finally(() => {
      state.loading.delete(view);
      state.loaded.add(view);
    });
  state.loading.set(view, promise);
}

function viewFromHash() {
  const candidate = location.hash.replace("#", "");
  return VIEWS[candidate] ? candidate : "overview";
}

/* ---------------------------------------------------------------------------
 * 4. Prediction schema, forms, and validation
 * ------------------------------------------------------------------------- */

async function getSchema() {
  if (!state.schema) {
    state.schema = await API.get("/api/prediction-schema");
  }
  return state.schema;
}

async function getSummary() {
  if (!state.summary) {
    state.summary = await API.get("/api/summary");
  }
  return state.summary;
}

async function getPerformance() {
  if (!state.performance) {
    state.performance = await API.get("/api/model-performance");
  }
  return state.performance;
}

async function getImportance() {
  if (!state.importance) {
    state.importance = await API.get("/api/feature-importance");
  }
  return state.importance;
}

function fieldId(track, field) {
  return `${track}-${field.name}`;
}

function fieldMarkup(track, field, { optional = false } = {}) {
  const id = fieldId(track, field);
  const helpId = `${id}-help`;
  const errorId = `${id}-error`;
  const requiredMark = optional ? "" : ` <span class="required" aria-hidden="true">*</span>`;
  const unit = field.unit ? `<span class="unit">${esc(field.unit)}</span>` : "";
  const help = field.help ? `<p class="field-help" id="${helpId}">${esc(field.help)}</p>` : "";
  const requiredAttr = optional ? "" : " required";
  const describedBy = `${helpId} ${errorId}`.trim();

  let control;
  if (field.type === "category") {
    const options = (field.allowed || [])
      .map((value) => {
        const selected = !optional && value === field.example ? " selected" : "";
        return `<option value="${esc(value)}"${selected}>${esc(value)}</option>`;
      })
      .join("");
    control = `<select id="${id}" name="${esc(field.name)}"${requiredAttr} aria-describedby="${describedBy}">${
      optional ? `<option value="">Not provided</option>` : ""
    }${options}</select>`;
  } else {
    const value = optional ? "" : ` value="${esc(field.example)}"`;
    control = `<input id="${id}" name="${esc(field.name)}" type="number" inputmode="decimal" min="${esc(field.min)}" max="${esc(field.max)}" step="${esc(field.step || 1)}"${value}${requiredAttr} aria-describedby="${describedBy}">`;
  }

  return `<div class="field">
    <label for="${id}"><span>${esc(field.label)}${requiredMark}</span>${unit}</label>
    ${control}
    ${help}
    <p class="field-error" id="${errorId}" hidden></p>
  </div>`;
}

function buildPredictionForm(track, fields) {
  const container = $(`#${track}-fields`);
  container.setAttribute("aria-busy", "false");
  container.innerHTML = fields.map((field) => fieldMarkup(track, field)).join("");
}

function fieldInput(track, field) {
  return $(`#${fieldId(track, field)}`);
}

function setFieldError(track, field, message) {
  const input = fieldInput(track, field);
  if (!input) return;
  const error = $(`#${fieldId(track, field)}-error`);
  input.setAttribute("aria-invalid", message ? "true" : "false");
  if (!error) return;
  error.textContent = message || "";
  error.hidden = !message;
}

function validateField(track, field, { optional = false } = {}) {
  const input = fieldInput(track, field);
  if (!input) return true;
  const raw = input.value.trim();
  if (!raw) {
    if (optional) {
      setFieldError(track, field, "");
      return true;
    }
    setFieldError(track, field, "This field is required.");
    return false;
  }
  if (field.type === "number") {
    const value = Number(raw);
    if (!Number.isFinite(value)) {
      setFieldError(track, field, "Enter a valid number.");
      return false;
    }
    if (value < Number(field.min) || value > Number(field.max)) {
      setFieldError(track, field, `Enter a value between ${field.min} and ${field.max}.`);
      return false;
    }
  } else if (!(field.allowed || []).includes(raw)) {
    setFieldError(track, field, "Choose one of the listed options.");
    return false;
  }
  setFieldError(track, field, "");
  return true;
}

function validateForm(track, fields) {
  let firstInvalid = null;
  for (const field of fields) {
    if (!validateField(track, field)) {
      if (!firstInvalid) firstInvalid = fieldInput(track, field);
    }
  }
  if (firstInvalid) {
    firstInvalid.focus();
    return false;
  }
  return true;
}

function readForm(form) {
  return Object.fromEntries(new FormData(form).entries());
}

function bindPredictionForm(track, fields) {
  const form = $(`#${track}-form`);
  if (!form || form.dataset.bound === "true") return;
  form.dataset.bound = "true";
  const submit = $(`#${track}-submit`);
  if (submit) submit.dataset.label = submit.textContent.trim();

  fields.forEach((field) => {
    const input = fieldInput(track, field);
    if (!input) return;
    input.addEventListener("blur", () => validateField(track, field));
    input.addEventListener("input", () => setFieldError(track, field, ""));
    input.addEventListener("change", () => setFieldError(track, field, ""));
  });

  form.addEventListener("reset", () => {
    window.setTimeout(() => {
      fields.forEach((field) => setFieldError(track, field, ""));
      renderResultEmpty(track);
    }, 0);
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!validateForm(track, fields)) return;
    setFormBusy(track, true);
    renderResultLoading(track);
    try {
      const data = await API.post(`/api/predict/${track}`, readForm(form));
      if (track === "exam") {
        renderExamResult(data);
      } else {
        renderPlacementResult(data);
      }
    } catch (error) {
      renderResultError(track, error);
      if (error.data && error.data.field) {
        const field = fields.find((candidate) => candidate.name === error.data.field);
        if (field) {
          setFieldError(track, field, error.message);
          const input = fieldInput(track, field);
          if (input) input.focus();
        }
      }
    } finally {
      setFormBusy(track, false);
    }
  });
}

function setFormBusy(track, busy) {
  const submit = $(`#${track}-submit`);
  if (!submit) return;
  submit.disabled = busy;
  submit.textContent = busy ? "Estimating…" : submit.dataset.label || "Estimate";
}

function renderLimitations(track, limitations) {
  const list = $(`#${track}-limitations`);
  if (!list) return;
  const items = Array.isArray(limitations) && limitations.length
    ? limitations
    : ["No limitations were supplied by the API."];
  list.innerHTML = items.map((item) => `<li>${esc(item)}</li>`).join("");
}

function renderProcess(track, meta, summary) {
  const container = $(`#${track}-process`);
  if (!container) return;
  const model = (summary && summary.deployed_models && summary.deployed_models[track]) || meta.model_label;
  const inputCount = (meta.fields || []).length;
  if (track === "exam") {
    const rmse = summary ? summary.exam_model_test_rmse : null;
    const interval = isNumber(rmse)
      ? `The result card uses an approximate ±1 held-out RMSE band (${fmt.dec(rmse, 3)} points), not a confidence interval.`
      : "No held-out RMSE is available, so no error band is shown.";
    container.innerHTML = `
      <p><b>Model:</b> ${esc(model)}</p>
      <p><b>Inputs:</b> ${esc(inputCount)} validated exam-track fields.</p>
      <p><b>Preprocessing:</b> imputation, scaling, and one-hot encoding are fitted inside the saved pipeline on the training split only.</p>
      <p><b>Uncertainty:</b> ${esc(interval)}</p>
    `;
  } else {
    container.innerHTML = `
      <p><b>Model:</b> ${esc(model)}</p>
      <p><b>Inputs:</b> ${esc(inputCount)} validated placement-track fields.</p>
      <p><b>Output:</b> the model's probability at a 0.5 decision threshold. It is not a calibrated real-world probability.</p>
      <p><b>Source:</b> the placement label is deterministic in the synthetic source data; high scores reflect dataset construction.</p>
    `;
  }
}

function renderResultEmpty(track) {
  const element = $(`#${track}-result`);
  if (!element) return;
  element.classList.remove("is-busy");
  if (track === "exam") {
    element.innerHTML = `<div class="result-empty">
      ${icon("chart")}
      <h3 id="exam-result-title">Waiting for inputs</h3>
      <p>Submit the form to call the saved Linear Regression pipeline.</p>
    </div>`;
  } else {
    element.innerHTML = `<div class="result-empty">
      ${icon("briefcase")}
      <h3 id="placement-result-title">Waiting for inputs</h3>
      <p>Submit the form to call the saved placement pipeline.</p>
    </div>`;
  }
}

function renderResultLoading(track) {
  const element = $(`#${track}-result`);
  if (!element) return;
  element.classList.add("is-busy");
  element.innerHTML = `<div class="empty-state" aria-busy="true">
    <div class="skeleton skeleton-line w-40"></div>
    <div class="skeleton skeleton-line w-80"></div>
    <div class="skeleton skeleton-line w-60"></div>
    <p class="muted small" style="margin-top:12px">Checking validated inputs and loading the saved pipeline…</p>
  </div>`;
}

function renderResultError(track, error) {
  const element = $(`#${track}-result`);
  if (!element) return;
  element.classList.remove("is-busy");
  element.innerHTML = `<div class="result-error">
    <b>Estimate unavailable</b>
    <p>${esc(error.message)}</p>
    <div class="form-actions" style="margin-top:10px">
      <button class="button small" type="button" data-retry-submit>${icon("refresh", "icon icon-sm")}Try again</button>
      <button class="button small quiet" type="button" data-focus-form>Review inputs</button>
    </div>
  </div>`;
  const retry = $("[data-retry-submit]", element);
  if (retry) retry.addEventListener("click", () => $(`#${track}-form`).requestSubmit());
  const focus = $("[data-focus-form]", element);
  if (focus) focus.addEventListener("click", () => $(`#${track}-form`).scrollIntoView({ behavior: "smooth", block: "start" }));
}

function advisoryBlock(items, note) {
  if (!Array.isArray(items) || items.length === 0) {
    return `<div class="advisory">
      <b>No advisory thresholds crossed</b>
      <p>No specific prompt was triggered for these inputs. This is not a guarantee of any outcome.</p>
      ${note ? `<small>${esc(note)}</small>` : ""}
    </div>`;
  }
  return `<div class="advisory">
    <b>Conversation starters</b>
    <ul>${items.map((item) => `<li>${esc(item)}</li>`).join("")}</ul>
    ${note ? `<small>${esc(note)}</small>` : ""}
  </div>`;
}

function limitationsBlock(limitations) {
  if (!Array.isArray(limitations) || limitations.length === 0) return "";
  return `<div class="result-limitations">
    <h4>Limitations</h4>
    <ul>${limitations.map((item) => `<li>${esc(item)}</li>`).join("")}</ul>
  </div>`;
}

function renderExamResult(data) {
  const element = $("#exam-result");
  element.classList.remove("is-busy");
  const tierClass = data.tier === "Strong" ? "ok" : data.tier === "On track" ? "info" : "review";
  const interval = Array.isArray(data.interval) && data.interval.length === 2
    ? `Approximate range: ${fmt.dec(data.interval[0], 2)} to ${fmt.dec(data.interval[1], 2)} points`
    : "No error band is available because the held-out metric is missing.";
  element.innerHTML = `
    <div class="result-head">
      <span class="eyebrow">Estimated exam score</span>
      <span class="badge badge-${tierClass}">${esc(data.tier)}</span>
    </div>
    <div class="result-value">${esc(fmt.dec(data.score, 1))}<span>/100</span></div>
    <p class="result-caption">${esc(interval)}</p>
    <div class="result-meta">
      <span><b>Model:</b> ${esc(data.model)}</span>
      <span><b>Error band:</b> ${esc(data.interval_basis || "Not available")}</span>
    </div>
    <div class="callout callout-info" style="margin-top:12px">
      ${icon("info")}
      <div>${esc(data.interval_note || "This estimate is not a guarantee.")}</div>
    </div>
    ${advisoryBlock(data.advisory_suggestions, data.advisory_note)}
    ${limitationsBlock(data.limitations)}
  `;
}

function renderPlacementResult(data) {
  const element = $("#placement-result");
  element.classList.remove("is-busy");
  const probability = Math.max(0, Math.min(1, Number(data.probability) || 0));
  const outcomeClass = data.placed ? "info" : "neutral";
  const outcomeLabel = data.placed ? "Model signals placed" : "Model signals not placed";
  element.innerHTML = `
    <div class="result-head">
      <span class="eyebrow">Estimated placement outcome</span>
      <span class="badge badge-${outcomeClass}">${esc(outcomeLabel)}</span>
    </div>
    <div class="result-value">${esc(fmt.dec(data.percentage, 1))}<span>% model output</span></div>
    <p class="result-caption">Decision threshold: ${esc(fmt.dec(data.decision_threshold, 2))}. The probability is not calibrated.</p>
    <div class="probability-meter" role="img" aria-label="Model output probability ${esc(fmt.pct(probability, 1))}"><span style="width:${(probability * 100).toFixed(1)}%"></span></div>
    <div class="result-meta">
      <span><b>Model:</b> ${esc(data.model)}</span>
      <span><b>Source track:</b> ${esc(data.source_track || "placement")}</span>
      <span><b>Exam records linked:</b> ${data.exam_records_linked ? "Yes" : "No"}</span>
    </div>
    <div class="callout callout-warning" style="margin-top:12px">
      ${icon("warning")}
      <div>${esc(data.warning)}</div>
    </div>
    <p class="muted small" style="margin-top:10px">${esc(data.probability_note || "")}</p>
    ${advisoryBlock(data.advisory_suggestions, data.advisory_note)}
    ${limitationsBlock(data.limitations)}
  `;
}

/* ---------------------------------------------------------------------------
 * 5. View loaders and renderers
 * ------------------------------------------------------------------------- */

async function loadOverview() {
  const [summaryResult, performanceResult, leakageResult] = await Promise.all([
    settled(getSummary()),
    settled(getPerformance()),
    settled(API.get("/api/leakage")),
  ]);

  const summary = summaryResult.ok ? summaryResult.data : null;
  const performance = performanceResult.ok ? performanceResult.data : null;
  const leakage = leakageResult.ok ? leakageResult.data : null;

  if (summary) {
    $("#hero-exam-rows").textContent = fmt.int(summary.exam_rows);
    $("#hero-placement-rows").textContent = fmt.int(summary.placement_rows);
    $("#overview-stats").innerHTML = [
      statCard("Exam records", fmt.int(summary.exam_rows), `${fmt.int(summary.exam_columns)} columns after encoding`),
      statCard("Average exam score", fmt.dec(summary.exam_mean, 2), `Range ${summary.exam_min} to ${summary.exam_max}`),
      statCard("Placement records", fmt.int(summary.placement_rows), `${fmt.int(summary.placement_columns)} columns after encoding`),
      statCard("Placed share", fmt.pct(summary.placement_rate), `${fmt.int(summary.placement_placed)} placed · ${fmt.int(summary.placement_unplaced)} not placed`),
    ].join("");
    if (summary.investigation) {
      $("#overview-alert-text").textContent = summary.investigation;
    }
  } else {
    $("#overview-stats").innerHTML = `<div class="card" style="grid-column:1/-1">${errorMarkup("Summary data is unavailable. Run python scripts/preprocess.py.", true)}</div>`;
    $("#hero-exam-rows").textContent = "—";
    $("#hero-placement-rows").textContent = "—";
  }

  if (performance) {
    $("#hero-model-count").textContent = fmt.int((performance.models || []).length);
    renderOverviewModelMetrics(performance);
  } else {
    $("#hero-model-count").textContent = "—";
    $("#overview-exam-metrics").innerHTML = metricChip("Deployed model", summary?.deployed_models?.exam || "Unavailable");
    $("#overview-placement-metrics").innerHTML = metricChip("Deployed model", summary?.deployed_models?.placement || "Unavailable");
  }

  renderOverviewFacts(summary, performance, leakage);
}

function renderOverviewModelMetrics(performance) {
  const models = performance.models || [];
  const exam = models.find((model) => model.task === "exam_score" && model.deployed);
  const placement = models.find((model) => model.task === "placement" && model.deployed);
  if (exam && exam.metrics && exam.metrics.test) {
    $("#overview-exam-metrics").innerHTML = [
      metricChip("Deployed model", exam.model),
      metricChip("Test MAE", fmt.dec(exam.metrics.test.MAE, 3)),
      metricChip("Test R²", fmt.dec(exam.metrics.test.R2, 3)),
    ].join("");
  }
  if (placement && placement.metrics && placement.metrics.test) {
    $("#overview-placement-metrics").innerHTML = [
      metricChip("Deployed model", placement.model),
      metricChip("Test F1", fmt.pct(placement.metrics.test.F1, 1)),
      metricChip("Test ROC-AUC", fmt.dec(placement.metrics.test.ROC_AUC, 3)),
    ].join("");
  }
}

function renderOverviewFacts(summary, performance, leakage) {
  const container = $("#overview-facts");
  if (!container) return;
  const items = [];
  if (summary) {
    items.push({
      state: summary.source_records_genuinely_linked ? "bad" : "warn",
      text: summary.source_records_genuinely_linked
        ? "The source files share a student identifier."
        : "No shared student identifier exists between the two source files.",
    });
    items.push({
      state: summary.row_order_pairing_status === "removed" ? "ok" : "warn",
      text: "Row-order pairing is removed and is not used in modeling.",
    });
    items.push({
      state: "ok",
      text: `${fmt.int(summary.exam_rows)} exam rows and ${fmt.int(summary.placement_rows)} placement rows are modeled independently.`,
    });
  }
  if (performance) {
    items.push({
      state: "ok",
      text: `${fmt.int((performance.models || []).length)} supervised models are registered, with held-out and cross-validation evidence.`,
    });
  }
  if (leakage && leakage.placement_rule) {
    items.push({
      state: "warn",
      text: `The placement rule reproduces ${fmt.pct(leakage.placement_rule.accuracy, 2)} of source labels with ${fmt.int(leakage.placement_rule.errors)} errors; high placement scores are a dataset-construction artifact.`,
    });
  }
  if (!items.length) {
    container.innerHTML = errorMarkup("Overview facts are unavailable. Run the pipeline first.", true);
    return;
  }
  container.innerHTML = `<ul class="check-list">${items
    .map((item) => {
      const iconName = item.state === "ok" ? "check" : item.state === "bad" ? "warning" : "info";
      return `<li class="${item.state === "ok" ? "" : item.state === "bad" ? "is-bad" : "is-warn"}">${icon(iconName)}<span>${esc(item.text)}</span></li>`;
    })
    .join("")}</ul>`;
}

async function loadExamView() {
  const [schemaResult, summaryResult, importanceResult] = await Promise.all([
    settled(getSchema()),
    settled(getSummary()),
    settled(getImportance()),
  ]);
  if (!schemaResult.ok) {
    showFormUnavailable("exam", schemaResult.error.message);
    return;
  }
  const schema = schemaResult.data;
  const meta = schema.tracks.exam;
  const fields = meta.fields;
  $("#exam-model-name").textContent = meta.model_label;
  $("#exam-model-badge").textContent = meta.model_label;
  buildPredictionForm("exam", fields);
  bindPredictionForm("exam", fields);
  renderLimitations("exam", meta.limitations);
  renderProcess("exam", meta, summaryResult.ok ? summaryResult.data : null);
  if (importanceResult.ok) {
    state.importance = importanceResult.data;
    renderImportance("exam_score", "exam-drivers", 6);
  } else {
    $("#exam-drivers").innerHTML = errorMarkup("Feature importance is unavailable. Run python scripts/explain_models.py.", true);
  }
  if (state.health && state.health.models && state.health.models.exam && !state.health.models.exam.ready) {
    $("#exam-submit").disabled = true;
    showGlobalAlert("The exam model artifact is missing. Run python scripts/train_models.py first.", "danger");
  }
}

async function loadPlacementView() {
  const [schemaResult, summaryResult] = await Promise.all([
    settled(getSchema()),
    settled(getSummary()),
  ]);
  if (!schemaResult.ok) {
    showFormUnavailable("placement", schemaResult.error.message);
    return;
  }
  const meta = schemaResult.data.tracks.placement;
  const fields = meta.fields;
  $("#placement-model-name").textContent = meta.model_label;
  $("#placement-model-badge").textContent = meta.model_label;
  buildPredictionForm("placement", fields);
  bindPredictionForm("placement", fields);
  renderLimitations("placement", meta.limitations);
  renderProcess("placement", meta, summaryResult.ok ? summaryResult.data : null);
  if (state.health && state.health.models && state.health.models.placement && !state.health.models.placement.ready) {
    $("#placement-submit").disabled = true;
    showGlobalAlert("The placement model artifact is missing. Run python scripts/train_models.py first.", "danger");
  }
}

function showFormUnavailable(track, message) {
  const container = $(`#${track}-fields`);
  container.setAttribute("aria-busy", "false");
  container.innerHTML = errorMarkup(message, true);
  const submit = $(`#${track}-submit`);
  if (submit) submit.disabled = true;
}

function renderImportance(taskKey, containerId, limit = 6) {
  const container = $(`#${containerId}`);
  if (!container) return;
  const block = state.importance && state.importance.tasks && state.importance.tasks[taskKey];
  if (!block || !Array.isArray(block.models) || block.models.length === 0) {
    container.innerHTML = errorMarkup("Feature importance is unavailable. Run python scripts/explain_models.py.", true);
    return;
  }
  const preferredName = taskKey === "exam_score" ? "LinearRegression" : "RandomForestClassifier";
  const model = block.models.find((item) => item.model === preferredName) || block.models[0];
  const rows = (model.top_permutation_features || []).slice(0, limit);
  if (!rows.length) {
    container.innerHTML = errorMarkup("This model has no permutation-importance rows.", true);
    return;
  }
  const max = Math.max(...rows.map((row) => Math.max(0, Number(row.importance) || 0)), 1e-9);
  container.innerHTML = `<div class="bar-group">
    <h4>${esc(model.model)}</h4>
    <p class="bar-caption">${esc(block.test_metric ? `Permutation importance on test ${block.test_metric}` : "Permutation importance on the held-out test split")}</p>
    ${rows.map((row) => barRow(row.feature, row.importance, max, fmt.dec(row.importance, 4))).join("")}
  </div>
  <p class="muted small" style="margin-top:10px">${esc((state.importance.caveats || [])[0] || "Association with prediction is not a causal effect.")}</p>`;
}

async function loadInsightsView() {
  const [clusterResult, fairnessResult, pcaResult, qualityResult] = await Promise.all([
    settled(API.get("/api/clusters")),
    settled(API.get("/api/fairness")),
    settled(API.get("/api/pca")),
    settled(API.get("/api/data-quality")),
  ]);
  renderClusters(clusterResult);
  renderFairness(fairnessResult);
  renderPca(pcaResult);
  renderQuality(qualityResult);
}

function renderClusters(result) {
  const summary = $("#cluster-summary");
  const table = $("#cluster-table");
  const image = $("#cluster-figure");
  const note = $("#cluster-note");
  const badge = $("#cluster-badge");

  if (!result.ok) {
    summary.innerHTML = errorMarkup(result.error.message, true);
    table.innerHTML = `<tr><td colspan="4">${errorMarkup("Cluster profiles are unavailable.", false)}</td></tr>`;
    image.hidden = true;
    note.textContent = "Run python scripts/clustering.py to generate the cluster summary and visualization.";
    badge.textContent = "Unavailable";
    return;
  }

  const data = result.data;
  badge.textContent = "Descriptive";
  summary.innerHTML = `
    <div class="card-head">
      <div><span class="eyebrow">Cohort method</span><h2>${esc(data.method)}</h2></div>
      <span class="badge badge-info">${esc(String(data.n_clusters))} clusters</span>
    </div>
    <div class="stats-grid" style="grid-template-columns:repeat(2,minmax(0,1fr));margin:14px 0 0">
      ${statCard("Records assigned", fmt.int(data.rows), `Fitted on ${fmt.int(data.train_rows)} training rows`)}
      ${statCard("Train silhouette", fmt.dec(data.silhouette_score, 3), "Higher means more separated")}
    </div>
    <p class="muted small" style="margin-top:12px">${esc(data.assignment_policy)}</p>
    <p class="muted small" style="margin-top:8px">${esc((data.limitations || []).join(" "))}</p>
  `;

  const centers = data.cluster_centers || [];
  table.innerHTML = centers
    .map((cluster) => {
      const numeric = cluster.numeric_center || {};
      const profileBits = [];
      if (isNumber(numeric.Attendance)) profileBits.push(`attendance ${fmt.dec(numeric.Attendance, 1)}`);
      if (isNumber(numeric.Hours_Studied)) profileBits.push(`study ${fmt.dec(numeric.Hours_Studied, 1)} h`);
      if (isNumber(numeric.Previous_Scores)) profileBits.push(`previous scores ${fmt.dec(numeric.Previous_Scores, 1)}`);
      return `<tr>
        <td><span class="badge badge-neutral">Cluster ${esc(String(cluster.cluster))}</span></td>
        <td class="num">${esc(fmt.int(cluster.count))}</td>
        <td class="num">${esc(fmt.pct(cluster.share, 1))}</td>
        <td>${esc(cluster.description)}${profileBits.length ? `<br><span class="muted small">${esc(profileBits.join(" · "))}</span>` : ""}</td>
      </tr>`;
    })
    .join("");

  image.hidden = false;
  const ratios = data.pca_explained_variance_ratio || [];
  note.textContent = ratios.length
    ? `PCA view fitted on the training split. PC1 explains ${fmt.pct(ratios[0], 1)} and PC2 ${fmt.pct(ratios[1], 1)} of variance.`
    : "PCA view fitted on the training split.";
  wireImageFallback(image, note, "The cluster visualization is unavailable. Run python scripts/clustering.py.");
}

function renderFairness(result) {
  const summary = $("#fairness-summary");
  const table = $("#fairness-table");
  const note = $("#fairness-note");
  const badge = $("#fairness-badge");

  if (!result.ok) {
    summary.textContent = result.error.message;
    table.innerHTML = `<tr><td colspan="4">${errorMarkup("Group comparison is unavailable.", false)}</td></tr>`;
    note.textContent = "Run python scripts/fairness_analysis.py to generate the comparison.";
    badge.textContent = "Unavailable";
    return;
  }

  const data = result.data;
  badge.textContent = "Descriptive";
  const gap = data.largest_group_gap;
  summary.innerHTML = `
    <p><b>Overall test MAE:</b> ${esc(fmt.dec(data.overall_mae, 3))}</p>
    ${gap ? `<p><b>Largest gap:</b> ${esc(fmt.dec(gap.absolute_gap, 3))} between ${esc(gap.lowest_error_group)} and ${esc(gap.highest_error_group)} in ${esc(gap.group_column)}.</p>` : ""}
    <p>${esc(data.interpretation || "")}</p>
  `;
  const rows = data.results || [];
  table.innerHTML = rows.length
    ? rows
        .map(
          (row) => `<tr>
            <td>${esc(row.group_column)}: ${esc(row.group)}</td>
            <td class="num">${esc(fmt.int(row.group_size))}</td>
            <td class="num">${esc(fmt.dec(row.mean_absolute_error, 3))}</td>
            <td class="num">${esc(fmt.signed(row.difference_from_overall, 3))}</td>
          </tr>`,
        )
        .join("")
    : `<tr><td colspan="4">${errorMarkup("No group met the minimum size threshold.", false)}</td></tr>`;
  const excluded = Object.keys(data.groups_excluded || {});
  note.textContent = `${(data.limitations || []).join(" ")}${excluded.length ? ` Excluded groups: ${excluded.join(", ")}.` : ""}`;
}

function renderPca(result) {
  const image = $("#pca-figure");
  const note = $("#pca-note");
  if (!result.ok) {
    image.hidden = true;
    note.textContent = "PCA output is unavailable. Run python scripts/pca_analysis.py.";
    return;
  }
  const data = result.data;
  const exam = data.exam_score;
  const placement = data.placement;
  if (!exam || !placement) {
    image.hidden = true;
    note.textContent = "PCA summary is incomplete. Run python scripts/pca_analysis.py.";
    return;
  }
  image.hidden = false;
  note.textContent = `Exam track: PC1 ${fmt.pct(exam.explained_variance_ratio[0], 1)}, PC2 ${fmt.pct(exam.explained_variance_ratio[1], 1)}. Placement track: PC1 ${fmt.pct(placement.explained_variance_ratio[0], 1)}, PC2 ${fmt.pct(placement.explained_variance_ratio[1], 1)}. Targets were used only for coloring the test split.`;
  wireImageFallback(image, note, "The PCA visualization is unavailable. Run python scripts/pca_analysis.py.");
}

function renderQuality(result) {
  const summary = $("#quality-summary");
  const details = $("#quality-details");
  const badge = $("#quality-badge");
  const alert = $("#quality-alert");
  if (!result.ok) {
    summary.innerHTML = `<div style="grid-column:1/-1">${errorMarkup(result.error.message, true)}</div>`;
    details.innerHTML = errorMarkup("Data-quality details are unavailable. Run python scripts/data_quality.py.", true);
    badge.textContent = "Unavailable";
    return;
  }

  const data = result.data;
  const exam = data.datasets.exam_processed;
  const placement = data.datasets.placement_processed;
  const link = data.linkage || {};
  const splitMatches = Object.values(data.split_row_totals || {}).every((item) => item.matches_processed);
  badge.textContent = splitMatches ? "Checked" : "Review";
  alert.hidden = !link.genuinely_linked ? false : true;

  summary.innerHTML = [
    statCard("Genuine student link", link.genuinely_linked ? "Yes" : "No", link.shared_student_identifier ? `Shared key: ${link.shared_student_identifier}` : "No shared identifier"),
    statCard("Row-order pairing", link.row_order_pairing_status === "removed" ? "Removed" : "Review", "Not used in modeling"),
    statCard("Split row check", splitMatches ? "Match" : "Mismatch", "Split rows equal processed rows"),
    statCard("Exam score range", `${fmt.int(exam.target_summary.min)} to ${fmt.int(exam.target_summary.max)}`, "After preprocessing"),
  ].join("");

  const splitRows = ["exam_score", "placement"]
    .map((track) => {
      const split = data.splits[track];
      const total = split.train.rows + split.val.rows + split.test.rows;
      return `<tr><td>${track === "exam_score" ? "Exam" : "Placement"}</td><td class="num">${fmt.int(split.train.rows)}</td><td class="num">${fmt.int(split.val.rows)}</td><td class="num">${fmt.int(split.test.rows)}</td><td class="num">${fmt.int(total)}</td></tr>`;
    })
    .join("");

  const processing = data.processing || {};
  const processingBits = [];
  if (processing.exam) {
    processingBits.push(`Exam: removed ${fmt.int(processing.exam.incomplete_rows_removed)} incomplete rows and ${fmt.int(processing.exam.duplicates_removed)} duplicate rows; bounded ${fmt.int(processing.exam.exam_scores_bounded)} exam score(s).`);
  }
  if (processing.placement) {
    processingBits.push(`Placement: removed ${fmt.int(processing.placement.duplicates_removed)} duplicate rows; bounded ${fmt.int(processing.placement.cgpa_values_bounded)} CGPA value(s).`);
  }

  details.innerHTML = `
    <div class="table-wrap" style="margin-top:10px">
      <table>
        <caption class="sr-only">Dataset row and column counts</caption>
        <thead><tr><th>Dataset</th><th class="num">Rows</th><th class="num">Columns</th><th class="num">Missing</th><th class="num">Duplicates</th></tr></thead>
        <tbody>
          <tr><td>Exam raw</td><td class="num">${fmt.int(data.datasets.exam_raw.rows)}</td><td class="num">${fmt.int(data.datasets.exam_raw.columns)}</td><td class="num">${fmt.int(data.datasets.exam_raw.missing_values_total)}</td><td class="num">${fmt.int(data.datasets.exam_raw.duplicate_rows)}</td></tr>
          <tr><td>Exam processed</td><td class="num">${fmt.int(exam.rows)}</td><td class="num">${fmt.int(exam.columns)}</td><td class="num">${fmt.int(exam.missing_values_total)}</td><td class="num">${fmt.int(exam.duplicate_rows)}</td></tr>
          <tr><td>Placement raw</td><td class="num">${fmt.int(data.datasets.placement_raw.rows)}</td><td class="num">${fmt.int(data.datasets.placement_raw.columns)}</td><td class="num">${fmt.int(data.datasets.placement_raw.missing_values_total)}</td><td class="num">${fmt.int(data.datasets.placement_raw.duplicate_rows)}</td></tr>
          <tr><td>Placement processed</td><td class="num">${fmt.int(placement.rows)}</td><td class="num">${fmt.int(placement.columns)}</td><td class="num">${fmt.int(placement.missing_values_total)}</td><td class="num">${fmt.int(placement.duplicate_rows)}</td></tr>
        </tbody>
      </table>
    </div>
    <div class="table-wrap" style="margin-top:12px">
      <table>
        <caption class="sr-only">Train, validation, and test row counts</caption>
        <thead><tr><th>Track</th><th class="num">Train</th><th class="num">Validation</th><th class="num">Test</th><th class="num">Total</th></tr></thead>
        <tbody>${splitRows}</tbody>
      </table>
    </div>
    <p class="muted small" style="margin-top:12px">${esc(processingBits.join(" "))}</p>
    <p class="muted small" style="margin-top:8px">${esc(link.investigation || "No shared student identifier was found.")}</p>
  `;
}

async function loadRecommendationsView() {
  const [schemaResult, rulesResult] = await Promise.all([
    settled(getSchema()),
    settled(API.get("/api/advisory-rules")),
  ]);
  if (!schemaResult.ok || !rulesResult.ok) {
    const message = !rulesResult.ok ? rulesResult.error.message : schemaResult.error.message;
    $("#advisory-fields").innerHTML = errorMarkup(message, true);
    $("#advisory-rules").innerHTML = `<tr><td colspan="3">${errorMarkup("Advisory thresholds are unavailable.", false)}</td></tr>`;
    return;
  }

  const schema = schemaResult.data;
  const ruleData = rulesResult.data;
  const rules = ruleData.rules || [];
  const schemaFields = {};
  Object.values(schema.tracks).forEach((track) => {
    (track.fields || []).forEach((field) => {
      schemaFields[field.name] = field;
    });
  });
  const ruleFields = [];
  const seen = new Set();
  rules.forEach((rule) => {
    if (seen.has(rule.field)) return;
    seen.add(rule.field);
    const schemaField = schemaFields[rule.field];
    if (schemaField) {
      ruleFields.push({ ...schemaField, example: "", help: rule.message });
    } else {
      ruleFields.push({
        name: rule.field,
        label: rule.label,
        type: "number",
        min: 0,
        max: Math.max(Number(rule.threshold) * 3, 100),
        step: 1,
        example: "",
        unit: rule.unit,
        help: rule.message,
      });
    }
  });
  $("#advisory-fields").setAttribute("aria-busy", "false");
  $("#advisory-fields").innerHTML = ruleFields.map((field) => fieldMarkup("advisory", field, { optional: true })).join("");
  $("#advisory-rules").innerHTML = rules
    .map(
      (rule) => `<tr>
        <td>${esc(rule.label)}</td>
        <td>${esc(rule.operator)} ${esc(String(rule.threshold))} ${esc(rule.unit || "")}</td>
        <td>${esc(rule.message)}</td>
      </tr>`,
    )
    .join("");
  $("#advisory-note").textContent = ruleData.note || "";

  const form = $("#advisory-form");
  if (form.dataset.bound === "true") return;
  form.dataset.bound = "true";
  ruleFields.forEach((field) => {
    const input = fieldInput("advisory", field);
    if (!input) return;
    input.addEventListener("blur", () => validateField("advisory", field, { optional: true }));
    input.addEventListener("input", () => setFieldError("advisory", field, ""));
  });
  form.addEventListener("reset", () => {
    window.setTimeout(() => ruleFields.forEach((field) => setFieldError("advisory", field, "")), 0);
  });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    let valid = true;
    ruleFields.forEach((field) => {
      if (!validateField("advisory", field, { optional: true })) valid = false;
    });
    if (!valid) return;
    const payload = {};
    ruleFields.forEach((field) => {
      const input = fieldInput("advisory", field);
      if (input && input.value.trim() !== "") payload[field.name] = input.value.trim();
    });
    const submit = $("#advisory-submit");
    submit.disabled = true;
    submit.textContent = "Checking…";
    $("#advisory-result").innerHTML = `<div class="empty-state" aria-busy="true">
      <div class="skeleton skeleton-line w-40"></div>
      <div class="skeleton skeleton-line w-80"></div>
      <p class="muted small" style="margin-top:12px">Checking transparent thresholds…</p>
    </div>`;
    try {
      const data = await API.post("/api/interventions", payload);
      $("#advisory-result").innerHTML = advisoryBlock(data.suggestions, data.note);
    } catch (error) {
      $("#advisory-result").innerHTML = `<div class="result-error"><b>Prompts unavailable</b><p>${esc(error.message)}</p></div>`;
    } finally {
      submit.disabled = false;
      submit.textContent = "Get advisory prompts";
    }
  });
}

async function loadModelsView() {
  const [performanceResult, cvResult, overfitResult, importanceResult, leakageResult, investigationResult] =
    await Promise.all([
      settled(getPerformance()),
      settled(API.get("/api/cross-validation")),
      settled(API.get("/api/overfitting")),
      settled(getImportance()),
      settled(API.get("/api/leakage")),
      settled(API.get("/api/placement-investigation")),
    ]);

  const overfit = overfitResult.ok ? overfitResult.data : null;
  renderModelComparison(performanceResult, overfit);
  renderCrossValidation(cvResult);
  renderSplitComparison(performanceResult);
  renderOverfit(overfitResult);
  renderConfusions(performanceResult);
  if (importanceResult.ok) {
    state.importance = importanceResult.data;
    renderAllImportance();
  } else {
    $("#importance-body").innerHTML = errorMarkup(importanceResult.error.message, true);
  }
  renderLeakage(leakageResult);
  renderInvestigation(investigationResult);
}

function renderModelComparison(result, overfit) {
  const body = $("#comparison-body");
  const badge = $("#comparison-badge");
  if (!result.ok) {
    body.innerHTML = `<tr><td colspan="5">${errorMarkup(result.error.message, true)}</td></tr>`;
    badge.textContent = "Unavailable";
    return;
  }
  const models = result.data.models || [];
  const statusByKey = new Map();
  ((overfit && overfit.summary) || []).forEach((row) => {
    statusByKey.set(`${row.task}|${row.model}`, row.status);
  });
  badge.textContent = `${models.length} models`;
  body.innerHTML = models
    .map((model) => {
      const test = (model.metrics && model.metrics.test) || {};
      const metricsText = model.task === "exam_score"
        ? `MAE ${fmt.dec(test.MAE, 3)} · RMSE ${fmt.dec(test.RMSE, 3)} · R² ${fmt.dec(test.R2, 3)}`
        : `Accuracy ${fmt.pct(test.Accuracy, 2)} · F1 ${fmt.pct(test.F1, 2)} · ROC-AUC ${fmt.dec(test.ROC_AUC, 3)}`;
      const status = statusByKey.get(`${model.task}|${model.model}`) || "unknown";
      return `<tr>
        <td>${model.task === "exam_score" ? "Exam" : "Placement"}</td>
        <td><strong>${esc(model.model)}</strong>${model.deployed ? ` <span class="badge badge-accent">Deployed</span>` : ""}</td>
        <td>${esc(model.preprocessing)}</td>
        <td>${esc(metricsText)}</td>
        <td>${statusBadge(status)}</td>
      </tr>`;
    })
    .join("");
}

function renderCrossValidation(result) {
  const body = $("#cv-body");
  if (!result.ok) {
    body.innerHTML = `<tr><td colspan="3">${errorMarkup(result.error.message, true)}</td></tr>`;
    return;
  }
  const groups = new Map();
  (result.data.results || []).forEach((row) => {
    const key = `${row.task}|${row.model}`;
    if (!groups.has(key)) groups.set(key, { task: row.task, model: row.model, metrics: {} });
    groups.get(key).metrics[row.metric] = row;
  });
  const rows = Array.from(groups.values());
  body.innerHTML = rows.length
    ? rows
        .map((group) => {
          const metrics = group.metrics;
          const text = group.task === "exam_score"
            ? `MAE ${fmt.meanStd(metrics.MAE?.mean, metrics.MAE?.std, 3)} · RMSE ${fmt.meanStd(metrics.RMSE?.mean, metrics.RMSE?.std, 3)} · R² ${fmt.meanStd(metrics.R2?.mean, metrics.R2?.std, 3)}`
            : `Accuracy ${fmt.meanStd(metrics.Accuracy?.mean, metrics.Accuracy?.std, 3)} · Precision ${fmt.meanStd(metrics.Precision?.mean, metrics.Precision?.std, 3)} · Recall ${fmt.meanStd(metrics.Recall?.mean, metrics.Recall?.std, 3)} · F1 ${fmt.meanStd(metrics.F1?.mean, metrics.F1?.std, 3)} · ROC-AUC ${fmt.meanStd(metrics.ROC_AUC?.mean, metrics.ROC_AUC?.std, 3)}`;
          return `<tr><td>${group.task === "exam_score" ? "Exam" : "Placement"}</td><td>${esc(group.model)}</td><td>${esc(text)}</td></tr>`;
        })
        .join("")
    : `<tr><td colspan="3">${errorMarkup("No cross-validation rows are available.", false)}</td></tr>`;
}

function renderSplitComparison(result) {
  const body = $("#split-body");
  if (!result.ok) {
    body.innerHTML = `<tr><td colspan="3">${errorMarkup(result.error.message, true)}</td></tr>`;
    return;
  }
  const models = result.data.models || [];
  body.innerHTML = models
    .map((model) => {
      const metrics = model.metrics || {};
      const train = metrics.train || {};
      const val = metrics.val || {};
      const test = metrics.test || {};
      const text = model.task === "exam_score"
        ? `<span class="small"><b>MAE</b> ${fmt.dec(train.MAE, 3)} / ${fmt.dec(val.MAE, 3)} / ${fmt.dec(test.MAE, 3)}<br><b>RMSE</b> ${fmt.dec(train.RMSE, 3)} / ${fmt.dec(val.RMSE, 3)} / ${fmt.dec(test.RMSE, 3)}<br><b>R²</b> ${fmt.dec(train.R2, 3)} / ${fmt.dec(val.R2, 3)} / ${fmt.dec(test.R2, 3)}</span>`
        : `<span class="small"><b>Accuracy</b> ${fmt.dec(train.Accuracy, 3)} / ${fmt.dec(val.Accuracy, 3)} / ${fmt.dec(test.Accuracy, 3)}<br><b>F1</b> ${fmt.dec(train.F1, 3)} / ${fmt.dec(val.F1, 3)} / ${fmt.dec(test.F1, 3)}<br><b>ROC-AUC</b> ${fmt.dec(train.ROC_AUC, 3)} / ${fmt.dec(val.ROC_AUC, 3)} / ${fmt.dec(test.ROC_AUC, 3)}</span>`;
      return `<tr><td>${model.task === "exam_score" ? "Exam" : "Placement"}</td><td>${esc(model.model)}</td><td>${text}</td></tr>`;
    })
    .join("");
}

function renderOverfit(result) {
  const warnings = $("#overfit-warnings");
  const body = $("#overfit-body");
  if (!result.ok) {
    warnings.innerHTML = errorMarkup(result.error.message, true);
    body.innerHTML = `<tr><td colspan="5">${errorMarkup("Overfitting analysis is unavailable.", false)}</td></tr>`;
    return;
  }
  const data = result.data;
  const globalWarnings = data.global_warnings || [];
  warnings.innerHTML = `<div>${globalWarnings.length ? `<ul style="margin:0;padding-left:18px">${globalWarnings.map((item) => `<li>${esc(item)}</li>`).join("")}</ul>` : "No global warnings were generated."}</div>`;
  const summaries = data.summary || [];
  body.innerHTML = summaries.length
    ? summaries
        .map((row) => {
          const gap = isNumber(row.headline_gap) ? fmt.signed(row.headline_gap, 3) : "—";
          return `<tr>
            <td>${row.task === "exam_score" ? "Exam" : "Placement"}</td>
            <td>${esc(row.model)}</td>
            <td>${esc(row.headline_metric)}: ${fmt.dec(row.headline_train, 3)} / ${fmt.dec(row.headline_validation, 3)} / ${fmt.dec(row.headline_test, 3)} / ${fmt.meanStd(row.headline_cv_mean, row.headline_cv_std, 3)}<br><span class="muted small">gap ${esc(gap)}</span></td>
            <td class="small">${esc(row.flag_text || "none")}</td>
            <td>${statusBadge(row.status)}</td>
          </tr>`;
        })
        .join("")
    : `<tr><td colspan="5">${errorMarkup("No overfitting summary is available.", false)}</td></tr>`;
}

function renderConfusions(result) {
  const grid = $("#confusion-grid");
  if (!result.ok) {
    grid.innerHTML = errorMarkup(result.error.message, true);
    return;
  }
  const models = (result.data.models || []).filter((model) => model.task === "placement" && model.confusion_matrix);
  if (!models.length) {
    grid.innerHTML = errorMarkup("No confusion matrices are available.", false);
    return;
  }
  grid.innerHTML = models
    .map((model) => {
      const matrix = model.confusion_matrix.matrix || [[0, 0], [0, 0]];
      const test = model.metrics.test || {};
      return `<div class="matrix-card">
        <h4>${esc(model.model)}</h4>
        <table class="matrix">
          <caption class="sr-only">Confusion matrix for ${esc(model.model)}</caption>
          <thead><tr><th></th><th>Predicted 0</th><th>Predicted 1</th></tr></thead>
          <tbody>
            <tr><th>Actual 0</th><td>${esc(fmt.int(matrix[0][0]))}</td><td>${esc(fmt.int(matrix[0][1]))}</td></tr>
            <tr><th>Actual 1</th><td>${esc(fmt.int(matrix[1][0]))}</td><td>${esc(fmt.int(matrix[1][1]))}</td></tr>
          </tbody>
        </table>
        <p class="matrix-note">Precision ${fmt.pct(test.Precision, 1)} · Recall ${fmt.pct(test.Recall, 1)} · F1 ${fmt.pct(test.F1, 1)} · ROC-AUC ${fmt.dec(test.ROC_AUC, 3)}</p>
      </div>`;
    })
    .join("");
}

function renderAllImportance() {
  const container = $("#importance-body");
  const tasks = state.importance && state.importance.tasks;
  if (!tasks) {
    container.innerHTML = errorMarkup("Feature importance is unavailable.", true);
    return;
  }
  const groups = [];
  ["exam_score", "placement"].forEach((taskKey) => {
    const block = tasks[taskKey];
    if (!block || !Array.isArray(block.models)) return;
    groups.push(`<div class="bar-group"><h4>${taskKey === "exam_score" ? "Exam score" : "Placement (experimental)"}</h4>`);
    block.models.forEach((model) => {
      const rows = (model.top_permutation_features || []).slice(0, 5);
      if (!rows.length) return;
      const max = Math.max(...rows.map((row) => Math.max(0, Number(row.importance) || 0)), 1e-9);
      groups.push(`<h4 style="font-size:12px;margin-top:10px">${esc(model.model)}</h4>`);
      groups.push(rows.map((row) => barRow(row.feature, row.importance, max, fmt.dec(row.importance, 4))).join(""));
    });
    groups.push("</div>");
  });
  const caveats = (state.importance.caveats || []).slice(0, 2).join(" ");
  container.innerHTML = `${groups.join("")}<p class="muted small" style="margin-top:12px">${esc(caveats)}</p>`;
}

function renderLeakage(result) {
  const container = $("#leakage-body");
  if (!result.ok) {
    container.innerHTML = errorMarkup(result.error.message, true);
    return;
  }
  const data = result.data;
  const rule = data.placement_rule || {};
  const checks = (data.checks || []).filter(
    (check) => !check.result && (check.severity === "critical" || check.severity === "review"),
  );
  const claim = data.leakage_free_claim === false
    ? `<span class="badge badge-review">${icon("warning", "icon icon-sm")}No leakage-free claim</span>`
    : `<span class="badge badge-neutral">Claim not stated</span>`;
  container.innerHTML = `
    <div class="card-head" style="margin-bottom:8px">
      <div><h3>Leakage verdict</h3><p class="muted small">${esc(data.verdict || "")}</p></div>
      ${claim}
    </div>
    <div class="callout callout-info" style="margin-top:10px">
      ${icon("info")}
      <div>
        <p><b>Placement rule check:</b> accuracy ${fmt.pct(rule.accuracy, 2)} with ${fmt.int(rule.errors)} error(s) across ${fmt.int(rule.rows)} rows.</p>
        <p>${esc(rule.interpretation || "")}</p>
      </div>
    </div>
    <h4 style="margin-top:14px">Checks that need attention</h4>
    ${
      checks.length
        ? `<ul class="muted-list">${checks.map((check) => `<li><b>${esc(check.task)}</b>: ${esc(check.check)} — ${esc(String(check.value))} (${esc(check.severity)})</li>`).join("")}</ul>`
        : `<p class="muted small" style="margin-top:6px">No critical or review checks failed.</p>`
    }
    <p class="muted small" style="margin-top:12px">${esc((data.method_notes || []).join(" "))}</p>
  `;
}

function renderInvestigation(result) {
  const container = $("#investigation-body");
  if (!result.ok) {
    container.innerHTML = errorMarkup(result.error.message, true);
    return;
  }
  const data = result.data;
  const rule = data.deterministic_label_rule || {};
  const unconstrained = data.unconstrained_random_forest || {};
  const constrained = data.constrained_random_forest || {};
  container.innerHTML = `
    <h3>Why the placement score is high</h3>
    <p class="muted small" style="margin-top:4px">${esc(rule.description || "")}</p>
    <ul class="muted-list">
      <li>Rule accuracy across ${fmt.int(rule.rows)} rows: ${fmt.pct(rule.accuracy, 2)} (${fmt.int(rule.errors)} errors).</li>
      <li>Majority-class baseline: ${fmt.pct(data.majority_class_baseline_accuracy, 2)}.</li>
      <li>Unconstrained Random Forest: accuracy ${fmt.pct(unconstrained.Accuracy, 2)}, F1 ${fmt.pct(unconstrained.F1, 2)}. Diagnostic only.</li>
      <li>Deployed depth-limited Random Forest: accuracy ${fmt.pct(constrained.Accuracy, 2)}, F1 ${fmt.pct(constrained.F1, 2)}.</li>
    </ul>
    <p class="muted small" style="margin-top:10px">${esc(data.conclusion || "")}</p>
  `;
}

function wireImageFallback(image, note, message) {
  if (!image || image.dataset.fallbackWired === "true") return;
  image.dataset.fallbackWired = "true";
  image.addEventListener("error", () => {
    image.hidden = true;
    if (note) note.textContent = message;
  });
}

function errorMarkup(message, retryable = false) {
  return `<div class="empty-state empty-state-error">
    ${icon("warning")}
    <h4>Data unavailable</h4>
    <p>${esc(message)}</p>
    ${retryable ? `<button class="button small" type="button" data-retry>${icon("refresh", "icon icon-sm")}Retry</button>` : ""}
  </div>`;
}

/* ---------------------------------------------------------------------------
 * Initialisation
 * ------------------------------------------------------------------------- */

function bindNavigation() {
  $$(".nav-item").forEach((button) => {
    button.addEventListener("click", () => showView(button.dataset.view));
  });
  $$("[data-go]").forEach((button) => {
    button.addEventListener("click", () => showView(button.dataset.go));
  });
  $$("[data-nav]").forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      showView(link.dataset.nav);
    });
  });
  $("#print-button").addEventListener("click", () => window.print());
  window.addEventListener("hashchange", () => showView(viewFromHash(), { updateHash: false }));
}

async function init() {
  bindNavigation();
  const initialView = viewFromHash();
  showView(initialView, { updateHash: false, scroll: false, focus: false });
  loadHealth();
}

init();
