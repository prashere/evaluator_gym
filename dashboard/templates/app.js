let DATA = null;
const TIER_FILTER = { retrieval: "1", reconciliation: "2", exception: "3" };

async function loadData() {
  const res = await fetch("data.json");
  if (!res.ok) throw new Error("Failed to load data.json");
  DATA = await res.json();
  return DATA;
}

function fmt(v, d = 3) {
  if (v === null || v === undefined) return "—";
  return Number(v).toFixed(d);
}

function validityBadge(runValidity) {
  const status = (runValidity || {}).status;
  if (!status || status === "valid") return "";
  const label = status === "invalid" ? "INVALID" : "INCOMPLETE";
  const reason = runValidity.reason ? ` title="${runValidity.reason}"` : "";
  return `<span class="validity-badge validity-${status}"${reason}>${label}</span>`;
}

function fmtPct(rate) {
  if (rate === null || rate === undefined) return "—";
  return (Number(rate) * 100).toFixed(1) + "%";
}

function escAttr(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;");
}

function infoTip(key) {
  const m = DATA.glossary.metrics[key];
  if (!m) return "";
  const text = m.simple || m.description || "";
  if (!text) return "";
  return `<span class="info-tip" tabindex="0" data-tip="${escAttr(text)}" aria-label="${escAttr(text)}">ⓘ</span>`;
}

function baselines() {
  return DATA.sanity_checks || [];
}

function hasBaselines() {
  return baselines().length > 0;
}

function baselineLabel(entry) {
  if (!entry.is_baseline) return "";
  if (entry.is_oracle) {
    return `<span class="baseline-badge baseline-oracle" title="Uses recorded ground truth — not a deployable agent">Upper bound</span>`;
  }
  return `<span class="baseline-badge" title="${escAttr(entry.purpose || "Calibration policy — not deployable")}">Calibration · not deployable</span>`;
}

function leaderboardEntries() {
  const models = DATA.models.map((m) => ({ ...m, is_baseline: false }));
  const rows = baselines().map((b) => ({
    slug: b.slug,
    name: b.name,
    is_baseline: true,
    is_oracle: b.is_oracle,
    purpose: b.purpose,
    run_validity: null,
    overall: b.overall || {},
    by_task_type: b.by_task_type || {},
    components_avg: b.components_avg || {},
    cost: {},
  }));
  return [...models, ...rows];
}

function shortColumnName(name, isBaseline) {
  if (isBaseline) return name.replace("baseline-", "").slice(0, 12);
  return name.replace("groq/", "");
}

function scoreCell(entry) {
  const o = entry.overall || {};
  if (entry.is_baseline) {
    return o.n ? fmt(o.mean) + " ± " + fmt(o.std) : "—";
  }
  const comparable = (entry.run_validity || {}).status !== "invalid";
  return comparable && o.n ? fmt(o.mean) + " ± " + fmt(o.std) : "—";
}

function tierCell(tierStats) {
  const t = tierStats || {};
  return t.n ? fmt(t.mean) + " (" + t.n + ")" : "—";
}

function collectComponentKeys(entries) {
  const keys = new Set();
  entries.forEach((e) => Object.keys(e.components_avg || {}).forEach((k) => keys.add(k)));
  return [...keys];
}

function routeFromHash() {
  const h = location.hash.slice(1) || "overview";
  const parts = h.split("/");
  return { view: parts[0], param: parts.slice(1).join("/") };
}

function navigate(view, param) {
  location.hash = param ? `${view}/${param}` : view;
}

function setActiveNav(view) {
  document.querySelectorAll(".main-nav a").forEach((a) => {
    a.classList.toggle("active", a.dataset.route === view || (view.startsWith("task") && a.dataset.route === "tasks") || (view === "inspect" && a.dataset.route === "tasks"));
  });
}

function renderShell() {
  const m = DATA.meta;
  document.getElementById("site-title").textContent = m.title;
  document.getElementById("site-subtitle").textContent = m.subtitle;
  const banner = document.getElementById("subset-banner");
  if (m.subset_warning && m.subset_message) {
    banner.textContent = m.subset_message;
    banner.classList.remove("hidden");
  } else {
    banner.classList.add("hidden");
  }
}

function metaGrid(extra) {
  const m = DATA.meta;
  return `<dl class="meta-grid">
    <div><dt>Run ID</dt><dd>${m.run_id}</dd></div>
    <div><dt>Tasks evaluated</dt><dd>${m.tasks_evaluated} / ${m.benchmark_tasks}</dd></div>
    <div><dt>Rollouts / task</dt><dd>${m.rollouts_per_task}</dd></div>
    <div><dt>Ruleset</dt><dd>${m.ruleset_version}</dd></div>
    <div><dt>Rubric</dt><dd>${m.rubric_version}</dd></div>
    <div><dt>Schema</dt><dd>${m.schema_version}</dd></div>
    ${extra || ""}
  </dl>`;
}

function renderOverview() {
  const m = DATA.meta;
  let cards = DATA.models.map((mod) => {
    const o = mod.overall;
    const comparable = (mod.run_validity || {}).status !== "invalid";
    const score = comparable && o.n > 0 ? fmt(o.mean) + " ± " + fmt(o.std) : "Not comparable";
    return `<article class="card">
      <h3>${mod.name} ${validityBadge(mod.run_validity)}</h3>
      <div class="card-metric card-metric-primary">
        <span class="metric-label">Average score ${infoTip("conditional_reward")}</span>
        <div class="score">${score}</div>
      </div>
      <div class="card-metric">
        <span class="metric-label">Parse success ${infoTip("parse_success_rate")}</span>
        <span>${fmtPct(o.parse_success_rate ?? o.scored_rate)}</span>
      </div>
      <div class="card-metric">
        <span class="metric-label">Overall usable ${infoTip("overall_usable")}</span>
        <span>${o.overall_usable_mean != null ? fmt(o.overall_usable_mean) : "—"}</span>
      </div>
      <div class="card-metric">
        <span class="metric-label">Reliability ${infoTip("reliability")}</span>
        <span>${fmt(o.reliability_pct, 1)}%</span>
      </div>
      <div class="card-meta">${o.n} scored rollouts · ${mod.tasks_evaluated} tasks · Est. cost $${fmt(mod.cost.estimated_cost_usd, 4)}</div>
      <div class="card-actions"><a class="btn btn-primary" href="#compare" data-slug="${mod.slug}">Explore performance</a></div>
    </article>`;
  }).join("");

  return `<section>
    <p class="lead">${m.purpose}</p>
    ${metaGrid()}
    <h2>Deployable models</h2>
    <div class="cards">${cards}</div>
    <p class="lead" style="margin-top:1rem;font-size:0.85rem">No global winner is declared when benchmark coverage is incomplete. Use Compare Agents and Task Explorer to investigate.</p>
  </section>`;
}

function renderCompare() {
  const { param } = routeFromHash();
  const selected = param ? param.split(",") : DATA.models.map((m) => m.slug);
  const opts = DATA.models.map((m) => `<option value="${m.slug}" ${selected.includes(m.slug) ? "selected" : ""}>${m.name}</option>`).join("");
  const board = leaderboardEntries();

  let rows = board.map((entry) => {
    const o = entry.overall || {};
    const rowCls = entry.is_baseline ? "row-baseline" : "";
    const parsePct = entry.is_baseline ? "—" : fmtPct(o.parse_success_rate ?? o.scored_rate);
    const usable = entry.is_baseline ? "—" : (o.overall_usable_mean != null ? fmt(o.overall_usable_mean) : "—");
    const fmtFail = entry.is_baseline ? "—" : (o.format_fail_pct ?? o.parse_fail_pct ?? "—") + "%";
    const provFail = entry.is_baseline ? "—" : (o.provider_fail_pct ?? "—") + "%";
    const infraFail = entry.is_baseline ? "—" : (o.infra_fail_pct ?? "—") + "%";
    return `<tr class="${rowCls}">
      <td>${entry.name} ${baselineLabel(entry)} ${validityBadge(entry.run_validity)}</td>
      <td class="num">${scoreCell(entry)}</td>
      <td class="num">${parsePct}</td>
      <td class="num">${o.n ?? "—"}</td>
      <td class="num">${usable}</td>
      <td class="num">${fmtFail}</td>
      <td class="num">${provFail}</td>
      <td class="num">${infraFail}</td>
    </tr>`;
  }).join("");

  const modelEntries = board.filter((e) => !e.is_baseline && selected.includes(e.slug));

  let typeRows = board.map((entry) => {
    const t = entry.by_task_type || {};
    return `<tr class="${entry.is_baseline ? "row-baseline" : ""}">
      <td>${entry.name} ${baselineLabel(entry)}</td>
      <td class="num">${tierCell(t.retrieval)}</td>
      <td class="num">${tierCell(t.reconciliation)}</td>
      <td class="num">${tierCell(t.exception)}</td>
    </tr>`;
  }).join("");

  let costRows = modelEntries.map((mod) => `<tr>
    <td>${mod.name}</td>
    <td class="num">$${fmt(mod.cost.estimated_cost_usd, 4)}</td>
    <td class="num">${fmt(mod.cost.tokens_input, 0)} / ${fmt(mod.cost.tokens_output, 0)}</td>
    <td class="num">${mod.cost.mean_latency_ms != null ? Math.round(mod.cost.mean_latency_ms) : "—"}</td>
  </tr>`).join("");

  const compKeys = collectComponentKeys(board);
  const compHeader = compKeys.map((k) => {
    const g = DATA.glossary.metrics[k];
    return `<th>${g?.label || k} ${infoTip(k)}</th>`;
  }).join("");
  const compRows = board.map((entry) =>
    `<tr class="${entry.is_baseline ? "row-baseline" : ""}"><td>${entry.name} ${baselineLabel(entry)}</td>${compKeys.map((k) => `<td class="num">${fmt(entry.components_avg[k])}</td>`).join("")}</tr>`
  ).join("");

  const baselineNote = hasBaselines()
    ? `<p class="lead" style="font-size:0.85rem">Calibration baselines test the rubric with fixed policies. They are not deployable agents.</p>`
    : "";

  return `<section>
    <h1>Compare Agents</h1>
    <div class="filters">
      <label>Filter deployable models <select id="model-filter" multiple size="3">${opts}</select></label>
      <button class="btn" id="apply-filter">Apply filter</button>
    </div>
    <p class="lead" style="font-size:0.85rem">Filter applies to cost table only. Leaderboard always includes calibration baselines when present.</p>

    <h2>Leaderboard</h2>
    ${baselineNote}
    <table><thead><tr>
      <th>Agent</th>
      <th>Mean score ± SD ${infoTip("conditional_reward")}</th>
      <th>Parse success ${infoTip("parse_success_rate")}</th>
      <th>Scored rollouts ${infoTip("scored_rollouts")}</th>
      <th>Overall usable ${infoTip("overall_usable")}</th>
      <th>Format fail % ${infoTip("parse_failure")}</th>
      <th>Provider fail % ${infoTip("provider_failure")}</th>
      <th>Scoring/infra fail % ${infoTip("infrastructure_failure")}</th>
    </tr></thead><tbody>${rows}</tbody></table>

    <h2>Performance by work type</h2>
    <p class="lead" style="font-size:0.85rem">T1 = Retrieval · T2 = Reconciliation · T3 = Exception / Trap</p>
    <table><thead><tr><th>Agent</th><th>Retrieval</th><th>Reconciliation</th><th>Exception / Trap</th></tr></thead><tbody>${typeRows}</tbody></table>

    <h2>Reward breakdown by rubric component</h2>
    <p class="lead" style="font-size:0.85rem">Average score per rubric component — not one aggregate number.</p>
    <table><thead><tr><th>Agent</th>${compHeader}</tr></thead><tbody>${compRows || `<tr><td colspan="${compKeys.length + 1}">No component data</td></tr>`}</tbody></table>

    <h2>Cost and speed (deployable models)</h2>
    <table><thead><tr><th>Model</th><th>Estimated cost ${infoTip("estimated_cost")}</th><th>Tokens in/out</th><th>Mean latency (ms)</th></tr></thead><tbody>${costRows || "<tr><td colspan=4>No models selected</td></tr>"}</tbody></table>
    <p class="lead" style="font-size:0.85rem">${DATA.models[0]?.cost?.pricing_note || ""}</p>
  </section>`;
}

function renderTasks(filter) {
  filter = filter || {};
  let tasks = DATA.tasks;
  if (filter.tier) tasks = tasks.filter((t) => String(t.tier) === filter.tier);
  if (filter.evaluated === "yes") tasks = tasks.filter((t) => t.evaluated_in_run);
  if (filter.q) {
    const q = filter.q.toLowerCase();
    tasks = tasks.filter((t) => t.task_id.includes(q) || (t.description || "").toLowerCase().includes(q));
  }

  const rows = tasks.map((t) => {
    const chips = Object.entries(t.models)
      .filter(([, v]) => v.is_baseline === false)
      .map(([slug, v]) => {
        if (!v.evaluated) return `<span class="chip">—</span>`;
        const st = v.rollouts[0]?.status || "—";
        const cls = st.includes("Parse") ? "chip-parse" : v.cell?.mean != null && v.cell.mean < 0.5 ? "chip-bad" : "";
        return `<span class="chip ${cls}" title="${slug}">${v.display_name.split("/").pop()}: ${st}</span>`;
      }).join("");
    return `<article class="task-row">
      <header>
        <strong>${t.task_id}</strong>
        <span class="task-type">${t.task_type} (${t.task_type_tier})</span>
        ${t.evaluated_in_run ? "" : '<span style="color:var(--muted)">Not evaluated in this run</span>'}
      </header>
      <p class="task-desc">${t.description || "No description in coverage metadata."}</p>
      <div class="model-chips">${chips}</div>
      <div style="margin-top:0.5rem"><a class="btn" href="#task/${t.task_id}">Inspect task</a></div>
    </article>`;
  }).join("");

  return `<section>
    <h1>Task Explorer</h1>
    <div class="filters">
      <select id="f-tier"><option value="">All task types</option>
        <option value="1">Retrieval</option><option value="2">Reconciliation</option><option value="3">Exception / Trap</option>
      </select>
      <select id="f-eval"><option value="">All tasks</option><option value="yes">Evaluated only</option></select>
      <input id="f-q" placeholder="Search task ID or description" />
      <button class="btn" id="f-apply">Filter</button>
    </div>
    ${rows || "<p>No tasks match filter.</p>"}
  </section>`;
}

function renderTaskDetail(taskId) {
  const task = DATA.tasks.find((t) => t.task_id === taskId);
  if (!task) return "<p>Task not found.</p>";

  const modelRows = Object.entries(task.models)
    .filter(([, v]) => !v.is_baseline && v.evaluated)
    .map(([slug, v]) => `<tr>
      <td>${v.display_name}</td>
      <td class="num">${v.cell.mean != null ? fmt(v.cell.mean) : "—"}</td>
      <td>${v.cell.bucket}</td>
      <td class="num">${v.rollouts.length}</td>
    </tr>`).join("");

  const rolloutBlocks = Object.entries(task.models)
    .filter(([, v]) => v.evaluated)
    .map(([slug, v]) => {
      const rolls = v.rollouts.map((r) =>
        `<li>Rollout ${r.rollout_index + 1}: <strong>${r.status}</strong>
          <a class="btn" href="#inspect/${r.inspector_id}">Inspect evaluation</a></li>`
      ).join("");
      return `<div><strong>${v.display_name}</strong><ul>${rolls}</ul></div>`;
    }).join("");

  return `<section>
    <div class="breadcrumb"><a href="#tasks">Task Explorer</a> → ${taskId}</div>
    <h1>${taskId}</h1>
    <p class="task-type">${task.task_type} (${task.task_type_tier})</p>
    <p>${task.description || ""}</p>
    <p><strong>Rules under test:</strong> ${(task.rules_under_test || []).join(", ") || "—"}</p>

    <h2>Model comparison on this task</h2>
    <table><thead><tr><th>Model</th><th>Mean score</th><th>Status</th><th>Rollouts</th></tr></thead><tbody>${modelRows || "<tr><td colspan=4>Not evaluated</td></tr>"}</tbody></table>

    <h2>Rollouts</h2>
    ${rolloutBlocks || "<p>No rollouts in this run.</p>"}
  </section>`;
}

function renderInspector(inspectorId) {
  const rec = DATA.inspectors[inspectorId];
  if (!rec) return "<p>Inspection record not found.</p>";
  const c = rec.chain;
  const checks = (c.recorded_audit.rubric_checks || []).map((ch) => {
    const icon = ch.passed === true ? "✓" : ch.passed === false ? "✗" : "⚠";
    const cls = ch.passed === true ? "pass" : ch.passed === false ? "fail" : "warn";
    let extra = "";
    if (ch.missing?.length) extra += `<div>Missing evidence: ${JSON.stringify(ch.missing)}</div>`;
    if (ch.extra?.length) extra += `<div>Extra evidence: ${JSON.stringify(ch.extra)}</div>`;
    return `<div class="check ${cls}"><span class="icon">${icon}</span><div><strong>${ch.label}</strong> (${ch.technical}) — ${fmt(ch.score)}<br/><span style="color:var(--muted)">${ch.detail || ""}</span>${extra}</div></div>`;
  }).join("");

  const gate = c.score_summary.gate_applied ? `<div class="gate-box">
    <strong>Gate applied</strong>: ${c.score_summary.gate_reason || "yes"}<br/>
    ${c.score_summary.gate_note || ""}<br/>
    Base: ${fmt(c.score_summary.base_reward)} → Final: ${fmt(c.score_summary.final_reward)}
  </div>` : "";

  return `<section class="inspector">
    <div class="breadcrumb"><a href="#task/${rec.task_id}">${rec.task_id}</a> → ${rec.display_name} rollout ${rec.rollout_index + 1}</div>
    <h1>Evaluation Inspector</h1>

    <div class="chain-step"><h4>1. Task</h4>
      <p>${c.task.task_type} · ${c.task.description || ""}</p>
      ${c.task.user_excerpt ? `<pre>${c.task.user_excerpt}</pre>` : ""}
    </div>

    <div class="chain-step"><h4>2. Agent response</h4>
      ${rec.is_baseline ? `<p>Calibration policy: ${rec.baseline_policy || "—"}</p>` : ""}
      ${c.agent_response.has_tool_calls ? `<p>${c.agent_response.tool_trace_note}</p>` : ""}
      ${c.agent_response.assistant ? `<pre>${c.agent_response.assistant}</pre>` : "<p>No assistant content recorded.</p>"}
    </div>

    <div class="chain-step"><h4>3. Parsed output</h4>
      ${c.parsed_output.ok === false ? `<p><strong>Parse failure:</strong> ${c.parsed_output.error_class}: ${c.parsed_output.error_message || ""}</p>` : ""}
      ${c.parsed_output.data ? `<pre>${JSON.stringify(c.parsed_output.data, null, 2)}</pre>` : "<p>No structured parse data.</p>"}
    </div>

    <div class="chain-step"><h4>4. Recorded evaluation audit</h4>
      <p style="font-size:0.85rem;color:var(--muted)">${c.recorded_audit.note}</p>
      ${checks || "<p>No rubric checks recorded.</p>"}
    </div>

    <div class="chain-step"><h4>5. Final score</h4>
      <p>Failure class: ${c.score_summary.failure_class || "none"}</p>
      ${gate}
      <p><strong>Final score:</strong> ${fmt(c.score_summary.final_reward)}</p>
    </div>
  </section>`;
}

function cellClass(cell) {
  if (cell.bucket === "not_run") return "cell-not_run";
  if (cell.bucket === "parse" || cell.bucket === "format_failure") return "cell-parse";
  if (cell.bucket === "provider_failure") return "cell-provider";
  if (cell.bucket === "infra" || cell.bucket === "scoring_failure" || cell.bucket === "infrastructure_failure") return "cell-infra";
  const m = cell.mean;
  if (m == null) return "cell-infra";
  if (m >= 0.9) return "cell-scored-high";
  if (m >= 0.5) return "cell-scored-mid";
  return "cell-scored-low";
}

function cellLabel(cell) {
  if (cell.bucket === "not_run") return "—";
  if (cell.bucket === "parse" || cell.bucket === "format_failure") return "format";
  if (cell.bucket === "provider_failure") return "provider";
  if (cell.bucket === "infra" || cell.bucket === "scoring_failure" || cell.bucket === "infrastructure_failure") return "infra";
  return cell.n > 1 ? fmt(cell.mean, 2) + "*" : fmt(cell.mean, 2);
}

function heatmapColumns(mode) {
  const models = DATA.heatmap.model_columns || [];
  const bl = DATA.heatmap.baseline_columns || [];
  if (mode === "baselines") return bl.map((c) => ({ ...c, is_baseline: true }));
  if (mode === "models") return models.map((c) => ({ ...c, is_baseline: false }));
  return [
    ...models.map((c) => ({ ...c, is_baseline: false })),
    ...bl.map((c) => ({ ...c, is_baseline: true })),
  ];
}

function renderPatterns() {
  const { param } = routeFromHash();
  const colMode = ["models", "baselines", "all"].includes(param) ? param : (hasBaselines() ? "all" : "models");
  const cols = heatmapColumns(colMode);
  const colOpts = [
    ["models", "Deployable models"],
    ["baselines", "Calibration baselines"],
    ["all", "All"],
  ].map(([v, label]) => `<option value="${v}" ${colMode === v ? "selected" : ""}>${label}</option>`).join("");

  let html = `<section><h1>Failure Patterns</h1>
    <p class="lead">Task × agent heatmap. White = not evaluated · Hatch = parse · Gray = infra · Color = scored. * = mean over ${DATA.meta.rollouts_per_task} rollouts. Click a cell to inspect.</p>
    <div class="filters">
      <label>Columns <select id="hm-columns">${colOpts}</select></label>
      <select id="p-tier"><option value="">All types</option><option value="1">Retrieval</option><option value="2">Reconciliation</option><option value="3">Exception</option></select>
    </div>
    <div class="heatmap-wrap"><table class="heatmap" id="hm-table"><thead><tr><th>Task</th>`;
  cols.forEach((c) => {
    html += `<th class="${c.is_baseline ? "col-baseline" : ""}" title="${c.name}">${shortColumnName(c.name, c.is_baseline)}</th>`;
  });
  html += "</tr></thead><tbody>";

  DATA.heatmap.tasks.forEach((t) => {
    html += `<tr data-tier="${t.tier}"><td class="task-label" title="${t.description || ""}">${t.task_id}<br/><small>${t.task_type || ""}</small></td>`;
    cols.forEach((col) => {
      const key = `${t.task_id}|${col.slug}`;
      const cell = DATA.heatmap.cells[key] || { bucket: "not_run" };
      const task = DATA.tasks.find((x) => x.task_id === t.task_id);
      const perf = task?.models[col.slug];
      const iid = perf?.rollouts?.[0]?.inspector_id;
      html += `<td class="cell ${cellClass(cell)}" data-iid="${iid || ""}" title="${t.description || ""} (n=${cell.n || 0})">${cellLabel(cell)}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody></table></div></section>";
  return html;
}

function renderMethodology() {
  const m = DATA.meta;
  const keyMetrics = [
    "conditional_reward",
    "score_variability",
    "parse_success_rate",
    "overall_usable",
    "reliability",
    "parse_failure",
    "provider_failure",
  ];
  const metrics = keyMetrics.map((k) => {
    const v = DATA.glossary.metrics[k];
    if (!v) return "";
    return `<li><strong>${v.label}</strong> — ${v.simple || v.description}</li>`;
  }).join("");
  const types = Object.values(DATA.glossary.task_types).map(
    (t) => `<li><strong>${t.label}</strong> (${t.tier}): ${t.description}</li>`
  ).join("");

  return `<section><h1>Methodology</h1>
    <p class="lead">This page reads saved evaluation results from <code>results/</code>. It does not re-run models or re-score answers.</p>
    <h2>Run</h2>
    ${metaGrid()}
    <h2>Task types</h2>
    <ul class="method-list">${types}</ul>
    <h2>Headline metrics</h2>
    <ul class="method-list">${metrics}</ul>
  </section>`;
}

function bindEvents(view) {
  if (view === "compare") {
    document.getElementById("apply-filter")?.addEventListener("click", () => {
      const sel = [...document.getElementById("model-filter").selectedOptions].map((o) => o.value);
      navigate("compare", sel.join(","));
    });
  }
  if (view === "tasks") {
    document.getElementById("f-apply")?.addEventListener("click", () => {
      const tier = document.getElementById("f-tier").value;
      const ev = document.getElementById("f-eval").value;
      const q = document.getElementById("f-q").value;
      navigate("tasks", [tier && "tier=" + tier, ev && "eval=" + ev, q && "q=" + encodeURIComponent(q)].filter(Boolean).join("&"));
    });
  }
  if (view === "patterns") {
    document.getElementById("hm-columns")?.addEventListener("change", (e) => {
      navigate("patterns", e.target.value);
    });
    document.getElementById("p-tier")?.addEventListener("change", (e) => {
      const v = e.target.value;
      document.querySelectorAll("#hm-table tbody tr").forEach((tr) => {
        tr.style.display = !v || tr.dataset.tier === v ? "" : "none";
      });
    });
    document.querySelectorAll("#hm-table .cell[data-iid]").forEach((el) => {
      el.addEventListener("click", () => {
        const iid = el.dataset.iid;
        if (iid) navigate("inspect", iid);
      });
    });
  }
}

function parseTaskFilter(param) {
  const f = {};
  if (!param) return f;
  param.split("&").forEach((p) => {
    const [k, v] = p.split("=");
    if (k === "tier") f.tier = v;
    if (k === "eval") f.evaluated = v;
    if (k === "q") f.q = decodeURIComponent(v);
  });
  return f;
}

function render() {
  const { view, param } = routeFromHash();
  setActiveNav(view);
  const app = document.getElementById("app");
  let html = "";
  switch (view) {
    case "overview": html = renderOverview(); break;
    case "compare": html = renderCompare(); break;
    case "tasks": html = renderTasks(parseTaskFilter(param)); break;
    case "task": html = renderTaskDetail(param); break;
    case "inspect": html = renderInspector(param); break;
    case "patterns": html = renderPatterns(); break;
    case "sanity": html = renderOverview(); break;
    case "methodology": html = renderMethodology(); break;
    default: html = renderOverview();
  }
  app.innerHTML = html;
  bindEvents(view);
  window.scrollTo(0, 0);
}

async function main() {
  await loadData();
  renderShell();
  window.addEventListener("hashchange", render);
  render();
}

main().catch((e) => {
  document.body.innerHTML = `<p style="color:red;padding:2rem">Dashboard failed: ${e.message}</p>`;
});
