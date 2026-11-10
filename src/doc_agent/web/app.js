"use strict";

const TOKEN_KEY = "doc-agent-token";
const $ = (id) => document.getElementById(id);

function getToken() {
  try {
    return localStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

function setToken(value) {
  try {
    value ? localStorage.setItem(TOKEN_KEY, value) : localStorage.removeItem(TOKEN_KEY);
  } catch {}
}

class ApiError extends Error {
  constructor(status, detail) {
    super(typeof detail === "string" ? detail : JSON.stringify(detail));
    this.status = status;
  }
}

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}), Authorization: `Bearer ${getToken()}` };
  if (options.json !== undefined) {
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(options.json);
  }
  const res = await fetch(path, { ...options, headers });
  const body = await res.json().catch(() => ({}));
  if (res.status === 401) showSignIn();
  if (!res.ok) throw new ApiError(res.status, body.detail ?? res.statusText);
  return body;
}

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = text;
  if (className) node.className = className;
  return node;
}

// ---- Session --------------------------------------------------------------

function showSignIn() {
  $("sign-in").hidden = false;
  $("workspace").hidden = true;
  // Dev tokens only exist outside prod, so only hint at them locally
  $("dev-tokens").hidden = !["localhost", "127.0.0.1"].includes(location.hostname);
}

async function showWorkspace() {
  const [me, health] = await Promise.all([api("/api/me"), api("/api/health")]);
  $("who").textContent = `Signed in as ${me.user_id}`;
  $("avatar").textContent = me.user_id.slice(0, 1).toUpperCase();
  $("model-info").textContent = `${health.model} · ${health.embedder}`;
  const demo = health.model === "heuristic";
  const badge = $("mode-badge");
  badge.textContent = demo ? "Demo model" : health.model;
  badge.classList.toggle("demo", demo);
  badge.title = demo
    ? "Running the deterministic offline model. Set OPENAI_API_KEY for a real LLM."
    : `Model: ${health.model}`;
  $("sign-in").hidden = true;
  $("workspace").hidden = false;
  await loadDocuments();
  $("question").focus();
}

// ---- Documents ------------------------------------------------------------

const DATE = new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", year: "numeric" });

function fileKind(title) {
  const ext = (title.split(".").pop() || "").toLowerCase();
  return ["pdf", "docx", "csv", "txt", "md"].includes(ext) ? ext : "txt";
}

async function loadDocuments() {
  const docs = await api("/api/documents");
  const list = $("documents");
  list.replaceChildren();
  $("doc-count").textContent = docs.length ? String(docs.length) : "";
  if (!docs.length) list.append(el("li", "No documents yet. Upload one below.", "doc-empty"));
  for (const d of docs) {
    const li = el("li", undefined, "doc");
    li.dataset.documentId = d.id;
    const kind = fileKind(d.title);
    li.append(el("span", kind, `file-icon ${kind}`));
    const text = el("div", undefined, "doc-text");
    text.append(el("span", d.title, "doc-title"));
    const chunks = `${d.chunk_count} chunk${d.chunk_count === 1 ? "" : "s"}`;
    text.append(el("span", `${chunks} · ${DATE.format(new Date(d.created_at))}`, "doc-meta"));
    li.append(text);
    if (d.stale) {
      const badge = el("span", "stale", "tag tag-warn");
      badge.title = "Older than 180 days. Answers from it get a warning.";
      li.append(badge);
    }
    list.append(li);
  }
}

async function upload(file) {
  if (!file) return;
  const status = $("upload-status");
  const zone = $("upload-form");
  const form = new FormData();
  form.append("file", file);
  zone.classList.add("busy");
  status.className = "upload-status";
  status.textContent = `Uploading ${file.name}…`;
  try {
    const doc = await api("/api/documents", { method: "POST", body: form });
    status.classList.add("ok");
    status.textContent = `Added ${doc.title} (${doc.chunk_count} chunks)`;
    await loadDocuments();
  } catch (err) {
    status.classList.add("bad");
    status.textContent = `Upload failed: ${err.message}`;
  } finally {
    zone.classList.remove("busy");
    zone.reset();
  }
}

// ---- Runs -----------------------------------------------------------------

const STATUS_LABEL = {
  completed: "Answered",
  awaiting_approval: "Waiting for approval",
  failed: "Failed",
  refused: "Refused",
  running: "Working",
};

let runs = [];
let pendingQuestion = null;

function stepLabel(s) {
  const d = s.detail || {};
  switch (s.kind) {
    case "input_check":
      return ["Input check", d.ok ? "passed" : `blocked: ${d.reason}`];
    case "model":
      return ["Model", d.type === "tool_calls" ? `calls ${d.tool_calls.join(", ")}` : "writes the answer"];
    case "tool": {
      if (!d.ok) return [d.tool, d.error, "bad"];
      const hits = d.hits === null || d.hits === undefined ? "" : ` · ${d.hits} hits`;
      return [d.tool, `ok${hits}`];
    }
    case "approval_requested":
      return ["Approval requested", d.tool, "warn"];
    case "approval_decision":
      return ["Operator decision", d.approve ? "approved" : "rejected", d.approve ? "" : "warn"];
    case "output_check":
      return ["Output check", d.grounded ? "grounded" : "not grounded", d.grounded ? "" : "warn"];
    case "error":
      return [`Error in ${d.source}`, d.message, "bad"];
    default:
      return [s.kind, ""];
  }
}

// Turns "[chunk:id]" markers into numbered source chips
function renderAnswer(target, text, citations) {
  const index = new Map(citations.map((c, i) => [c.chunk_id, i + 1]));
  const parts = text.split(/(\[chunk:[^\]]*\])/g);
  for (const part of parts) {
    const m = part.match(/^\[chunk:([^\]]*)\]$/);
    if (m && index.has(m[1])) {
      const n = index.get(m[1]);
      const chip = el("sup", String(n), "cite");
      chip.title = citations[n - 1].document_title;
      target.append(chip);
    } else if (part) {
      target.append(document.createTextNode(part));
    }
  }
}

function buildTurn(run, isLatest) {
  const node = $("run-template").content.firstElementChild.cloneNode(true);
  const q = node.querySelector(".bubble.q");
  q.textContent = run.question;

  const pill = node.querySelector(".status-pill");
  pill.textContent = STATUS_LABEL[run.status] ?? run.status;
  pill.classList.add(run.status);
  pill.dataset.status = run.status;

  const meta = node.querySelector(".meta");
  const ms = run.trace.reduce((sum, s) => sum + (s.duration_ms || 0), 0);
  meta.textContent = run.trace.length ? `${run.trace.length} steps · ${Math.round(ms)} ms` : "";

  const answer = node.querySelector(".answer");
  if (run.answer) renderAnswer(answer, run.answer, run.citations);
  answer.hidden = !run.answer;

  const error = node.querySelector(".run-error");
  error.textContent = run.error ? `${run.error.message} (${run.error.code})` : "";
  error.hidden = !run.error;

  const warnings = node.querySelector(".warnings");
  warnings.replaceChildren(...run.warnings.map((w) => el("li", w)));
  warnings.hidden = run.warnings.length === 0;

  const cites = node.querySelector(".citations");
  cites.replaceChildren(
    ...run.citations.map((c, i) => {
      const li = el("li", undefined, "source");
      li.append(el("span", String(i + 1), "source-n"));
      li.append(el("span", c.document_title, "source-title"));
      li.append(el("span", c.chunk_id, "source-chunk"));
      return li;
    }),
  );
  node.querySelector(".sources").hidden = run.citations.length === 0;

  const approval = node.querySelector(".approval");
  const pending = run.pending_action;
  if (pending) {
    approval.querySelector(".approval-reason").textContent = pending.reason;
    approval.querySelector(".approval-args").textContent = `${pending.tool}(${JSON.stringify(pending.args)})`;
    approval.querySelector(".approve").addEventListener("click", () => decide(run.id, true, approval));
    approval.querySelector(".reject").addEventListener("click", () => decide(run.id, false, approval));
  } else {
    approval.remove();
  }

  const trace = node.querySelector(".trace");
  trace.replaceChildren(
    ...run.trace.map((s) => {
      const [title, detail, tone] = stepLabel(s);
      const li = el("li", undefined, `step ${tone || ""}`);
      li.append(el("span", undefined, "step-dot"));
      const body = el("div", undefined, "step-body");
      const head = el("div", undefined, "step-head");
      head.append(el("span", title, "step-title"));
      if (s.duration_ms !== null && s.duration_ms !== undefined) {
        head.append(el("span", `${Math.round(s.duration_ms)} ms`, "step-ms"));
      }
      body.append(head);
      if (detail) body.append(el("span", detail, "step-detail"));
      li.append(body);
      li.title = JSON.stringify(s.detail);
      return li;
    }),
  );
  node.querySelector(".trace-count").textContent = `${run.trace.length} steps`;
  node.querySelector(".trace-wrap").hidden = run.trace.length === 0;

  // Test hooks point at the newest turn only
  if (isLatest) {
    node.dataset.testid = "run";
    pill.dataset.testid = "run-status";
    answer.dataset.testid = "answer";
    cites.dataset.testid = "citations";
    warnings.dataset.testid = "warnings";
    trace.dataset.testid = "trace";
    if (pending) approval.dataset.testid = "approval";
  }
  return node;
}

function buildThinking(question) {
  const node = el("article", undefined, "turn");
  const row = el("div", undefined, "bubble-row");
  row.append(el("div", question, "bubble q"));
  const reply = el("div", undefined, "reply");
  const avatar = el("div", undefined, "reply-avatar");
  avatar.append(el("span", undefined, "logo sm"));
  const body = el("div", undefined, "reply-body thinking");
  body.append(el("span", "Searching your documents", "muted small"));
  const dots = el("span", undefined, "dots");
  dots.append(el("i"), el("i"), el("i"));
  body.append(dots);
  reply.append(avatar, body);
  node.append(row, reply);
  return node;
}

let animatedCount = 0;

function renderThread() {
  const thread = $("thread");
  const turns = runs.map((r, i) => buildTurn(r, i === runs.length - 1));
  if (pendingQuestion) turns.push(buildThinking(pendingQuestion));
  // Only turns that just appeared slide in; re-renders stay still
  turns.slice(animatedCount).forEach((t) => t.classList.add("fresh"));
  animatedCount = runs.length;
  $("empty").hidden = turns.length > 0;
  thread.replaceChildren($("empty"), ...turns);
  thread.scrollTo({ top: thread.scrollHeight, behavior: "smooth" });
}

function upsertRun(run) {
  const i = runs.findIndex((r) => r.id === run.id);
  if (i === -1) runs.push(run);
  else runs[i] = run;
  renderThread();
}

async function ask(question) {
  if (!question.trim() || pendingQuestion) return;
  const button = $("ask-button");
  button.disabled = true;
  pendingQuestion = question;
  $("question").value = "";
  autosize();
  renderThread();
  let run;
  try {
    run = await api("/api/runs", { method: "POST", json: { question } });
  } catch (err) {
    run = {
      id: `local-${Date.now()}`,
      question,
      status: "failed",
      answer: null,
      citations: [],
      warnings: [],
      pending_action: null,
      trace: [],
      error: { code: `http_${err.status ?? "network"}`, message: err.message },
    };
  } finally {
    pendingQuestion = null;
    button.disabled = false;
  }
  upsertRun(run);
  $("question").focus();
}

async function decide(runId, approve, card) {
  const buttons = card.querySelectorAll("button");
  buttons.forEach((b) => (b.disabled = true));
  try {
    const note = card.querySelector(".approval-note").value.trim() || undefined;
    const run = await api(`/api/runs/${runId}/decision`, { method: "POST", json: { approve, note } });
    upsertRun(run);
    await loadDocuments();
  } catch (err) {
    buttons.forEach((b) => (b.disabled = false));
    let msg = card.querySelector(".form-error");
    if (!msg) {
      msg = el("p", undefined, "form-error");
      msg.setAttribute("role", "alert");
      card.append(msg);
    }
    msg.textContent = `Decision failed: ${err.message}`;
  }
}

// ---- Wiring ---------------------------------------------------------------

function autosize() {
  const q = $("question");
  q.style.height = "auto";
  q.style.height = `${Math.min(q.scrollHeight, 200)}px`;
}

$("sign-in-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  setToken($("token").value.trim());
  $("sign-in-error").textContent = "";
  try {
    await showWorkspace();
  } catch (err) {
    setToken(null);
    showSignIn();
    $("sign-in-error").textContent = err.status === 401 ? "That token was not accepted." : err.message;
  }
});

$("sign-out").addEventListener("click", () => {
  setToken(null);
  runs = [];
  animatedCount = 0;
  renderThread();
  $("token").value = "";
  showSignIn();
});

$("new-chat").addEventListener("click", () => {
  runs = [];
  animatedCount = 0;
  renderThread();
  $("question").focus();
});

$("ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  ask($("question").value);
});

$("question").addEventListener("input", autosize);
$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    $("ask-form").requestSubmit();
  }
});

for (const b of document.querySelectorAll(".suggestion")) {
  b.addEventListener("click", () => ask(b.dataset.q));
}

$("file").addEventListener("change", () => upload($("file").files[0]));
const zone = $("upload-form");
zone.addEventListener("submit", (e) => e.preventDefault());
for (const type of ["dragenter", "dragover"]) {
  zone.addEventListener(type, (e) => {
    e.preventDefault();
    zone.classList.add("drag");
  });
}
for (const type of ["dragleave", "drop"]) {
  zone.addEventListener(type, () => zone.classList.remove("drag"));
}
zone.addEventListener("drop", (e) => {
  e.preventDefault();
  upload(e.dataTransfer.files[0]);
});

if (getToken()) {
  showWorkspace().catch(showSignIn);
} else {
  showSignIn();
}
