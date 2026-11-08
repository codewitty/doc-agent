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
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function showSignIn() {
  $("sign-in").hidden = false;
  $("workspace").hidden = true;
  $("sign-out").hidden = true;
  $("who").textContent = "";
}

async function showWorkspace() {
  const [me, health] = await Promise.all([api("/api/me"), api("/api/health")]);
  $("who").textContent = `Signed in as ${me.user_id}`;
  $("model-info").textContent = `${health.model} · ${health.embedder}`;
  $("sign-in").hidden = true;
  $("workspace").hidden = false;
  $("sign-out").hidden = false;
  await loadDocuments();
}

async function loadDocuments() {
  const docs = await api("/api/documents");
  const list = $("documents");
  list.replaceChildren();
  if (!docs.length) list.append(el("li", "No documents yet.", "muted"));
  for (const d of docs) {
    const li = el("li");
    li.dataset.documentId = d.id;
    li.append(el("span", d.title, "title"));
    const chunks = `${d.chunk_count} chunk${d.chunk_count === 1 ? "" : "s"}`;
    li.append(el("span", `${chunks} · ${d.created_at.slice(0, 10)}`, "muted"));
    if (d.stale) li.append(el("span", "stale", "badge warn"));
    list.append(li);
  }
}

let currentRun = null;

function renderRun(run) {
  currentRun = run;
  $("run").hidden = false;
  const status = $("run-status");
  status.textContent = run.status.replace("_", " ");
  status.className = `badge ${run.status}`;
  status.dataset.status = run.status;

  $("answer").textContent = run.answer ?? "";
  $("answer").hidden = !run.answer;
  $("run-error").textContent = run.error ? `${run.error.message} (${run.error.code})` : "";

  const cites = $("citations");
  cites.replaceChildren(...run.citations.map((c) => el("li", `${c.document_title} · ${c.chunk_id}`)));
  $("citations-wrap").hidden = run.citations.length === 0;

  $("warnings").replaceChildren(...run.warnings.map((w) => el("li", w)));

  const pending = run.pending_action;
  $("approval").hidden = !pending;
  if (pending) {
    $("approval-reason").textContent = pending.reason;
    $("approval-args").textContent = `${pending.tool}(${JSON.stringify(pending.args)})`;
    $("approval-note").value = "";
  }

  $("trace").replaceChildren(
    ...run.trace.map((s) => {
      const tr = el("tr");
      tr.append(el("td", String(s.index)), el("td", s.kind), el("td", s.duration_ms ?? ""));
      tr.append(el("td", JSON.stringify(s.detail)));
      return tr;
    }),
  );
}

function busy(button, on) {
  button.disabled = on;
  button.setAttribute("aria-busy", String(on));
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
  showSignIn();
});

$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const button = e.submitter;
  const form = new FormData();
  form.append("file", $("file").files[0]);
  busy(button, true);
  $("upload-status").textContent = "Uploading…";
  try {
    const doc = await api("/api/documents", { method: "POST", body: form });
    $("upload-status").textContent = `Added ${doc.title} (${doc.chunk_count} chunks).`;
    $("upload-form").reset();
    await loadDocuments();
  } catch (err) {
    $("upload-status").textContent = `Upload failed: ${err.message}`;
  } finally {
    busy(button, false);
  }
});

$("ask-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const button = $("ask-button");
  busy(button, true);
  try {
    const run = await api("/api/runs", { method: "POST", json: { question: $("question").value } });
    renderRun(run);
  } catch (err) {
    renderRun({
      status: "failed",
      answer: null,
      citations: [],
      warnings: [],
      pending_action: null,
      trace: [],
      error: { code: `http_${err.status ?? "network"}`, message: err.message },
    });
  } finally {
    busy(button, false);
  }
});

async function decide(approve) {
  const buttons = [$("approve"), $("reject")];
  buttons.forEach((b) => busy(b, true));
  try {
    const note = $("approval-note").value.trim() || undefined;
    const run = await api(`/api/runs/${currentRun.id}/decision`, {
      method: "POST",
      json: { approve, note },
    });
    renderRun(run);
    await loadDocuments();
  } catch (err) {
    $("run-error").textContent = `Decision failed: ${err.message}`;
  } finally {
    buttons.forEach((b) => busy(b, false));
  }
}

$("approve").addEventListener("click", () => decide(true));
$("reject").addEventListener("click", () => decide(false));

if (getToken()) {
  showWorkspace().catch(showSignIn);
} else {
  showSignIn();
}
