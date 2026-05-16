// Streaming demo client.
// Consumes the SSE stream from /stream and renders phase cards live.
//
// We use plain fetch() + a TextDecoder rather than EventSource because
// EventSource doesn't expose readyState transitions cleanly for one-shot
// streams; the manual parse here matches the server's frame shape
// (`event: <type>\ndata: <json>\n\n`).

const form = document.getElementById("form");
const qInput = document.getElementById("q");
const goBtn = document.getElementById("go");
const phasesEl = document.getElementById("phases");
const answerEl = document.getElementById("answer");

function fmtMs(ms) {
  return ms < 10 ? ms.toFixed(2) + " ms" : Math.round(ms) + " ms";
}

function appendCard(type, label, detail, elapsedMs, opts) {
  const card = document.createElement("div");
  card.className = "card" + (opts && opts.start ? " start" : "");
  const head = document.createElement("div");
  head.className = "card-head";
  const left = document.createElement("span");
  left.className = "phase-name " + type;
  left.textContent = label;
  const right = document.createElement("span");
  right.className = "elapsed";
  right.textContent = fmtMs(elapsedMs);
  head.appendChild(left);
  head.appendChild(right);
  card.appendChild(head);
  if (detail) {
    const d = document.createElement("div");
    d.className = "phase-detail";
    d.textContent = detail;
    card.appendChild(d);
  }
  phasesEl.appendChild(card);
  return card;
}

function appendChunks(card, chunks) {
  for (const c of chunks) {
    const div = document.createElement("div");
    div.className = "chunk";
    const id = document.createElement("span");
    id.className = "id";
    id.textContent = "[" + c.external_id + "] ";
    div.appendChild(id);
    div.appendChild(document.createTextNode(c.text));
    card.appendChild(div);
  }
}

function reset() {
  phasesEl.innerHTML = "";
  answerEl.textContent = "";
}

async function runQuery(q) {
  reset();
  goBtn.disabled = true;
  let resp;
  try {
    resp = await fetch("/stream?q=" + encodeURIComponent(q) + "&k=3");
  } catch (e) {
    appendCard("error", "error", "fetch failed: " + e.message, 0);
    goBtn.disabled = false;
    return;
  }
  if (!resp.ok || !resp.body) {
    appendCard("error", "error", "HTTP " + resp.status, 0);
    goBtn.disabled = false;
    return;
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      handleFrame(frame);
    }
  }
  goBtn.disabled = false;
}

function handleFrame(frame) {
  let type = "";
  let dataLine = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event: ")) type = line.slice(7).trim();
    else if (line.startsWith("data: ")) dataLine += line.slice(6);
  }
  if (!type || !dataLine) return;
  let parsed;
  try {
    parsed = JSON.parse(dataLine);
  } catch (e) {
    appendCard("error", "error", "bad frame JSON", 0);
    return;
  }
  const payload = parsed.payload || {};
  const elapsed = parsed.elapsed_ms || 0;

  switch (type) {
    case "retrieving":
      appendCard("retrieving", "retrieving", "query: " + payload.query + " · k=" + payload.k, elapsed, { start: true });
      return;
    case "retrieved": {
      const card = appendCard("retrieved", "retrieved", "count=" + payload.count + " · phase " + fmtMs(payload.phase_ms), elapsed);
      appendChunks(card, payload.chunks);
      return;
    }
    case "reranking":
      appendCard("reranking", "reranking", "candidates=" + payload.candidates, elapsed, { start: true });
      return;
    case "reranked": {
      const card = appendCard("reranked", "reranked", "count=" + payload.count + " · phase " + fmtMs(payload.phase_ms), elapsed);
      appendChunks(card, payload.chunks);
      return;
    }
    case "generating":
      appendCard("generating", "generating", "context chunks=" + payload.context_chunks, elapsed, { start: true });
      return;
    case "token":
      answerEl.textContent += payload.text;
      return;
    case "generated":
      appendCard("generated", "generated", "phase " + fmtMs(payload.phase_ms), elapsed);
      return;
    case "done":
      appendCard("done", "done", "total " + fmtMs(payload.total_ms), elapsed);
      return;
    case "error":
      appendCard("error", "error", payload.exception + ": " + payload.message, elapsed);
      return;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = qInput.value.trim();
  if (q) runQuery(q);
});
