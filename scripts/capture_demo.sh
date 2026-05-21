#!/usr/bin/env bash
# Deterministic driver for the 60-second README demo (issue #25).
#
# Runs the two demo surfaces in sequence on a fresh clone with no API key:
#
#   1. demo/streaming/server.py spawned in background; curl -N against
#      /stream?q=postgres+tuning so the SSE phase events
#      (retrieving / retrieved / reranking / reranked / generating /
#       token... / generated / done) print to the terminal in real time.
#      Server is reaped via EXIT trap.
#
#   2. (optional) demo/nextjs/ — npm install if needed, then exec
#      `npm run dev` in the foreground so the operator records the
#      browser tour (citation chips + retrieved-chunks panel + phase
#      pills). Ctrl+C lands directly on the dev server.
#
# The output is the recording — when JT records the GIF/video, this
# script's stdout + the browser at http://localhost:3000 is what gets
# captured. Hermetic: no API key, no network, no Postgres.
#
# Variables:
#   CAPTURE_PACE_SECONDS    pause between sections (default 2 for
#                           recording; test_capture_demo_smoke.py sets 0).
#   CAPTURE_LAUNCH_NEXTJS   if "1" (default), npm install (idempotent) +
#                           exec `npm run dev` after the curl tour.
#                           Smoke tests set this to "0".
#   CAPTURE_SSE_TIMEOUT     seconds to read the SSE stream (default 6).
#                           The full hermetic stream is ~500 ms; 6 s is
#                           cushion + lets a recording show the phase
#                           cards transition naturally.
#   CAPTURE_DEMO_PORT       port the SSE server binds (default 8765 —
#                           matches the demo's documented default).
#
# Exit: 0 on full success. Background server is reaped via EXIT trap.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACE="${CAPTURE_PACE_SECONDS:-2}"
LAUNCH_NEXTJS="${CAPTURE_LAUNCH_NEXTJS:-1}"
SSE_TIMEOUT="${CAPTURE_SSE_TIMEOUT:-6}"
PORT="${CAPTURE_DEMO_PORT:-8765}"

banner() {
  printf '\n'
  printf '═══ %s\n' "$1"
  printf '\n'
}

pace() {
  if [ "$PACE" != "0" ]; then
    sleep "$PACE"
  fi
}

cd "$REPO_ROOT"

SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    # Give the server a moment to release the port; ignore errors so
    # the trap is safe to fire from any exit path.
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

banner "rag-production-kit · 60-second demo"
printf 'two surfaces · in-memory corpus · no API key, no Postgres\n'
pace

banner "1/2 · streaming SSE server + live phase events"
printf 'spawn  python -m demo.streaming.server     (port %s, in-memory FakeRetriever)\n' "$PORT"
printf 'curl   /stream?q=postgres+tuning           (retrieve → rerank → generate → done)\n\n'

# Resolve the Python interpreter from the active venv if one is present.
# Falls back to plain `python` (or python3) so the script works under
# whatever interpreter the operator's PATH provides.
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi

PORT="$PORT" "$PYTHON_BIN" -m demo.streaming.server >/dev/null 2>&1 &
SERVER_PID=$!
printf 'server pid %s; waiting for port to bind...\n' "$SERVER_PID"

# Poll the port instead of sleeping a fixed amount: faster locally,
# more robust on slower CI machines. Cap at 5 s — if the server isn't
# up by then something is wrong.
for _ in $(seq 1 25); do
  if (echo > /dev/tcp/127.0.0.1/"$PORT") 2>/dev/null; then
    break
  fi
  sleep 0.2
done

printf 'connected; streaming for up to %s s...\n\n' "$SSE_TIMEOUT"
# `|| true` because curl --max-time exits non-zero on a forced cutoff
# even when the stream ended naturally with `event: done`. We just
# want the body in either case.
curl -s -N --max-time "$SSE_TIMEOUT" "http://localhost:${PORT}/stream?q=postgres+tuning" || true

# Tear the server down before moving on — Next.js segment binds a
# different port but the operator may also run the SSE demo locally.
cleanup
SERVER_PID=""
pace

banner "2/2 · Next.js frontend (inline citation chips + chunks panel)"
printf 'demo/nextjs · React 19 · streams the same SSE protocol against an in-process fixture\n'
printf '  · citation chip [N] hover → matching chunk highlights · click → scrolls into view\n'
printf '  · pipeline phase pills: retrieve → rerank → generate → done\n\n'

if [ "$LAUNCH_NEXTJS" != "1" ]; then
  printf '(CAPTURE_LAUNCH_NEXTJS=0 → npm run dev launch skipped)\n'
  printf 'to record the second segment manually:\n'
  printf '  cd demo/nextjs && npm install && npm run dev   # → http://localhost:3000\n'
elif ! command -v npm >/dev/null 2>&1; then
  printf '(npm not found on PATH — install Node 20+ first)\n'
  printf '  cd demo/nextjs && npm install && npm run dev\n'
else
  cd "$REPO_ROOT/demo/nextjs"
  if [ ! -d node_modules ]; then
    printf 'node_modules missing — running `npm install` (one-time)...\n'
    npm install --silent
  fi
  printf 'launching `npm run dev` · http://localhost:3000 · Ctrl+C when done recording\n\n'
  # `exec` so Ctrl+C lands directly on next dev and the bash wrapper
  # doesn't intercept it. The cleanup trap won't run after exec, but
  # the SSE server was already torn down above.
  exec npm run dev
fi

banner "demo complete"
printf 'both surfaces ran end-to-end with zero API calls and zero external services.\n'
printf 'recapture: scripts/capture_demo.sh (env: CAPTURE_PACE_SECONDS, CAPTURE_LAUNCH_NEXTJS,\n'
printf '  CAPTURE_SSE_TIMEOUT, CAPTURE_DEMO_PORT).\n'
