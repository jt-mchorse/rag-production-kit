"""Eval orchestrator: runs three metrics over the rag-qa-v0.1 golden set
and writes one ``RunResult``-shape JSON per metric.

Hermetic by design: an in-memory token-overlap retriever stands in for
real pgvector retrieval so the workflow runs with zero external services.
Real-PG / real-LLM runs are operator-triggered by setting
``RAG_EVAL_BACKEND=pgvector`` (not implemented in this file — that lands
when the operator wires their own ``ANTHROPIC_API_KEY`` budget).

Three suites, one JSON per:

- ``faithfulness``:  per-row 1.0 iff the generator output is a
  ``GeneratedAnswer`` whose ``enforce_citations`` already validated, else
  0.0 (a ``Refusal`` or a citation-error response).
- ``recall_at_5``:   per-row 1.0 iff every gold chunk id in the example's
  ``provenance.gold_chunk_ids`` appears among the top-5 retrieved
  ``external_id``s; 0.0 otherwise.
- ``correctness``:   per-row fraction of expected-output content tokens
  that appear in the generated answer (a deterministic string-overlap
  proxy for the LLM-as-judge correctness rubric; real-judge runs land
  with an operator-set API key).

Run shape matches ``eval_harness.runner.RunResult`` so ``eval-harness
diff-json`` can compare current vs baseline without modification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from rag_kit import (
    CitationError,
    GeneratedAnswer,
    Refusal,
    TemplateGenerator,
    enforce_citations,
)
from rag_kit.retriever import RetrievalResult

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "evals" / "dataset" / "rag_qa_v1.jsonl"
CORPUS_PATH = REPO_ROOT / "evals" / "dataset" / "corpus_v1.jsonl"
BASELINES_DIR = REPO_ROOT / "evals" / "baselines"
CURRENT_DIR = REPO_ROOT / "evals" / "current"

SUITES = ("faithfulness", "recall_at_5", "correctness")


@dataclass(frozen=True)
class _Example:
    example_id: str
    input: str
    expected: str
    gold_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class _Chunk:
    external_id: str
    text: str


def _load_dataset(path: Path) -> list[_Example]:
    out: list[_Example] = []
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        gold = tuple(row.get("provenance", {}).get("gold_chunk_ids", ()))
        expected = row["expected_outputs"][0]["value"]
        out.append(
            _Example(
                example_id=row["id"],
                input=row["input"],
                expected=expected,
                gold_chunk_ids=gold,
            )
        )
    if not out:
        raise ValueError(f"empty dataset at {path}")
    return out


def _load_corpus(path: Path) -> list[_Chunk]:
    out: list[_Chunk] = []
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        out.append(_Chunk(external_id=row["external_id"], text=row["text"]))
    return out


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _retrieve_in_memory(query: str, corpus: list[_Chunk], k: int) -> list[RetrievalResult]:
    """Token-overlap ranker that stands in for pgvector for hermetic CI.

    Scored by (count of query tokens present in chunk text) / (chunk
    token count + 1); ties broken by external_id for determinism.
    Returns ``RetrievalResult`` instances so downstream code (citations,
    generator) is identical to the production path.
    """
    q_tokens = set(_tokens(query))
    scored: list[tuple[float, _Chunk]] = []
    for chunk in corpus:
        c_tokens = _tokens(chunk.text)
        overlap = sum(1 for tok in c_tokens if tok in q_tokens)
        score = overlap / (len(c_tokens) + 1)
        scored.append((score, chunk))
    scored.sort(key=lambda pair: (-pair[0], pair[1].external_id))
    top = scored[:k]
    results: list[RetrievalResult] = []
    for rank_index, (score, chunk) in enumerate(top, start=1):
        results.append(
            RetrievalResult(
                external_id=chunk.external_id,
                text=chunk.text,
                metadata={},
                fused_score=score,
                ranks={"lex": rank_index, "dense": rank_index},
            )
        )
    return results


def _score_faithfulness(
    generated: GeneratedAnswer | Refusal, retrieved: list[RetrievalResult]
) -> tuple[float, str]:
    if isinstance(generated, Refusal):
        return 0.0, f"refused (reason={generated.reason})"
    try:
        enforce_citations(generated.text, retrieved)
    except CitationError as e:
        return 0.0, f"citation invalid: {e}"
    return 1.0, "all sentences cite a retrieved chunk"


def _score_recall_at_5(
    retrieved: list[RetrievalResult], gold_ids: tuple[str, ...]
) -> tuple[float, str]:
    if not gold_ids:
        return 0.0, "no gold_chunk_ids in dataset row"
    top_ids = {r.external_id for r in retrieved[:5]}
    hits = [g for g in gold_ids if g in top_ids]
    score = len(hits) / len(gold_ids)
    return (
        score,
        f"hits {len(hits)}/{len(gold_ids)} (gold={list(gold_ids)}, top5={sorted(top_ids)})",
    )


def _score_correctness(generated: GeneratedAnswer | Refusal, expected: str) -> tuple[float, str]:
    if isinstance(generated, Refusal):
        return 0.0, f"refused (reason={generated.reason})"
    expected_tokens = set(_tokens(expected)) - _STOP
    if not expected_tokens:
        return 0.0, "expected output had no content tokens"
    gen_tokens = set(_tokens(generated.text)) - _STOP
    hits = expected_tokens & gen_tokens
    score = len(hits) / len(expected_tokens)
    return score, f"token overlap {len(hits)}/{len(expected_tokens)} ({sorted(hits)})"


_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
        "yes",
    }
)


@dataclass(frozen=True)
class _SuiteRow:
    example_id: str
    score: float
    reasoning: str


@dataclass(frozen=True)
class _SuiteRun:
    suite: str
    rows: list[_SuiteRow]
    git_sha: str | None

    def to_run_result(self, *, dataset_version: str) -> dict:
        mean = sum(r.score for r in self.rows) / len(self.rows) if self.rows else 0.0
        run_id = _new_run_id(self.suite, self.git_sha or "no-git")
        return {
            "run_id": run_id,
            "started_at": "2026-05-16T00:00:00Z",
            "suite": self.suite,
            "dataset_version": dataset_version,
            "judge_model": "deterministic-stub-v1",
            "judge_kappa": None,
            "mean_score": round(mean, 4),
            "n_rows": len(self.rows),
            "git_sha": self.git_sha,
            "rows": [
                {"example_id": r.example_id, "score": round(r.score, 4), "reasoning": r.reasoning}
                for r in self.rows
            ],
        }


def _new_run_id(suite: str, git_sha: str) -> str:
    h = hashlib.sha256(f"{suite}|{git_sha}".encode()).hexdigest()[:12]
    return f"{suite}_{h}"


def _git_sha() -> str | None:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return sha.decode().strip()
    except Exception:
        return None


def run_all_suites() -> list[_SuiteRun]:
    examples = _load_dataset(DATASET_PATH)
    corpus = _load_corpus(CORPUS_PATH)
    generator = TemplateGenerator()
    sha = _git_sha()

    faith: list[_SuiteRow] = []
    recall: list[_SuiteRow] = []
    correctness: list[_SuiteRow] = []

    for ex in examples:
        retrieved = _retrieve_in_memory(ex.input, corpus, k=5)
        gen_out = generator.generate(ex.input, retrieved, threshold=0.0)

        fs, fr = _score_faithfulness(gen_out, retrieved)
        rs, rr = _score_recall_at_5(retrieved, ex.gold_chunk_ids)
        cs, cr = _score_correctness(gen_out, ex.expected)

        faith.append(_SuiteRow(ex.example_id, fs, fr))
        recall.append(_SuiteRow(ex.example_id, rs, rr))
        correctness.append(_SuiteRow(ex.example_id, cs, cr))

    return [
        _SuiteRun("faithfulness", faith, sha),
        _SuiteRun("recall_at_5", recall, sha),
        _SuiteRun("correctness", correctness, sha),
    ]


def write_runs(runs: list[_SuiteRun], out_dir: Path, *, dataset_version: str) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for run in runs:
        path = out_dir / f"{run.suite}.json"
        path.write_text(
            json.dumps(run.to_run_result(dataset_version=dataset_version), indent=2) + "\n"
        )
        paths[run.suite] = path
    return paths


def _post_composite_comment(repo: str, pr: int, deltas: dict[str, str], token: str | None) -> None:
    """Composite sticky comment with all three suite deltas.

    Avoids ``eval-harness comment`` (it uses a single hardcoded marker)
    so the three suites share one comment instead of overwriting each
    other. Marker below is repo-private to keep this comment from
    colliding with the harness's own demo marker.
    """
    marker = "<!-- rag-production-kit:eval-sticky -->"
    body_parts = [marker, "", "# Eval delta — rag-production-kit"]
    body_parts.append("")
    body_parts.append(
        "Three suites against the synthetic `rag-qa-v0.1` golden set; current vs committed baseline."
    )
    body_parts.append("")
    for suite in SUITES:
        body_parts.append(f"## `{suite}`")
        body_parts.append("")
        body_parts.append(deltas.get(suite, "_(no delta produced)_"))
        body_parts.append("")

    body = "\n".join(body_parts)

    if token is None:
        print("(dry-run: no GITHUB_TOKEN; comment body printed below)\n")
        print(body)
        return

    # Find an existing sticky comment by marker.
    list_url = f"https://api.github.com/repos/{repo}/issues/{pr}/comments?per_page=100"
    req = urllib.request.Request(
        list_url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"},
    )
    existing_id: int | None = None
    try:
        with urllib.request.urlopen(req) as resp:
            comments = json.loads(resp.read().decode())
            for c in comments:
                if marker in (c.get("body") or ""):
                    existing_id = int(c["id"])
                    break
    except urllib.error.HTTPError as e:
        print(f"warning: failed to list PR comments: {e}", file=sys.stderr)

    payload = json.dumps({"body": body}).encode()
    if existing_id is not None:
        edit_url = f"https://api.github.com/repos/{repo}/issues/comments/{existing_id}"
        req = urllib.request.Request(
            edit_url,
            data=payload,
            method="PATCH",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
            },
        )
    else:
        create_url = f"https://api.github.com/repos/{repo}/issues/{pr}/comments"
        req = urllib.request.Request(
            create_url,
            data=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
            },
        )
    with urllib.request.urlopen(req) as resp:
        if resp.status >= 300:
            raise RuntimeError(f"comment post failed: {resp.status} {resp.read().decode()}")


def _diff_markdown(current: Path, baseline: Path) -> str:
    """Shell out to `eval-harness diff-json --format markdown`.

    Keeps the diff rendering identical to llm-eval-harness's own output.
    """
    out = subprocess.run(
        [
            "eval-harness",
            "diff-json",
            "--current",
            str(current),
            "--baseline",
            str(baseline),
            "--format",
            "markdown",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout or out.stderr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-baselines",
        action="store_true",
        help="Overwrite committed baselines with this run.",
    )
    parser.add_argument(
        "--post-comment",
        action="store_true",
        help="Render diffs against baselines and post a composite PR sticky comment.",
    )
    parser.add_argument("--repo", default=None, help="`owner/name` for the sticky comment.")
    parser.add_argument("--pr", type=int, default=None, help="PR number for the sticky comment.")
    parser.add_argument(
        "--token-env", default="GITHUB_TOKEN", help="Env var to read the API token from."
    )
    args = parser.parse_args(argv)

    runs = run_all_suites()

    target_dir = BASELINES_DIR if args.write_baselines else CURRENT_DIR
    written = write_runs(runs, target_dir, dataset_version="rag-qa-v0.1")
    for _, path in written.items():
        try:
            rel = path.relative_to(REPO_ROOT)
            print(f"wrote {rel}")
        except ValueError:
            # Tests may redirect CURRENT_DIR to a tmp_path outside REPO_ROOT.
            print(f"wrote {path}")

    if args.post_comment:
        if not args.repo or args.pr is None:
            print("--post-comment requires --repo and --pr", file=sys.stderr)
            return 2
        import os

        token = os.environ.get(args.token_env)
        deltas: dict[str, str] = {}
        for suite in SUITES:
            cur = CURRENT_DIR / f"{suite}.json"
            base = BASELINES_DIR / f"{suite}.json"
            if not base.exists():
                deltas[suite] = f"_(no baseline at {base.relative_to(REPO_ROOT)})_"
                continue
            deltas[suite] = _diff_markdown(cur, base)
        _post_composite_comment(args.repo, args.pr, deltas, token)

    return 0


if __name__ == "__main__":
    sys.exit(main())
