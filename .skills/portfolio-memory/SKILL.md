---
name: portfolio-memory
description: Use this skill on every session that touches a portfolio repo (any repo under jt-mchorse/ following the portfolio handoff). It enforces the memory protocol — reading MEMORY/full_history_ai.md and MEMORY/core_decisions_ai.md before planning, and writing append-only updates to both human and AI versions at session end. Triggers whenever Cowork is operating on rag-production-kit, agent-orchestration-platform, llm-eval-harness, prompt-regression-suite, ai-app-integration-tests, nextjs-streaming-ai-patterns, python-async-llm-pipelines, embedding-model-shootout, chunking-strategies-lab, llm-cost-optimizer, vector-search-at-scale, mcp-server-cookbook, or portfolio-ops. Do NOT skip memory reads even for short sessions or memory-only updates.
---

# Portfolio Memory Protocol

## Why this exists
The portfolio is twelve interconnected repos worked on across many sessions over many months. Without enforced memory, every session re-litigates settled decisions and duplicates prior work. This skill makes memory the load-bearing first step of every session.

## The four files (per repo)

```
MEMORY/
├── full_history_ai.md        # structured, append-only session log (read every session)
├── full_history_human.md     # prose version of same (written at session end)
├── core_decisions_ai.md      # YAML decision log (read every session)
└── core_decisions_human.md   # prose version of same
```

## Mandatory protocol

### At session start (BEFORE planning anything)

1. `cat MEMORY/full_history_ai.md` — read at minimum the last 5 session entries.
2. `cat MEMORY/core_decisions_ai.md` — read entire file (it's small).
3. Check: does the issue you're about to work conflict with any non-superseded decision? If yes, do not proceed; comment on the issue and pause.

If `MEMORY/` doesn't exist yet, this is a new repo bootstrap. Create the four files using the templates in this skill (below) and write the D-001 baseline decision before any code.

### At session end (BEFORE the final commit)

1. **Write the AI version first.** Append a new YAML block to `full_history_ai.md`:

```yaml
---
session: <ISO-8601 timestamp>
duration_min: <integer>
issue: <issue number, or null for memory-only sessions>
focus: <snake_case label>
delta:
  files_changed: <int>
  tests_added: <int>
  benchmarks: { <metric>: <value>, ... }   # only if measured this session
context_for_next_session:
  - <bullet>
  - <bullet>
decisions_made: [D-NNN, ...]               # IDs from core_decisions, empty list if none
followups: [#<issue>, ...]                  # new issues filed this session
---
```

2. **Derive the human version.** Append a section to `full_history_human.md`:

```markdown
## YYYY-MM-DD — Issue #NN: <human-readable focus>
**Duration:** ~<N> min · **Branch:** <branch-name>

<1–3 prose bullets on what got done>

**Why this work, this session:** <one sentence>

**Open questions / blockers:** <one sentence or "none">

**Next session:** <one sentence>
```

3. **If a core decision was made this session**, update both `core_decisions_*` files. Use a separate commit for memory updates (`memory: session YYYY-MM-DD repo-name`).

### Decision entry format

`core_decisions_ai.md`:

```yaml
- id: D-NNN
  date: YYYY-MM-DD
  decision: <snake_case_summary>
  rationale: <snake_case_one_clause>
  alternatives_rejected: [<a>, <b>]
  reversibility: cheap | expensive | one-way
  related_issues: [#NN, #NN]
  superseded_by: null   # set to D-NNN when superseded; never delete entries
```

`core_decisions_human.md`:

```markdown
## D-NNN — <Title> (YYYY-MM-DD)
**Decision:** <one sentence>

**Why:** <paragraph>

**Alternatives considered:**
- <alt> — rejected because <reason>

**Reversibility:** <cheap | expensive | one-way>

**Related issues:** #NN, #NN
```

## Hard rules

- **Append-only.** Never edit or delete past session entries or decisions. To change a decision, supersede it.
- **AI version is the source of truth.** Human version is derived. If they diverge, AI wins; rewrite the human entry.
- **Memory commits are separate.** `memory: session YYYY-MM-DD <repo>` — never bundled with code.
- **No silent decisions.** If you made a tradeoff in code that wasn't trivial, it's a decision. Log it.
- **Bootstrap exception.** First session on a brand-new repo writes D-001 from the portfolio handoff §2 spec, then proceeds normally.

## Starter file contents

When initializing MEMORY/ for a new repo:

`full_history_ai.md`:
```markdown
# Session History (AI-readable, append-only)

Schema: see .skills/portfolio-memory/SKILL.md
```

`full_history_human.md`:
```markdown
# Session History (human-readable)

Chronological log of work sessions. Most recent first below the divider.

---
```

`core_decisions_ai.md`:
```yaml
# Core Decisions (AI-readable, YAML, append-only)
# Schema: see .skills/portfolio-memory/SKILL.md

- id: D-001
  date: <bootstrap date>
  decision: scope_per_portfolio_handoff_section_2
  rationale: locked_scope_prevents_drift
  alternatives_rejected: []
  reversibility: expensive
  related_issues: []
  superseded_by: null
```

`core_decisions_human.md`:
```markdown
# Core Decisions

Strategic decisions for this repo, with reasoning. Append-only — superseded decisions are marked, not removed.

## D-001 — Scope locked to portfolio handoff §2 (<date>)
**Decision:** Scope of this repo is fixed by the portfolio handoff document, section 2.

**Why:** The handoff spec was deliberated; ad-hoc scope expansion within a session is the failure mode this prevents.

**Alternatives considered:** None — this is a baseline.

**Reversibility:** Expensive. Scope changes require a deliberate revisit and a new decision entry.

**Related issues:** —
```
