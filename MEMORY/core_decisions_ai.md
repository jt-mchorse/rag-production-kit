# Core Decisions (AI-readable, YAML, append-only)
# Schema: see .skills/portfolio-memory/SKILL.md

- id: D-001
  date: 2026-05-10
  decision: scope_per_portfolio_handoff_section_2
  rationale: locked_scope_prevents_drift
  alternatives_rejected: []
  reversibility: expensive
  related_issues: []
  superseded_by: null

- id: D-002
  date: 2026-05-14
  decision: only_required_runtime_dep_is_psycopg_everything_else_is_optional_extras
  rationale: keep_import_cost_low_for_downstream_consumers_embedders_are_pluggable_eval_harness_is_imported_not_vendored
  alternatives_rejected: [bundle_anthropic_sdk_as_required, bundle_eval_harness_as_required]
  reversibility: cheap
  related_issues: [1, 2, 7]
  superseded_by: null

- id: D-003
  date: 2026-05-14
  decision: dense_vector_dimensionality_configured_per_deployment_default_64_matches_hashembedder
  rationale: real_embedders_have_different_dims_512_768_1024_3072_and_schema_must_change_in_lockstep_with_embedder
  alternatives_rejected: [hardcoded_1024_dim, store_as_jsonb_with_no_typed_column]
  reversibility: cheap
  related_issues: [1, 2]
  superseded_by: null

- id: D-004
  date: 2026-05-14
  decision: reciprocal_rank_fusion_with_k_60_as_default_returns_per_method_ranks
  rationale: cormack_et_al_2009_default_strong_baseline_per_method_ranks_enable_debugging_of_which_channel_surfaced_each_doc
  alternatives_rejected: [weighted_score_fusion, condorcet_voting, dense_only, lexical_only]
  reversibility: cheap
  related_issues: [1]
  superseded_by: null
