# Session History (AI-readable, append-only)

Schema: see .skills/portfolio-memory/SKILL.md

---
session: 2026-05-14T14:30:00Z
duration_min: 70
issue: 1
focus: hybrid_retrieval_bm25_plus_pgvector_with_rrf
delta:
  files_added: 18
  files_changed: 5
  tests_added: 21
  unit_coverage_pct: 64
context_for_next_session:
  - hybrid_retrieval_api_shipped_indexer_retriever_rrf
  - schema_dim_is_64_to_match_hashembedder_real_embedders_change_vector_n_and_embedding_dim_together_per_d003
  - docker_compose_uses_pgvector_pgvector_pg16_init_sql_mounted_to_docker_entrypoint_initdb_d
  - ci_has_three_jobs_lint_unit_matrix_integration_pg_with_service_container
  - recall_at_5_intentionally_deferred_to_issue_7_eval_harness_integration
  - retriever_returns_per_method_ranks_so_callers_can_debug_which_channel_surfaced_each_doc
decisions_made: [D-002, D-003, D-004]
followups: []
---

---
session: 2026-05-15T14:37Z
duration_min: 60
issue: 2
focus: cross_encoder_reranking_layer
delta:
  files_added: 1
  files_changed: 5
  tests_added: 18
  test_pass_rate: "33/33 hermetic + 2 new pg integration"
context_for_next_session:
  - reranker_protocol_plus_lexical_overlap_default_plus_cohere_production
  - retriever_search_reranker_kwarg_opt_in_d007_backwards_compatible
  - rerank_delta_ndcg_telemetry_helper_for_tracking_reranker_movement
  - recall_at_5_with_without_reranker_intentionally_deferred_to_issue_7
  - cohere_extra_lazy_imported_so_module_loads_without_it
  - portfolio_pattern_single_method_protocol_for_test_substitution_now_in_three_repos_eval_harness_rag_kit
decisions_made: [D-005, D-006, D-007]
followups: []
---
