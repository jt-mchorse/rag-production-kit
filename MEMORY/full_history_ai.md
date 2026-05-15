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

---
session: 2026-05-15T19:15Z
duration_min: 60
issue: 4
focus: citation_enforcement_and_weak_context_refusal
delta:
  files_added: 2
  files_changed: 3
  tests_added: 18
  test_pass_rate: "51/51 hermetic (7 pg-integration skipped as before)"
context_for_next_session:
  - generator_protocol_plus_template_default_plus_anthropic_extra_lands_d008
  - two_refusal_paths_pre_llm_threshold_and_post_llm_citation_validation_d009
  - enforce_citations_is_a_free_function_so_alt_generators_can_reuse_d008
  - faithfulness_in_ci_via_eval_harness_is_the_remaining_box_on_issue_4_belongs_with_7
  - rag_anthropic_extra_added_to_pyproject_anthropic_sdk_lazy_imported
  - default_threshold_0_02_tuned_to_existing_hybrid_retrieval_fixture_scores
decisions_made: [D-008, D-009]
followups: []
---

---
session: 2026-05-16T03:09Z
duration_min: 75
issue: 5
focus: streaming_intermediate_events_sse_pipeline
delta:
  files_added: 5
  files_changed: 4
  tests_added: 23
  test_pass_rate: "56/56 hermetic + 7 pg-integration skipped"
  benchmarks:
    streaming_p50_total_ms: 0.11
    streaming_p95_total_ms: 0.14
    streaming_throughput_q_per_s: 8553
    n: 1000
    host: "apple_silicon_arm64_python_3_14_0"
context_for_next_session:
  - streaming_pipeline_is_sync_generator_yielding_typed_streamevents
  - phases_retrieving_retrieved_reranking_reranked_generating_token_generated_done_error
  - to_sse_serializes_one_frame_per_event_per_html_spec
  - tokenstream_protocol_is_the_seam_used_with_4_anthropicgenerator
  - phasetimings_does_linear_interp_percentile_no_numpy_dep
  - demo_under_demo_streaming_uses_stdlib_http_server_runs_without_postgres
  - bench_script_under_scripts_bench_streaming_drives_n_queries_prints_p50_p95_per_phase
decisions_made: [D-010, D-011]
followups: []
---
