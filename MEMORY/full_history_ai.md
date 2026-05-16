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

---
session: 2026-05-16T15:05Z
duration_min: 35
issue: 4
focus: rebase_pr_11_onto_main_after_streaming_pr_landed
delta:
  files_changed: 6  # __init__.py, README, 4 memory files
  tests_added: 0
  test_pass_rate: "74/74 hermetic + 7 pg-integration skipped"
context_for_next_session:
  - pr_11_rebased_on_main_with_d_008_d_009_and_d_010_d_011_interleaved_chronologically
  - readme_what_this_is_now_describes_1_2_4_5_together_one_paragraph_per_layer
  - rag_kit_init_exports_generator_and_streaming_layers_both_sets
  - issue_4_criterion_3_faithfulness_in_ci_re_scoped_to_issue_7_where_it_belongs
  - pr_11_state_clean_ready_for_review_per_d_004_next_scheduled_session_can_squash_merge
  - issue_7_eval_harness_integration_is_the_next_repo_target_in_this_run
decisions_made: []
followups: []
---

---
session: 2026-05-16T15:30Z
duration_min: 30
issue: 7
focus: eval_harness_integration_three_metrics_workflow
delta:
  files_added: 13  # __init__, run_eval, dataset+corpus, 3 baselines, 3 current, calibration, workflow, test
  files_changed: 2  # README, pyproject
  tests_added: 13
  test_pass_rate: "87/87 hermetic + 7 pg-integration skipped"
  benchmarks:
    faithfulness_mean: 1.0
    recall_at_5_mean: 1.0
    correctness_mean: 0.9008
    n_rows: 8
    judge: deterministic-stub-v1
context_for_next_session:
  - eval_extra_pins_eval_harness_to_2398cc3_hatch_allow_direct_references_enabled
  - three_suites_faithfulness_recall_at_5_correctness_each_writes_one_runresult_json
  - in_memory_token_overlap_retriever_stands_in_for_pgvector_for_hermetic_ci
  - composite_pr_comment_via_direct_github_api_repo_specific_marker_avoids_clobbering_d_012
  - corpus_chunks_are_single_sentence_for_templategenerator_compatibility_d_013
  - real_llm_eval_path_operator_triggered_with_anthropic_api_key_not_yet_wired
  - workflow_eval_yml_on_pr_install_eval_extra_run_eval_post_composite_sticky
decisions_made: [D-012, D-013]
followups: []
---

---
session: 2026-05-16T19:30Z
duration_min: 60
issue: 3
focus: query_rewriting_decomposition_template_plus_anthropic
delta:
  files_added: 3  # rewriter.py, test_rewriter.py, test_retriever_rewriter.py, bench_rewriter.py = 4 actually
  files_changed: 3  # __init__.py, retriever.py, README.md
  tests_added: 29
  test_pass_rate: "116/116 hermetic + 7 pg-integration skipped"
  benchmarks:
    rewriter_recall_at_2_baseline: 0.625
    rewriter_recall_at_2_rewriter: 0.688
    rewriter_recall_at_3_baseline: 0.625
    rewriter_recall_at_3_rewriter: 0.812
    rewriter_recall_at_5_baseline: 0.875
    rewriter_recall_at_5_rewriter: 0.938
    n_queries: 8
    n_corpus_chunks: 18
context_for_next_session:
  - rewriter_protocol_plus_templaterewriter_plus_anthropicrewriter_lazy_imported_via_existing_rag_anthropic_extra
  - retriever_search_rewriter_kwarg_default_none_keeps_existing_behavior_parallel_to_d_007
  - multi_sub_query_path_rrf_fuses_across_subqueries_per_method_ranks_dict_carries_subquery_i_keys
  - reranker_with_rewriter_scores_against_original_user_query_not_any_one_subquery
  - templaterewriter_patterns_compare_then_multi_question_and_with_priority_order
  - bench_script_scripts_bench_rewriter_runs_in_memory_token_overlap_no_pg_no_api_keys
  - anthropic_rewriter_numbers_pending_operator_triggered_anthropic_api_key_run
  - next_repo_targets_6_cost_telemetry_or_loop_to_a_different_portfolio_repo
decisions_made: [D-014]
followups: []
---
