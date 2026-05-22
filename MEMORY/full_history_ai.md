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

---
session: 2026-05-16T20:30Z
duration_min: 50
issue: 6
focus: cost_telemetry_per_request_blob_sqlite_store_stdlib_dashboard
delta:
  files_added: 3  # rag_kit/telemetry.py, tests/test_telemetry.py, scripts/telemetry_dashboard.py
  files_changed: 2  # __init__.py, README.md
  tests_added: 21
  test_pass_rate: "108/108 hermetic + 7 pg-integration skipped"
context_for_next_session:
  - cost_telemetry_layer_costrecord_pricetable_telemetrystore_aggregate_lives_at_rag_kit_telemetry
  - sqlite_via_stdlib_no_new_runtime_dep_d_002_preserved
  - price_table_ships_no_defaults_unknown_model_raises_d_015
  - aggregate_percentile_uses_same_nist_type_7_math_as_streaming_phase_timings
  - dashboard_at_scripts_telemetry_dashboard_stdlib_http_server_inline_svg_no_external_deps_airgapped
  - dashboard_endpoints_get_root_html_get_api_last_24h_json
  - dashboard_seed_flag_inserts_synthetic_records_clearly_labeled_synthetic_n
  - issue_6_acceptance_telemetry_schema_documented_done_dashboard_chart_renders_last_24h_done_unit_test_for_cost_calc_with_fixture_prices_done
  - this_branch_is_off_main_pre_d_014_rewriter_pr_so_init_py_exports_only_telemetry_not_rewriter
decisions_made: [D-015]
followups: []
---

---
session: 2026-05-18T04:50Z
duration_min: 55
issue: 8
focus: nextjs_15_demo_with_inline_citations_seventh_issue_of_night
delta:
  files_changed: 16
  tests_added: 13
context_for_next_session:
  - d_016_nextjs_demo_speaks_same_sse_protocol_as_python_demo
  - new_demo_nextjs_subdirectory_alongside_existing_demo_streaming
  - fresh_clone_runs_with_no_postgres_no_anthropic_no_python_backend_deterministic_in_repo_fixtures
  - citation_chips_hover_highlight_plus_click_scroll_to_matching_chunk_in_panel
  - production_build_static_root_dynamic_api_stream
  - pr_body_flags_browser_walkthrough_as_not_done_in_pr
decisions_made: [D-016]
followups: []
---

---
session: 2026-05-18T15:27Z
duration_min: 25
issue: 17
focus: architecture_doc_covers_all_eight_shipped_layers
delta:
  files_changed: 2  # README.md, docs/architecture.md
  files_added: 0
  tests_added: 0   # pure docs
  test_pass_rate: "137/137 + 7 pg-integration skipped"
context_for_next_session:
  - docs_architecture_md_rewritten_one_integrated_index_plus_query_lifecycle_mermaid_at_top_plus_one_section_per_shipped_layer
  - eight_layer_sections_hybrid_retrieval_reranker_rewriter_generator_streaming_telemetry_eval_nextjs_demo
  - readme_architecture_dropped_ascii_for_one_paragraph_summary_link
  - pending_section_removed_every_runtime_layer_in_section_2_has_shipped
  - mermaid_labels_with_parens_fully_double_quoted_same_lint_as_cost_optimizer_doc
  - no_new_d_entry_pure_docs_references_d_002_through_d_016
  - quality_bar_section_1_architecture_diagram_now_met_was_previously_partial
decisions_made: []
followups: []
---

---
session: 2026-05-18T20:45Z
duration_min: 20
issue: 19
focus: snapshot_test_locks_readme_rewriter_recall_table_to_bench_rewriter_output
delta:
  files_added: 1   # tests/test_rewriter_bench_snapshot.py
  files_changed: 0
  tests_added: 4   # parametrized over k in {2,3,5} plus row-count guard
  test_pass_rate: "141/141 + 7 pg-integration skipped"
context_for_next_session:
  - snapshot_imports_scripts_bench_rewriter_run_and_summary_directly_no_subprocess
  - readme_table_located_by_header_signature_robust_to_surrounding_prose_edits
  - parametrized_k_in_2_3_5_each_cell_matched_means_at_abs_5e_4_counts_exact
  - failure_messages_name_three_regen_commands_one_per_k_plus_git_diff_readme_md
  - tamper_verified_by_editing_k_3_rewriter_mean_0_812_to_0_999_test_fired_then_reverted
  - row_count_test_guards_against_silently_dropping_or_adding_a_k_row
  - pattern_parallel_to_llm_cost_optimizer_iter_3_and_prompt_regression_suite_iter_4_landed_same_day
decisions_made: []
followups: []
---

---
session: 2026-05-19T15:09Z
duration_min: 25
issue: 21
focus: snapshot_test_locks_readme_eval_mean_score_table_to_run_eval_output
delta:
  files_added: 1   # tests/test_eval_bench_snapshot.py
  files_changed: 0
  tests_added: 4   # parametrized over the three suites plus row-count guard
  test_pass_rate: "145/145 + 7 pg-integration skipped"
context_for_next_session:
  - second_readme_table_now_locked_third_snapshot_test_in_this_repo
  - parametrized_over_faithfulness_recall_at_5_correctness_each_cell_matched_means_at_abs_5e_3
  - module_scoped_fixture_caches_one_run_all_suites_call_across_the_three_parametrized_rows
  - failure_messages_name_python_m_evals_run_eval_write_baselines_plus_git_diff_readme_md
  - tamper_verified_by_editing_correctness_cell_0_90_to_0_99_test_fired_then_reverted
  - row_count_test_guards_against_silently_dropping_or_adding_a_suite_row
  - follow_up_19_explicitly_recommended_was_now_the_filing_thread_of_21
  - eval_orchestrator_remains_deterministic_d_012_plus_d_013_still_load_bearing_no_new_decision
decisions_made: []
followups: []
---

---
session: 2026-05-20T03:11Z
duration_min: 35
issue: 23
focus: public_surface_snapshot_test_locks_rag_kit_top_level_init
delta:
  files_added: 1   # tests/test_public_surface.py
  files_changed: 1   # rag_kit/__init__.py (+__version__)
  tests_added: 13   # 4 standalone + 9 parametrized submodule anchors
  test_pass_rate: "158/158 + 7 pg-integration skipped"
context_for_next_session:
  - public_surface_pattern_portable_to_remaining_four_python_repos_embedding_model_shootout_chunking_strategies_lab_python_async_llm_pipelines_mcp_server_cookbook_python_example
  - rag_kit_now_publishes_dunder_version_str_0_0_1_mirror_of_pyproject
  - ast_parser_filters_on_level_geq_1_for_relative_imports_same_fix_as_prompt_regression_suite_pr_20
  - no_importlib_reload_workaround_needed_no_entry_points_pytest_plugin_in_this_repo_init_already_at_100pct_coverage
  - tamper_verified_three_axes_bad_version_drop_document_alias_rename_hashembedder
  - readme_quickstart_line_115_now_pinned_document_hashembedder_indexer_retriever
  - nine_submodule_anchors_embedder_fusion_generator_indexer_reranker_retriever_rewriter_streaming_telemetry
decisions_made: []
followups: []
---

---
session: 2026-05-22T16:00Z
duration_min: 30
issue: 27
focus: readme_what_this_is_section_now_lists_all_eight_shipped_layers
delta:
  files_changed: 1   # README.md
  files_added: 1     # tests/test_readme_what_this_is_lists_shipped_layers.py
  tests_added: 4
  test_pass_rate: "162/162 + 7 pg-integration skipped"
context_for_next_session:
  - readme_what_this_is_section_closing_paragraph_l83_87_said_everything_beyond_layers_1_to_6_was_staged_but_7_shipped_2026_05_16_and_8_shipped_2026_05_18
  - architecture_section_already_said_eight_runtime_layers_ship_today_so_one_part_of_readme_contradicted_another
  - replaced_stale_paragraph_with_two_new_bold_sub_sections_eval_harness_integration_and_nextjs_demo_frontend_same_shape_as_the_existing_six
  - added_missing_link_reference_for_issue_8
  - new_snapshot_test_four_axes_canonical_layer_set_match_no_extra_bold_openers_stale_phrase_absent_architecture_section_agrees
  - canonical_layers_frozenset_lives_in_the_test_so_a_future_ninth_layer_must_update_both_the_section_and_the_test
  - stale_phrase_matched_via_re_staged_s_plus_in_s_plus_follow_up_s_plus_issues_because_the_original_wrapped_mid_line_at_follow_up_newline_issues
  - tamper_verified_by_reinjecting_the_wrapped_stale_phrasing_third_assertion_fires_with_the_matched_wrapped_form_quoted_in_the_message
  - issue_27_was_filed_in_session_after_phase_a_pr_review_merged_seven_prs_and_session_one_closed_llm_cost_optimizer_25_no_other_actionable_high_or_med_issues_were_open_anywhere_the_portfolio_is_pure_drift_hunting_at_this_point
  - tenth_post_v0_1_readme_vs_code_drift_fix_in_the_portfolio_pattern
decisions_made: []
followups: []
---
