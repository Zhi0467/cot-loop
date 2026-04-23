# Prompt-Profile RFM Artifact Schema

Last updated: 2026-04-22 00:03 UTC

## Purpose

- Keep the prompt-profile RFM stage auditable without Slack archaeology.
- Make every vector bundle and steering run carry one machine-readable JSON record with the same core keys.
- Split provenance cleanly:
  - artifact clock: when this record was materialized;
  - source clock: which prompts, model/tokenizer revisions, and git state defined the object being reported.

## Stage-0 Registry Validation Record

- Schema: `prompt_profile_rfm_stage_registry_validation.v1`
- One validation record should capture:
  - resolved archive root
  - dataset key and display name
  - prompt field
  - train/test counts
  - active-stage flag
  - feature key
  - sample shape
  - ordered train/test prompt-ID hashes
  - ordered train/test prompt-text hashes
  - target name
  - manifest path
  - prompt-rollout archive file
  - prompt-profile files
  - exact train/test prompt IDs

## Stage-0.5 Screening Summary Record

- Current implementation surface:
  - aggregate JSON written by `scripts/collect_model_stats.py`
  - incremental `__progress.json` written by the same collector during long screens
  - schema fields already exposed in the live screen:
    - `status`
    - `metadata.timestamp`
    - `metadata.prompt_profile_summary`
    - `metadata.prompt_profile_file`
    - `metadata.prompt_rollout_archive_file`
    - `metadata.prompt_rollout_archive_schema`
    - `counts.num_prompt_profiled`
    - `counts.num_prompt_majority_tail_positive`
    - `metrics.majority_s_0.5_positive_rate`
    - `metrics.completion_tail_fraction`
    - `metrics.median_generation_length`
- Required provenance keys for stage-0.5 summaries:
  - dataset and split
  - task kind
  - prompt formatter
  - generation config
  - loop-detector config
  - prompt-profile summary
  - archive schema name
  - row filter used for the candidate slice
  - any prompt-exclusion source used to keep the screen disjoint from an older stage object
- This is the stage-0.5 ledger for deciding whether a candidate pool clears the repaired positive-rate gate before any activation dataset is built from it.

## Stage-0.5 Incremental Progress Record

- File pattern: `<out_base>__progress.json`
- Purpose:
  - expose partial positive-rate / tail-rate / length signals before the full `300`-prompt summary lands
  - make long shared-node screens restart-safe instead of all-or-nothing
  - point the next session at the live sidecars without requiring log archaeology
- Required top-level keys:
  - `status`
  - `metadata`
  - `counts`
  - `metrics`
- Required metadata keys:
  - `task_kind`
  - `model_id`
  - `seed`
  - `timestamp`
  - `prompt_profile_summary`
  - `prompt_profile_file`
  - `prompt_rollout_archive_file`
  - `prompt_rollout_archive_schema`
- Required count keys:
  - `num_samples`
  - `num_generated`
  - `num_looped`
  - `num_max_length_hits`
  - `num_prompt_too_long`
  - `num_prompt_profiled`
  - `num_prompt_majority_tail_positive`
- Required metric keys:
  - `loop_fraction`
  - `max_length_hit_fraction`
  - `avg_generation_length`
  - `median_generation_length`
  - `majority_s_0.5_positive_rate`
  - `completion_tail_fraction`

## Stage-0.5 Screening Archive Row

- Schema: `prompt_rollout_archive.v2`
- Required top-level keys per prompt:
  - `split`
  - `sample_id`
  - `record_id`
  - `question_id` where the benchmark has one
  - `prompt_style`
  - `prompt`
  - `prompt_token_ids`
  - `prompt_token_count`
  - `effective_max_tokens`
  - `target_name`
  - `target_value`
  - `majority_tail`
  - `p_cap`
  - `p_loop`
  - `loop_budget_share`
  - `mean_relative_length`
  - `num_rollouts`
  - `record_metadata`
  - `rollouts`
- Required per-rollout keys:
  - `rollout_index`
  - `completion_text`
  - `completion_token_ids`
  - `finish_reason`
  - `length`
  - `relative_length`
  - `cap_hit`
  - `loop_flag`
  - `tail_hit`
  - `first_loop_prefix_length`
  - `correct`
- This row schema is the replay surface for later activation lookup, relabeling, prompt-text deduplication, and attention or prompt-profile follow-ups. If exact completion token IDs are present, downstream analyses should prefer them over retokenizing `completion_text`.

## Detector Run Record

- Schema: `prompt_profile_rfm_detector_run.v1`
- Required keys:
  - `benchmark`
  - `layer`
  - `prompt_ids.train`
  - `prompt_ids.val`
  - `prompt_ids.test`
  - `prompt_id_hashes`
  - `feature_key`
  - `preprocessing`
  - `rfm_hyperparameters`
  - `selection`
  - `sign_convention`
  - `score_sign`
  - `decision_threshold`
  - `train_metrics`
  - `val_metrics`
  - `test_metrics`
  - `git_commit`
  - `model_id`
  - `model_revision`
  - `tokenizer_revision`
  - `random_seed`
  - `output_path`
  - `checkpoint_path`
  - `artifact_sha256`

This is the stage-1 ledger record for one selected benchmark/layer detector.
It should be written next to the exported checkpoint so later lookups can answer:

- which exact prompt IDs defined train / val / test;
- whether train balancing or other preprocessing changed the natural archive split;
- which bandwidth / regularization / iteration won on validation;
- which score orientation and threshold were used for threshold diagnostics.

## Stage Materialization Split Manifest

- Supporting record: `split_manifest.json` emitted by `scripts/materialize_prompt_profile_stage_binary_data.py`
- Required keys:
  - `benchmark`
  - `prompt_ids.train`
  - `prompt_ids.val`
  - `prompt_ids.test`
  - `prompt_id_hashes.train`
  - `prompt_id_hashes.val`
  - `prompt_id_hashes.test`
  - `prompt_text_hashes.train`
  - `prompt_text_hashes.val`
  - `prompt_text_hashes.test`
  - `num_positive.train`
  - `num_positive.val`
  - `num_positive.test`
  - `materialization.archive_data_dir`
  - `materialization.prompt_rollout_archive_file`
  - `materialization.feature_key`
  - `materialization.model_id`
  - `materialization.model_revision`
  - `materialization.tokenizer_revision`
  - `materialization.seed`

This is the reusable March-object ledger for matched baseline reruns. It should
be treated as the source of truth for which prompt IDs defined the train / val /
test comparison object before any activation or prompt-only baseline is trained.

## RFM Vector Bundle Record

- Schema: `prompt_profile_rfm_vector_bundle.v1`
- Required keys:
  - `benchmark`
  - `layer`
  - `prompt_ids.train`
  - `prompt_ids.val`
  - `prompt_ids.test`
  - `feature_key`
  - `preprocessing`
  - `rfm_hyperparameters`
  - `vector_extraction`
  - `sign_convention`
  - `raw_vector_norm`
  - `normalized_vector_checksum`
  - `git_commit`
  - `model_revision`
  - `tokenizer_revision`
  - `random_seed`
  - `artifact_sha256`

- Optional keys now used once direction-stability replay exists:
  - `direction_bootstrap.record_path`
  - `direction_bootstrap.num_requested`
  - `direction_bootstrap.num_completed`
  - `direction_bootstrap.seed`
  - `direction_bootstrap.selection_iteration`
  - `direction_bootstrap.cosine_to_reference`

## Vector Direction Bootstrap Record

- Schema: `prompt_profile_rfm_vector_direction_bootstrap.v1`
- Required keys:
  - `benchmark`
  - `layer`
  - `prompt_ids.train`
  - `prompt_ids.val`
  - `prompt_ids.test`
  - `feature_key`
  - `preprocessing`
  - `rfm_hyperparameters`
  - `vector_extraction_formula`
  - `sign_convention`
  - `reference_vector_checksum`
  - `bootstrap`
  - `cosine_to_reference`
  - `git_commit`
  - `model_revision`
  - `tokenizer_revision`
  - `random_seed`
  - `artifact_sha256`

This is the stage-2 direction-stability ledger for one benchmark/layer bundle.
It should answer:

- how many fit-train bootstrap resamples were replayed;
- which selected detector iteration and sign convention defined the replayed direction;
- how tightly the resampled signed vectors align with the exported reference direction.

## Steering Run Record

- Schema: `prompt_profile_rfm_steering_run.v1`
- Required keys:
  - `condition_name`
  - `vector_artifact_hash`
  - `hook_site`
  - `t`
  - `seeds`
  - `prompt_ids`
  - `generation_config`
  - `grader_version`
  - `output_path`
  - `artifact_sha256`

## Implementation Surface

- Shared helpers live in `src/loop_probe/stage_artifacts.py`.
- Registry emission and validation live in:
  - `scripts/emit_prompt_profile_rfm_stage_registry.py`
  - `scripts/validate_prompt_profile_rfm_stage_registry.py`
- The stage registry itself lives in `src/loop_probe/prompt_profile_rfm_stage_registry.py`.

## Notes

- `normalized_vector_checksum` should be computed after the sign convention is fixed and after normalization, so two records with opposite sign do not collide semantically.
- `artifact_sha256` is the hash of the canonical JSON payload, not a model checkpoint digest.
- Store these records next to the generated run outputs, not only in summary tables.
- For stage-1 detector checkpoints, the checkpoint payload may keep the learned
  Mahalanobis matrix and kernel weights in a Torch file, but the JSON record is
  still the source of truth for searchable provenance.
