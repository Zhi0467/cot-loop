# Prompt-Profile RFM Artifact Schema

Last updated: 2026-04-21 09:13 UTC

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
