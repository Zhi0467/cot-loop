# `rollout_bundle.v1` schema

Every run of `scripts/rollout/collect_model_stats.py` on a dataset now emits exactly
two files, written through `src/probe/bundle_io.py`:

| File | Contents |
| --- | --- |
| `<base>.jsonl.gz` | One JSON line per prompt (see "Bundle row"). Always gzipped. |
| `<base>.json` | Small sidecar with run metadata + aggregate counts + metrics. |

There are no sibling `__lcb_records.json`, `__prompt_profile.jsonl`,
`__prompt_rollout_archive.jsonl(.gz)`, `__rollout_archive.jsonl.gz`,
`__progress.json`, or `__lcb_records_checkpoint.json` files anymore. Any
legacy bundle with those siblings must be converted with
`scripts/rollout/migrate_legacy_rollout_bundle.py` before it can be consumed by the
current tools.

## Bundle row (`<base>.jsonl.gz`)

```jsonc
{
  "schema": "rollout_bundle.v1",
  "sample_id": 0,
  "split": "test",
  "record_id": "1873_A",                  // question_id for LCB/TACO, stable prompt id otherwise
  "record_metadata": { /* task-kind-specific raw row */ },
  "prompt": "...",                         // rendered prompt as fed to vLLM
  "prompt_token_ids": [151644, 872, ...],
  "prompt_token_count": 527,
  "effective_max_tokens": 40433,
  "max_model_len": 40960,
  "prompt_too_long": false,
  "prompt_profile": {                      // null when `prompt_too_long`
    "p_cap": 0.0, "p_loop": 0.1,
    "loop_budget_share": 0.0,
    "mu_log_rel": -2.1,
    "mean_length": 2500.0, "mean_relative_length": 0.061,
    "tail_threshold": 0.5,
    "tail_hit_count": 0, "majority_tail": 0,
    "num_rollouts": 10
    // task-kind extensions (e.g. "p_correct") may be added here
  },
  "rollouts": [
    {
      "rollout_index": 0,
      "completion_text": "...",            // raw vLLM text, never the extracted code
      "completion_token_ids": [...],       // may be null on migrated legacy bundles
      "completion_token_count": 2611,
      "total_token_count": 3138,
      "finish_reason": "stop",
      "loop_flag": false, "max_length_hit": false,
      "first_loop_prefix_length": null,
      "loop_trigger": null,                // or {"n", "k", "ngram"}
      "length": 2611, "relative_length": 0.064,
      "cap_hit": 0, "tail_hit": 0,
      "grading": {                         // null when the task is ungraded
        // livecodebench_codegen / taco_codegen:
        "code_output": "...",
        "passed": true,
        "pass_fraction": 1.0
        // math_freeform / multiple_choice_*:
        // "correct": 0 | 1
      }
    }
  ]
}
```

Migrated legacy rows additionally carry a row-level `degraded` marker (e.g.
`"no_raw_completion_text"` or
`"no_raw_completion_text,no_raw_prompt"`) when the source artifacts did not
retain the fields we need for full replay.

## Sidecar (`<base>.json`)

```jsonc
{
  "schema": "rollout_bundle.v1",
  "metadata": {
    "dataset": "livecodebench/code_generation_lite",
    "config": null,
    "split": "test",
    "task_kind": "livecodebench_codegen",
    "model_id": "Qwen/Qwen3-1.7B",
    "generation_config": { /* temperature, num_generations, max_tokens, ... */ },
    "stats_contract_version": "rollout_stats_v2",
    "seed": 0,
    "statistics": "success_fraction,loop_fraction,...",
    "loop_detector": { "n": 30, "k": 20 },
    "tail_threshold": 0.5,
    "prompt_token_summary": { "mean": 527.0, "p95": 901, ... },
    "timestamp": "...",
    "max_samples": 1000,
    "release_version": "release_v6",
    "lm_style": "HFChatTemplate",
    "thinking_mode": "on",
    "bundle_file": "<base>.jsonl.gz",
    "lcb_native_metrics": { /* populated only for livecodebench_codegen runs */ }
  },
  "counts": { "num_samples": 400, "num_generated": 4000, ... },
  "metrics": { "success_fraction": 0.41, "loop_fraction": 0.07, ... }
}
```

## Read path

```python
from probe.bundle_io import (
    bundle_paths,
    iter_bundle_rows,
    read_bundle_sidecar,
)

sidecar_path, bundle_path = bundle_paths("outputs/.../livecodebench_release_v6__test__Qwen_Qwen3-1.7B")
sidecar = read_bundle_sidecar(sidecar_path)
for row in iter_bundle_rows(bundle_path):
    for rollout in row["rollouts"]:
        ...
```

`iter_bundle_rows` transparently reads across concatenated gzip members, so a
bundle that was written by multiple `BundleWriter` opens (crash / resume /
distributed worker merge) is indistinguishable from one written in a single
pass.

## Resume / distributed writes

- `scripts/rollout/collect_model_stats.py --resume` scans the existing
  `<base>.jsonl.gz` via `completed_sample_ids(...)` and skips any `sample_id`
  that is already present. There are no separate checkpoint files.
- Distributed (`--dp > 1`) runs write to temporary per-rank bundle files; the
  main process merges them with `concat_bundles(...)` into the final
  `<base>.jsonl.gz` at the end of collection.

## Legacy migration

`scripts/rollout/migrate_legacy_rollout_bundle.py --in-dir <old> --out-dir <new>`
joins whichever of `<base>.json`, `__lcb_records.json`,
`__prompt_rollout_archive.jsonl(.gz)`, `__rollout_archive.jsonl.gz`, and
`__prompt_profile.jsonl` exist together and emits a fresh
`<base>.jsonl.gz` + `<base>.json` pair. Rows that cannot retain raw prompt or
completion text are flagged with the `degraded` field.
