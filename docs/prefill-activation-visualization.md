# Prefill Activation Visualization

Last updated: 2026-03-21 23:53 UTC

## Purpose

This note records the corrected activation-first visualization path for repeated-rollout
datasets. The key object is the prompt prefill hidden state, not rollout text.

## Saved Files That Are Sufficient

For a prompt-profile dataset build, the following files are enough to regenerate the
plot without another rollout pass:

- `manifest.json`
- `train/shard-*.pt`
- `test/shard-*.pt`
- `diagnostics/train_prompt_profile.jsonl`
- `diagnostics/test_prompt_profile.jsonl`
- `diagnostics/prompt_rollout_archive.jsonl`

What each file contributes:

- the shard files store the saved prefill activations plus sample IDs;
- the prompt-profile JSONLs store prompt-level aggregate rates such as `p_cap`,
  `p_loop`, and `mean_relative_length`;
- the rollout archive stores per-rollout labels (`cap_hit`, `loop_flag`,
  `finish_reason`, rollout length) and completion text so correctness can be
  reconstructed for multiple-choice tasks.

## Scripts

Exporter:

```bash
python scripts/export_prefill_activation_projection.py \
  --data-dir /path/to/prompt_profile_dataset/data \
  --out-dir outputs/gpqa_prefill_activation_projection/export
```

Renderer:

```bash
python3 scripts/render_prefill_activation_projection.py \
  --summary-json outputs/gpqa_prefill_activation_projection/export/projection_summary.json \
  --prompt-csv outputs/gpqa_prefill_activation_projection/export/prompt_projection.csv \
  --rollout-csv outputs/gpqa_prefill_activation_projection/export/rollout_projection.csv \
  --out-dir outputs/gpqa_prefill_activation_projection/figures
```

Default behavior:

- slice the saved stacked prefill tensor to the final layer (`last_layer`);
- fit one shared 2D PCA plane across all prompts;
- export one prompt table and one repeated-rollout table;
- render separate prompt-level and rollout-level panels on the same fixed plane.

## GPQA Pilot Read

On the saved `48`-prompt / `480`-rollout GPQA pilot from 2026-03-21:

- the first two PCA axes explain `15.9%` and `9.7%` of prompt variance;
- correctness is the clearest large-scale pattern in the plane;
- max-length risk is visible but fragmented across multiple prompt islands rather
  than one single failure cluster;
- loop risk is broader than max-length risk and remains highly mixed with
  non-loop prompts, even though every cap hit still lies inside the loop set on
  this slice;
- because repeated rollouts share one prefill coordinate per prompt, the plot is
  a prompt-propensity view, not a within-prompt branching view.

Three high-`p_cap` prompts (`p_cap >= 0.5`) occupy three distinct regions in the
plane rather than collapsing into one cluster, which is the concrete reason to
describe the `max_length` structure as fragmented rather than cleanly separable.
