# Prompt-Profile Projection

Last updated: 2026-03-22 00:52 UTC

## Purpose

This note records the prompt-level activation-visualization path for repeated-rollout
prompt-profile datasets. The object is one prompt prefill activation vector, not one
rollout text embedding. Each plotted point is a prompt, and prompt labels come either
from majority vote over that prompt's rollouts or from prompt-level derived rates.

## Saved Files That Are Sufficient

For a saved prompt-profile dataset bundle, the required source files are:

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
- the rollout archive stores per-rollout `relative_length`, `cap_hit`, `loop_flag`,
  `finish_reason`, and completion text so prompt-majority labels and alternate
  thresholds can be recomputed without rerolling.

## What The Export Produces

`scripts/export_prompt_profile_projection.py` writes:

- `prompt_projection.csv`
- `projection_summary.json`

`scripts/render_prompt_profile_projection.py` renders:

- `prompt_binary_panels.png` / `.pdf`
- `prompt_continuous_panels.png` / `.pdf`
- `prompt_separability_summary.png` / `.pdf`

The binary figure uses one dot per prompt and includes:

- an unsupervised cluster panel on the shared PCA plane;
- prompt-majority `cap`, `loop`, and `finish_reason == length`;
- prompt-majority correctness when correctness can be reconstructed;
- prompt-majority `s_t` panels for the requested thresholds.

The continuous figure uses the same fixed PCA plane and colors prompts by:

- `p_cap`
- `p_loop`
- `finish_reason == length` rate
- `mean_relative_length`
- `correct_rate` when available
- `s_t` for each requested threshold

## Quantitative Separability

The exporter also writes a prompt-level separability summary:

- first fit PCA on the prompt prefill vectors;
- then choose one unsupervised KMeans partition on the 2D plane from `k = 2..6`
  by silhouette score;
- compare each binary label to that partition with prevalence, cluster-vote balanced
  accuracy, adjusted mutual information, and label-silhouette score;
- compare each continuous statistic with cluster `R^2`, 2D linear `R^2`, and the
  largest absolute Spearman correlation with either PCA axis.

This keeps the visual read and the quantitative read on the same fixed prompt plane.

## Scripts

Export:

```bash
python scripts/export_prompt_profile_projection.py \
  --data-dir /path/to/prompt_profile_dataset/data \
  --out-dir outputs/prompt_profile_projection/export \
  --tail-thresholds 0.5 0.6 0.9
```

Render:

```bash
python scripts/render_prompt_profile_projection.py \
  --summary-json outputs/prompt_profile_projection/export/projection_summary.json \
  --prompt-csv outputs/prompt_profile_projection/export/prompt_projection.csv \
  --out-dir outputs/prompt_profile_projection/figures
```

One-GPU Slurm entrypoint:

```bash
sbatch slurm/run_prompt_profile_projection.sbatch
```

The Slurm path builds the repeated-rollout prompt-profile bundle on one GPU, exports
the prompt-level projection tables, and renders panels only when `matplotlib` is
available in the runtime environment.

## GPQA Validation Read

On the saved `48`-prompt / `480`-rollout GPQA prompt-profile pilot:

- the 2D plane still exists, and the unsupervised prompt partition prefers `k = 4`;
- correctness aligns with that partition much more than prompt-majority loop or
  prompt-majority cap-hit;
- prompt-majority `s_0.5` and `s_0.6` are almost the same on this slice, so lowering
  the threshold changes prevalence only slightly here;
- the broad activation geometry still looks more like prompt correctness / prompt type
  than like one clean loop-failure cluster.
