# Attention-Residual PCA Across Repeated N-Grams - 2026-04-25

Last updated: 2026-04-26

## Active First Experiment

The first deeper-attention experiment is a concrete extension of the existing
loop-trigger-attention analysis:

```text
For each selected looped rollout, plot the trajectory of A_i = a^L_{b_i}
and P_i = h^{L-1}_{b_i} + a^L_{b_i} across repeated n-gram occurrences.
```

Here `a^L_{b_i}` is the final-layer attention residual write, and
`P_i = h^{L-1}_{b_i} + a^L_{b_i}` is the post-attention residual state entering
the final MLP block.

The v1 experiment compares two strict rollout-level buckets:

- looped rollouts that are still correct and in budget;
- looped rollouts that hit max length and are wrong.

The source of truth is the current W17 `rollout_bundle.v1` rebuild, not the
older March prompt-profile archives.

## Dataset And Rollout Selection

Run this selection independently for every retained dataset and thinking mode:

- `LiveCodeBench`
- `TACO-hard`
- `MATH level-5`
- `Omni-MATH >= 7`
- thinking `on`
- thinking `off`

For each dataset/mode bundle, select up to ten rollout-level examples:

- `loop_correct_in_budget`: up to `5` rollouts with
  - `rollout.loop_flag == true`
  - `rollout.max_length_hit == false`
  - rollout correctness is true
- `loop_wrong_max_length`: up to `5` rollouts with
  - `rollout.loop_flag == true`
  - `rollout.max_length_hit == true`
  - rollout correctness is false

Keep these filters strict. If a dataset/mode has fewer than `5` qualifying
rollouts in either bucket, use all available rows in that bucket and mark the
bucket as underfilled. Do not backfill with near misses.

Correctness comes from `rollout.grading`:

- code tasks (`LiveCodeBench`, `TACO-hard`): `grading.passed`
- math tasks (`MATH level-5`, `Omni-MATH >= 7`): `grading.correct`

Rows with missing grading are not eligible until CPU-only finalization has
filled the grading field.

## 1. Repeated N-Gram Setup

Take one selected rollout with completion token sequence

```text
y_0, y_1, ..., y_{T-1}.
```

Use the saved loop detector settings:

```text
n = 30
k = 20
```

The saved `rollout.loop_trigger.ngram_token_ids` defines the repeated 30-gram

```text
G = (g_0, g_1, ..., g_29).
```

Rescan the full `completion_token_ids` and record every start position where
that exact 30-gram appears:

```text
s_1 < s_2 < ... < s_m,
```

where

```text
(y_{s_i}, y_{s_i + 1}, ..., y_{s_i + 29}) = G.
```

Do not limit the trajectory to the first `20` starts saved by the original
trigger detector. The saved trigger is only the detector event; this PCA
experiment uses all later repeats too.

For each occurrence, define the completion-relative boundary:

```text
b_i_completion = s_i - 1.
```

The model-position boundary is:

```text
b_i_model = prompt_len + s_i - 1.
```

When `s_i == 0`, keep the point. In that case the boundary is the final prompt
token:

```text
b_i_model = prompt_len - 1.
```

This is the autoregressive state that predicts the first token of the repeated
30-gram:

```text
y_{s_i} = g_0.
```

So the sequence

```text
b_1, b_2, ..., b_m
```

gives the repeated loop-entry boundaries for this rollout.

## 2. Transformer-Block Notation

Use the final pre-norm Qwen3 decoder block. Let
`h_t^{ell-1}` be the residual stream at position `t` before layer `ell`.

For the final layer `L`, the normalized attention input is

```text
u_t^L = LN_1^L(h_t^{L-1}).
```

The final-layer attention residual write is

```text
a_t^L = Attn^L(u_{\le t}^L).
```

This is the actual self-attention output after the output projection, added
back into the residual stream.

The post-attention residual state is

```text
p_t^L = h_t^{L-1} + a_t^L.
```

The final MLP acts on the normalized version of that state:

```text
m_t^L = MLP^L(LN_2^L(p_t^L)).
```

The final residual output of the block is

```text
h_t^L = p_t^L + m_t^L.
```

The two main objects are therefore:

```text
A_i = a^L_{b_i}
```

and

```text
P_i = p^L_{b_i} = h^{L-1}_{b_i} + a^L_{b_i}.
```

`A_i` asks:

```text
How is the final-layer attention write changing across repeated n-grams?
```

`P_i` asks:

```text
How is the actual residual state entering the final MLP changing across
repeated n-grams?
```

These differ by the incoming residual stream:

```text
P_i - A_i = h^{L-1}_{b_i}.
```

If `A_i` and `P_i` show different geometry, that tells us whether the movement
is mainly from the final attention write or from the accumulated prior residual
state.

## 3. Primary Per-Rollout PCA

For a single rollout, construct the attention-output dataset

```text
X_A = [
  A_1^T
  A_2^T
  ...
  A_m^T
] in R^{m x d}.
```

Center it:

```text
Atilde_i = A_i - mean(A).
```

Compute the top two PCA directions:

```text
v_1^A, v_2^A in R^d.
```

Plot the 2D coordinates

```text
c_i^A = (<Atilde_i, v_1^A>, <Atilde_i, v_2^A>).
```

This gives a 2D trajectory

```text
c_1^A, c_2^A, ..., c_m^A.
```

Draw arrows

```text
c_i^A -> c_{i+1}^A
```

so the plot shows how the attention output evolves over repeated appearances
of the same 30-gram.

Then do the same for the post-attention residual state:

```text
X_P = [
  P_1^T
  P_2^T
  ...
  P_m^T
] in R^{m x d}.
```

Center:

```text
Ptilde_i = P_i - mean(P),
```

compute top PCA directions

```text
v_1^P, v_2^P,
```

and plot

```text
c_i^P = (<Ptilde_i, v_1^P>, <Ptilde_i, v_2^P>).
```

For each rollout, the two core local plots are:

```text
{a^L_{b_i}}_{i=1}^m
```

and

```text
{h^{L-1}_{b_i} + a^L_{b_i}}_{i=1}^m.
```

## 4. What To Color Or Annotate

For each boundary `b_i`, attach metadata.

The simplest annotation is occurrence index:

```text
i = 1, 2, ..., m.
```

Coloring by `i` shows whether the trajectory drifts monotonically through the
loop.

Also annotate the repeat probability when practical. Let `logit_{b_i}(v)` be
the next-token logit for token `v` at position `b_i`. Since the next repeated
30-gram begins with `g_0`, define

```text
q_i = P(y_{s_i} = g_0 | prefix through b_i).
```

Equivalently, define a repeat logit margin:

```text
M_i = logit_{b_i}(g_0) - max_{v != g_0} logit_{b_i}(v).
```

This separates two cases:

```text
the model strongly favors continuing the repeated n-gram
```

versus

```text
the model's boundary state is already less committed to the repeat.
```

For this first experiment, this annotation is a per-boundary behavioral readout
for both selected rollout buckets.

## 5. Interpretation Of The Two Plots

Compare the `A` plot and the `P` plot within each rollout and across the two
selected outcome buckets.

### Case 1: `A_i` and `P_i` both separate by bucket

Then the final-layer attention write itself is probably carrying bucket-relevant
signal, and that signal survives after adding the incoming residual stream:

```text
p^L_{b_i} = h^{L-1}_{b_i} + a^L_{b_i}.
```

This is the cleanest evidence that the final-layer attention output is involved
in the difference between correct in-budget loops and wrong max-length loops.

### Case 2: `A_i` does not separate, but `P_i` does

Then the final attention write is not the main source of the visible movement.
The accumulated residual input

```text
h^{L-1}_{b_i}
```

is probably already different before the last attention layer. The relevant
signal may come from earlier layers, earlier MLPs, or the residual stream state
built up over repeated tokens.

### Case 3: `A_i` separates, but `P_i` does not

Then the final attention write is moving, but it may be small relative to, or
partially canceled by, the incoming residual stream:

```text
a^L_{b_i}
```

has an interesting trajectory, but

```text
h^{L-1}_{b_i} + a^L_{b_i}
```

does not.

This would suggest that final attention is doing something dynamic, but not
enough to move the actual state entering the MLP.

### Case 4: neither `A_i` nor `P_i` separates by bucket

Then this particular final-layer attention-residual view may not explain the
bucket contrast. The next object to inspect would be the final residual state
after the last MLP:

```text
h^L_{b_i} = p^L_{b_i} + m^L_{b_i}.
```

Other possible explanations include earlier layers, final layer norm,
unembedding directions, or sampling effects.

## 6. Recommended Plot Set

For each selected rollout, produce three panels if possible.

### Panel 1: final-layer attention write

```text
A_i = a^L_{b_i}.
```

This is the most attention-specific plot.

### Panel 2: post-attention residual / pre-MLP residual

```text
P_i = h^{L-1}_{b_i} + a^L_{b_i}.
```

This is the state handed to the final MLP residual sublayer.

### Panel 3: optional final residual after MLP

```text
H_i = h^L_{b_i}.
```

This is closest to the final logits, after the final block has completed and
before the model's final norm.

The two core plots are Panel 1 and Panel 2. Panel 3 is useful if Panel 1 and
Panel 2 do not explain the bucket contrast.

## 7. Per-Rollout PCA Versus Shared PCA

Do PCA separately for each rollout to see the within-rollout trajectory.

For rollout `r`, this gives local PCA coordinates:

```text
c_{i,r}^A
c_{i,r}^P
```

Local PCA axes are not directly comparable across rollouts. The PC1 direction
in one rollout is not necessarily the same direction as PC1 in another rollout.

Therefore use two modes.

### Local PCA

For each rollout independently:

```text
{A_{i,r}}_{i=1}^{m_r}
{P_{i,r}}_{i=1}^{m_r}
```

This answers:

```text
What is the trajectory shape inside this rollout?
```

### Global PCA

Pool vectors across the selected rollouts within each dataset/mode:

```text
All_A = {A_{i,r}: r in selected rollouts, i = 1, ..., m_r}
```

or

```text
All_P = {P_{i,r}: r in selected rollouts, i = 1, ..., m_r}.
```

Fit one shared PCA basis on the pooled set. Then plot all selected rollouts in
the same 2D space, colored by bucket:

```text
loop_correct_in_budget
loop_wrong_max_length
```

Local PCA is better for discovering individual trajectory shapes. Global PCA is
better for comparing whether the two rollout buckets occupy different regions.

## 8. Important Sanity Checks

First, track explained variance:

```text
lambda_1 + lambda_2
```

relative to total variance. If the top two PCs explain little variance, the 2D
plot may be visually misleading.

Second, plot vector norms:

```text
||A_i||_2
||P_i||_2
```

A PCA trajectory can be dominated by norm growth rather than directional
change. As a robustness check, also try direction-normalized vectors:

```text
A_i / ||A_i||_2
P_i / ||P_i||_2
```

Third, always annotate the repeat logit margin when practical:

```text
M_i = logit_{b_i}(g_0) - max_{v != g_0} logit_{b_i}(v).
```

Fourth, do not interpret local PCA orientation too strongly. In one rollout,
"left" or "right" has no absolute meaning. What matters is whether the
trajectory forms a stable cluster, a monotone drift, a loop, or a final jump.

Fifth, keep underfilled buckets visible. A dataset/mode with fewer than five
strictly qualifying rows is still useful, but it should not be visually
presented as a balanced 5-vs-5 comparison.

## 9. Implementation Plan

Build this as a new mechanism-analysis script rather than overloading the
existing attention-mass report:

```text
scripts/mechanism_analysis/analyze_attention_residual_pca_ngrams.py
```

Reuse the existing logic from
`scripts/mechanism_analysis/analyze_loop_trigger_attention.py` for:

- reading `rollout_bundle.v1` rows through `probe.bundle_io`;
- replaying prompt plus completion prefixes;
- Qwen3-specific attention internals and final-layer indexing;
- Slurm sharding shape if the full run needs multiple GPUs.

The new script should add:

- strict two-bucket selection from finalized grading fields;
- rescanning all occurrences of the saved triggering n-gram;
- final-layer hooks that capture `h^{L-1}`, `a^L`, `p^L`, and optionally `h^L`
  only at selected boundary positions;
- repeat-token probability or logit-margin extraction when practical;
- local and pooled PCA exports;
- a selection ledger and underfilled-bucket report.

Suggested output root:

```text
outputs/weeks/2026-W17/attention_residual_pca_ngrams_20260426/
```

Required output files:

- `analysis_config.json`
- `selection_ledger.jsonl`
- `selection_summary.csv`
- `pca_points.jsonl`
- `pca_summary.csv`
- per-rollout local PCA figures for `A` and `P`
- per-dataset/mode pooled PCA figures for `A` and `P`

`analysis_config.json` should record source bundles, model id, loop detector
settings, selection limits, thinking modes, and git commit if available.

## 10. Acceptance Checks

Before running the full analysis:

1. Run a selection-only dry run on each available dataset/mode bundle.
2. Confirm every selected rollout has `loop_flag == true` and non-null
   `loop_trigger`.
3. Confirm the n-gram rescan finds at least the saved trigger starts.
4. Confirm `s_i == 0` maps to `prompt_len - 1`, not to a dropped point.
5. Run one model replay smoke on a single selected rollout and verify shapes:
   - `A`: `[num_boundaries, hidden_size]`
   - `P`: `[num_boundaries, hidden_size]`
   - optional `H`: `[num_boundaries, hidden_size]`
6. Confirm local PCA writes a figure and explained-variance values for that one
   rollout.

Done means the report can say, for each dataset/mode, whether the correct
in-budget looped rollouts and wrong max-length looped rollouts show visibly
different final-layer attention-write or post-attention-residual trajectories
across repeated n-gram occurrences.

## 11. Final Experiment Definition

For every retained dataset and thinking mode:

1. Read the current finalized `rollout_bundle.v1` bundle.
2. Select up to `5` `loop_correct_in_budget` rollouts and up to `5`
   `loop_wrong_max_length` rollouts using strict criteria.
3. For every selected rollout, identify the saved triggering 30-gram:

```text
G = (g_0, ..., g_29).
```

4. Rescan the full completion to record every occurrence start:

```text
s_1, ..., s_m.
```

5. Define model-position boundaries:

```text
b_i = prompt_len + s_i - 1,
```

with `s_i == 0` mapped to `prompt_len - 1`.

6. At each boundary, extract:

```text
A_i = a^L_{b_i}
P_i = h^{L-1}_{b_i} + a^L_{b_i}.
```

7. Optionally also extract:

```text
H_i = h^L_{b_i}.
```

8. Run local PCA per rollout on `{A_i}` and `{P_i}` separately.
9. Run pooled PCA per dataset/mode on `{A_i}` and `{P_i}` separately.
10. Compare the two buckets:

```text
loop_correct_in_budget
loop_wrong_max_length
```

The first object shows how the final attention output residual evolves. The
second shows how the actual state entering the last MLP evolves.
