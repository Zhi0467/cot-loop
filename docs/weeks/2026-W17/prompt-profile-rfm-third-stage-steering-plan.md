# Prompt-Profile RFM Third-Stage Steering Grand Plan

Last updated: 2026-04-23 23:28 UTC

This is the only active steering-stage plan. It replaces the earlier steering
notes, which mixed design, status, queue notes, and superseded assumptions.

## One-Line Object

The steering stage tests whether mode-local, benchmark-local RFM concept
vectors trained from prompt-prefill activations can causally reduce long-rollout
and loop risk under the same model, dataset, prompt surface, thinking mode, and
decode contract that produced the rollouts used to train those vectors.

The main rule is simple:

`stats collection -> prompt-level materialization -> probe/RFM training -> vector export -> steering`

must close inside the same `(dataset, thinking mode)` path. No cross-mode
steering row is stage evidence.

## Current Ground Truth

### Active data foundation

The active foundation is the paired four-dataset rollout-stat rebuild described
in `main-four-dataset-rollout-rebuild-2026-04-23.md`.

Datasets:

| Dataset | Active size | Thinking modes |
| --- | ---: | --- |
| `LiveCodeBench` | full `1055` | `on`, `off` |
| `TACO-hard` | `1000 / 5536` | `on`, `off` |
| `MATH level-5` | `1000 / 2304` | `on`, `off` |
| `Omni-MATH >= 7` | full `916` | `on`, `off` |

Shared collection contract:

- model: `Qwen/Qwen3-1.7B`
- temperature: `0.2`
- generations per prompt: `10`
- max tokens / max model length: as recorded in the collector manifest
- per-prompt archives must keep prompt text, prompt token ids, rollout text,
  completion token ids, raw row metadata, correctness where available, length
  metrics, loop flags, and cap-hit flags

The current live output root still has the historical name
`main_five_dataset_rebuild_full_or_1k_20260423`, but the active manifest is the
four-dataset surface above. `LiveCodeBench-extra` is dropped because it is a
strict subset of `LiveCodeBench release_v6`, not a separate benchmark lane.
The `LiveCodeBench` row is intentionally full `release_v6` (`1055`) in the
current suite: the code leaves `max_samples` unset for that dataset, while
larger filtered pools such as `TACO-hard` and `MATH level-5` are capped at
`1000`.

### Stage exclusions

The stage starts from the rebuilt mode-local rollout archives. Anything outside
that chain can be kept only as a smoke, infrastructure receipt, or historical
debugging artifact.

- reduced decode-budget steering pilots
- last-prompt-token-only steering pilots
- prompt-subset steering tables
- `LiveCodeBench-extra` as an independent dataset
- mixed thinking-on / thinking-off averaged vectors

## Claims To Keep Separate

The stage has four separable claims. Do not merge them in prose or reporting.

1. Detector quality:
   RFM must be compared against prompt-only, activation-linear, and activation
   MLP baselines on the same mode-local materialized prompt object.
2. Direction quality:
   Exported vectors need their own stability and projection diagnostics. A good
   detector is not automatically a good steering direction.
3. In-distribution steering:
   A mode-local vector bundle must reduce long-rollout / loop / cap-hit metrics
   relative to `no_steer` and control directions without unacceptable accuracy
   loss.
4. Transfer:
   Average-vector transfer is a later experiment and must stay mode-local.
   There is no mixed thinking-on / thinking-off average vector.

## Mode-Local Pipeline

Run the following chain separately for each dataset and thinking mode.

### P0: Stats receipt

Use the rebuilt collector archive as the only source of truth. The receipt must
include:

- dataset name and row filter
- thinking mode requested and resolved
- prompt formatter / LM style
- model and tokenizer revisions when available
- generation config
- prompt ids and record ids
- prompt text and prompt token ids
- completion text and completion token ids
- correctness where a grader exists
- per-rollout length, relative length, cap hit, max-length hit, loop flag, and
  first-loop prefix length where available

### P1: Prompt-level materialization

Materialize prompt-level targets from the mode-local archive. Do not reuse
labels from any earlier bundle or from the opposite thinking mode.

Required prompt-level targets:

- `majority_s_0.5`: prompt is positive when a strict majority of rollouts have
  `relative_length >= 0.5`
- `mean_relative_length`
- `p_loop`
- `p_cap` / cap-hit rate where the archive supports it

`majority_s_0.5` is a long-rollout / budget-usage risk proxy, not a pure loop
label. The report must keep that wording.

Dataset admission gate for steering:

- the mode-local train or fit-train object must have at least `10%` positives
  under `majority_s_0.5`
- if a dataset/mode falls below that gate, keep it diagnostic-only unless
  Wangzhi explicitly accepts a sparse-positive steering run

### P2: Detector and baseline table

Train the mode-local RFM as a sibling detector, not as a replacement for the
existing probe surface.

For each admitted `(dataset, mode)` object, report:

- prompt-only baselines:
  - prompt length
  - strongest available prompt-shape baseline
- activation baselines:
  - linear last-layer
  - linear ensemble
  - MLP last-layer
  - MLP ensemble
- RFM:
  - one layerwise detector per block
  - selected layer and selected hyperparameters
  - `PR-AUC`, `ROC-AUC`, prevalence, and threshold diagnostics

Primary detector metric is `PR-AUC`; `ROC-AUC` and threshold metrics are
secondary context.

### P3: Vector export and direction diagnostics

Export one signed vector per transformer block from the mode-local RFM bundle.

Sign convention:

- `+v_l` means higher predicted `majority_s_0.5` risk at block `l`
- `-v_l` is the anti-risk direction
- for spherical steering, the anti-risk pole is `-normalize(v_l)`

Required diagnostics before steering interpretation:

- vector checksum and artifact hash for every block
- selected RFM hyperparameters and sign rule
- held-out one-dimensional projection score `h_l^T v_l`
- bootstrap or seed cosine stability against the full-fit reference vector
- cross-layer cosine structure
- cross-benchmark cosine only after at least two admitted datasets exist inside
  the same thinking mode

Interpretation gate:

- if fewer than `20 / 28` layers clear mean bootstrap cosine `>= 0.7`, the
  steering table may still run as infrastructure, but it cannot be presented as
  a stable-vector causal read without an explicit caveat
- if the selected detector layer is unstable, report that separately instead of
  hiding it in an aggregate direction table

## Steering Contract

This contract is fixed for the first real third-stage steering table.

Hook and timing:

- hook site: `prefill_layer_output_all_tokens`
- intervention timing: prefill forward pass only
- token scope: every prompt token, not only the last prompt token
- block scope: every block uses its own block-specific vector `v_l`
- do not add top-`k` layer selection, probe gating, or an online controller in
  the first pass

Conditions:

| Condition | Meaning |
| --- | --- |
| `no_steer` | matched baseline |
| `minus_v_linear` | additive anti-risk direction |
| `plus_v_linear` | additive sign-flip / risk direction |
| `random_linear` | additive random-direction control |
| `minus_v_spherical` | norm-preserving anti-risk pole |
| `plus_v_spherical` | norm-preserving sign-flip pole |
| `random_spherical` | norm-preserving random-direction control |

First-pass strengths:

- spherical: `t = 0.3`
- linear: `epsilon = 0.2`
- no strength sweep until the seven-condition table exists for the relevant
  `(dataset, mode)` object

Decode budget:

- use the full source-manifest decode contract for the row
- never report a reduced-cap table as stage evidence
- if a smaller cap or prompt subset is needed for a smoke, label it as a smoke
  in the output directory, run record, and report

Required steering outputs:

- condition-level summary JSON
- prompt-level ledger
- `prompt_profile_rfm_steering_run.v1` record
- prompt ids and prompt-id hash
- vector artifact hash
- thinking mode
- hook site
- `t` and `epsilon`
- generation config
- grader version
- git commit and output root

Required metrics:

- `pass@1` / accuracy
- accuracy delta vs `no_steer`
- loop fraction
- max-length-hit fraction
- `>50%` budget fraction or stage-equivalent long-rollout fraction
- average generation length
- median generation length
- bootstrap interval for the key deltas when sample size supports it

Required intervention diagnostics:

- mean pre-intervention norm
- mean post-intervention norm
- norm-preservation error for spherical conditions
- mean starting angle to target/direction
- mean realized angular movement
- condition runtime and any timeout / cap-hit asymmetry

## Execution Order

### P0: Finish the rebuilt stats archives

Close the eight retained jobs from the current four-dataset suite:

- `2875` `LiveCodeBench` thinking `on`
- `2880` `LiveCodeBench` thinking `off`
- `2877` `TACO-hard` thinking `on`
- `2882` `TACO-hard` thinking `off`
- `2878` `MATH level-5` thinking `on`
- `2883` `MATH level-5` thinking `off`
- `2879` `Omni-MATH >= 7` thinking `on`
- `2884` `Omni-MATH >= 7` thinking `off`

Any repair must preserve the same dataset, row filter, sampling config, and
thinking tag. First-row failures are launch/runtime bugs, not science results.

### P1: Build mode-local prompt-profile objects

For each finished stats archive:

1. recompute prompt-level targets from the archive;
2. write split manifests with prompt ids, prompt text hashes, and positive
   counts;
3. record whether the `>= 10%` positive-rate gate passed.

### P2: Train mode-local detectors and export vectors

For each admitted object:

1. train prompt-only and activation baselines;
2. train layerwise RFM;
3. export signed block-specific vectors;
4. run direction diagnostics;
5. write one detector/vector summary table.

### P3: Run in-distribution steering

Start with `LiveCodeBench` because it has the cleanest grader and the least
ambiguous code-generation evaluation path. Build the vectors from the rebuilt
mode-local archives before steering.

For each mode:

1. run the seven-condition table on the full held-out prompt split;
2. verify prompt ids, thinking mode, hook site, vector hash, and full decode
   budget in the run records;
3. report steering metrics and intervention diagnostics together.

After `LiveCodeBench`, promote `TACO-hard`, `MATH level-5`, and
`Omni-MATH >= 7` only path-by-path after their rebuilt mode-local materialized
objects pass the admission gate and have detector/vector receipts.

### P4: Optional controls after the first seven-condition table

Only after the base table exists:

- shuffled-label linear and spherical controls
- layer-restriction ablations
- strength sweeps for `t` or `epsilon`
- controller / probe-gated steering

These are follow-ups, not substitutes for the first fixed-contract table.

### P5: Mode-local transfer

Average-vector transfer starts only after at least two non-identical datasets
inside the same thinking mode have:

- passed the positive-rate gate;
- produced detector/vector bundles;
- passed direction diagnostics;
- produced readable in-distribution steering receipts.

The average vector is built layerwise within one thinking mode. Never average
thinking-on and thinking-off vectors together.

## Positive Steering Claim Gate

A steering row can be called positive only if all of these are true on the same
mode-local object:

1. `minus_v_*` improves loop / max-hit / long-rollout metrics relative to
   `no_steer`.
2. The improvement is better than the matched random-direction control.
3. The sign-flip `plus_v_*` does not show the same improvement.
4. Accuracy does not show an unacceptable drop relative to `no_steer`; if the
   baseline accuracy is low, report raw counts and bootstrap uncertainty rather
   than only percentages.
5. The run used full held-out prompts and the full decode contract, or it is
   explicitly labeled as a pilot and not used for the stage claim.
6. The vector bundle is mode-local and benchmark-local, with prompt ids and
   vector hashes matching the steering run record.

If those conditions fail, the result is still useful, but the language should be
"negative steering read", "implementation receipt", or "diagnostic control", not
"steering works".

## Backlog

Immediate:

1. Finish all eight rebuilt stats archives and verify their prompt-rollout
   archive schemas.
2. Materialize mode-local prompt-profile objects and admission-gate tables.
3. Train `LiveCodeBench` thinking-on and thinking-off RFMs from rebuilt archives.
4. Export the two `LiveCodeBench` mode-local vector bundles and direction
   diagnostics.
5. Run the corrected seven-condition `LiveCodeBench` steering table in thinking
   `on`.
6. Run the corrected seven-condition `LiveCodeBench` steering table in thinking
   `off`.

Next:

1. Repeat detector/vector export for admitted `TACO-hard`, `MATH level-5`, and
   `Omni-MATH >= 7` mode-local objects.
2. Run in-distribution steering only for paths that pass admission and vector
   diagnostics.
3. Write one steering report that contains the mode-local detector table,
   direction diagnostics, and steering table in one artifact.
4. Only then open transfer or controller variants.

Documentation:

1. Keep this file as the only steering-stage plan.
2. Keep `prompt-profile-rfm-artifact-schema-2026-04-21.md` as the artifact
   schema reference.
3. Do not create another steering status document unless it replaces this file.

## What Not To Claim

- Do not use any vector bundle that was not exported from the rebuilt
  mode-local archive for the same `(dataset, thinking mode)` path.
- Do not claim `LiveCodeBench-extra` is a separate benchmark.
- Do not call `majority_s_0.5` a pure loop label.
- Do not treat detector `PR-AUC` ranking as a steering result.
- Do not treat reduced-cap or prompt-subset runs as full-stage evidence.
- Do not mix thinking-on and thinking-off vectors in transfer.
- Do not cite trigger-attention reports as proof that the RFM direction is
  mechanistically correct.
