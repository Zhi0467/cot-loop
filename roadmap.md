# Roadmap - CoT Loop Detection

Last updated: 2026-03-05 18:45 UTC

Scope:
- Build and validate a probe pipeline for CoT loop detection across prefill and completion feature views.
- Quantify generalization across natural eval splits and OOD tasks.

## Milestone 1 - Pipeline and feature views
Status: done (2026-03-05 18:45 UTC)
Success criteria:
- Dataset builder supports prefill and completion feature views in a single pass.
- Training supports MLP depth and width sweeps with configurable dropout.
- Slurm launchers exist for k=5 three-view dataset build and ablation sweep.

## Milestone 2 - Consolidated findings
Status: done (2026-03-05 18:45 UTC)
Success criteria:
- Consolidated findings doc and PDF reflect the latest k=5 three-view results.
- PR #2 includes the related codebase updates and review fixes.

## Milestone 3 - Cross-dataset validation
Status: next (set 2026-03-05 18:45 UTC)
Success criteria:
- Replicate three-view results on additional evaluation sets or models.
- Compare posterior vs prefill robustness under varied token budgets.

## Milestone 4 - Deployment readiness
Status: future (set 2026-03-05 18:45 UTC)
Success criteria:
- Recommend a default feature view and probe configuration for routine use.
- Document performance tradeoffs, expected failure modes, and cost profile.
