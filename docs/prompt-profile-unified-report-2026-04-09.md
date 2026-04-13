# Unified Prompt-Profile Report

Last updated: 2026-04-09 22:55 UTC

## Executive Summary

- This is the current single prompt-profile report surface for the thread.
  - Regression stays on the natural prompt-disjoint split with natural sampling.
  - Classification stays on balanced train with natural test.
  - The same report now carries setup, predictor views, metrics, results, and the plain-English metadata interpretation.
- Regression headline:
  - `mean_relative_length` is useful as a screening score, not as a clean calibrated regressor.
  - On `top_20p_capture`, ensemble beats prompt length on `GPQA`, `MMLU-Pro`, and `LiveCodeBench`.
- Classification headline:
  - `majority_s_0.5` is still the cleaner deployment-facing head.
  - The best single global activation surface is still `ensemble h256 d1`.
- Metadata headline:
  - The old reported metadata baseline is only a train-fit prompt-length baseline on this fixed-budget run.
  - Prompt-only cues are strong mostly because prompt surface structure already carries a lot of the same long-completion signal.

## Main Artifact

- Unified PDF: `outputs/prompt_profile_unified_report_20260409/prompt_profile_unified_report_20260409.pdf`
- Unified TeX: `outputs/prompt_profile_unified_report_20260409/prompt_profile_unified_report_20260409.tex`
- Figures: `outputs/prompt_profile_unified_report_20260409/figures/`

## Current Read

- Natural regression:
  - `GPQA`: prompt length `0.168`, ensemble `0.247 +/- 0.018`
  - `AIME`: prompt length `0.306`, ensemble `0.305 +/- 0.017`
  - `MATH-500`: prompt length `0.290`, ensemble `0.262 +/- 0.058`
  - `MMLU-Pro`: prompt length `0.292`, ensemble `0.355 +/- 0.025`
  - `LiveCodeBench`: prompt length `0.248`, ensemble `0.340 +/- 0.005`
- Balanced classification:
  - `GPQA`: prompt length `0.066`, best cheap prompt feature `Newline count = 0.167`, ensemble `0.583`
  - `AIME`: prompt length `0.898`, best cheap prompt feature `Dollar count = 0.937`, ensemble `0.912`
  - `MATH-500`: prompt length `0.100`, best cheap prompt feature `Dollar count = 0.117`, ensemble `0.166`
  - `MMLU-Pro`: prompt length `0.110`, best cheap prompt feature `Newline count = 0.506`, ensemble `0.212`
  - `LiveCodeBench`: prompt length `0.576`, best cheap prompt feature `Character length = 0.590`, ensemble `0.714`

## Interpretation

- Prompt length predicts completion length here mostly because, on these datasets, longer prompts usually mean the prompt already contains more work.
- That story is strongest on `AIME`, `MATH-500`, and much of `LiveCodeBench`.
- `GPQA` is the important exception: raw length is weak there, and prompt structure matters more than raw length.
- For classification, some of the same prompt cues carry over because the positive label is still largely asking whether rollouts cross about half the fixed budget.
- So the present results support lift over raw prompt length on some surfaces, but not yet a general lift over strong prompt-only controls.
