# Data Layout

## Default Benchmark File

`data/aime_2024_2025.jsonl` remains the default local corpus for quick experiments and script defaults.

The current main-rollout rebuild also vendors one screened local math pool:

- `data/omni_math_ge7_screen_300.jsonl`
  - `300` raw Omni-MATH problems from the finished `>= 7` screening merge
  - each row keeps `_source_sample_id`, `problem`, `answer`, `difficulty`, `domain`, and `source`
  - use `question_field=problem`, `answer_field=answer`, and `prompt_format=chat_template`

## Expected Record Format (Per Line JSON)

```json
{
  "id": "AIME24I-1",
  "year": 2024,
  "contest": "AIME I",
  "problem": 1,
  "question": "Full problem statement text",
  "answer": "Final answer (string)"
}
```

Notes:
- `question` is used as the prompt text.
- `answer` is required whenever you want correctness grading and loop analysis with ground truth.

## Custom Local JSONL Datasets

For non-default JSONL files, include the prompt field referenced by `--prompt-field` in `scripts/build_probe_dataset.py`.

To keep backward compatibility with existing command defaults, local datasets that do not use `question` should be paired with an explicit `--prompt-field` and `--test-dataset` when used with defaults.
