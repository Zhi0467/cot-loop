# Athena Length-Mechanism Context

Last updated: 2026-04-09 02:39 UTC

## Question Sent To Athena

Why does prompt length predict completion length at all on this natural regression surface, assuming there is no implementation bug?

Constraints:

- answer in plain English;
- stay on the literal input-output pair `prompt length -> completion length`;
- use the new length-mechanism note and summary JSON only;
- avoid drifting into the binary head unless strictly necessary.

## Main Athena Takeaways

- On this fixed setup, longer prompts are often the prompts that already contain more written work:
  - more math setup
  - more notation
  - more conditions to track
  - or a longer coding spec with more requirements
- Because the model and decode settings are fixed, that visible workload pattern stays stable enough to learn from prompt length alone.
- Raw length works best when "longer prompt" usually means "more of the same kind of task":
  - strongest on `AIME`
  - still real but milder on `MATH-500`
- Raw length weakens when prompts can be long for different reasons:
  - `GPQA` is the clearest case
  - the medium-long prompts are riskier than the very longest prompts there
- The safest summary Athena gave is:
  - prompt length works when it is standing in for plainly visible workload in the prompt
  - it works poorly when different prompt types end up with similar length

## Useful Athena Phrasing

This proves that, on this fixed regression setup, visible prompt size really does predict average completion length on test data, so the old prompt-length effect does not need a bug to explain it. It does not prove that prompt length by itself is what makes the model write more, and it does not support a rule like “longer prompt always means longer completion.”
