author: Zhi
date: 2026-04-03

This document is a plan to understand where loop and max-length come from.

clarification: we call both loop and max-length hits as degenerate rollouts, because our previous statistics collection experiments show that max-length hits form a subset of looped rollouts, and that mean rollout length plus binary length threshold is a good proxy for looped rollouts, and are easier to predict at prefill time.

so we care about where these degenerate rollouts come from.

## Plan

The hypothesis is the base model should barely have any degenerate rollouts. This is because the base model in a way learns only the semantics and syntax of the language, and should behave like a human.

So we do this:
1. take Olmo-3-7B, Olmo-3-7B-Instruct-SFT, and Olmo-3-7B-Instruct as our axis of progression from base -> SFT -> RLVR.
2. for each checkpoint we collect stats like we did with Qwen3, refer to code and our previous threads on this "statistics-collection" module, this should be a 100% reproducible process.
3. replace the visualization component with Sankey plot instead of the current bar plot when appropriate.
4. collect stats for all three checkpoints using our GPU node.
5. compare the stats across the three checkpoints.
6. draw conclusions on our hypothesis. I expect to prove or disprove it, meaning we need to check and plot the progression of each metric across the three checkpoints. For this kind of progression analysis, do not use Sankey plot, but rather a line plot.