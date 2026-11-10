# Eval report: anthropic

21 cases x 3 repeats, run 2026-11-10T15:28:17+00:00.
Overall pass rate: **94%**. Gate: **PASS**.

| Criterion | Pass rate | Applicable runs | Threshold |
|---|---|---|---|
| task_completion | 95% | 63 | 85% |
| groundedness | 96% | 27 | 90% |
| relevance | 100% | 27 | 90% |
| tool_accuracy | 100% | 48 | 90% |
| safety | 100% | 63 | 100% |
| failure_handling | 100% | 15 | 90% |

## Flaky cases

- delete-rejected
- msa-termination-notice
- unanswerable-sports

## Failed runs

| Case | Repeat | Criterion | Reason |
|---|---|---|---|
| unanswerable-sports | 0 | task_completion | expected the no-information reply |
| msa-termination-notice | 2 | groundedness | numbers not in cited chunks: ['24'] |
| unanswerable-sports | 2 | task_completion | expected the no-information reply |
| delete-rejected | 2 | task_completion | answer missing ['rejected'] |

