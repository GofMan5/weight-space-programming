# Experiment and acceptance protocol

## Before creating candidates

Freeze source hash/revision, architecture/config/tokenizer, runtime/library versions, hardware, input/preprocessing, split IDs, seeds, decoding and resource limits. Use an existing evaluator where possible. Save a runnable command and baseline per-example results.

Specify one primary endpoint and a practical improvement threshold, plus regression tolerances. Choose these for the task, not a universal magic percentage. For a deterministic bug, the original should fail a meaningful example and the candidate should pass it plus counterexamples. For stochastic performance, plan paired seeds/examples and enough repetitions to distinguish noise from the useful effect.

Separate data into discovery/practice, candidate selection/validation, and a final held-out evaluation. Record dataset revision, IDs and overlap checks. If final evaluation changes your edit, label that set as development data and obtain new held-out evidence before claiming generalization. Many candidate trials also increase selection bias.

## Gates

| Gate | Required evidence | On failure |
|---|---|---|
| Original recoverable | Verified backup hashes or verified pinned immutable artifact | Stop writes until recoverable |
| Structural integrity | Same intended keys/shapes/dtypes, finite values, allowed change names and slices, consistent shards/config | Reject candidate |
| Reload | Fresh process loads actual saved model; ties and runtime configuration preserved | Reject candidate |
| Smoke behavior | Valid outputs, expected I/O and sane latency/memory | Reject candidate |
| Target behavior | Same inputs/settings, sufficient examples and measured improvement | Reject or mark inconclusive |
| Regressions | Relevant non-target behavior within predeclared tolerances | Reject candidate |
| Rollback | Restore into a new path, hashes match, original probe replays | Do not promote |

The included `tensor_audit.py compare` gates structural integrity and allowed tensor names. It does not test slice intent, tied storage, model behavior, relative-norm thresholds or readiness for deployment. Implement task-specific checks for those. A zero exit status from the tool is not a release decision.

## Match metrics to the failure

For reasoning loops, record task accuracy, completion/answer-extraction rate, repetitive-span rate, generated-token distribution and token-limit hits. A response that merely stops sooner is not necessarily more correct. Keep max tokens and sampling identical for comparisons.

For classifiers, use calibration and class/edge-case performance as well as total accuracy. For policies, use episode outcomes, legal actions, failure rates, paired environment seeds and opponent/starting-side controls. For generative models, include output validity and a task-appropriate quality evaluation; a weight statistic alone does not measure quality.

For collateral checks, use a compact but relevant suite: unrelated domains/tasks, instruction following, supported languages, context lengths and rare inputs, as appropriate. Preserve applicable model safety behavior as part of regressions. Record what the suite does not cover.

## Statistics and reporting

Keep sample-level results. Report counts and paired deltas, not only rounded percentages. Bootstrap over independent problems/episodes or use another justified paired method; multiple samples of one prompt are clustered, not independent new problems. Distinguish measurement variability from nondeterministic runtime differences. If the confidence interval includes no useful gain, report that the improvement is not established.

Record every candidate's parent hash, hypothesis, patch hash and evaluation status, including rejected candidates. End at the budget or lack of supported progress. Do not silently extend inference/paid compute budgets or replace the task with training.

## Reproduction record

Use JSON or the project's existing experiment format. Include: source/candidate hashes; exact commands as argument arrays and cwd; input IDs; tool/runtime versions; seed and decoding settings; exit statuses; literal stdout/stderr paths; metric definitions and values; thresholds; sample counts; uncertainty; changed tensor names/slices/count/fraction; numerical norms; elapsed time and inference budget; acceptance decision; restore command and proof. Keep secrets out. Prefer local restricted raw data and redacted shareable summaries when examples contain private content.

If no inference is permitted, finish with structural evidence and explicitly unverified behavioral predictions. In that mode, do not promote an untested candidate as a proven repair. The same distinction applies when hardware, model implementation, tokenizer or evaluation data are missing.
