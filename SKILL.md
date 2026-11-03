---
name: weight-space-programming
description: Analyze, inspect, diagnose, directly edit, construct, and test neural-network weights with reproducible comparisons and verified rollback. Use for checkpoint inspection, tensor or circuit analysis, behavioral failures such as repetition loops, sparse weight surgery, and weight-space programming. Applies to LLMs, policies, and other neural networks; ordinary prompting, optimizer-based fine-tuning, and generic application debugging are separate tasks.
---

# Weight-Space Programming

Turn a natural-language behavioral goal into an evidence-backed analysis or a small, testable change to neural-network parameters. Preserve the original and accept a candidate only when measured results justify it. This skill is portable across agents; it does not depend on a particular LLM, MCP server, or cloud service.

Based on [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) and its [released experiments](https://github.com/RikiyaT/lm_understand-logs). The findings are early and model-specific. Reading weights suggests hypotheses; it does not prove behavior. A sparse edit can still cause broad regressions. No skill guarantees that arbitrary weight changes are harmless.

## Select the task and evidence budget

Use the user's requested mode:

| Mode | Work | Deliverable |
|---|---|---|
| Inspect | Inventory parameters, numerical health, architecture, differences | Read-only report with provenance and uncertainty |
| Diagnose | Trace a failure from behavior through components | Falsifiable hypothesis, evidence, competing explanations |
| Edit | Modify a supported mechanism in an existing checkpoint | Separate candidate, deterministic edit, tensor diff, evaluation, rollback |
| Construct | Implement a specified computation within a network architecture | Weight-construction script and behavioral tests |
| Reproduce | Recreate a particular published experiment | Pinned inputs, matched protocol, results and deviations |

Establish what is available: checkpoint and exact revision, architecture/forward code, tokenizer or input encoding, output/action semantics, permitted tools and inference, compute/time budget, and intended behavior. Infer routine details from local files. Ask only when a missing fact changes the experiment or risks touching the wrong model. If weights are absent, provide the analysis/experiment design and identify the missing artifact; never invent inspected tensors or measured scores.

An inspection request authorizes reading, not changing weights. Work on an isolated candidate for an edit request; deployment, uploading data, or replacing a live checkpoint requires the user's corresponding authorization. Keep model weights, prompts, and datasets local unless their transfer is explicitly allowed.

Distinguish **weights-only**, **activation-assisted**, and **behavior-assisted** evidence. Respect an explicitly specified zero-forward-pass constraint, including synthetic inputs and individual-layer calls. Matrix algebra on weights is not a forward pass. Agree on what counts as one inference call/batch, then track samples, tokens, time and calls as well as the nominal pass count. Ordinary repair should include inference tests when available; published restrictions are reproduction settings, not universal rules. No optimizer, backpropagation, or gradient-free fitting loop is part of direct weight-space programming unless the user changes the method.

## Preserve and characterize the baseline

1. Read architecture code and checkpoint metadata before interpreting tensor names. Record matrix orientation, nonlinearities, normalization, residual paths, tied weights, quantization, positional encoding, and sharding. Include preprocessing, tokenizer, chat template and generation settings in the baseline.
2. Record exact source/revision and file hashes, tool versions, hardware, seeds and evaluation split IDs. For a mutable local checkpoint, make a verified independent copy before editing. For a pinned immutable store, verify that its exact bytes remain recoverable; do not treat a moving branch or URL as a backup. Check available disk/RAM first.
3. Reproduce the failure and measure a baseline using the existing evaluator. Check prompt/template, tokenizer, decoding, stop conditions, truncation, dtype and runtime bugs before attributing a failure to weights. Store raw per-example outputs, not just aggregate scores.
4. Inspect tensor keys, shapes, dtypes and non-finite values. Read summaries and relevant slices instead of dumping a large checkpoint into context. Preserve original bytes; read-only inspection does not need a duplicated model.

Use [checkpoint-tools.md](references/checkpoint-tools.md) for the bundled tools, supported formats, exact commands and limitations. They audit artifacts; they do not certify model quality or replace the model's own loader/evaluator.

## Analyze before editing

Read [analysis-and-edits.md](references/analysis-and-edits.md) for weight algebra, circuit hypotheses, direct construction, quantization and tied-weight pitfalls.

For each hypothesis, connect **observed failure → candidate mechanism → discriminating probe → predicted intervention effect**. Separate measured facts from plausible explanations. For example, repeated answers may implicate a residual direction, but a large singular value alone does not establish that direction's semantics.

Use the smallest relevant view: selected rows/columns, per-head blocks, weight products, singular values, norms and comparisons against a matched baseline. Compare coordinates only when checkpoints share the same parameterization; hidden-unit permutations or different architectures invalidate naive subtraction. Label ablation/activation evidence explicitly. If the evidence cannot distinguish hypotheses, return that uncertainty or use an allowed probe instead of declaring a diagnosis.

## Make one reversible candidate

Before writing, specify exact tensor names/slices, the operation and mechanism, invariants, primary improvement threshold, regression tolerances, and budget. Choose these before looking at candidate results. Derive each candidate from the same baseline, or record its parent hash explicitly; do not silently accumulate notebook edits.

- Write a deterministic patch/construction script bound to baseline hashes. Preserve shapes, dtype, key set, metadata and untouched tensor values. Use explicit index/shape assertions and finite checks. Reopen the saved candidate and inspect the serialized values; casting can erase or amplify an edit.
- Do not hard-code the blog's layers 15/19, columns, direction or percentage into another model. The mechanism and architecture must justify them.
- For a task requiring no training, implement the proposed computation directly. Searching coefficients against a loss is fitting even without gradients.
- Save into a new directory. Do not overwrite the source, edit a shared cache, use hard links for mutable candidates, or switch a serving process during experimentation. A verified rollback means the original artifact and settings can actually be loaded again.
- Gate the tensor diff against the exact allowed names and separately assert allowed slices. The bundled comparator checks names, not the semantics or intended slices of an edit.

## Evaluate, decide, and roll back

Read [evaluation.md](references/evaluation.md) before an edit or a behavioral claim. Use the same loader, prompts, seeds, decoding, precision and budget for baseline and candidate; record unavoidable differences. Keep discovery/practice data separate from validation and held-out final evaluation. Once final tests influence selection, they are no longer untouched tests.

Check in this order: serialized integrity → clean reload → smoke inference → target behavior → unrelated capability/operational regressions → restore and replay baseline. A failed gate prevents promotion. An inconclusive improvement is an inconclusive result; retaining the baseline is a valid outcome. Stop at the agreed budget, when evidence contradicts the mechanism, or when the next experiment needs resources not available.

For repeated or stochastic runs, use paired examples/seeds and report uncertainty. Measure both desired behavior and potential collateral effects, including output validity, answer completion, throughput/memory and domain-appropriate capability tests. A successful file load or a smaller edit fraction is not a behavioral safety test.

Promotion is a separate operation after validation. For a user-authorized deployment, test the actual packaged checkpoint and runtime, use the project's normal atomic switch, and retain the prior version. Restore to a fresh directory first, verify every hash, reload, and replay the baseline probe. Do not delete backup evidence as cleanup.

## Hand off verifiable evidence

For inspection, report what was actually inspected and the remaining uncertainty. For an edit/construction, deliver:

- The candidate checkpoint and unchanged original/backup location with hashes.
- The deterministic patch or construction script plus file/tensor diff (names, slices, changed count, norms, maximum delta).
- An experiment record: hypothesis, permitted evidence, commands/working directories, inputs, versions, literal outputs/exit statuses, sample counts, settings, metrics, uncertainty, budget used and pass/fail decisions. Preserve relevant raw outputs without credentials or private text in public reports.
- A runnable rollback command/script and its executed verification result.

State separately what passed, what failed, and what was not run. Do not claim replication of the blog, universal interpretability, or compatibility with all models from a toy test. Answer in the user's language; keep reproducible code and tensor identifiers exact.

## Resources and self-check

- [analysis-and-edits.md](references/analysis-and-edits.md): analytical methods and intervention mechanics.
- [evaluation.md](references/evaluation.md): experiment design and acceptance gates.
- [checkpoint-tools.md](references/checkpoint-tools.md): backup, integrity and tensor-audit commands.
- [source-notes.md](references/source-notes.md): reported results, negative findings and precise provenance.
- [evals/evals.json](evals/evals.json): realistic agent evaluation tasks and expected observable behavior; these are not a claim of cross-model evaluation.

Test the bundled tools from this skill's directory with `python -m unittest discover -s tests -v`. Run the executable miniature edit/restore experiment with `python scripts/demo_experiment.py --output <new-directory>`. Python 3.10+ is required; `checkpoint_guard.py` uses only the standard library. The tensor tools/demo additionally need PyTorch, NumPy and safetensors in the chosen environment. Use existing compatible installations; do not modify the user's global environment just to run a check.
