# Checkpoint tools

`<SKILL_DIR>` means the directory containing SKILL.md. Substitute actual paths; all output destinations below must be new. Scripts emit JSON on stdout and fail with a nonzero exit code on a rejected operation. Capture stdout/stderr/exit status in the experiment record. They do not run model-supplied code or contact a network.

## Backup and restore (standard library)

```text
python <SKILL_DIR>/scripts/checkpoint_guard.py snapshot <original-model-directory> <new-backup-directory>
python <SKILL_DIR>/scripts/checkpoint_guard.py verify <backup-directory>
python <SKILL_DIR>/scripts/checkpoint_guard.py restore <backup-directory> <new-restored-model-directory>
```

A backup contains `model/` and `manifest.json` (SHA-256 and byte size of every file). Snapshot copies all files in the chosen model directory, verifies the copy and rechecks the source. Choose a model artifact directory, not an entire home directory or shared cache. It checks free space before copying; metadata and filesystem overhead still require headroom. No hard links are created. Empty directories are not part of the manifest.

Links/junctions/reparse points, overlapping source/destination paths, existing destinations, empty source directories and invalid manifests are rejected. Materialize files from a symlink-based model cache into an isolated real directory first using its supported download/export mechanism. Snapshot does not make the source read-only: stop other writers and keep both source and backup quiescent. Hashes detect changes but are not a concurrency lock or authenticity signature.

Restore verifies the backup before and after copying and writes only a fresh destination. Repointing a service to that restored path is a separate deployment action. Partial destinations from failed copies are retained for inspection and never presented as verified; retry with a fresh path after resolving the cause. Do not manually trust a backup that lacks a successful verification result.

## Tensor inventory and diff

Requires PyTorch, NumPy and safetensors. Supports a single `.safetensors`, a numeric `.npz`, or a directory containing top-level safetensors shards (optionally `model.safetensors.index.json`). It does not support arbitrary pickle checkpoints, GGUF, offloaded tensors, nested shard layouts, or custom packed/quantized editing. Use a reviewed native adapter for other formats.

```text
python <SKILL_DIR>/scripts/tensor_audit.py inspect <checkpoint>
python <SKILL_DIR>/scripts/tensor_audit.py compare <baseline> <candidate> --allow layer.weight --allow layer.bias
```

Without `--allow`, comparison permits no tensor value changes. Each allow entry is an exact tensor name, not a regex. Unknown names are errors. Added/removed tensors, changed shapes/dtypes, non-finite values or edits outside the allowlist fail. An unchanged allowed tensor is not an error. Empty/duplicate/malformed tensor inventories fail.

Reports include dtype/shape/element count, finite checks, tensor norms and per-tensor changed-element count, maximum absolute delta, L2 delta and relative L2 delta (null for a zero baseline norm). Comparisons use exact values after serialization; no tolerance hides small changes. Integer/bool changes are counted exactly; floating-point norms are diagnostics and can lose precision for very large integers. Signed zero is treated as numerically equal. File hashing detects byte-level differences separately.

Safetensors are loaded one tensor at a time on CPU; the comparator may hold one tensor from each checkpoint plus full working arrays in float64. A single enormous tensor can still exceed RAM. NumPy arrays must be numeric non-object arrays; unsupported dtypes fail explicitly. Sparse, complex and float8 tensors require a reviewed adapter. Metadata and non-weight configuration are protected by the file manifest and native loader tests, not the tensor comparator.

## Executable example and tests

```text
python -m unittest discover -s <SKILL_DIR>/tests -v
python <SKILL_DIR>/scripts/demo_experiment.py --output <new-experiment-directory>
```

The demo creates a tiny ReLU network with a known sign-path defect, copies its original artifacts, patches one specified weight, runs actual forward inference on development and separate final points, checks an intentionally unrelated mutation is rejected, restores the original and verifies that its original behavior returns. It writes `verification.json`, baseline/candidate outputs, `patch.py` and a runnable `rollback.py`. This tests mechanics; it does not reproduce the blog's LLM scores.
