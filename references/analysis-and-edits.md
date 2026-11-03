# Analyze and intervene

## Recover the actual computation

Read the forward implementation and state-dict schema. For PyTorch `Linear`, stored `W` is `[out, in]`, and batched row inputs compute `x @ W.T + b`. Other frameworks may store the transpose. Verify orientation with code and dimensions before selecting rows or columns. A column of an MLP down-projection often represents one intermediate feature's contribution to residual channels; a row does not represent the same object.

Map layer order, head packing, GQA/MQA sharing, biases, gating, residual addition, norm placement and positional mechanisms. Derive formulas for the actual architecture. For nonlinear networks, multiplying successive matrices alone does not describe the full computation. Attention QK/OV analyses depend on convention and scaling; a large product is not automatically an induction circuit.

Start with shape/dtype inventory and non-finite checks. Then inspect norms, outliers, singular spectra, rank estimates and selected correlations where they answer a hypothesis. Compare homologous components, account for scale symmetries and neuron permutations, and distinguish numerical anomalies from legitimate outliers. Do not label a network's corpus, honesty or capabilities from tensor names, high norms, or low rank alone.

With permitted inference, choose discriminating inputs: repeated strings for copying, controlled context changes for attention, and positive/negative examples for the suspected mechanism. Use counterexamples and matched controls. Feature correlations suggest a location; targeted ablations or restoration provide stronger causal evidence. Avoid tuning on held-out answers.

## Targeted edit mechanics

Record an edit proposal containing the mechanism, exact tensor and slice, formula, expected local effect, likely collateral effects and falsification probe. Choose the smallest change supported by evidence; small changed-element count is a description, not a safety certificate.

A possible operation, only when a measured output direction is implicated, is a partial projection:

`W[:, S] = W[:, S] - alpha * u[:, None] * (u @ W[:, S])[None, :]`

Here `W` is verified `[output, input]`, `u` is a finite, nonzero unit vector in the output space, `S` is an explicit set of unique input-feature indices, and `alpha` is an experiment parameter justified by the hypothesis. Compute on a copy in an appropriate working precision and cast back to the original dtype. Assert dimensions, bounds, finiteness, and exact equality outside `S`. Reopen and test after serialization. This formula is an example, not the default repair for repetition or a prescription to erase arbitrary behavior.

Other justified interventions may include repairing a known corrupted scalar, modifying a feature's output contribution, constructing a small control path or restoring a component from an exact matched checkpoint. Keep the corresponding control: unchanged model, candidate, and where useful an ablation or equally sized irrelevant edit. Do not multiply arbitrary whole layers or sweep parameters until a benchmark happens to improve while calling that mechanistic diagnosis.

## Construct weights as a program

Write a mathematical specification of inputs, outputs and expected behavior. Factor the computation into features and layer-wise operations that the fixed architecture supports. With ReLU, `ReLU(x) - ReLU(-x) = x`; threshold and piecewise-linear behavior can be composed explicitly. Tanh only approximates certain gates and can saturate; choose magnitudes from the domain and test boundary behavior. Account for finite precision and activation range.

Test truth tables, boundaries, symmetries, legal-action masks, action selection and stochastic settings. A linearized or handcrafted policy must still respect the environment's real encoding and evaluator. Do not replace the network with a side-channel algorithm at evaluation time. Keep synthesis-time and evaluation-time inference budgets distinct.

## Serialization and architecture traps

- **Tied weights:** document alias relationships from model code/config. A checkpoint format may store one copy of a shared tensor. Use the native exporter/reloader, preserve ties, and verify behavior after reload; a generic key-set comparison does not verify aliasing.
- **Quantized/packed weights:** integer storage, scales, zero points and group layout are a single representation. Do not treat packed bytes as float weights. Use the framework's dequantize/edit/requantize path on a copy and evaluate the resulting model. Requantization can change many values; report the full delta. If that path is unverified, limit the result to inspection/design.
- **Adapters:** distinguish adapter deltas from base tensors. Record merge scaling and avoid double-merging. Compare a consistently merged or unmerged representation.
- **Shards:** preserve key-to-file index consistency and metadata. Check every shard. The bundled auditor detects duplicate keys and index mismatches but does not export models.
- **Dtype/device:** preserve the deployment dtype, use `eval()` and inference/no-grad mode for probes, and record precision changes. BF16/FP16 rounding may remove small edits or create overflow.
- **Memory-mapped files:** safetensors loaders can retain mapped storage. Clone tensors for mutable work and save to a fresh file/directory; in particular, overwriting a still-mapped file fails on Windows.
- **Loading:** prefer safetensors and NumPy archives with `allow_pickle=False`. For trusted PyTorch state dictionaries use a current runtime and `weights_only=True`; never enable arbitrary unpickling as a silent fallback. Review custom model code before execution; leave remote-code execution disabled unless the task requires it and it has been reviewed.
- **Resources:** load tensors/shards incrementally when feasible. Estimate memory and disk requirements; larger-than-memory or exotic formats need a format-native adapter, not a claim of completed inspection.
