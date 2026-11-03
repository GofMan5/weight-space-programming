<div align="center">

# Weight-Space Programming

**An agent skill for working directly with neural-network weights.**

Inspect checkpoints. Investigate mechanisms. Test precise edits.
Keep the original recoverable and let evidence decide what happens next.

[![Agent Skills](https://img.shields.io/badge/Agent%20Skills-compatible-111827?style=flat-square)](https://agentskills.io/) [![License: MIT](https://img.shields.io/badge/license-MIT-22a06b?style=flat-square)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square)](#try-the-demo) [![GitHub stars](https://img.shields.io/github/stars/GofMan5/weight-space-programming?style=flat-square&color=da9d24)](https://github.com/GofMan5/weight-space-programming/stargazers)

[Install](#install) &nbsp; / &nbsp; [Capabilities](#capabilities) &nbsp; / &nbsp; [Workflow](#workflow) &nbsp; / &nbsp; [Demo](#try-the-demo) &nbsp; / &nbsp; [Research](#research--credits)

</div>

---

```text
  CHECKPOINT  →  HYPOTHESIS  →  CANDIDATE
  map & save     test a cause   edit a copy
                                  ↓
                       EVALUATE & RESTORE
```

Neural-network weights encode a computation, not just a file to load. This skill gives an agent a disciplined way to inspect that computation and, when asked, investigate whether a particular intervention improves a model.

Inspired by [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/), it translates early, model-specific research into a practical workflow. It does **not** guarantee that an arbitrary model can be understood, repaired, or changed without side effects.

> [!IMPORTANT]
> **Small changes can have large effects.** Preserve the baseline, compare behavior under matched settings, check regressions, and verify restoration. If evidence is inconclusive, keep the baseline.

## Install

Install using the [Skills CLI](https://github.com/vercel-labs/skills):

```bash
npx skills add GofMan5/weight-space-programming
```

The CLI discovers `weight-space-programming` from this repository and guides installation for supported agents. Add `-g` for a user-level installation. For other agents, consult their skill installation instructions or provide the [`SKILL.md`](SKILL.md) folder directly. An agent also needs access to the checkpoint, architecture, and tools appropriate to the task.

<details>
<summary><b>Manual installation for Codex</b></summary>

macOS / Linux:

```sh
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
git clone https://github.com/GofMan5/weight-space-programming.git \
  "${CODEX_HOME:-$HOME/.codex}/skills/weight-space-programming"
```

Windows / PowerShell:

```powershell
$CodexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $HOME '.codex' }
New-Item -ItemType Directory -Force -Path (Join-Path $CodexHome 'skills') | Out-Null
git clone https://github.com/GofMan5/weight-space-programming.git (Join-Path $CodexHome 'skills/weight-space-programming')
```

Clone into a new destination; if the skill is already installed, update it using the method that installed it. Restart your agent session if needed.

</details>

### Give it a task

```text
Use $weight-space-programming to inspect this local checkpoint. Do not edit it
or run inference. Report the tensor inventory and distinguish observations
from hypotheses.
```

More starting points:

- **Investigate:** "Find plausible causes of repetition in this model and design tests to distinguish them."
- **Edit:** "Test a targeted weight edit on a copy; compare behavior and verify that the original can be restored."
- **Construct:** "Implement this specified computation in a small network by setting weights directly, then test the boundaries."

## Capabilities

| Mode | Question | Result |
|:---|:---|:---|
| **Inspect** | What is actually in this checkpoint? | Shapes, dtypes, numerical checks, provenance and limits |
| **Diagnose** | What mechanism could explain the behavior? | A falsifiable hypothesis and discriminating probes |
| **Edit** | Does a targeted change improve the model? | An isolated candidate, tensor diff, evaluation and rollback |
| **Construct** | Can specified behavior be expressed in weights? | A construction script and behavioral tests |
| **Reproduce** | Can a published experiment be matched? | Pinned inputs, a matched protocol and documented deviations |

The workflow separates observations from hypotheses. A read-only request stays read-only; behavioral claims require behavioral evidence. It does not silently train a model, deploy a candidate, or upload weights and data.

## Workflow

```text
  01  BASELINE      Identify the model, runtime, input settings and hashes.
  02  MECHANISM     Connect an observed failure to a testable explanation.
  03  CANDIDATE     Apply a precise, deterministic edit to an isolated copy.
  04  GATES         Reload; compare behavior, regressions and tensor changes.
  05  ROLLBACK      Restore exact bytes and replay the original baseline.
```

For example, a request to investigate repeated outputs starts by checking the tokenizer, generation settings and runtime—not by copying layer numbers from a different model. If the evidence does not identify a supported mechanism, the outcome is an investigation report rather than an invented fix.

### Tools included

| Tool | Purpose | Requirements |
|:---|:---|:---|
| [`checkpoint_guard.py`](scripts/checkpoint_guard.py) | Snapshot, verify and restore checkpoint directories with SHA-256 manifests | Python standard library |
| [`tensor_audit.py`](scripts/tensor_audit.py) | Inspect or compare safetensors / numeric NPZ tensors | PyTorch, NumPy, safetensors |
| [`demo_experiment.py`](scripts/demo_experiment.py) | Run a miniature edit, rejection, evaluation and rollback experiment | PyTorch, NumPy, safetensors |

The tensor comparator checks exact **tensor names**, not whether an edit changed only the intended slices. It does not validate model behavior, tied storage or deployment readiness. GGUF and arbitrary pickle checkpoints are not supported by this helper. See the [tool guide](references/checkpoint-tools.md) for formats, commands, memory limits and other boundaries.

## Try the demo

The example uses a tiny ReLU network with a deliberately incorrect sign—not a real LLM. Run from the repository root with **Python 3.10+**, PyTorch, NumPy and safetensors installed in your chosen environment:

```sh
python -m unittest discover -s tests -v
python scripts/demo_experiment.py --output ./demo-run
```

`demo-run` must be a **new** path. The experiment writes the original, a candidate, a deliberately rejected candidate, a patch, a rollback script and a verification record. It checks the network's outputs before and after the edit and verifies the restored original. The demo is a test of the mechanics, **not a reproduction of the paper's LLM results**.

For a real checkpoint, start with the [checkpoint tools](references/checkpoint-tools.md) and [evaluation protocol](references/evaluation.md). Use a reviewed model-native loader and evaluator, and preserve private inputs and weights locally unless their transfer is explicitly authorized.

## Explore the skill

```text
SKILL.md                       Agent entry point and decision workflow
references/analysis-and-edits  Architecture, circuits and edit mechanics
references/checkpoint-tools    Format support and exact tool commands
references/evaluation          Baselines, acceptance gates and reporting
references/source-notes        Research claims and evidence boundaries
scripts/                       Backup, tensor audit and executable demo
tests/                         Automated checks for the bundled tools
evals/                         Agent evaluation scenarios—not benchmark scores
```

## Research & credits

Inspired by [*Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*](https://rikiyat.github.io/blog/weight-space-programming/) by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab, and their [published experiments](https://github.com/RikiyaT/lm_understand-logs). This project is an independent engineering workflow, not an official release from the research authors. See [source notes](references/source-notes.md) for specific results, negative findings and provenance.

## License

This repository is available under the [MIT License](LICENSE): use, modify and redistribute it, including commercially, subject to the license notice. The license applies to this repository's own material; the linked research, third-party models, datasets and tools retain their respective licenses.

---

<div align="center">
<sub>Curiosity in the weights. Evidence in the results.</sub>
</div>
