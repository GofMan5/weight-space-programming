<div align="center">

<br>

**A SKILL FOR AGENTS WORKING INSIDE NEURAL NETWORKS**

# Weight-Space Programming

### Read the weights. Form a hypothesis. Test the change.

An evidence-first agent skill for inspecting neural-network checkpoints,<br>
investigating behavior, and making **measured, reversible** weight edits.

[Get started](#get-started) &nbsp;·&nbsp; [What it does](#what-it-does) &nbsp;·&nbsp; [Try the demo](#try-the-demo) &nbsp;·&nbsp; [Research](#research--credits)

<br>

`INSPECT` &nbsp; → &nbsp; `HYPOTHESIZE` &nbsp; → &nbsp; `EDIT A COPY` &nbsp; → &nbsp; `EVALUATE` &nbsp; → &nbsp; `RESTORE`

<br>
</div>

---

## The idea

Model weights are not just files to train and load. They can also be examined as the parameters of a computation. This skill gives an AI agent a practical workflow for exploring that computation—and, when requested, testing a specific intervention without overwriting the original checkpoint.

It is inspired by [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/). The research results are early and model-specific; this repository provides an engineering workflow, **not a guarantee that an arbitrary model can be understood or repaired**.

> **A small edit is not automatically a safe edit.** The skill asks for a recoverable baseline, matched evaluations, regression checks, and a verified restore. If evidence is inconclusive, the baseline stays in place.

## What it does

| Mode | Question | Result |
|:---|:---|:---|
| **Inspect** | What is actually in this checkpoint? | Shapes, dtypes, numerical checks, provenance and limits |
| **Diagnose** | What mechanism could explain the behavior? | A falsifiable hypothesis and discriminating probes |
| **Edit** | Does a targeted change improve the model? | An isolated candidate, tensor diff, evaluation and rollback |
| **Construct** | Can specified behavior be expressed in weights? | A construction script and behavioral tests |
| **Reproduce** | Can a published experiment be matched? | Pinned inputs, a matched protocol and documented deviations |

The workflow separates observations from hypotheses. A read-only request stays read-only; behavioral claims require behavioral evidence. It does not silently train a model, deploy a candidate, or upload weights and data.

## Get started

Clone this repository into your agent's skills directory. For **Codex**:

<details open>
<summary><b>macOS / Linux</b></summary>

```sh
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
git clone https://github.com/GofMan5/weight-space-programming.git \
  "${CODEX_HOME:-$HOME/.codex}/skills/weight-space-programming"
```

</details>

<details>
<summary><b>Windows · PowerShell</b></summary>

```powershell
$CodexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $HOME '.codex' }
New-Item -ItemType Directory -Force -Path (Join-Path $CodexHome 'skills') | Out-Null
git clone https://github.com/GofMan5/weight-space-programming.git (Join-Path $CodexHome 'skills/weight-space-programming')
```

</details>

Restart your agent session if needed, then ask:

```text
Use $weight-space-programming to inspect this local checkpoint. Do not edit it
or run inference. Report the tensor inventory and distinguish observations
from hypotheses.
```

The core [`SKILL.md`](SKILL.md) and its references are portable instructions. Other agents can use them if they support the Agent Skills layout or can be given the files directly; installation and invocation vary by agent. An agent still needs access to the actual checkpoint, architecture, and appropriate tools to perform the work.

## From evidence to intervention

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
