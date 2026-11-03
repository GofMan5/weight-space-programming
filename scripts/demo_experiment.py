#!/usr/bin/env python3
"""Exercise an actual ReLU weight repair, rejection gate, and byte-exact rollback."""

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from checkpoint_guard import entries, file_hash


PATCH = '''import sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
sys.path.insert(0, SCRIPTS)
from checkpoint_guard import file_hash
source, destination = map(Path, sys.argv[1:])
if file_hash(source / "model.safetensors") != BASELINE_HASH:
    raise ValueError("baseline hash mismatch")
weights = load_file(str(source / "model.safetensors"))
assert tuple(weights["2.weight"].shape) == (1, 2)
assert weights["2.weight"].dtype == torch.float32
assert weights["2.weight"].tolist() == [[1.0, -1.0]]
original = {name: value.clone() for name, value in weights.items()}
weights["2.weight"][0, 1] = 1.0
assert all(torch.equal(value, original[name]) for name, value in weights.items() if name != "2.weight")
assert weights["2.weight"][0, 0] == original["2.weight"][0, 0]
assert all(torch.isfinite(value).all() for value in weights.values())
destination.mkdir()
save_file(weights, str(destination / "model.safetensors"))
(destination / "config.json").write_bytes((source / "config.json").read_bytes())
reopened = load_file(str(destination / "model.safetensors"))
assert all(torch.equal(value, reopened[name]) for name, value in weights.items())
print('{"patched": "2.weight[0,1]", "before": -1, "after": 1}')
'''

ROLLBACK = '''import subprocess
import sys
from pathlib import Path
if len(sys.argv) != 2:
    raise SystemExit("usage: python rollback.py NEW_RESTORE_DIRECTORY")
result = subprocess.run([sys.executable, GUARD, "restore", str(Path(__file__).parent / "backup"), sys.argv[1]])
raise SystemExit(result.returncode)
'''


def evaluate(checkpoint, split):
    model = torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.ReLU(), torch.nn.Linear(2, 1))
    model.load_state_dict(load_file(str(checkpoint / "model.safetensors")), strict=True)
    model.eval()
    points = [-2., -1., 0., 1., 2.] if split == "development" else [-8., -0.25, 0.25, 8.]
    with torch.inference_mode():
        actual = model(torch.tensor(points).reshape(-1, 1)).reshape(-1).tolist()
    expected = [abs(x) for x in points]
    result = {"split": split, "inputs": points, "expected": expected, "actual": actual,
              "correct": sum(a == b for a, b in zip(actual, expected)), "count": len(points)}
    print(json.dumps(result))
    return result


def experiment(output):
    if output.exists() or not output.parent.is_dir():
        raise ValueError("output must be a new directory under an existing parent")
    output.mkdir()
    scripts = Path(__file__).resolve().parent
    original, candidate = output / "original", output / "candidate"
    original.mkdir()
    save_file({"0.weight": torch.tensor([[1.], [-1.]]), "0.bias": torch.zeros(2),
               "2.weight": torch.tensor([[1., -1.]]), "2.bias": torch.zeros(1)}, str(original / "model.safetensors"))
    (original / "config.json").write_text('{"architecture": "Linear(1,2)-ReLU-Linear(2,1)", "goal": "abs(x)"}\n', encoding="utf-8")
    baseline_files = entries(original)
    patch = PATCH.replace("SCRIPTS", repr(str(scripts))).replace("BASELINE_HASH", repr(file_hash(original / "model.safetensors")))
    (output / "patch.py").write_text(patch, encoding="utf-8")
    (output / "rollback.py").write_text(ROLLBACK.replace("GUARD", repr(str(scripts / "checkpoint_guard.py"))), encoding="utf-8")
    commands = []

    def run(arguments, expected_exit=0):
        result = subprocess.run([sys.executable, *map(str, arguments)], cwd=output, capture_output=True, text=True, encoding="utf-8")
        record = {"command": [sys.executable, *map(str, arguments)], "cwd": str(output), "exit_status": result.returncode,
                  "stdout": result.stdout, "stderr": result.stderr}
        commands.append(record)
        if result.returncode != expected_exit:
            raise RuntimeError(json.dumps(record))
        return json.loads(result.stdout) if result.stdout.strip() else None

    run([scripts / "checkpoint_guard.py", "snapshot", original, output / "backup"])
    run([scripts / "tensor_audit.py", "inspect", original])
    baseline_dev = run([Path(__file__).resolve(), "--evaluate", original, "--split", "development"])
    run([output / "patch.py", original, candidate])
    difference = run([scripts / "tensor_audit.py", "compare", original, candidate, "--allow", "2.weight"])
    candidate_dev = run([Path(__file__).resolve(), "--evaluate", candidate, "--split", "development"])
    baseline_final = run([Path(__file__).resolve(), "--evaluate", original, "--split", "final"])
    candidate_final = run([Path(__file__).resolve(), "--evaluate", candidate, "--split", "final"])
    bad = output / "rejected-candidate"
    shutil.copytree(candidate, bad)
    values = {name: value.clone() for name, value in load_file(str(bad / "model.safetensors")).items()}
    values["2.bias"] += 1
    save_file(values, str(bad / "model.safetensors"))
    run([scripts / "tensor_audit.py", "compare", original, bad, "--allow", "2.weight"], expected_exit=1)
    run([output / "rollback.py", output / "restored"])
    restored = run([Path(__file__).resolve(), "--evaluate", output / "restored", "--split", "final"])
    assert difference["changed_elements"] == 1
    assert candidate_dev["correct"] == candidate_dev["count"]
    assert baseline_final["correct"] == 2 and candidate_final["correct"] == 4
    assert restored == baseline_final
    assert entries(original) == entries(output / "restored") == baseline_files
    assert all(a == b for x, a, b in zip(baseline_final["inputs"], baseline_final["actual"], candidate_final["actual"]) if x > 0)
    record = {"ok": True, "scope": "mechanics demonstration, not an LLM benchmark reproduction",
              "hypothesis": "the negative-input ReLU path has the wrong output sign",
              "acceptance": {"target": "exact abs(x) on final points", "regression": "positive-input outputs unchanged", "rollback": "identical files and baseline outputs"},
              "versions": {"python": platform.python_version(), "torch": torch.__version__},
              "baseline_files": baseline_files, "candidate_files": entries(candidate), "tensor_diff": difference,
              "baseline_development": baseline_dev, "candidate_development": candidate_dev,
              "baseline_final": baseline_final, "candidate_final": candidate_final,
              "rollback_verified": True, "unrelated_edit_rejected": True, "commands": commands}
    (output / "verification.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "verification": str(output / "verification.json"), "baseline_final_correct": 2, "candidate_final_correct": 4, "rollback_verified": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--evaluate", type=Path)
    parser.add_argument("--split", choices=("development", "final"), default="final")
    args = parser.parse_args()
    if args.evaluate:
        evaluate(args.evaluate, args.split)
    else:
        experiment(args.output.resolve())


if __name__ == "__main__":
    main()
