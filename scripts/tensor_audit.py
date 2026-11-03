#!/usr/bin/env python3
"""Inventory safetensors/NPZ checkpoints and gate structural tensor changes."""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from safetensors import SafetensorError, safe_open


def inventory(path):
    weights = None
    if path.is_dir():
        shards = sorted(path.glob("*.safetensors"))
        if not shards:
            raise ValueError("no top-level safetensors shards")
        index_file = path / "model.safetensors.index.json"
        if index_file.exists():
            data = json.loads(index_file.read_text(encoding="utf-8"))
            weights = data.get("weight_map") if isinstance(data, dict) else None
            if not isinstance(weights, dict) or not weights or set(weights.values()) != {p.name for p in shards}:
                raise ValueError("invalid shard index")
    elif path.is_file() and path.suffix == ".safetensors":
        shards = [path]
        weights = None
    elif path.is_file() and path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            names = archive.files
            if len(names) != len(set(names)) or not names:
                raise ValueError("duplicate or empty NPZ inventory")
            for name in names:
                array = archive[name]
                if array.dtype.kind not in "biuf":
                    raise ValueError(f"unsupported dtype: {name}: {array.dtype}")
        return {name: (path, "npz") for name in names}
    else:
        raise ValueError("expected safetensors file, NPZ file or safetensors directory")
    found = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in found:
                    raise ValueError(f"duplicate tensor: {name}")
                found[name] = (shard, "safetensors")
                if path.is_dir() and index_file.exists() and weights.get(name) != shard.name:
                    raise ValueError(f"shard index mismatch: {name}")
    if not found or (path.is_dir() and index_file.exists() and set(found) != set(weights)):
        raise ValueError("empty/incomplete tensor inventory")
    return found


def read_tensor(entry, name):
    path, kind = entry
    if kind == "npz":
        with np.load(path, allow_pickle=False) as archive:
            value = torch.from_numpy(np.array(archive[name], copy=True))
    else:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            value = handle.get_tensor(name)
    if value.dtype not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                           torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"unsupported tensor dtype: {name}: {value.dtype}")
    if value.is_floating_point() and not torch.isfinite(value).all().item():
        raise ValueError(f"non-finite tensor: {name}")
    return value


def norm(value):
    return float(torch.linalg.vector_norm(value.to(torch.float64)).item())


def inspect(path):
    found = inventory(path)
    return {"path": str(path), "tensors": {
        name: {"shape": list(value.shape), "dtype": str(value.dtype), "elements": value.numel(), "l2": norm(value)}
        for name in sorted(found) for value in [read_tensor(found[name], name)]
    }}


def compare(base, candidate, allowed):
    original, changed = inventory(base), inventory(candidate)
    if set(original) != set(changed):
        raise ValueError(f"tensor key mismatch: missing={sorted(set(original)-set(changed))}, added={sorted(set(changed)-set(original))}")
    unknown = allowed - set(original)
    if unknown:
        raise ValueError(f"unknown allowed tensor names: {sorted(unknown)}")
    results, total, count = {}, 0, 0
    for name in sorted(original):
        a, b = read_tensor(original[name], name), read_tensor(changed[name], name)
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"shape/dtype mismatch: {name}")
        different = int(torch.count_nonzero(a != b).item())
        if different and name not in allowed:
            raise ValueError(f"change outside allowed tensors: {name}")
        total += a.numel()
        count += different
        if different:
            delta = b.to(torch.float64) - a.to(torch.float64)
            size = norm(a)
            l2 = norm(delta)
            results[name] = {"changed_elements": different, "max_abs_delta": float(delta.abs().max().item()),
                             "l2_delta": l2, "relative_l2_delta": l2 / size if size else None}
    return {"baseline": str(base), "candidate": str(candidate), "changed_tensors": results,
            "changed_elements": count, "total_elements": total, "changed_fraction": count / total if total else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("inspect").add_argument("checkpoint", type=Path)
    command = commands.add_parser("compare")
    command.add_argument("baseline", type=Path)
    command.add_argument("candidate", type=Path)
    command.add_argument("--allow", action="append", default=[])
    args = parser.parse_args()
    try:
        report = inspect(args.checkpoint) if args.action == "inspect" else compare(args.baseline, args.candidate, set(args.allow))
        print(json.dumps({"ok": True, **report}, allow_nan=False))
    except (OSError, ValueError, KeyError, RuntimeError, TypeError, OverflowError, EOFError, zipfile.BadZipFile, SafetensorError) as error:
        print(json.dumps({"ok": False, "error": str(error)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
