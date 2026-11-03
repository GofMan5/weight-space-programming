import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


guard = module("checkpoint_guard")
audit = module("tensor_audit")


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.original = self.root / "original"
        self.original.mkdir()
        (self.original / "weights.bin").write_bytes(bytes(range(256)))
        (self.original / "config.json").write_text('{"width": 3}', encoding="utf-8")
        self.backup = self.root / "backup"

    def test_snapshot_restore_byte_identical_and_independent(self):
        expected = guard.entries(self.original)
        guard.snapshot(self.original, self.backup)
        (self.original / "weights.bin").write_bytes(b"changed after snapshot")
        restored = self.root / "restored"
        guard.restore(self.backup, restored)
        self.assertEqual(guard.entries(restored), expected)
        self.assertNotEqual(guard.entries(self.original), expected)

    def test_nested_unicode_files_roundtrip(self):
        nested = self.original / "nested folder" / "данные"
        nested.mkdir(parents=True)
        (nested / "tensor.bin").write_bytes(b"nested")
        expected = guard.entries(self.original)
        guard.snapshot(self.original, self.backup)
        target = self.root / "restored"
        guard.restore(self.backup, target)
        self.assertEqual(guard.entries(target), expected)

    def test_existing_destination_is_preserved(self):
        self.backup.mkdir()
        sentinel = self.backup / "keep"
        sentinel.write_text("keep")
        with self.assertRaises(ValueError):
            guard.snapshot(self.original, self.backup)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_nested_destination_rejected_before_write(self):
        with self.assertRaises(ValueError):
            guard.snapshot(self.original, self.original / "nested")
        self.assertFalse((self.original / "nested").exists())

    def test_corrupt_backup_rejected_before_restore(self):
        guard.snapshot(self.original, self.backup)
        (self.backup / "model" / "weights.bin").write_bytes(b"corrupted")
        target = self.root / "restored"
        with self.assertRaises(ValueError):
            guard.restore(self.backup, target)
        self.assertFalse(target.exists())

    def test_extra_and_missing_backup_file_rejected(self):
        guard.snapshot(self.original, self.backup)
        extra = self.backup / "model" / "extra"
        extra.write_text("x")
        with self.assertRaises(ValueError):
            guard.verify(self.backup)
        extra.unlink()
        (self.backup / "model" / "config.json").unlink()
        with self.assertRaises(ValueError):
            guard.verify(self.backup)

    def test_traversal_manifest_rejected(self):
        guard.snapshot(self.original, self.backup)
        manifest_path = self.backup / "manifest.json"
        data = json.loads(manifest_path.read_text())
        data["files"]["../outside"] = data["files"].pop("weights.bin")
        manifest_path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            guard.restore(self.backup, self.root / "restored")

    def test_empty_source_rejected(self):
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(ValueError):
            guard.snapshot(empty, self.backup)

    def test_restore_never_overwrites(self):
        guard.snapshot(self.original, self.backup)
        with self.assertRaises(ValueError):
            guard.restore(self.backup, self.original)
        self.assertEqual(guard.entries(self.original), guard.verify(self.backup))

    def test_symlink_rejected(self):
        link = self.original / "linked.bin"
        try:
            link.symlink_to(self.original / "weights.bin")
        except OSError as error:
            self.skipTest(f"OS symlink permission: {error}")
        with self.assertRaises(ValueError):
            guard.snapshot(self.original, self.backup)

    @unittest.skipUnless(sys.platform == "win32", "Windows junction test")
    def test_windows_junction_rejected_without_admin(self):
        import _winapi
        target = self.root / "target"
        target.mkdir()
        (target / "data").write_bytes(b"external")
        _winapi.CreateJunction(str(target), str(self.original / "junction"))
        with self.assertRaisesRegex(ValueError, "reparse"):
            guard.snapshot(self.original, self.backup)
        self.assertTrue((target / "data").exists())


class TensorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "base.safetensors"
        self.candidate = self.root / "candidate.safetensors"
        self.values = {"weight": torch.tensor([[1., 2.], [3., 4.]]), "bias": torch.zeros(2)}
        save_file(self.values, str(self.base))

    def save(self, **updates):
        save_file({**self.values, **updates}, str(self.candidate))

    def test_exact_diff_counts(self):
        self.save(weight=torch.tensor([[1., 2.], [3., 5.]]))
        report = audit.compare(self.base, self.candidate, {"weight"})
        self.assertEqual(report["changed_elements"], 1)
        self.assertEqual(report["total_elements"], 6)
        self.assertEqual(report["changed_tensors"]["weight"]["max_abs_delta"], 1)

    def test_default_rejects_any_change(self):
        self.save(bias=torch.ones(2))
        with self.assertRaisesRegex(ValueError, "outside allowed"):
            audit.compare(self.base, self.candidate, set())

    def test_unrelated_change_rejected(self):
        self.save(bias=torch.ones(2))
        with self.assertRaises(ValueError):
            audit.compare(self.base, self.candidate, {"weight"})

    def test_unknown_allow_name_rejected(self):
        self.save()
        with self.assertRaises(ValueError):
            audit.compare(self.base, self.candidate, {"typo"})

    def test_shapes_and_dtypes_rejected(self):
        for value in (torch.ones(4), self.values["weight"].to(torch.float64)):
            with self.subTest(value=value):
                self.save(weight=value)
                with self.assertRaises(ValueError):
                    audit.compare(self.base, self.candidate, {"weight"})

    def test_missing_and_added_keys_rejected(self):
        for values in ({"weight": self.values["weight"]}, {**self.values, "extra": torch.ones(1)}):
            save_file(values, str(self.candidate))
            with self.assertRaises(ValueError):
                audit.compare(self.base, self.candidate, set(values))

    def test_nonfinite_baseline_and_candidate_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            self.save(bias=torch.tensor([value, 0.]))
            for a, b in ((self.base, self.candidate), (self.candidate, self.base)):
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    audit.compare(a, b, {"bias"})

    def test_unchanged_passes(self):
        self.save()
        self.assertEqual(audit.compare(self.base, self.candidate, set())["changed_elements"], 0)

    def test_zero_baseline_relative_delta(self):
        self.save(bias=torch.ones(2))
        self.assertIsNone(audit.compare(self.base, self.candidate, {"bias"})["changed_tensors"]["bias"]["relative_l2_delta"])

    def test_bfloat16_values_and_scalars(self):
        save_file({"s": torch.tensor(1., dtype=torch.bfloat16)}, str(self.base))
        save_file({"s": torch.tensor(2., dtype=torch.bfloat16)}, str(self.candidate))
        self.assertEqual(audit.compare(self.base, self.candidate, {"s"})["changed_elements"], 1)

    def test_large_integer_change_count_is_exact(self):
        save_file({"n": torch.tensor([2**60], dtype=torch.int64)}, str(self.base))
        save_file({"n": torch.tensor([2**60 + 1], dtype=torch.int64)}, str(self.candidate))
        self.assertEqual(audit.compare(self.base, self.candidate, {"n"})["changed_elements"], 1)

    def test_npz_numeric_and_object_arrays(self):
        path = self.root / "arrays.npz"
        np.savez(path, w=np.arange(3, dtype=np.float32))
        self.assertEqual(audit.inspect(path)["tensors"]["w"]["elements"], 3)
        np.savez(path, w=np.array([{"malicious": "object"}], dtype=object))
        with self.assertRaises(ValueError):
            audit.inspect(path)

    def test_shard_index_and_duplicate_keys(self):
        directory = self.root / "shards"
        directory.mkdir()
        save_file({"weight": self.values["weight"]}, str(directory / "one.safetensors"))
        save_file({"bias": self.values["bias"]}, str(directory / "two.safetensors"))
        index = directory / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"weight": "one.safetensors", "bias": "two.safetensors"}}))
        self.assertEqual(set(audit.inventory(directory)), {"weight", "bias"})
        index.write_text(json.dumps({"weight_map": {"weight": "two.safetensors", "bias": "one.safetensors"}}))
        with self.assertRaises(ValueError):
            audit.inventory(directory)
        index.unlink()
        save_file({"weight": self.values["weight"]}, str(directory / "two.safetensors"))
        with self.assertRaises(ValueError):
            audit.inventory(directory)

    def test_zero_element_checkpoint_has_defined_result(self):
        save_file({"empty": torch.zeros(0)}, str(self.base))
        save_file({"empty": torch.zeros(0)}, str(self.candidate))
        self.assertEqual(audit.compare(self.base, self.candidate, set())["changed_fraction"], 0)

    def test_invalid_safetensors_cli_fails_with_json(self):
        path = self.root / "invalid.safetensors"
        path.write_bytes(b"broken")
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / "tensor_audit.py"), "inspect", str(path)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(json.loads(result.stderr)["ok"])

    def test_malformed_shard_index_rejected(self):
        directory = self.root / "malformed"
        directory.mkdir()
        save_file(self.values, str(directory / "model.safetensors"))
        (directory / "model.safetensors.index.json").write_text("[]")
        with self.assertRaisesRegex(ValueError, "invalid shard index"):
            audit.inventory(directory)

    def test_readonly_inspection_preserves_files(self):
        before = guard.file_hash(self.base)
        report = audit.inspect(self.base)
        self.assertEqual(report["tensors"]["weight"]["shape"], [2, 2])
        self.assertEqual(guard.file_hash(self.base), before)


if __name__ == "__main__":
    unittest.main()
