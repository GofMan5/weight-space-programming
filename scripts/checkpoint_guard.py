#!/usr/bin/env python3
"""Copy, hash-check, and restore an isolated checkpoint directory."""

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path, PurePosixPath


MANIFEST = "manifest.json"


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_link(path):
    info = path.lstat()
    return path.is_symlink() or bool(getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def entries(root):
    if not root.is_dir() or is_link(root):
        raise ValueError(f"not a real directory: {root}")
    found = {}
    for current, folders, files in os.walk(root, followlinks=False):
        for name in folders + files:
            path = Path(current, name)
            if is_link(path):
                raise ValueError(f"link/reparse point rejected: {path}")
        for name in files:
            path = Path(current, name)
            if not path.is_file():
                raise ValueError(f"non-regular file rejected: {path}")
            rel = path.relative_to(root).as_posix()
            found[rel] = {"bytes": path.stat().st_size, "sha256": file_hash(path)}
    if not found:
        raise ValueError(f"no files: {root}")
    return dict(sorted(found.items()))


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("invalid manifest version")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("empty/invalid manifest")
    for rel, info in files.items():
        if not isinstance(rel, str) or not rel or "\\" in rel or ":" in rel:
            raise ValueError("invalid manifest path")
        parts = PurePosixPath(rel).parts
        if rel.startswith("/") or any(p in (".", "..") for p in parts) or "/".join(parts) != rel:
            raise ValueError("unsafe manifest path")
        if not isinstance(info, dict) or set(info) != {"bytes", "sha256"}:
            raise ValueError("invalid manifest entry")
        if type(info["bytes"]) is not int or info["bytes"] < 0:
            raise ValueError("invalid file size")
        if not isinstance(info["sha256"], str) or len(info["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in info["sha256"]):
            raise ValueError("invalid SHA-256")
    return files


def verify(backup):
    if not backup.is_dir() or is_link(backup):
        raise ValueError("backup is not a real directory")
    for item in backup.iterdir():
        if is_link(item) or item.name not in (MANIFEST, "model"):
            raise ValueError("backup contains unexpected entries")
    manifest_path = backup / MANIFEST
    if not manifest_path.is_file():
        raise ValueError("missing manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = validate_manifest(manifest)
    if entries(backup / "model") != expected:
        raise ValueError("backup file hashes/sizes or inventory do not match")
    return expected


def separate(source, dest):
    if dest.exists() or dest.is_symlink():
        raise ValueError(f"destination already exists: {dest}")
    if not dest.parent.is_dir() or is_link(dest.parent):
        raise ValueError("destination parent must be a real directory")
    a, b = source.resolve(strict=True), dest.resolve(strict=False)
    if a == b or a in b.parents or b in a.parents:
        raise ValueError("source and destination overlap")


def copy_files(source, dest, index):
    dest.mkdir()
    for rel in index:
        target = dest.joinpath(*PurePosixPath(rel).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source.joinpath(*PurePosixPath(rel).parts), target)


def capacity(dest, index):
    total = sum(info["bytes"] for info in index.values())
    if shutil.disk_usage(dest.parent).free < total + 1024 * 1024:
        raise ValueError("insufficient free disk space for copy and manifest")


def snapshot(source, dest):
    index = entries(source)
    separate(source, dest)
    capacity(dest, index)
    dest.mkdir()
    copy_files(source, dest / "model", index)
    (dest / MANIFEST).write_text(json.dumps({"version": 1, "files": index}, indent=2) + "\n", encoding="utf-8")
    verify(dest)
    if entries(source) != index:
        raise ValueError("source changed during snapshot")
    return {"backup": str(dest), "files": len(index), "bytes": sum(v["bytes"] for v in index.values())}


def restore(backup, dest):
    index = verify(backup)
    separate(backup, dest)
    capacity(dest, index)
    copy_files(backup / "model", dest, index)
    if entries(dest) != index:
        raise ValueError("restored bytes do not match manifest")
    verify(backup)
    return {"restored": str(dest), "files": len(index), "bytes": sum(v["bytes"] for v in index.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for name, arguments in (("snapshot", ("source", "destination")), ("verify", ("backup",)), ("restore", ("backup", "destination"))):
        command = commands.add_parser(name)
        for argument in arguments:
            command.add_argument(argument, type=Path)
    args = parser.parse_args()
    try:
        if args.action == "snapshot":
            report = snapshot(args.source, args.destination)
        elif args.action == "verify":
            report = {"backup": str(args.backup), "files": len(verify(args.backup))}
        else:
            report = restore(args.backup, args.destination)
        print(json.dumps({"ok": True, **report}))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
