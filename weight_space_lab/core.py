"""Single-file, floating-point safetensors prototype. Plans use explicit axes."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import threading
from collections import OrderedDict

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


_readonly_hashes = OrderedDict()
_hash_lock = threading.Lock()
_hash_watch_fd = None
_hash_watch_ids = set()
_hash_watch_epoch = 0


def _watch_readonly(path):
    """Local Linux inotify invalidates even changes hidden by coarse stat timestamps."""
    global _hash_watch_fd, _hash_watch_epoch
    import ctypes
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if _hash_watch_fd is None or len(_hash_watch_ids) >= 128:
            if _hash_watch_fd is not None:
                os.close(_hash_watch_fd)
            _readonly_hashes.clear()
            _hash_watch_ids.clear()
            _hash_watch_epoch += 1
            _hash_watch_fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
            if _hash_watch_fd < 0:
                _hash_watch_fd = None
                return False
        # Refresh every call so deleted/recreated files never retain a stale watch.
        wd = libc.inotify_add_watch(_hash_watch_fd, os.fsencode(path), 0x2 | 0x4 | 0x400 | 0x800)
        if wd < 0:
            return False
        _hash_watch_ids.add(wd)
        return True
    except (AttributeError, OSError):
        return False  # unsupported systems always compute the full digest


def _drain_hash_changes():
    global _hash_watch_epoch
    if _hash_watch_fd is None:
        return
    try:
        while os.read(_hash_watch_fd, 65536):
            _readonly_hashes.clear()
            _hash_watch_epoch += 1
    except BlockingIOError:
        pass


def _file_identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_mode)


def sha256(path: Path) -> str:
    # Large immutable research checkpoints otherwise get rehashed repeatedly
    # during one verified operation. Writable files are NEVER cached. Any write,
    # replacement, chmod or timestamp restoration invalidates this stat identity.
    before = path.stat()
    readonly = not (before.st_mode & 0o222)
    key = _file_identity(before)
    if readonly:
        with _hash_lock:
            readonly = _watch_readonly(path)
            _drain_hash_changes()
            epoch = _hash_watch_epoch
            if readonly and key in _readonly_hashes:
                _readonly_hashes.move_to_end(key)
                return _readonly_hashes[key]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if _file_identity(path.stat()) != key:
        raise ValueError('file changed while hashing')
    result = digest.hexdigest()
    if readonly:
        with _hash_lock:
            _drain_hash_changes()
            if _hash_watch_epoch != epoch:
                raise ValueError('watched file changed while hashing; retry')
            _readonly_hashes[key] = result
            while len(_readonly_hashes) > 64:
                _readonly_hashes.popitem(last=False)
    return result


def local(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("path must stay inside workspace")
    return path


def inspect(root: Path, checkpoint: str) -> dict:
    path = local(root, checkpoint)
    with safe_open(path, framework="np") as handle:
        tensors = [{"name": key, "shape": handle.get_slice(key).get_shape(),
                    "dtype": handle.get_slice(key).get_dtype()} for key in handle.keys()]
    return {"checkpoint": checkpoint, "sha256": sha256(path), "tensors": tensors}


def read_matrix(root: Path, checkpoint: str, tensor: str,
                rows: list[int], columns: list[int]) -> dict:
    if not rows or not columns or len(rows) * len(columns) > 4096:
        raise ValueError("request between 1 and 4096 values")
    with safe_open(local(root, checkpoint), framework="np") as handle:
        shape = handle.get_slice(tensor).get_shape()
        if len(shape) != 2:
            raise ValueError("matrix required")
        _indices(rows, shape[0])
        _indices(columns, shape[1])
        # Slice before materialization; no entire checkpoint load.
        values = np.concatenate([handle.get_slice(tensor)[i:i + 1, :][:, columns]
                                 for i in rows], axis=0)
    return {"tensor": tensor, "shape": shape, "rows": rows,
            "columns": columns, "values": values.tolist()}


def _indices(values: list[int], size: int) -> None:
    if (not values or len(set(values)) != len(values)
            or any(type(i) is not int or i < 0 or i >= size for i in values)):
        raise ValueError("indices must be unique nonnegative integers within axis")


def prepare(root: Path, plan: dict) -> tuple[dict, dict]:
    """Recompute candidate; budgets and hashes are enforced at write time too."""
    if not isinstance(plan.get("hypothesis"), str) or not plan["hypothesis"].strip():
        raise ValueError("hypothesis required")
    path = local(root, plan["checkpoint"])
    before_hash = sha256(path)
    if before_hash != plan["source_sha256"]:
        raise ValueError("source hash mismatch")
    fraction = plan["limits"]["max_changed_fraction"]
    relative = plan["limits"]["max_relative_l2"]
    if not (math.isfinite(fraction) and 0 < fraction <= 1
            and math.isfinite(relative) and 0 < relative <= 1):
        raise ValueError("limits must be finite in (0, 1]")
    operations = plan["operations"]
    if not operations or len(operations) > 32:
        raise ValueError("between 1 and 32 operations required")
    changed = {}
    summaries = []
    with safe_open(path, framework="np") as handle:
        total = sum(math.prod(handle.get_slice(k).get_shape()) for k in handle.keys())
        for operation in operations:
            key = operation["tensor"]
            if key in changed:
                raise ValueError("one operation per tensor; use separate experiments")
            original = handle.get_tensor(key)
            if original.ndim != 2 or original.dtype.kind != "f":
                raise ValueError("only floating-point matrices supported")
            if not np.isfinite(original).all():
                raise ValueError("nonfinite source")
            candidate = _apply_operation(original, operation)
            candidate = candidate.astype(original.dtype)
            if not np.isfinite(candidate).all():
                raise ValueError("nonfinite candidate")
            delta = candidate.astype(np.float64) - original.astype(np.float64)
            count = int(np.count_nonzero(delta))
            ratio = float(np.linalg.norm(delta) / max(np.linalg.norm(original.astype(np.float64)), 1e-30))
            if ratio > relative:
                raise ValueError("relative L2 budget exceeded")
            summaries.append({"tensor": key, "changed_parameters": count,
                              "relative_l2": ratio, "max_absolute_delta": float(np.abs(delta).max())})
            changed[key] = np.ascontiguousarray(candidate)
    changed_count = sum(s["changed_parameters"] for s in summaries)
    if changed_count / max(total, 1) > fraction:
        raise ValueError("changed parameter budget exceeded")
    if sha256(path) != before_hash:
        raise ValueError("source changed during preparation")
    report = {"source_sha256": before_hash, "total_parameters": total,
              "changed_parameters": changed_count, "changed_fraction": changed_count / max(total, 1),
              "tensors": summaries, "status": "unevaluated"}
    return changed, report


def _apply_operation(original: np.ndarray, operation: dict) -> np.ndarray:
    if operation["kind"] == "sequence":
        steps = operation.get("steps")
        if not isinstance(steps, list) or not 1 <= len(steps) <= 32:
            raise ValueError("sequence needs1..32 explicit steps")
        current = original.copy()
        for step in steps:
            if (not isinstance(step, dict) or step.get("kind") == "sequence"
                    or step.get("tensor", operation["tensor"]) != operation["tensor"]):
                raise ValueError("sequence steps must be non-nested operations on the same tensor")
            current = _apply_operation(current, step).astype(original.dtype)
            if not np.isfinite(current).all():
                raise ValueError("nonfinite intermediate candidate")
        return current
    if operation["kind"] != "set_values":
        return _transform(original, operation)
    rows, columns = operation["rows"], operation["columns"]
    _indices(rows, original.shape[0]); _indices(columns, original.shape[1])
    if len(rows) * len(columns) > 4096:
        raise ValueError("set_values permits at most4096 explicitly supplied values")
    values, expected = np.asarray(operation["values"], dtype=np.float64), np.asarray(operation["expected_values"], dtype=np.float64)
    shape = (len(rows), len(columns))
    if (values.shape != shape or expected.shape != shape
            or not np.isfinite(values).all() or not np.isfinite(expected).all()):
        raise ValueError("values/expected_values must be finite and match rows x columns")
    candidate = original.astype(np.float64)
    selector = np.ix_(rows, columns)
    if not np.array_equal(candidate[selector], expected):
        raise ValueError("expected_values differ from actual source/current coordinates")
    candidate[selector] = values
    return candidate


def _transform(original: np.ndarray, operation: dict) -> np.ndarray:
    """Explicit matrix transformations; no loss, gradients or optimizer."""
    axis = operation["axis"]
    if type(axis) is not int or axis not in (0, 1):
        raise ValueError("axis must be 0 (rows) or 1 (columns)")
    ids = operation["indices"]
    _indices(ids, original.shape[axis])
    alpha = operation["strength"]
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha < 0:
        raise ValueError("strength must be finite and nonnegative")
    if operation["kind"] == "scale" and alpha > 1:
        raise ValueError("attenuation scale strength must be in [0, 1]")
    candidate = original.astype(np.float64)
    selector = (ids, slice(None)) if axis == 0 else (slice(None), ids)
    block = candidate[selector]
    if operation["kind"] == "scale":
        # strength 0 = identity, 1 = ablation.
        candidate[selector] = block * (1 - alpha)
    elif operation["kind"] == "project_out":
        direction = np.asarray(operation["direction"], dtype=np.float64)
        expected = original.shape[1 - axis]
        if direction.shape != (expected,) or not np.isfinite(direction).all():
            raise ValueError("direction must match opposite axis")
        norm = np.linalg.norm(direction)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("direction must have finite nonzero norm")
        unit = direction / norm
        projection = (np.outer(block @ unit, unit) if axis == 0
                      else np.outer(unit, unit @ block))
        candidate[selector] = block - alpha * projection
    elif operation["kind"] == "rank_one":
        left = np.asarray(operation["left"], dtype=np.float64)
        right = np.asarray(operation["right"], dtype=np.float64)
        if (left.shape != (block.shape[0],) or right.shape != (block.shape[1],)
                or not np.isfinite(left).all() or not np.isfinite(right).all()):
            raise ValueError("rank_one vectors must be finite and match selected block rows/columns")
        candidate[selector] = block + alpha * np.outer(left, right)
    else:
        raise ValueError("unsupported edit kind")
    return candidate


def write_candidate(root: Path, plan: dict, output: str, *, lineage=None) -> dict:
    tensors, report = prepare(root, plan)
    target = local(root, output)
    if target.exists():
        raise ValueError("output already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".candidate-", dir=target.parent))
    try:
        save_file(tensors, str(staging / "replacements.safetensors"))
        report["patch_sha256"] = sha256(staging / "replacements.safetensors")
        manifest = {"plan": plan, "report": report}
        if lineage is not None:
            manifest["lineage"] = lineage
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
        # Empty target reservation prevents concurrent writers overwriting a candidate.
        target.mkdir()
        try:
            os.replace(staging, target)
        except BaseException:
            target.rmdir()
            raise
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"output": output, **report}
