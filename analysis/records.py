"""
Per-item records + run provenance.

The artifact of a probe run is a JSONL file with ONE ROW PER (probe, arm, item),
not a printed table. Aggregation and statistics happen in a separate pass over
those rows (analysis/report.py), so every table, interval and paired test can be
recomputed, re-sliced or re-tested without touching a GPU again. The v1 suite
printed means straight to stdout, which is why none of its numbers could be
given error bars after the fact.

Provenance matters as much as the numbers here: the revparityreg_24957997
comparison was run on crashed checkpoints with the pre-sdpa attention kernel, and
neither fact was visible anywhere in the output. Every results file therefore
carries a manifest naming the git SHA, each checkpoint's hash, its arch/rev args,
attn_impl, torch version, device and seed.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import Any


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(("git",) + args, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return "unknown"


def file_digest(path: str, chunk: int = 1 << 20, limit: int = 64 << 20) -> str:
    """sha256 of the first `limit` bytes of a file.

    Checkpoints here are ~1.1GB each; hashing the leading 64MB is enough to tell
    two checkpoints apart (the state dict starts immediately) while keeping a
    5-arm manifest near-instant. The prefix length is recorded in the digest
    string so nobody mistakes it for a whole-file hash.
    """
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as f:
        while read < limit:
            b = f.read(min(chunk, limit - read))
            if not b:
                break
            h.update(b)
            read += len(b)
    return f"sha256:{h.hexdigest()[:32]}/first{read}B"


@dataclass
class RunManifest:
    """Everything needed to know whether two results files are comparable."""
    suite_version: str = "2.0"
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    git_sha: str = field(default_factory=lambda: _git("rev-parse", "HEAD"))
    git_dirty: bool = field(default_factory=lambda: bool(_git("status", "--porcelain")))
    host: str = field(default_factory=socket.gethostname)
    user: str = field(default_factory=lambda: getpass.getuser())
    python: str = field(default_factory=platform.python_version)
    torch: str = ""
    numpy: str = ""
    device: str = ""
    seed: int = 0
    ckpt_dir: str = ""
    arms: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def add_arm(self, name: str, path: str, meta: dict) -> None:
        """Record one checkpoint. `meta` is the loader's metadata dict."""
        self.arms[name] = {
            "path": path,
            "digest": file_digest(path),
            "arch": meta.get("arch"),
            "regime": meta.get("regime"),
            "embed": meta.get("embed"),
            "attn_impl": meta.get("attn_impl"),
            "best_val": meta.get("best_val"),
            "step": meta.get("step"),
            "train_seed": meta.get("train_seed"),
            "max_steps": meta.get("max_steps"),
            "n_params": meta.get("n_params"),
        }

    def warnings(self) -> list[str]:
        """Comparability problems worth printing loudly before any table."""
        out = []
        impls = {a.get("attn_impl") for a in self.arms.values()}
        if len(impls) > 1:
            out.append(f"arms mix attention kernels {impls}: reduction order differs, "
                       f"so cross-arm differences are confounded")
        seeds = {a.get("train_seed") for a in self.arms.values()}
        if len(self.arms) > 1 and len(seeds) == 1:
            out.append(f"all arms share train seed {seeds.pop()} (1 seed/arm): architecture "
                       f"and seed are confounded — differences are descriptive, not inferential")
        steps, maxes = {}, set()
        for n, a in self.arms.items():
            if a.get("step") is not None:
                steps[n] = a["step"]
            if a.get("max_steps"):
                maxes.add(a["max_steps"])
        if steps and maxes:
            ms = max(maxes)
            short = {n: s for n, s in steps.items() if s < 0.98 * ms}
            if short:
                out.append(f"checkpoints below 98% of max_steps={ms} (truncated/crashed run): "
                           + ", ".join(f"{n}@{s}" for n, s in sorted(short.items())))
        if self.git_dirty:
            out.append("working tree dirty: git_sha does not fully describe this code")
        return out

    def to_dict(self) -> dict:
        return asdict(self)


class RecordWriter:
    """Append-only JSONL sink for per-item rows, plus a sidecar manifest.

    Usage:
        with RecordWriter(out_dir, "lens_depth", manifest) as w:
            w.write(arm="baseline", item_id="story42:tok17", layer=3, loss=2.9)
    """

    def __init__(self, out_dir: str, name: str, manifest: RunManifest | None = None):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"{name}.jsonl")
        self.name = name
        self._f = open(self.path, "w")
        self.n = 0
        if manifest is not None:
            with open(os.path.join(out_dir, "manifest.json"), "w") as mf:
                json.dump(manifest.to_dict(), mf, indent=2, default=str)

    def write(self, **row: Any) -> None:
        row.setdefault("probe", self.name)
        self._f.write(json.dumps(row, default=_jsonable) + "\n")
        self.n += 1

    def write_many(self, rows) -> None:
        for r in rows:
            self.write(**r)

    def close(self) -> None:
        if self._f and not self._f.closed:
            self._f.flush()
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _jsonable(o: Any):
    """Fallback encoder: numpy scalars/arrays and anything with tolist()."""
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def read_records(path: str) -> list[dict]:
    """Read a JSONL record file back for aggregation."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
