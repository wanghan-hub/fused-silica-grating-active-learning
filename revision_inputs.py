"""Portable, checksummed inputs for the PIER revision; no ML/simulation imports.

Copy only the audited split labels, historical confirmation data, and original
three fitted models. The old run stays read-only. New repeats contain separate
copies of their baseline data, never mutable hardlinks to published results.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping


ROLES = ("train_initial", "validation", "calibration", "iid", "ood")
ORIGINAL = ("random", "uncertainty", "task_diverse")
SNAPSHOT = "v3_input_snapshot.json"
PREPARED = "v3_repeat_inputs.json"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _disjoint(first: Path, second: Path) -> None:
    if first == second or first in second.parents or second in first.parents:
        raise ValueError(f"Input and output trees must be disjoint: {first} vs {second}")


def _atomic_json(path: Path, obj: Mapping[str, Any]) -> None:
    payload = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False,
                                     mode="w", encoding="utf-8") as stream:
        tmp = Path(stream.name)
        try:
            stream.write(payload)
        except BaseException:
            stream.close()
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _install_file(source: Path, dest: Path, sha: str) -> None:
    """Never replace an existing file with different contents."""
    if dest.exists():
        if not dest.is_file() or dest.is_symlink() or _sha(dest) != sha:
            raise RuntimeError(f"Conflicting existing input (will not overwrite): {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=dest.parent, prefix=".import-", suffix=".tmp")
    os.close(fd)
    temp = Path(name)
    try:
        shutil.copyfile(source, temp)
        if _sha(temp) != sha:
            raise RuntimeError(f"Copy checksum mismatch: {source}")
        os.replace(temp, dest)
    finally:
        temp.unlink(missing_ok=True)


def _repeat_files() -> list[str]:
    return ["data/manifest.json", "data/manifest.parquet", "data/audit.json",
            *[f"data/{role}.parquet" for role in ROLES],
            "validation/physics_summary.json", "validation/comsol_summary.json"]


def _source_metadata(source: Path) -> tuple[list[str], dict]:
    """Check the v2 frozen protocol and archived QC, without loading pickles."""
    reuse = _json(source / "reuse_manifest.json")
    if reuse.get("passed") is not True:
        raise ValueError("The old run does not have a passing reuse manifest")
    protocol = _json(source / "confirmation_tests/protocol.json")
    sig = protocol["signature"]
    if sig != _fingerprint({k: v for k, v in protocol.items() if k != "signature"}):
        raise ValueError("Invalid v2 confirmation protocol fingerprint")
    confirmation = _json(source / "confirmation_tests/audit.json")
    if confirmation["protocol_signature"] != sig or not confirmation.get("passed"):
        raise ValueError("Confirmation audit does not pass / match frozen protocol")
    source_cfg = _json(source / "logs/active_learning_invocation.json")["config"]
    data_seed = int(reuse["data_seed"])
    counts = source_cfg["dataset"]["counts"]
    nstates = int(_json(source / "validation/physics_summary.json")["single_geometry_states"])
    if protocol["physical_config"] != {k: source_cfg[k] for k in ("grid", "rcwa", "geometry")}:
        raise ValueError("Frozen confirmation and original AL physical settings differ")
    base_audit = _json(source / "data/audit.json")
    for role in ROLES:
        check = base_audit[role]["audit"]
        if not check["passed"] or check["geometries"] != counts[role] or check["rows"] != nstates * counts[role]:
            raise ValueError(f"Initial split {role} failed recorded QC/counts")
    for role in ("iid", "ood"):
        check = confirmation[role]["audit"]
        if not check["passed"] or check["geometries"] != counts[role] or check["rows"] != nstates * counts[role]:
            raise ValueError(f"Confirmation split {role} failed recorded QC/counts")
    data_sig = _fingerprint({**{k: source_cfg[k] for k in ("geometry","sampling","dataset","grid")},
                             "seed": data_seed})
    if _json(source / "data/manifest.json")["signature"] != data_sig:
        raise ValueError("Original dataset manifest does not match fixed data seed and physics")
    physics = _json(source / "validation/physics_summary.json")
    cross = _json(source / "validation/comsol_summary.json")
    if not (physics.get("passed") and cross.get("passed") and
            physics["physical_signature"] == cross["physical_signature"]):
        raise ValueError("Inconsistent initial RCWA / COMSOL certificates")
    if _sha(source / "optimization/candidates.csv") != protocol["candidate_sha256"]:
        raise ValueError("Archived finalist list does not match frozen reference")
    files = _repeat_files() + [
        "logs/active_learning_invocation.json", "reuse_manifest.json", "optimization/candidates.csv",
        *["confirmation_tests/" + n for n in ("iid.parquet", "ood.parquet", "iid_manifest.parquet",
          "ood_manifest.parquet", "protocol.json", "audit.json")]]
    frozen = {k.replace(chr(92), "/"): s for k, s in protocol["models"].items()}
    last = int(source_cfg["active_learning"]["rounds"])
    for strategy in ORIGINAL:
        al = f"active_learning/{strategy}"
        rd = f"{al}/round_{last:03d}"
        model = rd + "/model"
        meta = _json(source / model / "ensemble.json")
        calibrated = _json(source / model / "calibration.json")
        if (meta["training_signature"] != frozen[f"{model}/ensemble.json"] or
                meta["training_signature"] != calibrated["model_signature"] or
                not meta["training_qc"]["passed"]):
            raise ValueError(f"Unverified original fitted model: {strategy}")
        members = meta["members"]
        if (len(members) != source_cfg["model"]["ensemble_size"] or len(set(members)) != len(members)
                or any(not isinstance(m,str) or Path(m).name != m or not m.endswith(".pt") for m in members)):
            raise ValueError(f"Invalid model member list for {strategy}")
        files += [f"{al}/learning_curve.csv", f"{al}/protocol.json", f"{rd}/round.json",
                  f"{rd}/membership.parquet", *[f"{model}/{name}" for name in
                  ("ensemble.json", "calibration.json", "training_qc.json", *members)]]
    return sorted(files), {"schema": 1, "base_seed": int(source_cfg["seed"]),
                           "data_seed": data_seed, "original_strategies": list(ORIGINAL),
                           "physics_signature": physics["physical_signature"],
                           "manifest_signature": data_sig, "confirmation_signature": sig}


def import_inputs(source_v2_run: str | Path, dest_baseline: str | Path) -> dict:
    """Import a self-contained subset. Re-imports verify and never overwrite changes."""
    source, dest = Path(source_v2_run).resolve(), Path(dest_baseline).resolve()
    _disjoint(source, dest)
    files, meta = _source_metadata(source)
    source_attestation = {f["path"].replace(chr(92), "/"):f["sha256"]
                          for f in _json(source / "reuse_manifest.json")["files"]}
    info = {}
    for name in files:
        p = source / name
        if not p.is_file() or source not in p.resolve().parents:
            raise FileNotFoundError(f"Missing v2 input: {p}")
        digest = _sha(p)
        if name in source_attestation and source_attestation[name] != digest:
            raise RuntimeError(f"Historical v2 data checksum mismatch: {name}")
        info[name] = {"bytes": p.stat().st_size, "sha256": digest}
    snapshot = {**meta, "files": info}
    snapshot["signature"] = _fingerprint(snapshot)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / SNAPSHOT).exists() and _json(dest / SNAPSHOT) != snapshot:
        raise RuntimeError("Destination belongs to a different frozen input snapshot")
    for name, item in info.items():
        target = dest / name
        if dest not in target.resolve().parents:
            raise ValueError(f"Input path escapes destination: {target}")
        _install_file(source / name, target, item["sha256"])
    _atomic_json(dest / SNAPSHOT, snapshot)
    return {"baseline_root": str(dest), "files": len(info), "bytes": sum(v["bytes"] for v in info.values()),
            "signature": snapshot["signature"], "original_seed": snapshot["base_seed"]}


def verify_snapshot(base: str | Path) -> dict:
    """Verify bytes after upload, without access to the original E: drive or v2."""
    root = Path(base).resolve()
    snap = _json(root / SNAPSHOT)
    if snap["signature"] != _fingerprint({k:v for k,v in snap.items() if k != "signature"}):
        raise ValueError("Input inventory signature mismatch")
    for name,item in snap["files"].items():
        path = root / name
        if (root not in path.resolve().parents or not path.is_file() or
                path.stat().st_size != item["bytes"] or _sha(path) != item["sha256"]):
            raise RuntimeError(f"Frozen input is missing or changed: {path}")
    return snap


def prepare_repeat(cfg: Mapping[str, Any]) -> dict:
    """Create an isolated repeat, with unchanged split IDs for all model seeds."""
    base, dest = Path(cfg["_baseline_root"]).resolve(), Path(cfg["_root"]).resolve()
    _disjoint(base, dest)
    snap = verify_snapshot(base)
    source_cfg = _json(base / "logs/active_learning_invocation.json")["config"]
    for key in ("grid", "geometry", "rcwa", "validation", "comsol", "sampling", "dataset"):
        if cfg[key] != source_cfg[key]:
            raise ValueError(f"Config {key!r} differs from the validated v2 baseline")
    if int(cfg["reuse"]["data_seed"]) != snap["data_seed"]:
        raise ValueError("Vary the model/acquisition seed, not the fixed split seed")
    dest.mkdir(parents=True, exist_ok=True)
    payload = {"schema": 1, "input_signature": snap["signature"], "seed": int(cfg["seed"]),
               "data_seed": snap["data_seed"], "manifest_signature": snap["manifest_signature"]}
    marker = dest / PREPARED
    if marker.exists() and _json(marker) != payload:
        raise ValueError(f"Destination was already prepared for a different run: {dest}")
    if not marker.exists() and any((dest/p).exists() for p in ("active_learning", "optimization", "models")):
        raise ValueError(f"Refusing to prepare existing experiments: {dest}")
    for name in _repeat_files():
        target = dest / name
        if dest not in target.resolve().parents:
            raise ValueError(f"Repeat input path escapes directory: {target}")
        _install_file(base / name, target, snap["files"][name]["sha256"])
    _atomic_json(marker, payload)
    return {"prepared": True, "root": str(dest), **payload}


if __name__ == "__main__":
    import argparse
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source_v2_repeat", type=Path)
    cli.add_argument("destination_snapshot", type=Path)
    args = cli.parse_args()
    print(json.dumps(import_inputs(args.source_v2_repeat, args.destination_snapshot), indent=2))
