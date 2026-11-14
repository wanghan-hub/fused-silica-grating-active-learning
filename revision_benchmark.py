"""Paired cold-RCWA timing, nominal DE repeats and dense finalist verification.

Cold means no precomputed RCWA labels/checkpoints at the start of a measured
build or search. This is an application-cache experiment, not an OS cache flush.
Only certificates, fixed geometry lists and held-out identities are shared.
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata as md
import json
import logging
import os
from pathlib import Path
import platform
import re
import shutil
from time import perf_counter
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from dataset import (ExecutionSettings, audit_frame, close_rcwa_pool, compute_batch,
                     make_manifest, write_csv, write_parquet)
from optics import Geometry, OpticalGrid, fingerprint, save_json
from optimize import optical_metrics, run_optimization
from rcwa_solver import SolverConfig
from validate_physics import require_validation

LOG = logging.getLogger(__name__)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@contextlib.contextmanager
def _timing_lock(path: Path):
    """Advisory cross-process lock; a process exit releases it without deletion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if not f.tell():
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as err:
            raise RuntimeError("Another revision timing run holds " + str(path)) from err
        try:
            yield
        finally:
            if os.name == "nt":
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _hardware(config: Mapping[str, Any]) -> dict:
    from surrogate import torch, select_device
    # Identical when the build and all pairs run in one process, and when a
    # later optimizer seed is launched from a fresh CLI process with a saved build.
    torch.set_num_threads(int(config["model"]["cpu_threads"]))
    torch.use_deterministic_algorithms(bool(config["model"]["deterministic"]), warn_only=False)
    device = select_device(config["model"]["device"])
    cpu = platform.processor() or platform.machine()
    info = Path("/proc/cpuinfo")
    if info.is_file():
        text = info.read_text(errors="replace")
        match = re.search(r"^model name\s*:\s*(.+)$", text, re.MULTILINE)
        if match:
            cpu = match.group(1).strip()
    elif os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    versions = {}
    for name in ("torch", "numpy", "pandas", "pyarrow", "scipy", "grcwa"):
        versions[name] = md.version(name)
    result = {"host": platform.node(), "os": platform.platform(),
              "python": platform.python_version(), "cpu": cpu, "logical_cpus": os.cpu_count(),
              "scheduling": asdict(ExecutionSettings.from_config(config)),
              "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
              "torch_cpu_threads": torch.get_num_threads(),
              "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
              "training_device": device, "versions": versions,
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "cuda_runtime": torch.version.cuda}
    if device.startswith("cuda"):
        props = torch.cuda.get_device_properties(torch.device(device))
        result["gpu"] = {"name": props.name, "memory_bytes": props.total_memory,
                         "capability": [props.major, props.minor]}
    return result


def _sync(config: Mapping[str, Any]) -> None:
    from surrogate import torch, select_device
    device = select_device(config["model"]["device"])
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _phase(folder: Path, signature: str, producer: Callable[[Path], dict]) -> dict:
    """Reuse a complete measured phase or allocate a new empty attempt. No partial timings."""
    done = folder / "complete.json"
    if done.exists():
        item = _read(done)
        if item.get("signature") != signature or not item.get("complete"):
            raise RuntimeError(f"Refusing a different or incomplete protocol at {folder}")
        return item
    folder.mkdir(parents=True, exist_ok=True)
    number = 1
    while (folder / f"attempt_{number:03d}").exists():
        number += 1
    dest = folder / f"attempt_{number:03d}"
    dest.mkdir()
    LOG.info("Starting fresh measured phase: %s (earlier attempts are never counted)", dest)
    result = producer(dest)
    result.update(signature=signature, complete=True, attempt=dest.name,
                  finished_utc=datetime.now(timezone.utc).isoformat())
    save_json(done, result)
    return result


def _empty_config(cfg: Mapping[str, Any], dest: Path) -> dict:
    c = copy.deepcopy(dict(cfg))
    # Do not inherit *any* external shards from prepare_v2 or neighbouring repeats.
    c["reuse"] = {"data_seed": cfg["reuse"]["data_seed"], "sibling_repeat_cache": False}
    c["_root"] = str(dest)
    c["_learning_data_root"] = str(dest)
    c["runtime"] = {**c.get("runtime", {}), "stop_after_subtask": None}
    return c


def _fixed_inputs(c: dict) -> pd.DataFrame:
    source, dest = Path(c["_baseline_root"]), Path(c["_root"])
    names = ["data/manifest.json", "data/manifest.parquet", "validation/physics_summary.json"]
    if c["dataset"]["require_initial_comsol"]:
        names.append("validation/comsol_summary.json")
    for name in names:
        where = dest/name
        where.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, where)
    require_validation(c)
    manifest = make_manifest(c)
    if manifest.geometry_id.duplicated().any() or (manifest.groupby("family_id").role.nunique() != 1).any():
        raise ValueError("Invalid frozen manifest")
    if manifest.groupby("role").size().to_dict() != c["dataset"]["counts"]:
        raise ValueError("Frozen geometry counts disagree with config")
    extras = []
    for role in ("iid", "ood"):
        m = source/"confirmation_tests"/f"{role}_manifest.parquet"
        ids = pd.read_parquet(m, columns=["geometry_id"]).geometry_id.tolist()
        if len(ids) != c["dataset"]["counts"][role] or len(set(ids)) != len(ids):
            raise ValueError(f"Unexpected frozen {role} IDs")
        extras.extend(ids)
    if len(set(extras)) != len(extras) or set(extras) & set(manifest.geometry_id):
        raise ValueError("Original and fresh confirmation sets must be disjoint")
    c["_extra_protected_ids"] = sorted(extras)
    return manifest


def _cold_build(cfg: Mapping[str, Any], dest: Path) -> dict:
    from revision_learning import run_revision_learning
    c = _empty_config(cfg, dest)
    manifest = _fixed_inputs(c)
    grid = OpticalGrid.from_config(c["grid"])
    first, stats = ("train_initial", "validation", "calibration"), {}
    close_rcwa_pool()
    _sync(c)
    start = perf_counter()
    for role in first:
        group = manifest[manifest.role == role]
        gs = [Geometry.from_mapping(r) for r in group.to_dict("records")]
        # Audited training/validation/calibration labels MUST be solved here, not copied.
        resolution = "base" if role == "train_initial" else c["dataset"]["evaluation_solver"]
        solver = SolverConfig.from_mapping(c["rcwa"][resolution])
        t = perf_counter()
        labels, cost = compute_batch(gs, grid, c, solver)
        qc = audit_frame(labels, gs, grid, solver)
        if not qc["passed"] or cost["cache_hits"] or cost["new_state_solves"] != cost["logical_state_solves"]:
            raise RuntimeError(f"{role} failed cold-run accounting or QC: {cost}, {qc}")
        labels["role"] = role
        labels["family_id"] = labels.geometry_id.map(dict(zip(group.geometry_id, group.family_id)))
        write_parquet(labels, dest/"data"/f"{role}.parquet")
        stats[role] = {**cost, "resolution": resolution, "phase_wall_s": perf_counter() - t}
        del labels
    run_revision_learning(c, ["task_diverse"])
    _sync(c)
    close_rcwa_pool()
    elapsed = perf_counter() - start
    al_path = dest/"active_learning"/"task_diverse"
    curve = pd.read_csv(al_path/"learning_curve.csv")
    rounds = c["active_learning"]["rounds"]
    if list(curve["round"]) != list(range(rounds + 1)) or curve["resumed_partial"].astype(str).str.lower().isin(["true","1"]).any():
        raise RuntimeError("Fresh benchmark must have complete, uninterrupted AL rounds")
    extra = curve.loc[curve["round"] > 0]
    if extra.cache_hits.sum() or (extra.new_state_solves <= 0).any():
        raise RuntimeError("Unexpected cached/empty new AL queries in cold benchmark")
    n_new = int(sum(s["new_state_solves"] for s in stats.values()) + extra["new_state_solves"].sum())
    logical = int(curve.logical_state_solves.iloc[-1] + sum(stats[r]["logical_state_solves"] for r in first[1:]))
    if n_new != logical:
        raise RuntimeError(f"Model ledger mismatch: {n_new} actual, {logical} logical")
    model = al_path / f"round_{rounds:03d}" / "model"
    meta = _read(model/"ensemble.json")
    if not meta.get("training_qc", {}).get("passed") or _read(model/"calibration.json")["model_signature"] != meta["training_signature"]:
        raise RuntimeError("Cold ensemble did not pass training/calibration guards")
    digests = {n: _hash(model/n) for n in ["ensemble.json", "calibration.json", *meta["members"]]}
    return {"al_seed": c["seed"], "model_relative": model.relative_to(dest).as_posix(),
            "model_sha256": digests, "wall_s": elapsed, "logical_states": logical, "new_states": n_new,
            "initial_label_phases": stats,
            "active_label_wall_s": float(extra.labels_wall_s.sum()),
            "training_s": float(curve.training_s.sum()), "acquisition_s": float(curve.acquisition_s.sum()),
            "pool_generation_s": float(curve.pool_generation_s.sum()),
            "calibration_s": float(curve.calibration_s.sum()),
            "initial_and_active_states_at_base": int(curve.logical_state_solves.iloc[-1]),
            "validation_calibration_states": int(sum(stats[r]["logical_state_solves"] for r in first[1:])),
            "timing_scope": "wall includes fresh model labels, AL loop, all fits/calibrations, QC and writes. Phase components are nested, not additive."}


def _cold_search(cfg: Mapping[str, Any], dest: Path, method: str, seed: int, model: Path) -> dict:
    c = _empty_config(cfg, dest)
    c["seed"] = seed
    c["optimization"].update(methods=[method], objectives=["nominal"], model_directory=str(model))
    _fixed_inputs(c)
    close_rcwa_pool()
    _sync(c)
    started = perf_counter()
    summary = run_optimization(c)[method + "_nominal"]
    _sync(c)
    close_rcwa_pool()
    wall = perf_counter() - started
    root = dest/"optimization"
    history = pd.read_csv(root/(method+"_nominal")/"true_history.csv", float_precision="round_trip")
    if summary.get("wall_time_is_lower_bound_after_resume") or history.cache_hits.sum() or int(history.new_state_solves.sum()) != summary["logical_state_solves"]:
        raise RuntimeError(f"{method} search is not a cold, complete timing")
    candidate = pd.read_parquet(root/"candidates.parquet").iloc[0].to_dict()
    return {"candidate": candidate, "search_grid_mean_R": candidate["mean_R"],
            "search_wall_s": wall, "search_logical_states": int(summary["logical_state_solves"]),
            "search_new_states": int(history.new_state_solves.sum()), "search_cache_hits": int(history.cache_hits.sum()),
            "search_geometries": len(history), "termination": summary["termination"],
            "rcwa_evaluation_call_wall_s": float(history.evaluation_wall_s.sum()),
            "sum_solver_elapsed_s_not_wall": float(history.solver_elapsed_s.sum()),
            "history": (root/(method+"_nominal")/"true_history.csv").relative_to(dest).as_posix()}


def _dense(cfg: Mapping[str, Any], dest: Path, nominal: Mapping[str, Any]) -> dict:
    g = Geometry.from_mapping(nominal["candidate"])
    c = _empty_config(cfg, dest)
    grid = OpticalGrid.from_config(c["final_validation"]["grid"])
    solver = SolverConfig.from_mapping(c["rcwa"]["reference"])
    close_rcwa_pool()
    try:
        frame, cost = compute_batch([g], grid, c, solver)
    finally:
        close_rcwa_pool()
    qc = audit_frame(frame, [g], grid, solver)
    if not qc["passed"] or cost["cache_hits"] or cost["new_state_solves"] != cost["logical_state_solves"]:
        raise RuntimeError("Dense cross-check invalid or cached")
    return {"geometry_id": g.id, "dense_grid_metrics": optical_metrics(frame, grid, c),
            "cost_excluded_from_search_timing": cost, "qc": qc}


def run_benchmark(cfg: Mapping[str, Any], optimizer_seed: int, *, campaign: str = "primary") -> dict:
    """One pair of nominal DE searches, sharing one independently measured build per AL seed.

    A complete build/search can be reused at its *original* measured cost; interrupted
    phases get a newly numbered empty attempt. Never rerun over an incomplete cache.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", campaign):
        raise ValueError("campaign must be a short directory name without separators")
    if not 0 <= int(optimizer_seed) <= 2**32-102:
        raise ValueError("optimizer_seed out of range")
    root = Path(cfg["_root"]).resolve()
    baseline = Path(cfg["_baseline_root"]).resolve()
    if root == baseline or root in baseline.parents or baseline in root.parents:
        raise ValueError("Baseline and output must be disjoint")
    base = root/"benchmark"/campaign
    with _timing_lock(root.parent/".rcwa_benchmark.lock"):
        try:
            return _run_pair(cfg, int(optimizer_seed), base)
        finally:
            close_rcwa_pool()


def _run_pair(cfg: Mapping[str, Any], seed: int, base: Path) -> dict:
    base.mkdir(parents=True, exist_ok=True)
    hardware = _hardware(cfg)
    source = Path(__file__).parent
    # Universal newline decoding makes source hashes stable on Windows/Linux checkouts.
    code = {name: fingerprint((source/name).read_text(encoding="utf-8")) for name in
            ("revision_benchmark.py","revision_learning.py","active_learning.py","optimize.py",
             "surrogate.py","dataset.py","optics.py","rcwa_solver.py","validate_physics.py")}
    baseline = Path(cfg["_baseline_root"])
    names = ["data/manifest.parquet","data/manifest.json","validation/physics_summary.json",
             "confirmation_tests/iid_manifest.parquet","confirmation_tests/ood_manifest.parquet",
             "confirmation_tests/protocol.json","confirmation_tests/audit.json"]
    if cfg["dataset"]["require_initial_comsol"]:
        names.append("validation/comsol_summary.json")
    science = {k: cfg[k] for k in ("seed","geometry","sampling","grid","rcwa","model","metrics","calibration",
             "dataset","comsol","validation","tolerance","optimization","active_learning","final_validation")}
    key = fingerprint({"science": science, "hardware": hardware, "code": code,
                       "frozen_inputs": {n: _hash(baseline/n) for n in names}, "data_seed": cfg["reuse"]["data_seed"]})
    protocol = base/"protocol.json"
    if protocol.exists() and _read(protocol)["signature"] != key:
        raise RuntimeError(f"Configuration/code/hardware/frozen manifest changed: use a new --campaign ({base})")
    save_json(protocol, {"signature": key, "hardware": hardware, "code": code, "science": science})
    built = _phase(base/"model_build", key, lambda d: _cold_build(cfg, d))
    model = base/"model_build"/built["attempt"]/built["model_relative"]
    if any(_hash(model/n) != value for n,value in built["model_sha256"].items()):
        raise RuntimeError("Frozen cold-build model changed")
    trial = base/f"opt_{seed}"
    pair_key = fingerprint([key, seed, built["model_sha256"]])
    # Preassigned deterministic order independent of the resulting scores or times.
    # Same DE seed and initial Sobol sequence for the two methods in each pair.
    methods = ["direct","surrogate"] if int(fingerprint(seed)[-1], 16) % 2 == 0 else ["surrogate","direct"]
    data = {}
    for method in methods:
        directory = trial/method
        subkey = fingerprint([pair_key, method])
        summary = _phase(directory, subkey, lambda d: _cold_search(cfg, d, method, seed, model))
        data[method] = {**summary, "standalone_logical_states": summary["search_logical_states"] + (built["logical_states"] if method=="surrogate" else 0),
                        "standalone_wall_s": summary["search_wall_s"] + (built["wall_s"] if method=="surrogate" else 0)}
    # Dense finalists are measured after both searches and never charged to either route.
    for method in methods:
        check = _phase(trial/(method+"_dense"), fingerprint([pair_key, method, data[method]["candidate"]["geometry_id"]]),
                       lambda d: _dense(cfg, d, data[method]))
        data[method]["dense_validation"] = check
        data[method]["dense_grid_mean_R"] = check["dense_grid_metrics"]["mean_R"]
    sur_budget = int(data["surrogate"]["standalone_logical_states"])
    direct = data["direct"]
    h = pd.read_csv(trial/"direct"/direct["attempt"]/direct["history"], float_precision="round_trip")
    fair = h[h.logical_state_solves <= sur_budget]
    hits = h[h.true_objective <= data["surrogate"]["search_grid_mean_R"]]
    budget_report = {"surrogate_inclusive_states": sur_budget,
                     "direct_actual_states_at_or_below_budget": int(fair.logical_state_solves.max()) if len(fair) else 0,
                     "direct_search_grid_best_at_or_below_budget": float(fair.true_objective.min()) if len(fair) else None,
                     "direct_reached_budget": direct["search_logical_states"] >= sur_budget,
                     "direct_first_states_matching_surrogate_score": int(hits.logical_state_solves.iloc[0]) if len(hits) else None}
    record = {"complete": True, "timing_valid": True, "protocol_signature": pair_key,
              "campaign_signature": key, "al_seed": cfg["seed"], "optimizer_seed": seed,
              "search_order": methods, "hardware": hardware, "model_build": built,
              "budget_matched": budget_report, **data,
              "note": "Each standalone surrogate time includes the same measured model build once. Dense RCWA excluded for both. Across optimizer seeds with this fixed model, amortize build only once."}
    save_json(trial/"paired_result.json", record)
    pairs = [_read(p) for p in sorted(base.glob("opt_*/paired_result.json"))]
    if any(p["campaign_signature"] != key for p in pairs):
        raise RuntimeError("Mixed benchmark campaigns")
    n = len(pairs)
    rows = [{"optimizer_seed": p["optimizer_seed"], "al_seed": p["al_seed"], "method": m,
             **{k: p[m][k] for k in ("search_grid_mean_R","dense_grid_mean_R","standalone_wall_s",
                                       "search_wall_s","search_logical_states","standalone_logical_states",
                                       "search_new_states","search_cache_hits","termination")},
             "model_build_share_s_for_completed_pairs": built["wall_s"]/n if m=="surrogate" else 0,
             "amortized_wall_s_for_completed_pairs": p[m]["search_wall_s"] + (built["wall_s"]/n if m=="surrogate" else 0)}
            for p in pairs for m in ("direct","surrogate")]
    table = pd.DataFrame(rows)
    write_csv(table, base/"summary.csv")
    grouped = table.groupby("method")[["search_grid_mean_R","dense_grid_mean_R","standalone_wall_s","search_wall_s"]].agg(["mean","std"])
    grouped.columns = ["_".join(a) for a in grouped.columns]
    grouped["optimizer_seeds"] = n
    write_csv(grouped.reset_index(), base/"across_optimizer_seeds.csv")
    return record

