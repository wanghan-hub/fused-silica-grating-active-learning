"""Grouped manifests, constrained sampling, resumable RCWA shards and audits."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from dataclasses import asdict, dataclass, field, replace
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping
import json
import logging
import os
import atexit
import signal
import shutil
from hashlib import sha256

import numpy as np
import pandas as pd

from optics import (Geometry, OpticalGrid, GEOMETRY_COLUMNS, STATE_COLUMNS,
                    domain_bounds, feasibility, robust_feasible, fingerprint, save_json)
from rcwa_solver import SolverConfig, qc_errors, solve_state, solver_identity

LOGGER = logging.getLogger(__name__)

# This compatibility entry is restricted to the inspected, unchanged physics
# implementation. Old shards still undergo the same full QC before reuse.
_LEGACY_PHYSICS_SOURCE = "293c5cda3a9fe6cfa619729e10b9e897830d2216d0cead6b38db35d438caad34"
_LEGACY_CACHE_SOURCE = "6a1f2f9c1e91a59f8e334bfc6bae113c8f91ce8c214f4950f3b8f9038856e678"
_CACHE_CONTRACT = "independent_states_same_retry_and_qc_v1"
_POOL: ProcessPoolExecutor | None = None
_POOL_SETTINGS: tuple[int, int] | None = None


@dataclass(frozen=True)
class ExecutionSettings:
    """Scheduling only: never part of dataset splits or physical certificates."""
    workers: int
    blas_threads: int
    chunk_size: int
    log_interval_s: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ExecutionSettings:
        options = config.get("runtime", {})
        count = options.get("rcwa_workers", config["dataset"]["workers"])
        if count == "auto":
            count = max(1, min(16, (os.cpu_count() or 1) - 2))
        obj = cls(int(count), int(options.get("rcwa_blas_threads", config["dataset"]["blas_threads"])),
                  int(options.get("state_chunk_size", 8)), float(options.get("log_interval_s", 30)))
        if min(obj.workers, obj.blas_threads, obj.chunk_size) < 1 or obj.log_interval_s <= 0:
            raise ValueError("Runtime workers/threads/chunk_size/log_interval_s must be positive")
        if os.name == "nt" and obj.workers > 61:
            raise ValueError("Windows ProcessPoolExecutor supports at most 61 workers")
        return obj


def close_rcwa_pool() -> None:
    """Cancel queued chunks; let running chunks finish their atomic checkpoint."""
    global _POOL, _POOL_SETTINGS
    pool, _POOL, _POOL_SETTINGS = _POOL, None, None
    if pool is not None:
        pool.shutdown(wait=True, cancel_futures=True)


def _worker_initialize() -> None:
    # The parent handles Ctrl+C; workers finish their current short chunk safely.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _get_pool(options: ExecutionSettings) -> ProcessPoolExecutor:
    global _POOL, _POOL_SETTINGS
    signature = (options.workers, options.blas_threads)
    if _POOL is not None and _POOL_SETTINGS != signature:
        close_rcwa_pool()
    if _POOL is None:
        for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
            os.environ[name] = str(options.blas_threads)
        _POOL = ProcessPoolExecutor(max_workers=options.workers, mp_context=get_context("spawn"),
                                    initializer=_worker_initialize)
        _POOL_SETTINGS = signature
        LOGGER.info("Persistent RCWA pool: %d CPU workers, %d BLAS thread(s)/worker, %d states/chunk",
                    options.workers, options.blas_threads, options.chunk_size)
    return _POOL


atexit.register(close_rcwa_pool)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Atomic history/progress table replacement, including on Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    os.replace(temporary, path)


def sample_geometries(count: int, config: Mapping[str, Any], seed: int,
                      domain: str = "cover", exclude: set[str] | None = None,
                      robust: bool = False) -> list[Geometry]:
    """Rejection-filtered independent scrambled blocks (not a pristine Sobol net)."""
    from scipy.stats import qmc
    if count < 1:
        raise ValueError("Geometry count must be positive")
    bounds = domain_bounds(config, domain)
    excluded = set(exclude or ())
    result: list[Geometry] = []
    block_size = int(config["sampling"]["block_size"])
    if block_size < 2 or block_size & (block_size - 1):
        raise ValueError("Sampling block_size must be a power of two")
    tested = 0
    block = 0
    while len(result) < count and tested < config["sampling"]["max_candidates"]:
        method = config["sampling"]["method"]
        if method == "sobol":
            unit = qmc.Sobol(4, scramble=True, seed=seed + block).random_base2(int(np.log2(block_size)))
        elif method == "lhs":
            unit = qmc.LatinHypercube(4, seed=seed + block).random(block_size)
        elif method == "random":
            unit = np.random.default_rng(seed + block).random((block_size, 4))
        else:
            raise ValueError("sampling.method must be sobol, lhs or random")
        for values in qmc.scale(unit, bounds[:, 0], bounds[:, 1]):
            tested += 1
            g = Geometry.from_array(values)
            if g.id in excluded or feasibility(g, config, domain):
                continue
            if domain != "ood":
                ood = domain_bounds(config, "ood")
                if np.all(g.array() >= ood[:, 0]) and np.all(g.array() <= ood[:, 1]):
                    continue
            if robust and not robust_feasible(g, config):
                continue
            result.append(g)
            excluded.add(g.id)
            if len(result) == count:
                break
        block += 1
    if len(result) < count:
        raise RuntimeError(f"Only {len(result)}/{count} feasible geometries after {tested} candidates")
    return result


def manifest_signature(config: Mapping[str, Any]) -> str:
    values = {k: config[k] for k in ("geometry", "sampling", "dataset", "grid", "seed")}
    values["seed"] = config.get("reuse", {}).get("data_seed", config["seed"])
    return fingerprint(values)


def make_manifest(config: Mapping[str, Any]) -> pd.DataFrame:
    directory = Path(config["_root"]) / "data"
    path, meta_path = directory / "manifest.parquet", directory / "manifest.json"
    signature = manifest_signature(config)
    if path.exists():
        if not meta_path.exists() or json.loads(meta_path.read_text())["signature"] != signature:
            raise RuntimeError("Manifest settings changed. Use a new experiment_id")
        return pd.read_parquet(path)
    rows, excluded = [], set()
    for i, (role, count) in enumerate(config["dataset"]["counts"].items()):
        domain = "ood" if role == "ood" else "cover"
        group = sample_geometries(int(count), config, int(config.get("reuse", {}).get("data_seed", config["seed"])) + i * 100003,
                                  domain, excluded)
        for g in group:
            rows.append({**g.record(), "family_id": g.id, "role": role, "active_round": 0})
            excluded.add(g.id)
    manifest = pd.DataFrame(rows)
    if manifest.geometry_id.duplicated().any():
        raise RuntimeError("Duplicate geometries across manifest roles")
    write_parquet(manifest, path)
    save_json(meta_path, {"signature": signature, "rows": len(manifest)})
    return manifest


def _cache_key(g: Geometry, grid: OpticalGrid, settings: SolverConfig,
               retry: Mapping[str, Any]) -> str:
    source_hash = fingerprint({"physics": physics_source_hash(), "contract": _CACHE_CONTRACT})
    return _key_with_source(g, grid, settings, retry, source_hash)


def physics_source_hash() -> str:
    return fingerprint({name: Path(__file__).with_name(name).read_text(encoding="utf-8")
                        for name in ("optics.py", "rcwa_solver.py")})


def _key_with_source(g: Geometry, grid: OpticalGrid, settings: SolverConfig,
                     retry: Mapping[str, Any], source_hash: str) -> str:
    return fingerprint({"geometry": asdict(g), "grid": grid.record(),
                        "solver": asdict(settings), "retry": dict(retry),
                        "identity": solver_identity(), "source_hash": source_hash})


def _resume_key(g: Geometry, grid: OpticalGrid, settings: SolverConfig,
                retry: Mapping[str, Any], directory: Path) -> str:
    current = _cache_key(g, grid, settings, retry)
    def exists(key: str) -> bool:
        return ((directory / f"{key}.parquet").exists() or
                (directory / "attempts" / f"{key}.0.parquet").exists() or
                (directory / "chunks" / key).exists())
    if exists(current) or physics_source_hash() != _LEGACY_PHYSICS_SOURCE:
        return current
    legacy = _key_with_source(g, grid, settings, retry, _LEGACY_CACHE_SOURCE)
    return legacy if exists(legacy) else current


def audit_frame(frame: pd.DataFrame, geometries: list[Geometry], grid: OpticalGrid,
                settings: SolverConfig) -> dict[str, Any]:
    problems: list[str] = []
    required = {"geometry_id", *GEOMETRY_COLUMNS, *STATE_COLUMNS, "R", "T", "status",
                "orders_json", "energy_error", "solver_config_hash", "attempt", "elapsed_s"}
    if not required <= set(frame.columns):
        return {"passed": False, "problems": [f"missing_columns:{required - set(frame.columns)}"]}
    if frame.duplicated(["geometry_id", *STATE_COLUMNS]).any():
        problems.append("duplicate_states")
    expected_ids = {g.id for g in geometries}
    if set(frame.geometry_id) != expected_ids:
        problems.append("geometry_set_mismatch")
    expected_states = set(grid.states())
    for g in geometries:
        group = frame[frame.geometry_id == g.id]
        states = set(group[STATE_COLUMNS].itertuples(index=False, name=None))
        if states != expected_states:
            problems.append(f"incomplete_grid:{g.id}")
        if len(group) and not np.allclose(group[GEOMETRY_COLUMNS].values, g.array(), rtol=0, atol=1e-10):
            problems.append(f"geometry_mismatch:{g.id}")
    for row in frame.to_dict("records"):
        if row["status"] != "ok" or qc_errors(row, settings):
            problems.append(f"failed_state:{row['geometry_id']}")
        try:
            orders = json.loads(row["orders_json"])
            if not orders or abs(sum(v["R"] for v in orders) - row["R"]) > settings.order_sum_tolerance or \
                    abs(sum(v["T"] for v in orders) - row["T"]) > settings.order_sum_tolerance:
                problems.append("order_sum_mismatch")
            if not np.isclose(row["energy_error"], abs(row["R"] + row["T"] - 1), atol=1e-12, rtol=0):
                problems.append("energy_metadata_mismatch")
        except (ValueError, TypeError, KeyError):
            problems.append("invalid_orders")
    return {"passed": not problems, "rows": len(frame), "geometries": len(geometries),
            "problems": sorted(set(problems)),
            "max_energy_error": float(frame.energy_error.max()) if len(frame) else None}


@dataclass
class _GeometryWork:
    geometry: Geometry
    key: str
    collected: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict)
    attempt_rows: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict)
    logical_states: int = 0
    solver_seconds: float = 0.0
    new_states: int = 0
    started: float = field(default_factory=perf_counter)


def _state_chunk_job(task: tuple[Any, ...]) -> str:
    """Worker writes its own checkpoint before returning; no nested process pools."""
    values, states, solver_dict, attempt, path, threads = task
    from threadpoolctl import threadpool_limits
    g, settings = Geometry.from_array(values), SolverConfig.from_mapping(solver_dict)
    rows = []
    with threadpool_limits(limits=threads):
        for state in states:
            row = solve_state(g, *state, settings)
            row["attempt"] = attempt
            rows.append(row)
    write_parquet(pd.DataFrame(rows), Path(path))
    return path


def _read_attempt(path: Path, work: _GeometryWork, states: set[tuple[Any, ...]],
                  settings: SolverConfig, attempt: int, *, complete: bool = False) -> dict:
    """Validate checkpoint identity; failed physical states are retained for retry."""
    rows = pd.read_parquet(path).to_dict("records")
    result = {}
    for row in rows:
        state = tuple(row[k] for k in STATE_COLUMNS)
        if (state not in states or state in result or row["geometry_id"] != work.geometry.id or
                row["solver_config_hash"] != settings.id or row["attempt"] != attempt or
                not np.allclose([row[k] for k in GEOMETRY_COLUMNS], work.geometry.array(),
                                rtol=0, atol=1e-10)):
            raise RuntimeError(f"Checkpoint identity mismatch: {path}")
        result[state] = row
    if complete and set(result) != states:
        raise RuntimeError(f"Incomplete attempt checkpoint: {path}")
    return result


def compute_batch(geometries: list[Geometry], grid: OpticalGrid, config: Mapping[str, Any],
                  settings: SolverConfig | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parallelize states, including a single geometry, with a persistent CPU pool.

    Candidate order, per-state Maxwell calculation, retries, QC and logical cost
    definitions are unchanged. Complete legacy shards/attempts remain reusable.
    Chunk files also preserve partial geometries across graceful interruption.
    """
    if not geometries or len({g.id for g in geometries}) != len(geometries):
        raise ValueError("Batch must contain nonempty, unique geometries")
    solver_identity()
    settings = settings or SolverConfig.from_mapping(config["rcwa"]["base"])
    directory = Path(config["_root"]) / "data" / "shards"
    directory.mkdir(parents=True, exist_ok=True)
    retry, execution = config["rcwa"]["retry"], ExecutionSettings.from_config(config)
    states = grid.states()
    state_index = {state: index for index, state in enumerate(states)}
    start = perf_counter()
    results, active = [], []
    source_roots = []
    if config.get("reuse", {}).get("source_repeat_root"):
        source_roots.append(Path(config["reuse"]["source_repeat_root"]))
    if config.get("reuse", {}).get("sibling_repeat_cache", False):
        own_root = Path(config["_root"]).resolve()
        source_roots += [p for p in sorted(own_root.parent.glob("repeat_*"))
                         if p.is_dir() and p.resolve() != own_root]
    for g in geometries:
        key = _resume_key(g, grid, settings, retry, directory)
        path, cost_path = directory / f"{key}.parquet", directory / f"{key}.cost.json"
        if path.exists() and cost_path.exists():
            cached = pd.read_parquet(path)
            if audit_frame(cached, [g], grid, settings)["passed"]:
                cost = json.loads(cost_path.read_text(encoding="utf-8"))
                results.append((g.id, str(path), {**cost, "cache_hit": True,
                                                 "actual_elapsed_s": 0.0, "new_state_solves": 0}))
                continue
        reused = False
        for source_root in source_roots:
            # Read complete, content-keyed source shards only; never write source attempts/logs.
            source_directory = Path(source_root) / "data" / "shards"
            source_key = _resume_key(g, grid, settings, retry, source_directory)
            source_path = source_directory / f"{source_key}.parquet"
            source_cost = source_directory / f"{source_key}.cost.json"
            if source_path.exists() and source_cost.exists():
                cached = pd.read_parquet(source_path)
                if audit_frame(cached, [g], grid, settings)["passed"]:
                    cost = json.loads(source_cost.read_text(encoding="utf-8"))
                    results.append((g.id, str(source_path), {**cost, "cache_hit": True,
                        "actual_elapsed_s": 0.0, "new_state_solves": 0}))
                    reused = True
                    break
        if reused:
            continue
        active.append(_GeometryWork(g, key))
    for attempt in range(int(retry["max_retries"]) + 1):
        if not active:
            break
        factor = float(retry["growth"]) ** attempt
        cfg = replace(settings, harmonics=int(np.ceil(settings.harmonics * factor)),
                      nz=int(np.ceil(settings.nz * factor)), nx=int(np.ceil(settings.nx * factor)))
        jobs = []
        expected_by_key = {}
        for work in active:
            needed = [s for s in states if s not in work.collected or work.collected[s]["status"] != "ok"]
            expected_by_key[work.key] = set(needed)
            attempt_path = directory / "attempts" / f"{work.key}.{attempt}.parquet"
            if attempt_path.exists():
                work.attempt_rows = _read_attempt(attempt_path, work, set(needed), cfg, attempt, complete=True)
                continue
            work.attempt_rows = {}
            chunk_directory = directory / "chunks" / work.key / str(attempt)
            for fragment in sorted(chunk_directory.glob("*.parquet")):
                rows = _read_attempt(fragment, work, set(needed), cfg, attempt)
                if set(rows) & set(work.attempt_rows):
                    raise RuntimeError(f"Overlapping state checkpoints: {fragment}")
                work.attempt_rows.update(rows)
            missing = [s for s in needed if s not in work.attempt_rows]
            for index in range(0, len(missing), execution.chunk_size):
                chunk = missing[index:index + execution.chunk_size]
                # The geometry/grid/solver are identified by the full parent key.
                # A short unique starting-state index avoids Windows MAX_PATH.
                path = chunk_directory / f"{state_index[chunk[0]]:06d}_{len(chunk):04d}.parquet"
                task = (work.geometry.array().tolist(), chunk, asdict(cfg), attempt,
                        str(path), execution.blas_threads)
                jobs.append((work, task))
        total = sum(len(task[1]) for _, task in jobs)
        completed, last_log = 0, perf_counter()
        if total:
            LOGGER.info("RCWA batch: %d geometries, attempt %d, %d new states, %d CPU workers",
                        len(active), attempt, total, min(execution.workers, len(jobs)))

        def accept(work: _GeometryWork, path: str) -> None:
            nonlocal completed, last_log
            rows = _read_attempt(Path(path), work, expected_by_key[work.key], cfg, attempt)
            if set(rows) & set(work.attempt_rows):
                raise RuntimeError(f"Repeated state result: {path}")
            work.attempt_rows.update(rows)
            work.new_states += len(rows)
            completed += len(rows)
            if perf_counter() - last_log >= execution.log_interval_s or completed == total:
                LOGGER.info("RCWA states %d/%d (attempt %d); batch elapsed %.1f s",
                            completed, total, attempt, perf_counter() - start)
                last_log = perf_counter()

        if execution.workers == 1:
            for work, task in jobs:
                accept(work, _state_chunk_job(task))
        elif jobs:
            pool, iterator, pending = _get_pool(execution), iter(jobs), {}
            try:
                # Bound queued work and memory; Ctrl+C only waits for running chunks.
                exhausted = False
                while pending or not exhausted:
                    while len(pending) < 2 * execution.workers and not exhausted:
                        item = next(iterator, None)
                        if item is None:
                            exhausted = True
                        else:
                            work, task = item
                            pending[pool.submit(_state_chunk_job, task)] = work
                    if not pending:
                        break
                    ready, _ = wait(pending, timeout=min(5.0, execution.log_interval_s),
                                    return_when=FIRST_COMPLETED)
                    for future in ready:
                        accept(pending.pop(future), future.result())
                    if not ready and perf_counter() - last_log >= execution.log_interval_s:
                        LOGGER.info("RCWA states %d/%d; %d chunks in flight", completed, total, len(pending))
                        last_log = perf_counter()
            except BaseException:
                for future in pending:
                    future.cancel()
                LOGGER.warning("Stopping RCWA submission; finishing running chunks and preserving checkpoints")
                close_rcwa_pool()
                raise
        remaining = []
        for work in active:
            if set(work.attempt_rows) != expected_by_key[work.key]:
                raise RuntimeError(f"Missing state results for {work.geometry.id}, attempt {attempt}")
            ordered = [work.attempt_rows[s] for s in states if s in work.attempt_rows]
            attempt_path = directory / "attempts" / f"{work.key}.{attempt}.parquet"
            if not attempt_path.exists():
                write_parquet(pd.DataFrame(ordered), attempt_path)
            for row in ordered:
                work.collected[tuple(row[k] for k in STATE_COLUMNS)] = row
                work.logical_states += 1
                work.solver_seconds += float(row["elapsed_s"])
            done = len(work.collected) == len(states) and all(r["status"] == "ok" for r in work.collected.values())
            if not done and attempt < int(retry["max_retries"]):
                remaining.append(work)
                continue
            frame = pd.DataFrame([work.collected[s] for s in states if s in work.collected])
            audit = audit_frame(frame, [work.geometry], grid, settings)
            if not audit["passed"]:
                save_json(directory / f"{work.key}.failure.json", audit)
                raise RuntimeError(f"RCWA/QC failed for geometry {work.geometry.id}; attempts preserved at {directory / 'attempts'}")
            cost = {"logical_state_solves": work.logical_states, "solver_elapsed_s": work.solver_seconds,
                    "requested_states": len(states), "cache_hit": False,
                    "actual_elapsed_s": perf_counter() - work.started, "new_state_solves": work.new_states,
                    "execution": asdict(execution), "cache_contract": _CACHE_CONTRACT}
            path = directory / f"{work.key}.parquet"
            write_parquet(frame, path)
            save_json(directory / f"{work.key}.cost.json", cost)
            results.append((work.geometry.id, str(path), cost))
            LOGGER.info("Completed %d/%d geometry shards (%s)", len(results), len(geometries), work.geometry.id)
        active = remaining
    results.sort(key=lambda item: item[0])
    frame = pd.concat([pd.read_parquet(path) for _, path, _ in results], ignore_index=True)
    costs = [cost for _, _, cost in results]
    summary = {"geometries": len(geometries), "requested_states": len(frame),
               "logical_state_solves": sum(v["logical_state_solves"] for v in costs),
               "solver_elapsed_s": sum(v["solver_elapsed_s"] for v in costs),
               "cache_hits": sum(v["cache_hit"] for v in costs),
               "new_state_solves": sum(v["new_state_solves"] for v in costs),
               "execution": asdict(execution),
               "wall_time_s": perf_counter() - start}
    return frame, summary


def run_dataset(config: Mapping[str, Any]) -> dict[str, Any]:
    from validate_physics import require_validation
    require_validation(config)
    manifest = make_manifest(config)
    grid = OpticalGrid.from_config(config["grid"])
    summary = {}
    for role, group in manifest.groupby("role", sort=False):
        geometries = [Geometry.from_mapping(row) for row in group.to_dict("records")]
        settings_name = "base" if role == "train_initial" else config["dataset"]["evaluation_solver"]
        settings = SolverConfig.from_mapping(config["rcwa"][settings_name])
        frame, cost = compute_batch(geometries, grid, config, settings)
        audit = audit_frame(frame, geometries, grid, settings)
        if not audit["passed"]:
            raise RuntimeError(str(audit))
        frame["role"] = role
        frame["family_id"] = frame.geometry_id.map(dict(zip(group.geometry_id, group.family_id)))
        write_parquet(frame, Path(config["_root"]) / "data" / f"{role}.parquet")
        summary[role] = {"audit": audit, "cost": cost}
    save_json(Path(config["_root"]) / "data" / "audit.json", summary)
    return summary


def audit_dataset(config: Mapping[str, Any]) -> dict[str, Any]:
    manifest = make_manifest(config)
    if (manifest.groupby("family_id").role.nunique() > 1).any():
        raise RuntimeError("Cross-role perturbation-family leakage")
    result, seen = {}, set()
    for role, group in manifest.groupby("role"):
        ids = set(group.geometry_id)
        if seen & ids:
            raise RuntimeError("Cross-role geometry leakage")
        seen.update(ids)
        settings_name = "base" if role == "train_initial" else config["dataset"]["evaluation_solver"]
        frame = pd.read_parquet(Path(config["_root"]) / "data" / f"{role}.parquet")
        result[role] = audit_frame(frame, [Geometry.from_mapping(r) for r in group.to_dict("records")],
                                   OpticalGrid.from_config(config["grid"]),
                                   SolverConfig.from_mapping(config["rcwa"][settings_name]))
    save_json(Path(config["_root"]) / "data" / "reaudit.json", result)
    if not all(v["passed"] for v in result.values()):
        raise RuntimeError("Dataset audit failed; see reaudit.json")
    return result


def load_split(config: Mapping[str, Any], role: str) -> pd.DataFrame:
    return pd.read_parquet(Path(config["_root"]) / "data" / f"{role}.parquet")


def prepare_v2(config: Mapping[str, Any]) -> dict[str, Any]:
    """Copy verified base data/certificates only; preserve models and AL history in v1."""
    from validate_physics import require_validation
    source = Path(config["reuse"]["source_repeat_root"]).resolve()
    target = Path(config["_root"]).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Source and target repeat directories must be disjoint")
    if not source.is_dir():
        raise FileNotFoundError(f"v1 source missing: {source}; configure reuse.source_candidates")
    source_config = {**config, "_root": str(source)}
    require_validation(source_config)
    manifest_meta = json.loads((source / "data/manifest.json").read_text(encoding="utf-8"))
    if manifest_meta["signature"] != manifest_signature(config):
        raise RuntimeError("Source dataset settings/data seed differ; do not import")
    manifest = pd.read_parquet(source / "data/manifest.parquet")
    if manifest.geometry_id.duplicated().any() or (manifest.groupby("family_id").role.nunique()>1).any():
        raise RuntimeError("Source manifest contains geometry/family leakage")
    if set(manifest.role) != set(config["dataset"]["counts"]):
        raise RuntimeError("Source manifest roles differ")
    for role, group in manifest.groupby("role"):
        if len(group) != config["dataset"]["counts"][role]:
            raise RuntimeError(f"Source manifest count differs: {role}")
        settings = "base" if role == "train_initial" else config["dataset"]["evaluation_solver"]
        check = audit_frame(pd.read_parquet(source / "data" / f"{role}.parquet"),
            [Geometry.from_mapping(row) for row in group.to_dict("records")],
            OpticalGrid.from_config(config["grid"]), SolverConfig.from_mapping(config["rcwa"][settings]))
        if not check["passed"]:
            raise RuntimeError(f"Source {role} QC failed: {check}")
    files = [source / "data" / name for name in ["manifest.json", "manifest.parquet",
             *[f"{role}.parquet" for role in config["dataset"]["counts"]]]]
    files += [p for p in (source / "validation").rglob("*") if p.is_file()]
    if (source / "data/audit.json").exists():
        files.append(source / "data/audit.json")
    copied = []
    for old in files:
        destination = target / old.relative_to(source)
        digest = sha256(old.read_bytes()).hexdigest()
        if destination.exists():
            if sha256(destination.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Refusing to overwrite different existing file: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, destination)
        copied.append({"path": str(old.relative_to(source)), "sha256": digest})
    require_validation(config)
    audit_dataset(config)
    result = {"passed": True, "source": str(source), "destination": str(target),
              "data_seed": config["reuse"]["data_seed"], "model_seed": config["seed"],
              "files": copied, "source_shards": "read-only cache; logical costs retained",
              "legacy_tests_already_opened": True,
              "note": "No v1 models, AL queries, optimization histories or final certificates imported."}
    save_json(target / "reuse_manifest.json", result)
    return {k: v for k, v in result.items() if k != "files"}


def confirmation_tests(config: Mapping[str, Any]) -> dict[str, Any]:
    """Generate fresh final IID/OOD labels after optimization is frozen."""
    from validate_physics import require_validation
    require_validation(config)
    root = Path(config["_root"])
    progress = root / "optimization/progress.json"
    if not progress.exists() or not json.loads(progress.read_text())["complete"]:
        raise RuntimeError("Finish all optimization tasks before generating confirmation tests")
    candidates = root / "optimization/candidates.csv"
    if not candidates.exists():
        raise FileNotFoundError(candidates)
    directory = root / "confirmation_tests"
    metadata_path = directory / "protocol.json"
    models = sorted(root.glob("models/*/ensemble.json")) + sorted(root.glob("active_learning/*/round_*/model/ensemble.json"))
    if not models:
        raise RuntimeError("No models to freeze for final evaluation")
    frozen = {}
    for path in models:
        metadata = json.loads(path.read_text())
        if not metadata.get("training_qc", {}).get("passed", False):
            raise RuntimeError(f"Cannot freeze a model without passing training QC: {path}")
        frozen[str(path.relative_to(root))] = metadata["training_signature"]
    protocol = {"options": config["confirmation_tests"], "models": frozen,
                "candidate_sha256": sha256(candidates.read_bytes()).hexdigest(),
                "physical_config": {k: config[k] for k in ("grid", "geometry", "rcwa")}}
    signature = fingerprint(protocol)
    if metadata_path.exists() and json.loads(metadata_path.read_text())["signature"] != signature:
        raise RuntimeError("Frozen confirmation protocol changed; use a new experiment")
    directory.mkdir(parents=True, exist_ok=True)
    save_json(metadata_path, {"signature": signature, **protocol})
    excluded = set(make_manifest(config).geometry_id)
    excluded.update(pd.read_csv(candidates).geometry_id)
    for base in [root, Path(config["reuse"]["source_repeat_root"])]:
        for membership in base.glob("active_learning/*/round_*/membership.parquet"):
            excluded.update(pd.read_parquet(membership, columns=["geometry_id"]).geometry_id)
    grid = OpticalGrid.from_config(config["grid"])
    settings = SolverConfig.from_mapping(config["rcwa"][config["dataset"]["evaluation_solver"]])
    outcome = {}
    for index, role in enumerate(("iid", "ood")):
        manifest_path = directory / f"{role}_manifest.parquet"
        if manifest_path.exists():
            manifest = pd.read_parquet(manifest_path)
            if set(manifest.geometry_id) & excluded:
                raise RuntimeError("Confirmation manifest overlaps protected/training geometries")
            geometries = [Geometry.from_mapping(row) for row in manifest.to_dict("records")]
        else:
            geometries = sample_geometries(config["dataset"]["counts"][role], config,
                int(config["confirmation_tests"]["seed"])+index*100003,
                domain="ood" if role == "ood" else "cover", exclude=excluded)
            manifest = pd.DataFrame([g.record() for g in geometries])
            write_parquet(manifest, manifest_path)
        excluded.update(g.id for g in geometries)
        frame, cost = compute_batch(geometries, grid, config, settings)
        audit = audit_frame(frame, geometries, grid, settings)
        if not audit["passed"]:
            raise RuntimeError(f"Confirmation {role} QC failed")
        frame["role"] = role
        frame["family_id"] = frame.geometry_id
        write_parquet(frame, directory / f"{role}.parquet")
        outcome[role] = {"audit": audit, "cost": cost}
    save_json(directory / "audit.json", {"passed": True, "protocol_signature": signature, **outcome})
    return {"passed": True, "roles": outcome, "note": "Labels generated; report --evaluate-tests opens them."}


def load_confirmation_split(config: Mapping[str, Any], role: str) -> pd.DataFrame:
    """Read final fresh test labels; never silently fall back to opened v1 tests."""
    root = Path(config["_root"])
    directory = root / "confirmation_tests"
    audit = json.loads((directory / "audit.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    if not audit["passed"] or audit["protocol_signature"] != protocol["signature"]:
        raise RuntimeError("Confirmation tests incomplete or stale")
    current_models = sorted(root.glob("models/*/ensemble.json")) + sorted(root.glob("active_learning/*/round_*/model/ensemble.json"))
    if {p.relative_to(root).as_posix() for p in current_models} != {name.replace("\\", "/") for name in protocol["models"]}:
        raise RuntimeError("Model set changed after confirmation freeze")
    if protocol["options"] != config["confirmation_tests"] or protocol["physical_config"] != {k: config[k] for k in ("grid", "geometry", "rcwa")}:
        raise RuntimeError("Confirmation configuration changed after freeze")
    for name, signature in protocol["models"].items():
        if json.loads((root/name.replace("\\", "/")).read_text())["training_signature"] != signature:
            raise RuntimeError("Model changed after confirmation freeze")
    if sha256((root / "optimization/candidates.csv").read_bytes()).hexdigest() != protocol["candidate_sha256"]:
        raise RuntimeError("Candidates changed after confirmation freeze")
    frame = pd.read_parquet(directory / f"{role}.parquet")
    manifest = pd.read_parquet(directory / f"{role}_manifest.parquet")
    settings = SolverConfig.from_mapping(config["rcwa"][config["dataset"]["evaluation_solver"]])
    check = audit_frame(frame, [Geometry.from_mapping(r) for r in manifest.to_dict("records")],
                        OpticalGrid.from_config(config["grid"]), settings)
    if not check["passed"]:
        raise RuntimeError("Confirmation test data failed audit")
    return frame
