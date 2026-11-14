"""PIER revision: paired acquisition runs and evaluation on the frozen v2 test labels.

One query is a geometry (a full base-grid spectrum). Test labels are never read
by the learning loop. All writes go under the new run, not the submitted run.
"""
from __future__ import annotations

from hashlib import sha256
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from active_learning import acquire
from dataset import (audit_frame, compute_batch, manifest_signature, sample_geometries,
                     write_parquet)
from optics import GEOMETRY_COLUMNS, Geometry, OpticalGrid, fingerprint, save_json
from rcwa_solver import SolverConfig
from surrogate import (Ensemble, calibrate, data_signature, prediction_table,
                       select_device, train_ensemble)
from validate_physics import require_validation


STRATEGIES = ("random", "uncertainty", "task_diverse", "no_performance",
              "no_disagreement", "no_diversity")
LOG = logging.getLogger(__name__)
FEATURES = ["geometry_id", *GEOMETRY_COLUMNS, "wavelength_nm", "angle_deg", "polarization"]
TRAIN_COLS = [*FEATURES, "R", "T", "status", "family_id", "role"]


def _names(items: Iterable[str]) -> list[str]:
    names = [items] if isinstance(items, str) else list(items)
    if not names or len(set(names)) != len(names) or set(names) - set(STRATEGIES):
        raise ValueError(f"Expected nonempty, distinct strategies from {STRATEGIES}, got {names}")
    return names


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _roots(cfg: Mapping) -> tuple[Path, Path]:
    new, base = (Path(cfg[k]).resolve() for k in ("_root", "_baseline_root"))
    if new == base or base in new.parents or new in base.parents:
        raise ValueError("revision output and read-only baseline must be disjoint")
    return new, base


def _test_manifest(cfg: Mapping, base: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    directory = base / "confirmation_tests"
    p, audit = _read(directory / "protocol.json"), _read(directory / "audit.json")
    if (fingerprint({k: v for k, v in p.items() if k != "signature"}) != p["signature"]
            or not audit["passed"] or audit["protocol_signature"] != p["signature"]
            or p["physical_config"] != {k: cfg[k] for k in ("grid", "rcwa", "geometry")}
            or p["options"] != cfg["confirmation_tests"]
            or _digest(base / "optimization/candidates.csv") != p["candidate_sha256"]):
        raise RuntimeError("Frozen v2 confirmation protocol/candidate identity check failed")
    frames = {s: pd.read_parquet(directory / f"{s}_manifest.parquet") for s in ("iid", "ood")}
    ids = set()
    for s, frame in frames.items():
        row_ids = [Geometry.from_mapping(r).id for r in frame.to_dict("records")]
        if (len(frame) != cfg["dataset"]["counts"][s] or len(set(row_ids)) != len(row_ids)
                or row_ids != frame.geometry_id.tolist() or ids.intersection(row_ids)):
            raise RuntimeError(f"Invalid/overlapping {s} confirmation IDs")
        ids.update(row_ids)
    return p, frames


def _audit_table(frame: pd.DataFrame, manifest: pd.DataFrame, cfg: Mapping, solver: str) -> None:
    audit = audit_frame(frame, [Geometry.from_mapping(r) for r in manifest.to_dict("records")],
                        OpticalGrid.from_config(cfg["grid"]), SolverConfig.from_mapping(cfg["rcwa"][solver]))
    if not audit["passed"] or not np.isfinite(frame[["R", "T"]].to_numpy()).all():
        raise RuntimeError(f"RCWA label QC failed: {audit}")


def _check_members(m: pd.DataFrame, seed: int, strategy: str, expected: int, protected: set) -> set:
    ids = set(m.geometry_id)
    if (len(ids) != len(m) or len(ids) != expected or ids & protected
            or list(m.geometry_id) != [Geometry.from_mapping(r).id for r in m.to_dict("records")]
            or not m.method.eq(strategy).all() or not m.repeat_seed.eq(seed).all()
            or not m.family_id.eq(m.geometry_id).all()):
        raise RuntimeError(f"Unexpected, duplicate or protected geometry in {strategy} membership/query")
    return ids


def _model_token(d: Path) -> str:
    meta = _read(d / "ensemble.json")
    return fingerprint({name: _digest(d/name) for name in
                        ("ensemble.json", "calibration.json", *meta["members"])})


def run_revision_learning(cfg: Mapping[str, Any], strategies: Iterable[str]) -> dict:
    """Run chosen strategies under this seed; skip complete rounds, resume incomplete work.

    round_000 consumes supplied, audited initial labels (counted logically). If a
    cold timing experiment generated them, charge the generation in its prep stage.
    """
    strategies = _names(strategies)
    root, base = _roots(cfg)
    require_validation(cfg)
    if _read(root / "data/manifest.json")["signature"] != manifest_signature(cfg):
        raise RuntimeError("Initial split signature differs from the fixed data protocol")
    if _digest(root/"data/manifest.parquet") != _digest(base/"data/manifest.parquet"):
        raise RuntimeError("Prepared run did not retain the frozen split manifest")
    manifest = pd.read_parquet(root / "data/manifest.parquet")
    protocol, manifests = _test_manifest(cfg, base)
    test_ids = set(pd.concat(manifests.values()).geometry_id)
    if test_ids & set(manifest.geometry_id) or manifest.geometry_id.duplicated().any():
        raise RuntimeError("Original/confirmation split overlap")
    protected = test_ids | set(manifest.loc[manifest.role != "train_initial", "geometry_id"])
    # Protect submitted finalists (some might legitimately be original training
    # points; this constraint applies only to NEW queries).
    finalists = set(pd.read_csv(base/"optimization/candidates.csv").geometry_id)
    grid = OpticalGrid.from_config(cfg["grid"])
    tables = {}
    for role in ("train_initial", "validation", "calibration"):
        raw = pd.read_parquet(root / "data" / f"{role}.parquet")
        _audit_table(raw, manifest[manifest.role == role], cfg,
                     "base" if role == "train_initial" else cfg["dataset"]["evaluation_solver"])
        if not raw.role.eq(role).all() or not raw.family_id.eq(raw.geometry_id).all():
            raise RuntimeError(f"Invalid family/role in {role}")
        if role == "train_initial":
            initial_logical = int(len(raw) + raw["attempt"].sum())
        tables[role] = raw[TRAIN_COLS].copy()
        del raw
    initial, val, cal = (tables[r] for r in ("train_initial", "validation", "calibration"))
    members0 = manifest[manifest.role == "train_initial"].copy()
    al = cfg["active_learning"]
    seed, count = int(cfg["seed"]), int(al["batch_geometries"])
    probe = OpticalGrid.from_config(al["probe_grid"])
    settings = {k: v for k, v in al.items() if k != "strategies"}
    protocol_common = {"acquisition": settings, "model": cfg["model"], "seed": seed,
        "physics": protocol["physical_config"], "tests": sorted(protected), "finalists": sorted(finalists),
        "fixed_test_protocol": protocol["signature"], "metrics": cfg["metrics"],
        "sampling": cfg["sampling"], "calibration": cfg["calibration"],
        "sources": {f: fingerprint(Path(__file__).with_name(f).read_text(encoding="utf-8"))
                    for f in ("active_learning.py","revision_learning.py","surrogate.py","dataset.py")},
        "data": {k: data_signature(v) for k,v in tables.items()}}
    result = {}
    for strategy in strategies:
        directory = root / "active_learning" / strategy
        directory.mkdir(parents=True, exist_ok=True)
        sig_path = directory/"protocol.json"
        sig = fingerprint({**protocol_common, "strategy": strategy})
        if sig_path.exists() and _read(sig_path)["signature"] != sig:
            raise RuntimeError(f"Protocol changed at {directory}; use a new revision directory")
        save_json(sig_path, {"signature": sig, "strategy": strategy, "seed": seed})
        train, members, history = initial, members0.copy(), []
        members["method"], members["repeat_seed"], members["acquisition_score"] = strategy, seed, 0.0
        budget, previous_model = 0, None
        for index in range(al["rounds"] + 1):
            d = directory/f"round_{index:03d}"
            d.mkdir(exist_ok=True)
            record_path, model_dir = d/"round.json", d/"model"
            LOG.info("AL seed=%d strategy=%s round=%d/%d", seed, strategy, index, al["rounds"])
            started = perf_counter()
            marker = d/"attempt.json"
            finished = record_path.exists()
            resumed = marker.exists() and not finished
            if not finished:
                save_json(marker, {"seed": seed, "strategy": strategy, "round": index})
            acq_s = pool_s = label_s = aggregate_s = 0.0
            new_states = cache_hits = 0
            round_cost = initial_logical
            if index:
                q_path, l_path = d/"queries.parquet", d/"new_labels.parquet"
                excluded = protected | finalists | set(members.geometry_id)
                if q_path.exists():
                    q = pd.read_parquet(q_path)
                elif finished:
                    raise RuntimeError(f"Completed round lost its queries: {q_path}")
                else:
                    tic = perf_counter()
                    pool = sample_geometries(al["pool_size"], cfg, seed + 70001*index, exclude=excluded)
                    pool_s = perf_counter() - tic
                    if previous_model is None:
                        previous_model = Ensemble.load(directory/f"round_{index-1:03d}"/"model",
                                            cfg["model"]["device"], cfg["model"]["predict_batch_size"])
                    tic = perf_counter()
                    chosen, scores = acquire(previous_model, pool, probe, cfg, strategy, count, seed+index)
                    acq_s = perf_counter() - tic
                    q = pd.DataFrame([{**g.record(), "family_id":g.id,"role":"train_active","active_round":index,
                                      "acquisition_score":float(s),"method":strategy,"repeat_seed":seed}
                                      for g,s in zip(chosen, scores)])
                    write_parquet(q, q_path)
                    save_json(d/"query_timing.json", {"acquisition_s": acq_s,"pool_generation_s": pool_s})
                _check_members(q, seed, strategy, count, excluded)
                if not q.active_round.eq(index).all() or not q.role.eq("train_active").all():
                    raise ValueError(f"Invalid query provenance in {q_path}")
                if (d/"query_timing.json").is_file():
                    qt = _read(d/"query_timing.json")
                    acq_s, pool_s = qt["acquisition_s"], qt["pool_generation_s"]
                if l_path.exists() and (d/"label_timing.json").exists():
                    labels = pd.read_parquet(l_path)
                    cost = _read(d/"label_timing.json")
                    if cost["label_signature"] != data_signature(labels):
                        raise RuntimeError(f"Corrupt saved labels in {d}")
                elif finished:
                    raise RuntimeError(f"Completed round lost its RCWA records: {d}")
                else:
                    tic = perf_counter()
                    raw,cost = compute_batch([Geometry.from_mapping(r) for r in q.to_dict("records")], grid, cfg)
                    label_s = perf_counter() - tic
                    raw["role"], raw["family_id"] = "train_active", raw.geometry_id
                    labels = raw[TRAIN_COLS].copy()
                    del raw
                    cost = {**cost,"labels_wall_s":label_s,"label_signature":data_signature(labels)}
                    write_parquet(labels, l_path)
                    save_json(d/"label_timing.json",cost)
                if set(labels.geometry_id) != set(q.geometry_id) or len(labels) != count*len(grid.states()):
                    raise RuntimeError(f"Query/label mismatch in {d}")
                label_s, aggregate_s = cost["labels_wall_s"], cost["solver_elapsed_s"]
                new_states, cache_hits = cost["new_state_solves"], cost["cache_hits"]
                round_cost = cost["logical_state_solves"]
                members = pd.concat([members,q],ignore_index=True)
                train = pd.concat([train,labels], ignore_index=True)
            ids = _check_members(members,seed,strategy,len(members0)+index*count,protected)
            budget += round_cost
            if finished:
                record = _read(record_path)
                saved = pd.read_parquet(d/"membership.parquet")
                if (_check_members(saved,seed,strategy,len(ids),protected) != ids
                        or record["logical_state_solves"] != budget or record["protocol_signature"] != sig
                        or record["model_digest"] != _model_token(model_dir)
                        or data_signature(train) != record["train_signature"]):
                    raise RuntimeError(f"Stale completed AL round: {d}")
                previous_model = None
            else:
                write_parquet(members,d/"membership.parquet")
                tic=perf_counter()
                previous_model=train_ensemble(train,val,cfg,model_dir)
                train_wall=perf_counter()-tic
                tic=perf_counter()
                calibrate(previous_model,cal,cfg,model_dir)
                cal_s=perf_counter()-tic
                predicted=prediction_table(previous_model,val)
                error=float(np.sqrt(np.mean((predicted.R_pred-predicted.R)**2)))
                if not np.isfinite(error):
                    raise RuntimeError("Nonfinite validation RMSE")
                record={"strategy":strategy,"repeat_seed":seed,"round":index,"training_geometries":len(ids),
                        "round_logical_state_solves":round_cost,"logical_state_solves":budget,
                        "pool_generation_s":pool_s,"acquisition_s":acq_s,"labels_wall_s":label_s,
                        "new_state_solves":new_states,"cache_hits":cache_hits,"solver_work_s_round":aggregate_s,
                        "training_s":previous_model.metadata["training_time_s"],"training_call_wall_s":train_wall,
                        "training_device":previous_model.device, "calibration_s":cal_s,"validation_R_rmse":error,
                        "model_digest":_model_token(model_dir),"train_signature":data_signature(train),
                        "protocol_signature":sig,"resumed_partial":resumed,
                        "model_directory":str(model_dir.resolve()),"round_wall_s":perf_counter()-started}
                save_json(record_path,record)
            history.append(record)
            csv_path=directory/"learning_curve.csv"
            tmp_csv=csv_path.with_suffix(".tmp")
            pd.DataFrame(history).to_csv(tmp_csv,index=False)
            tmp_csv.replace(csv_path)
        result[strategy]=history[-1]
    return result


def evaluate_revision(cfg: Mapping[str, Any], strategies: Iterable[str],
                      model_root: str | Path | None = None) -> dict:
    """Score frozen final ensembles on the *same previously opened* v2 test sets.

    No optimization prerequisite, test-set tuning, new calibration or RCWA calls.
    Each run/arm locks checkpoint bytes, test bytes, and its calibration identity.
    """
    strategies = _names(strategies)
    root, base = _roots(cfg)
    source = Path(model_root).resolve() if model_root is not None else root
    source_is_v2 = source == base
    if source not in (base,root):
        raise ValueError("model_root must be the frozen baseline or this prepared seed run")
    out = root/"revision_evaluation"
    out.mkdir(exist_ok=True,parents=True)
    policy, manifests = _test_manifest(cfg,base)
    n = int(cfg["active_learning"]["rounds"])
    idx = int(cfg["seed"])
    if source_is_v2 and idx != int(cfg["reuse"]["data_seed"]):
        raise ValueError("Archived v2 results are available for the submitted seed only")
    archive_signatures = {k.replace(chr(92), "/"):v for k,v in policy["models"].items()}
    original = pd.read_parquet(base/"data/manifest.parquet")
    cal_frame = pd.read_parquet(source/"data/calibration.parquet",columns=FEATURES+["R","T"])
    cal_signature = data_signature(cal_frame)
    all_tests = set(pd.concat(manifests.values()).geometry_id)
    if set(original.geometry_id) & all_tests:
        raise ValueError("Fixed confirmation and original splits overlap")
    protected = all_tests | set(original.loc[original.role!="train_initial","geometry_id"])
    init_ids = set(original.loc[original.role=="train_initial","geometry_id"])
    if not set(cal_frame.geometry_id) == set(original.loc[original.role=="calibration","geometry_id"]):
        raise ValueError("Calibration set identity changed")
    files = {s:base/"confirmation_tests"/f"{s}.parquet" for s in manifests}
    test_tokens = {s:_digest(path) for s,path in files.items()}
    # Lock all requested arms before reading any test responses.
    locks, locations, results = {}, {}, {}
    for strategy in strategies:
        round_dir=source/"active_learning"/strategy/f"round_{n:03d}"
        d=round_dir/"model"
        meta,cal=_read(d/"ensemble.json"),_read(d/"calibration.json")
        membership=pd.read_parquet(round_dir/"membership.parquet")
        expected = len(init_ids)+n*cfg["active_learning"]["batch_geometries"]
        ids = _check_members(membership,idx,strategy,expected,protected)
        if (set(meta["train_geometry_ids"])!=ids or not init_ids.issubset(ids)
                or len(meta["members"]) != cfg["model"]["ensemble_size"]
                or not meta["training_qc"]["passed"]
                or meta["variant"]!=cfg["model"]["primary_variant"]
                or [m["seed"] for m in meta["training_qc"]["members"]]
                   != [idx+1009*k for k in range(len(meta["members"]))]
                or cal["model_signature"]!=meta["training_signature"]
                or cal["calibration_signature"]!=cal_signature
                or not np.isfinite(cal["factor"]) or cal["factor"] < 0
                or cal["target_coverage"]!=cfg["calibration"]["coverage"]
                or cal["sigma_floor"]!=cfg["calibration"]["sigma_floor"]
                or cal["calibration_geometries"]!=cal_frame.geometry_id.nunique()):
            raise RuntimeError(f"Model/calibration/split provenance mismatch: {d}")
        token = _model_token(d)
        if source_is_v2:
            key=(d/"ensemble.json").relative_to(source).as_posix()
            if archive_signatures.get(key)!=meta["training_signature"]:
                raise RuntimeError(f"Model does not match the submitted v2 freeze: {key}")
        elif _read(round_dir/"round.json").get("model_digest")!=token:
            raise RuntimeError(f"New checkpoint differs from completed round: {d}")
        lock={"schema":1,"strategy":strategy,"seed":idx,"model_digest":token,
              "model_training_signature":meta["training_signature"],
              "calibration_signature":cal_signature,"fixed_test_sha256":test_tokens,
              "test_protocol":policy["signature"],"submitted_comparator":source_is_v2}
        lock_file=out/f"{strategy}_freeze.json"
        if lock_file.exists() and _read(lock_file)!=lock:
            raise RuntimeError(f"Evaluation identity changed: {lock_file}; use a new experiment")
        save_json(lock_file,lock)
        locations[strategy],locks[strategy]=d,lock
    # Load/audit each table once for ALL requested models. No recalibration on test.
    frames={}
    for split,path in files.items():
        data=pd.read_parquet(path)
        _audit_table(data,manifests[split],cfg,cfg["dataset"]["evaluation_solver"])
        if not data.role.eq(split).all() or not data.family_id.eq(data.geometry_id).all():
            raise ValueError(f"Unexpected label roles in {path}")
        frames[split]=data[FEATURES+["R","T"]].copy()
        del data
    for strategy in strategies:
        ensemble=Ensemble.load(locations[strategy],cfg["model"]["device"],cfg["model"]["predict_batch_size"])
        rows=[]
        for split,frame in frames.items():
            pred=prediction_table(ensemble,frame)
            quantities=["R_pred","T_pred","R_std","R_lower","R_upper"]
            if not np.isfinite(pred[quantities].to_numpy()).all() or (pred.R_lower>pred.R_upper).any():
                raise RuntimeError(f"Invalid predictive distribution for {strategy}/{split}")
            err=(pred.R_pred-pred.R).to_numpy()
            pred["sq_error"]=err*err
            groups=pred.groupby("geometry_id",sort=True)
            geometry=groups.agg(mse_R=("sq_error","mean"),point_coverage=("covered","mean"),
                              simultaneous_coverage=("covered","all"),mean_R=("R","mean"),
                              mean_R_pred=("R_pred","mean"),height_nm=("height_nm","first"))
            geometry.index.name="geometry_id"
            # The revision study uses the submitted equal-state grid objective.
            from optimize import optical_weights
            grid=OpticalGrid.from_config(cfg["grid"])
            weights=optical_weights(grid,cfg)
            if not np.allclose(weights,1/len(weights),rtol=1e-12,atol=0):
                raise ValueError("Geometry objective summary requires the submitted equal-weight grid")
            gerr=geometry.mean_R_pred-geometry.mean_R
            geometry.to_csv(out/f"{strategy}_{split}_geometry.csv")
            row={"strategy":strategy,"seed":idx,"split":split,"n_geometries":len(geometry),
                 "n_states":len(pred),"rmse_R":float(np.sqrt(np.mean(err*err))),
                 "mae_R":float(np.mean(np.abs(err))),"maxae_R":float(np.max(np.abs(err))),
                 "point_coverage":float(pred.covered.mean()),
                 "geometry_coverage":float(geometry.simultaneous_coverage.mean()),
                 "mean_interval_width":float((pred.R_upper-pred.R_lower).mean()),
                 "objective_rmse":float(np.sqrt(np.mean(gerr**2))),
                 "geometry_all_covered_count":int(geometry.simultaneous_coverage.sum()),
                 "uncovered_states":int((~pred.covered).sum()),"calibration_q":ensemble.calibration_factor}
            rows.append(row)
        entry={**locks[strategy],"evaluation_set":"fixed_previously_opened_v2_confirmation",
               "model_root":str(source),"metrics":rows,"passed":True}
        save_json(out/f"{strategy}.json",entry)
        results[strategy]=entry
    return results
