"""Geometry-level active learning with protected evaluation sets and budget ledgers."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Mapping
import copy
import json

import numpy as np
import pandas as pd

from optics import (Geometry, OpticalGrid, domain_bounds, feature_matrix, fingerprint,
                    save_json, feasibility, robust_feasible)
from dataset import (audit_dataset, compute_batch, load_split, make_manifest,
                     sample_geometries, write_parquet)
from surrogate import calibrate, prediction_table, table_features, train_ensemble


def _standardize(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / max(float(values.std()), 1e-12)


def acquire(ensemble: Any, candidates: list[Geometry], grid: OpticalGrid,
            config: Mapping[str, Any], strategy: str, count: int, seed: int
            ) -> tuple[list[Geometry], np.ndarray]:
    from optimize import optical_weights
    rng = np.random.default_rng(seed)
    if count > len(candidates):
        raise ValueError("Acquisition batch exceeds pool")
    if strategy == "random":
        indices = rng.choice(len(candidates), count, replace=False)
        return [candidates[i] for i in indices], np.zeros(count)
    means, uncertainty = [], []
    chunk = int(config["active_learning"]["prediction_geometry_batch"])
    weights = optical_weights(grid, config)
    for start in range(0, len(candidates), chunk):
        group = candidates[start:start + chunk]
        prediction = ensemble.predict(feature_matrix(group, grid))
        members = prediction.members[:, :, 0].reshape(len(ensemble.models), len(group), -1)
        objectives = members @ weights
        means.extend(objectives.mean(axis=0))
        uncertainty.extend(objectives.std(axis=0, ddof=1))
    uncertainty, means = np.asarray(uncertainty), np.asarray(means)
    if strategy == "uncertainty":
        score = uncertainty
        indices = np.argsort(-score, kind="stable")[:count]
    elif strategy in {"task_diverse", "no_performance", "no_disagreement", "no_diversity"}:
        options = config["active_learning"]
        # Drop exactly one mechanism. Preserve the full strategy's sign, z-scoring,
        # shortlist, normalization, and deterministic tie handling in the other arms.
        u = _standardize(uncertainty)
        p = options["performance_weight"] * _standardize(-means)
        score = u if strategy == "no_performance" else p if strategy == "no_disagreement" else u + p
        if strategy == "no_diversity":
            indices = np.argsort(-score, kind="stable")[:count]
            return [candidates[i] for i in indices], score[indices]
        shortlist = np.argsort(-score, kind="stable")[:min(len(candidates), count * options["shortlist_factor"])]
        bounds = domain_bounds(config)
        x = (np.array([candidates[i].array() for i in shortlist]) - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
        selected = [0]
        distances = np.linalg.norm(x - x[0], axis=1)
        normalized_score = _standardize(score[shortlist])
        for _ in range(1, count):
            merit = distances + options["diversity_score_weight"] * normalized_score
            merit[selected] = -np.inf
            index = int(np.argmax(merit))
            selected.append(index)
            distances = np.minimum(distances, np.linalg.norm(x - x[index], axis=1))
        indices = shortlist[selected]
    else:
        raise ValueError(f"Unknown acquisition strategy: {strategy}")
    return [candidates[i] for i in indices], score[indices]


def run_active_learning(config: Mapping[str, Any]) -> dict[str, Any]:
    if "_baseline_root" in config:
        from revision_learning import run_revision_learning
        return run_revision_learning(config, config["active_learning"]["strategies"])
    from validate_physics import require_validation
    from optimize import optical_metrics
    require_validation(config)
    audit_dataset(config)
    manifest = make_manifest(config)
    protected = set(manifest.loc[manifest.role != "train_initial", "geometry_id"])
    validation, calibration = load_split(config, "validation"), load_split(config, "calibration")
    grid = OpticalGrid.from_config(config["grid"])
    probe = OpticalGrid.from_config(config["active_learning"]["probe_grid"])
    initial = manifest[manifest.role == "train_initial"]
    outcomes = {}
    for strategy in config["active_learning"]["strategies"]:
        directory = Path(config["_root"]) / "active_learning" / strategy
        directory.mkdir(parents=True, exist_ok=True)
        signature = fingerprint({"settings": config["active_learning"], "model": config["model"],
                                 "seed": config["seed"], "manifest": manifest.to_dict("records"),
                                 "training_source": Path(__file__).with_name("surrogate.py").read_text(encoding="utf-8"),
                                 "acquisition_source": Path(__file__).read_text(encoding="utf-8")})
        signature_path = directory / "protocol.json"
        if signature_path.exists() and json.loads(signature_path.read_text())["signature"] != signature:
            raise RuntimeError("AL protocol changed; choose a new experiment_id")
        save_json(signature_path, {"signature": signature})
        train = load_split(config, "train_initial").copy()
        membership = initial.copy()
        membership["acquisition_score"] = 0.0
        membership["method"] = strategy
        membership["repeat_seed"] = config["seed"]
        history = []
        budget, cumulative_seconds = 0, 0.0
        for round_id in range(config["active_learning"]["rounds"] + 1):
            current = directory / f"round_{round_id:03d}"
            current.mkdir(parents=True, exist_ok=True)
            started = perf_counter()
            acquisition_seconds = 0.0
            if round_id == 0:
                geometries = [Geometry.from_mapping(row) for row in initial.to_dict("records")]
                _, cost = compute_batch(geometries, grid, config)
            else:
                query_file = current / "queries.parquet"
                if query_file.exists():
                    query = pd.read_parquet(query_file)
                else:
                    excluded = protected | set(membership.geometry_id)
                    candidates = sample_geometries(config["active_learning"]["pool_size"], config,
                                                   config["seed"] + 70001 * round_id, exclude=excluded)
                    t0 = perf_counter()
                    chosen, scores = acquire(ensemble, candidates, probe, config, strategy,
                                              config["active_learning"]["batch_geometries"],
                                              config["seed"] + round_id)
                    acquisition_seconds = perf_counter() - t0
                    query = pd.DataFrame([{**g.record(), "family_id": g.id, "role": "train_active",
                                           "active_round": round_id, "acquisition_score": float(score),
                                           "method": strategy, "repeat_seed": config["seed"]}
                                          for g, score in zip(chosen, scores)])
                    write_parquet(query, query_file)
                if set(query.geometry_id) & (protected | set(membership.geometry_id)):
                    raise RuntimeError("AL query duplicates existing/protected geometry")
                geometries = [Geometry.from_mapping(row) for row in query.to_dict("records")]
                new_data, cost = compute_batch(geometries, grid, config)
                new_data["role"] = "train_active"
                new_data["active_round"] = round_id
                new_data["family_id"] = new_data.geometry_id
                train = pd.concat([train, new_data], ignore_index=True)
                membership = pd.concat([membership, query], ignore_index=True)
            budget += cost["logical_state_solves"]
            cumulative_seconds += cost["solver_elapsed_s"]
            write_parquet(membership, current / "membership.parquet")
            write_parquet(train, current / "training.parquet")
            ensemble = train_ensemble(train, validation, config, current / "model")
            calibrate(ensemble, calibration, config, current / "model")
            predictions = prediction_table(ensemble, validation)
            predictions.to_csv(current / "validation_predictions.csv", index=False)
            design_values = []
            for _, group in train.groupby("geometry_id"):
                geometry = Geometry.from_mapping(group.iloc[0])
                if feasibility(geometry, config, "design"):
                    continue
                if config["optimization"]["enforce_tolerance_feasible"] and not robust_feasible(geometry, config):
                    continue
                design_values.append(optical_metrics(group, grid, config)["mean_R"])
            best = min(design_values) if design_values else None
            record = {"strategy": strategy, "repeat_seed": config["seed"], "round": round_id,
                      "training_geometries": int(train.geometry_id.nunique()),
                      "logical_state_solves": budget, "solver_elapsed_s": cumulative_seconds,
                      "validation_R_rmse": float(np.sqrt(np.mean((predictions.R_pred - predictions.R) ** 2))),
                      "best_true_mean_R": best, "acquisition_s": acquisition_seconds,
                      "feasible_labelled_designs": len(design_values),
                      "training_s": ensemble.metadata["training_time_s"],
                      "round_wall_s": perf_counter() - started,
                      "model_directory": str((current / "model").resolve())}
            # Preserve original timings on resume instead of presenting cache hits as speedups.
            record_path = current / "round.json"
            if record_path.exists():
                record = json.loads(record_path.read_text())
            else:
                save_json(record_path, record)
            history.append(record)
            pd.DataFrame(history).to_csv(directory / "learning_curve.csv", index=False)
        outcomes[strategy] = history[-1]
    return outcomes
