"""One metric definition for all search methods, budgets and final verification."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Mapping
import json
import logging
from dataclasses import asdict

import numpy as np
import pandas as pd

from optics import (Geometry, OpticalGrid, domain_bounds, feasibility, robust_feasible,
                    perturbations, tolerance_draws, feature_matrix, fingerprint, save_json)
from dataset import (compute_batch, make_manifest, sample_geometries, write_parquet,
                     write_csv, physics_source_hash, ExecutionSettings)
from rcwa_solver import SolverConfig

LOGGER = logging.getLogger(__name__)
# Exact old-source protocol signatures, bound to the inspected refined config,
# metric definition, physics source and three configured repeat seeds. This is
# a narrow scheduling-only migration, NOT permission to ignore stale protocols.
_LEGACY_PROTOCOLS = {
    "bd07d4f3690eca16d950450caacc97ff3a37e2df0c69c66bbf7c26e53d2732b7": "dae0d26d802c8f9a26027f7be1677a735a6bdcf8435e826cb327bc65fcbc8fb5",
    "bea3953ddf130c64184c1a60530bc07b39f34fdfe60107daf1fc1c2ad5c0e601": "0b96aff90a995251aed7346fb6715c7428b95a2a597f9e3b549d704c13cc0a3f",
    "a1b189aef0b61b84b0d98bfbd30969555f72cc39f95af7ce007b4867d882c2be": "4158450c5d2d39f44eec9741f7c608b5e56752f56d3ec0515a358b01b7155044",
    "93bb5dbd0161644fef9e4ce3b510438f09d879c144aff082536339b115c71eb7": "6a155c543265fda3e73751bdf7848e72bfc8eca61bea56fcca603774c2142f8e",
    "84279a8e788473ae00df2f796e2ea8dd8c1bd96132eab46f00784548c10acfc1": "0aaec792c15ab6b85553b14b2f68602c9a9ca83bc7e4a2797decd5e3ba605d43",
    "c6bc104384d236163ffb7e6142b9118728437f9eaaaec1f2e5b88609b067459b": "95bc8a2de6b01479185d7e5b82c12defd443eaaaabaec46281ef3dec78820dcf",
}


def optical_weights(grid: OpticalGrid, config: Mapping[str, Any]) -> np.ndarray:
    """Separable normalized grid weights; same definition on coarse/dense grids."""
    def axis_weights(axis: tuple[float, ...], method: str) -> np.ndarray:
        if method == "equal" or len(axis) == 1:
            return np.ones(len(axis)) / len(axis)
        if method != "trapezoid":
            raise ValueError("Metric weighting must be equal or trapezoid")
        a = np.asarray(axis)
        if np.any(np.diff(a) <= 0):
            raise ValueError("Trapezoid quadrature requires increasing coordinates")
        w = np.r_[a[1] - a[0], a[2:] - a[:-2], a[-1] - a[-2]] / 2
        return w / w.sum()
    options = config["metrics"]
    ww = axis_weights(grid.wavelengths_nm, options["wavelength_weighting"])
    wa = axis_weights(grid.angles_deg, options["angle_weighting"])
    wp = np.array([options["polarization_weights"][p] for p in grid.polarizations], dtype=float)
    if np.any(wp <= 0):
        raise ValueError("Polarization weights must be positive")
    wp /= wp.sum()
    return (ww[:, None, None] * wa[None, :, None] * wp[None, None, :]).ravel()


def reflectance_vector(frame: pd.DataFrame, grid: OpticalGrid) -> np.ndarray:
    columns = ["wavelength_nm", "angle_deg", "polarization"]
    if frame.duplicated(columns).any():
        raise ValueError("Metrics expect exactly one geometry and one value per state")
    if "status" in frame and not (frame.status == "ok").all():
        raise ValueError("Failed physical states cannot enter an optical objective")
    indexed = frame.set_index(columns)
    expected = pd.MultiIndex.from_tuples(grid.states(), names=columns)
    if set(indexed.index) != set(expected):
        raise ValueError("Optical objective grid is incomplete or different")
    values = indexed.reindex(expected).R.to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite reflectance")
    return values


def optical_metrics(values: np.ndarray | pd.DataFrame, grid: OpticalGrid,
                    config: Mapping[str, Any]) -> dict[str, float]:
    r = reflectance_vector(values, grid) if isinstance(values, pd.DataFrame) else np.asarray(values, dtype=float).ravel()
    weights = optical_weights(grid, config)
    if len(r) != len(weights) or not np.isfinite(r).all():
        raise ValueError("Invalid optical metric input")
    order = np.argsort(r)
    qindex = min(int(np.searchsorted(np.cumsum(weights[order]), 0.95)), len(r) - 1)
    metrics = {"mean_R": float(r @ weights), "q95_R": float(r[order[qindex]]),
               "max_R": float(r.max())}
    cube = r.reshape(len(grid.wavelengths_nm), len(grid.angles_deg), len(grid.polarizations))
    if set(grid.polarizations) == {"TE", "TM"}:
        metrics["mean_absolute_TE_TM_gap"] = float(np.mean(abs(
            cube[:, :, grid.polarizations.index("TE")] - cube[:, :, grid.polarizations.index("TM")])))
    return metrics


def robust_metrics(objectives: np.ndarray, config: Mapping[str, Any]) -> dict[str, float]:
    values = np.asarray(objectives, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Robust statistics require at least two finite realizations")
    mean, std = float(values.mean()), float(values.std(ddof=1))
    return {"robust_mean_R": mean, "robust_std_R": std,
            "robust_q95_R": float(np.quantile(values, 0.95)),
            "robust_score": mean + config["optimization"]["kappa"] * std}


class BudgetExhausted(RuntimeError):
    """A search has consumed its configured logical state-evaluation budget."""


class PhysicsObjective:
    """Every accepted objective is Maxwell evaluated; no NN values enter this ledger."""
    def __init__(self, config: Mapping[str, Any], grid: OpticalGrid, kind: str,
                 directory: Path, seed: int):
        self.config, self.grid, self.kind, self.directory = config, grid, kind, directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.draws = tolerance_draws(config["optimization"]["mc_samples"], seed, config["tolerance"])
        self.used = 0
        self.rows: list[dict[str, Any]] = []
        self.memo: dict[str, float] = {}
        scientific = {"optimization": config["optimization"], "grid": grid.record(),
                      "geometry": config["geometry"], "tolerance": config["tolerance"],
                      "rcwa": config["rcwa"], "seed": seed, "kind": kind,
                      "metrics": config["metrics"], "physics_source": physics_source_hash()}
        if directory.name.startswith("surrogate_"):
            model_metadata = json.loads((resolve_model_directory(config)/"ensemble.json").read_text(encoding="utf-8"))
            if not model_metadata.get("training_qc", {}).get("passed", False):
                raise RuntimeError("Surrogate optimization requires passing training QC")
            scientific["surrogate_training_signature"] = model_metadata["training_signature"]
            calibration = resolve_model_directory(config)/"calibration.json"
            scientific["surrogate_calibration"] = json.loads(calibration.read_text(encoding="utf-8"))
        protocol = fingerprint({**scientific, "source": Path(__file__).read_text(encoding="utf-8")})
        protocol_path = directory / "protocol.json"
        metadata = {"signature": protocol, "scientific_signature": fingerprint(scientific)}
        if protocol_path.exists():
            previous = json.loads(protocol_path.read_text(encoding="utf-8"))
            if previous["signature"] != protocol:
                if _LEGACY_PROTOCOLS.get(previous["signature"]) != fingerprint(scientific):
                    raise RuntimeError("Optimization protocol changed; use a new experiment_id")
                backup = directory / "protocol.before_parallel_upgrade.json"
                if not backup.exists():
                    save_json(backup, previous)
                metadata["migrated_from"] = previous["signature"]
                metadata["migration"] = "Scheduling only; physics, search order, metrics and budgets unchanged"
                LOGGER.info("Reusing verified legacy optimization history: %s", directory.name)
            else:
                metadata = previous
        save_json(protocol_path, metadata)
        history_path = directory / "true_history.csv"
        if history_path.exists():
            history = pd.read_csv(history_path, float_precision="round_trip")
            if (history.empty or history.geometry_id.duplicated().any() or
                    not np.isfinite(history.true_objective.to_numpy(float)).all() or
                    not history.logical_state_solves.is_monotonic_increasing):
                raise RuntimeError(f"Invalid optimization history: {history_path}")
            self.rows = history.to_dict("records")
            self.used = int(history.logical_state_solves.iloc[-1])
            self.memo = dict(zip(history.geometry_id, history.true_objective))
        manifest = make_manifest(config)
        self.protected = set(manifest.loc[manifest.role != "train_initial", "geometry_id"])
        # V3 may compare optimizers on a fixed, previously opened confirmation set.
        # Keep those geometries out of both the physical and surrogate search routes.
        self.protected.update(config.get("_extra_protected_ids", ()))
        self.session_times: list[float] = []
        options = config["optimization"]
        method = directory.name.rsplit("_", 1)[0]
        per_evaluation = len(grid.states()) * (1 + options["mc_samples"] if kind == "robust" else 1)
        reserve = per_evaluation * (config["rcwa"]["retry"]["max_retries"] + 1)
        budget_count = max(0, (options["state_budget"] - reserve) // per_evaluation + 1)
        target = {"random": options["search_evaluations"], "bo": min(options["search_evaluations"], options["bo_pool_size"]),
                  "direct": max(5, options["de_population"]) * (options["de_generations"] + 1),
                  "surrogate": options["verify_candidates"]}.get(method, options["search_evaluations"])
        self.target = min(target, budget_count)
        self.update_progress()

    def update_progress(self) -> None:
        recent = float(np.median(self.session_times[-10:])) if self.session_times else None
        save_json(self.directory / "progress.json", {
            "subtask": self.directory.name, "verified_geometries": len(self.rows),
            "estimated_max_geometries_without_retries": self.target,
            "logical_state_solves": self.used, "state_budget": self.config["optimization"]["state_budget"],
            "recent_new_geometry_wall_s": recent,
            "rough_remaining_seconds": None if recent is None else max(0, self.target - len(self.rows)) * recent,
            "note": "Time estimate excludes GP/NN controller time and may shorten on optimizer convergence.",
            "execution": asdict(ExecutionSettings.from_config(self.config)),
        })

    def feasible(self, g: Geometry) -> bool:
        if g.id in self.protected or feasibility(g, self.config, "design"):
            return False
        if self.config["optimization"]["enforce_tolerance_feasible"] or self.kind == "robust":
            return robust_feasible(g, self.config)
        return True

    def __call__(self, array: Any) -> float:
        g = Geometry.from_array(array)
        if not self.feasible(g):
            return float(self.config["optimization"]["infeasible_penalty"])
        if g.id in self.memo:
            return self.memo[g.id]
        geometries = [g]
        if self.kind == "robust":
            geometries += perturbations(g, self.draws, self.config["tolerance"])
        if any(v.id in self.protected for v in geometries):
            return float(self.config["optimization"]["infeasible_penalty"])
        unique = list({v.id: v for v in geometries}.values())
        # Reserve all configured retry attempts; actual used work is separately recorded.
        reserve = len(unique) * len(self.grid.states()) * (self.config["rcwa"]["retry"]["max_retries"] + 1)
        if self.used + reserve > self.config["optimization"]["state_budget"]:
            raise BudgetExhausted("Logical RCWA state budget exhausted")
        started = perf_counter()
        frame, cost = compute_batch(unique, self.grid, self.config)
        self.used += cost["logical_state_solves"]
        grouped = {gid: optical_metrics(group, self.grid, self.config)
                   for gid, group in frame.groupby("geometry_id")}
        result = {**g.record(), **grouped[g.id]}
        if self.kind == "robust":
            result.update(robust_metrics(np.array([grouped[v.id]["mean_R"] for v in geometries[1:]]), self.config))
            score = result["robust_score"]
        else:
            score = result["mean_R"]
        row = {**result, "true_objective": score, "objective_kind": self.kind,
               "logical_state_solves": self.used, "evaluation_wall_s": perf_counter() - started,
               "solver_elapsed_s": cost["solver_elapsed_s"], "cache_hits": cost["cache_hits"],
               "new_state_solves": cost["new_state_solves"],
               "rcwa_workers": cost["execution"]["workers"]}
        self.rows.append(row)
        self.memo[g.id] = score
        write_csv(pd.DataFrame(self.rows), self.directory / "true_history.csv")
        if cost["new_state_solves"]:
            self.session_times.append(row["evaluation_wall_s"])
        self.update_progress()
        LOGGER.info("[%s] verified %d/~%d geometries; RCWA budget %d/%d; objective %.8g; %.1f s",
                    self.directory.name, len(self.rows), self.target, self.used,
                    self.config["optimization"]["state_budget"], score, row["evaluation_wall_s"])
        return score


def _surrogate_objective(g: Geometry, model: Any, grid: OpticalGrid, kind: str,
                         draws: np.ndarray, config: Mapping[str, Any]) -> float:
    if feasibility(g, config, "design") or ((kind == "robust" or
            config["optimization"]["enforce_tolerance_feasible"]) and not robust_feasible(g, config)):
        return float(config["optimization"]["infeasible_penalty"])
    geometries = [g] if kind == "nominal" else perturbations(g, draws, config["tolerance"])
    prediction = model.predict(feature_matrix(geometries, grid))
    member_r = prediction.members[:, :, 0].reshape(len(model.models), len(geometries), -1)
    member_j = member_r @ optical_weights(grid, config)
    if kind == "nominal":
        scores = member_j[:, 0]
    else:
        scores = member_j.mean(axis=1) + config["optimization"]["kappa"] * member_j.std(axis=1, ddof=1)
    # Model-risk penalty is not a claimed probabilistic confidence bound.
    return float(scores.mean() + config["optimization"]["uncertainty_penalty"] * scores.std(ddof=1))


def _bayesian_search(objective: PhysicsObjective, config: Mapping[str, Any], seed: int) -> None:
    from scipy.special import ndtr
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    options = config["optimization"]
    pool = sample_geometries(options["bo_pool_size"], config, seed, "design", objective.protected,
                              robust=options["enforce_tolerance_feasible"] or objective.kind == "robust")
    bounds = domain_bounds(config, "design")
    x = (np.array([g.array() for g in pool]) - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    used, observations = [], []
    for iteration in range(min(options["search_evaluations"], len(pool))):
        if iteration < options["bo_initial"]:
            index = iteration
        else:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.ones(4), nu=2.5) + WhiteKernel(1e-8, (1e-12, 1e-3))
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=seed,
                                           n_restarts_optimizer=options["bo_restarts"])
            gp.fit(x[used], np.asarray(observations))
            mu, sigma = gp.predict(x, return_std=True)
            sigma = np.maximum(sigma, options["bo_sigma_floor"])
            improvement = min(observations) - mu - options["bo_xi"]
            z = improvement / sigma
            ei = improvement * ndtr(z) + sigma * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
            ei[used] = -np.inf
            index = int(np.argmax(ei))
        value = objective(pool[index].array())
        used.append(index)
        observations.append(value)


def resolve_model_directory(config: Mapping[str, Any]) -> Path:
    requested = config["optimization"].get("model_directory")
    if requested:
        path = Path(requested)
        return path if path.is_absolute() else Path(config["_root"]) / path
    return Path(config["_root"]) / "models" / config["model"]["primary_variant"]


def run_optimization(config: Mapping[str, Any]) -> dict[str, Any]:
    from scipy.optimize import differential_evolution
    from validate_physics import require_validation
    require_validation(config)
    grid = OpticalGrid.from_config(config["grid"])
    options = config["optimization"]
    bounds = domain_bounds(config, "design")
    all_candidates = []
    model = None
    summaries = {}
    subtask_names = [f"{method}_{kind}" for kind in options["objectives"] for method in options["methods"]]
    stop_after = config.get("runtime", {}).get("stop_after_subtask")
    if stop_after is not None and stop_after not in subtask_names:
        raise ValueError(f"Unknown stop_after_subtask {stop_after}; choose one of {subtask_names}")
    progress_path = Path(config["_root"]) / "optimization" / "progress.json"
    for kind in options["objectives"]:
        if kind not in {"nominal", "robust"}:
            raise ValueError("Objectives must be nominal/robust")
        for method in options["methods"]:
            name = f"{method}_{kind}"
            LOGGER.info("Optimization subtask %d/%d: %s", len(summaries) + 1, len(subtask_names), name)
            save_json(progress_path, {"complete": False, "current_subtask": name,
                                     "completed_subtasks": list(summaries), "all_subtasks": subtask_names})
            directory = Path(config["_root"]) / "optimization" / f"{method}_{kind}"
            objective = PhysicsObjective(config, grid, kind, directory, config["seed"] + options["mc_seed_offset"])
            previous_evaluation_wall = sum(r["evaluation_wall_s"] for r in objective.rows)
            had_history = bool(objective.rows)
            start = perf_counter()
            termination = "completed"
            saved_summary = directory / "summary.json"
            try:
                if saved_summary.exists():
                    termination = "resumed_completed_run"
                    LOGGER.info("Skipping completed subtask %s (%d verified geometries)", name, len(objective.rows))
                elif method == "random":
                    pool = sample_geometries(options["search_evaluations"], config, config["seed"] + 101,
                                              "design", objective.protected,
                                              robust=options["enforce_tolerance_feasible"] or kind == "robust")
                    for g in pool:
                        objective(g.array())
                elif method == "bo":
                    _bayesian_search(objective, config, config["seed"] + 101)
                elif method == "direct":
                    initial = sample_geometries(max(5, options["de_population"]), config,
                                                 config["seed"] + 101, "design", objective.protected,
                                                 robust=options["enforce_tolerance_feasible"] or kind == "robust")
                    differential_evolution(objective, bounds, init=np.array([g.array() for g in initial]),
                                           seed=config["seed"], maxiter=options["de_generations"],
                                           polish=False, workers=1, tol=options["de_tolerance"])
                elif method == "surrogate":
                    if model is None:
                        from surrogate import Ensemble
                        model = Ensemble.load(resolve_model_directory(config), config["model"]["device"],
                                              config["model"]["predict_batch_size"])
                        LOGGER.info("Surrogate ensemble device: %s", model.device)
                    predicted: dict[str, tuple[Geometry, float]] = {}
                    def function(values: Any) -> float:
                        g = Geometry.from_array(values)
                        value = _surrogate_objective(g, model, grid, kind, objective.draws, config)
                        if objective.feasible(g) and np.isfinite(value):
                            predicted[g.id] = (g, value)
                        return value
                    initial = sample_geometries(max(5, options["de_population"]), config,
                                                 config["seed"] + 101, "design", objective.protected,
                                                 robust=options["enforce_tolerance_feasible"] or kind == "robust")
                    differential_evolution(function, bounds, init=np.array([g.array() for g in initial]),
                                           seed=config["seed"], maxiter=options["de_generations"],
                                           polish=False, workers=1, tol=options["de_tolerance"])
                    ranked = sorted(predicted.values(), key=lambda item: item[1])
                    pd.DataFrame([{**g.record(), "surrogate_objective": score} for g, score in ranked]).to_csv(
                        directory / "surrogate_screening.csv", index=False)
                    for g, _ in ranked[:options["verify_candidates"]]:
                        objective(g.array())
                else:
                    raise ValueError(f"Unknown optimization method {method}")
            except BudgetExhausted as exc:
                termination = str(exc)
            if not objective.rows:
                raise RuntimeError(f"{method}/{kind} produced no verified design; increase budget or check constraints")
            verified = pd.DataFrame(objective.rows).sort_values("true_objective")
            chosen = verified.head(options["representatives_per_method"]).copy()
            chosen["method"] = method
            chosen["repeat_seed"] = config["seed"]
            all_candidates.append(chosen)
            summary = {"method": method, "objective": kind, "termination": termination,
                       "logical_state_solves": objective.used,
                       "wall_time_s": previous_evaluation_wall + perf_counter() - start,
                       "wall_time_is_lower_bound_after_resume": had_history,
                       "previous_evaluation_wall_s": previous_evaluation_wall,
                       "execution": asdict(ExecutionSettings.from_config(config)),
                       "best_true_objective": float(verified.true_objective.min()),
                       "training_cost_included": False,
                       "note": "Search costs only. Add model training and AL/data costs for end-to-end comparisons."}
            if saved_summary.exists():
                summary = json.loads(saved_summary.read_text())
            summaries[f"{method}_{kind}"] = summary
            save_json(directory / "summary.json", summary)
            LOGGER.info("Finished subtask %s (%d/%d); summary: %s", name, len(summaries), len(subtask_names), saved_summary)
            if name == stop_after and len(summaries) < len(subtask_names):
                result = {"complete": False, "stopped_after_subtask": name,
                          "completed_subtasks": list(summaries), "remaining_subtasks": subtask_names[len(summaries):]}
                save_json(progress_path, result)
                LOGGER.info("Requested stop after %s. Restart optimize without --stop-after-subtask to continue.", name)
                return result
    candidates = pd.concat(all_candidates, ignore_index=True)
    write_parquet(candidates, Path(config["_root"]) / "optimization" / "candidates.parquet")
    candidates.to_csv(Path(config["_root"]) / "optimization" / "candidates.csv", index=False)
    save_json(progress_path, {"complete": True, "completed_subtasks": list(summaries), "all_subtasks": subtask_names})
    return summaries
