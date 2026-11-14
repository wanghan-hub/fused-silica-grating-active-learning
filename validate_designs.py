"""Dense, high-fidelity final validation and independent manufacturing trials."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from optics import (Geometry, OpticalGrid, feasibility, perturbations, tolerance_draws,
                    fingerprint, save_json)
from dataset import compute_batch, write_parquet
from rcwa_solver import SolverConfig
from optimize import optical_metrics, robust_metrics
from validate_physics import export_comsol, require_validation


def run_final_validation(config: Mapping[str, Any]) -> dict[str, Any]:
    from report_results import bootstrap_interval
    require_validation(config)
    root = Path(config["_root"])
    directory = root / "final_validation"
    directory.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_parquet(root / "optimization" / "candidates.parquet")
    if candidates.empty:
        raise RuntimeError("No RCWA-verified candidates")
    unique = candidates.drop_duplicates("geometry_id")
    geometries = [Geometry.from_mapping(row) for row in unique.to_dict("records")]
    grid = OpticalGrid.from_config(config["final_validation"]["grid"])
    probe = OpticalGrid.from_config(config["comsol"]["grid"])
    reference = SolverConfig.from_mapping(config["rcwa"]["reference"])
    refined = SolverConfig.from_mapping(config["rcwa"]["reference_check"])
    nominal, nominal_cost = compute_batch(geometries, grid, config, reference)
    write_parquet(nominal, directory / "nominal_spectra.parquet")
    nominal.to_csv(directory / "nominal_spectra.csv", index=False)
    cross, cross_cost = compute_batch(geometries, probe, config, reference)
    check, check_cost = compute_batch(geometries, probe, config, refined)
    keys = ["geometry_id", "wavelength_nm", "angle_deg", "polarization"]
    comparison = cross.merge(check[keys + ["R", "T"]], on=keys, suffixes=("", "_refined"), validate="one_to_one")
    comparison["refinement_error"] = np.maximum(abs(comparison.R - comparison.R_refined),
                                                 abs(comparison["T"] - comparison.T_refined))
    comparison.to_csv(directory / "final_convergence.csv", index=False)
    if float(comparison.refinement_error.max()) > config["validation"]["convergence_tolerance"]:
        save_json(directory / "summary.json", {"passed": False, "reason": "final_candidate_not_converged"})
        raise RuntimeError("Final candidate convergence failed; increase discretization before claims")
    export_comsol(geometries, probe, directory / "comsol_request.csv")
    cross.to_csv(directory / "comsol_rcwa_reference.csv", index=False)
    options = config["final_validation"]
    mc_seed = config["seed"] + options["mc_seed_offset"]
    if mc_seed == config["seed"] + config["optimization"]["mc_seed_offset"]:
        raise ValueError("Final Monte Carlo seed must be independent of search Monte Carlo seed")
    draws = tolerance_draws(options["mc_samples"], mc_seed, config["tolerance"])
    samples, summaries, costs = [], [], []
    for g in geometries:
        perturbed = perturbations(g, draws, config["tolerance"])
        valid = [v for v in perturbed if not feasibility(v, config, "cover")]
        metrics_by_id = {}
        if valid:
            unique_valid = list({v.id: v for v in valid}.values())
            values, cost = compute_batch(unique_valid, grid, config, reference)
            costs.append(cost)
            write_parquet(values, directory / "mc_spectra" / f"{g.id}.parquet")
            metrics_by_id = {gid: optical_metrics(group, grid, config)
                             for gid, group in values.groupby("geometry_id")}
        objectives = []
        for index, v in enumerate(perturbed):
            reasons = feasibility(v, config, "cover")
            row = {**v.record(), "family_id": g.id, "sample": index, "mc_seed": mc_seed,
                   "feasible": not reasons, "failure_reasons": ";".join(reasons), "mean_R": np.nan}
            if not reasons:
                row.update(metrics_by_id[v.id])
                objectives.append(row["mean_R"])
            samples.append(row)
        nominal_metrics = optical_metrics(nominal[nominal.geometry_id == g.id], grid, config)
        summary = {**g.record(), **nominal_metrics, "mc_total": len(perturbed),
                   "mc_feasible": len(objectives), "feasibility_rate": len(objectives) / len(perturbed),
                   "robust_statistics_conditional_on_feasibility": len(objectives) != len(perturbed)}
        if len(objectives) >= 2:
            summary.update(robust_metrics(np.array(objectives), config))
            low, high = bootstrap_interval(np.asarray(objectives), config["report"]["bootstrap_samples"],
                                            config["report"]["confidence"], mc_seed)
            summary.update(robust_mean_ci_low=low, robust_mean_ci_high=high)
        summaries.append(summary)
    sample_frame = pd.DataFrame(samples)
    sample_frame.to_csv(directory / "manufacturing_samples.csv", index=False)
    write_parquet(sample_frame, directory / "manufacturing_samples.parquet")
    summary_frame = pd.DataFrame(summaries)
    # Retain every method/objective attribution even if two methods found the same geometry.
    attribution = candidates[["geometry_id", "method", "objective_kind", "repeat_seed"]].drop_duplicates()
    summary_frame.merge(attribution, on="geometry_id", validate="one_to_many").to_csv(
        directory / "design_summary.csv", index=False)
    pairs = []
    for index, left in enumerate(geometries):
        for right in geometries[index + 1:]:
            a = sample_frame[sample_frame.family_id == left.id]
            b = sample_frame[sample_frame.family_id == right.id]
            joined = a.merge(b, on="sample", suffixes=("_left", "_right"))
            good = joined.feasible_left & joined.feasible_right
            if good.sum() >= 2:
                delta = joined.loc[good, "mean_R_left"].values - joined.loc[good, "mean_R_right"].values
                low, high = bootstrap_interval(delta, config["report"]["bootstrap_samples"],
                                                config["report"]["confidence"], mc_seed)
                pairs.append({"left": left.id, "right": right.id, "paired_count": int(good.sum()),
                              "mean_difference": float(delta.mean()), "ci_low": low, "ci_high": high,
                              "conditional_on_joint_feasibility": not bool(good.all())})
    pd.DataFrame(pairs).to_csv(directory / "paired_robust_comparisons.csv", index=False)
    summary = {"passed": True, "cross_solver_status": "pending_external_results",
               "unique_designs": len(geometries), "nominal_cost": nominal_cost,
               "cross_solver_reference_cost": cross_cost, "convergence_check_cost": check_cost,
               "manufacturing_costs": costs, "independent_mc_seed": mc_seed,
               "candidate_manifest_hash": fingerprint([g.record() for g in geometries])}
    save_json(directory / "summary.json", summary)
    return summary
