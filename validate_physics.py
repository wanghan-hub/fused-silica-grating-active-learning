"""Analytic, discretization and external COMSOL checks; no COMSOL dependency."""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping
import json
import logging

import numpy as np
import pandas as pd

from optics import (Geometry, OpticalGrid, GEOMETRY_COLUMNS, STATE_COLUMNS,
                    domain_bounds, feasibility, fingerprint, fresnel, save_json)
from rcwa_solver import SolverConfig, solve_geometry, solver_identity

LOGGER = logging.getLogger(__name__)


def physical_signature(config: Mapping[str, Any]) -> str:
    keys = ("grid", "geometry", "rcwa", "validation", "comsol")
    source = {name: Path(__file__).with_name(name).read_text(encoding="utf-8")
              for name in ("optics.py", "rcwa_solver.py", "validate_physics.py")}
    return fingerprint({"settings": {k: config[k] for k in keys},
                        "source": source, "backend": solver_identity()})


def condition_id(g: Geometry, wave: float, angle: float, pol: str) -> str:
    return fingerprint([g.id, float(wave), float(angle), pol])[:28]


def export_comsol(geometries: list[Geometry], grid: OpticalGrid, path: Path) -> pd.DataFrame:
    """Export a strict condition manifest; blank results are NOT zero observations."""
    records = [{**g.record(), "condition_id": condition_id(g, w, a, p),
                "wavelength_nm": w, "angle_deg": a, "polarization": p,
                "R_comsol": np.nan, "T_comsol": np.nan}
               for g in geometries for w, a, p in grid.states()]
    frame = pd.DataFrame(records).drop_duplicates("condition_id")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def compare_comsol(request_path: Path, result_path: Path, rcwa_frame: pd.DataFrame,
                   config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    """Strict one-to-one matching, completeness, coordinate and power checks."""
    expected = pd.read_csv(request_path, float_precision="round_trip")
    external = pd.read_csv(result_path, float_precision="round_trip")
    required = ["condition_id", *GEOMETRY_COLUMNS, *STATE_COLUMNS, "R_comsol", "T_comsol"]
    missing = set(required) - set(external.columns)
    if missing:
        raise ValueError(f"COMSOL CSV missing columns: {sorted(missing)}")
    if external.condition_id.duplicated().any() or expected.condition_id.duplicated().any():
        raise ValueError("Duplicate COMSOL condition_id")
    if set(expected.condition_id) != set(external.condition_id):
        raise ValueError("COMSOL conditions must exactly match the exported manifest")
    aligned = external.set_index("condition_id").loc[expected.condition_id].reset_index()
    for column in GEOMETRY_COLUMNS + ["wavelength_nm", "angle_deg"]:
        if not np.allclose(expected[column], aligned[column], rtol=0, atol=1e-8):
            raise ValueError(f"COMSOL geometry/state mismatch: {column}")
    if not np.array_equal(expected.polarization.values, aligned.polarization.values):
        raise ValueError("COMSOL polarization mismatch")
    rcwa = rcwa_frame.copy()
    rcwa["condition_id"] = [
        condition_id(Geometry.from_mapping(r), r["wavelength_nm"], r["angle_deg"], r["polarization"])
        for r in rcwa.to_dict("records")]
    if rcwa.condition_id.duplicated().any():
        raise ValueError("Duplicate RCWA conditions in comparison")
    joined = aligned.merge(rcwa[["condition_id", "R", "T", "status"]],
                           on="condition_id", how="left", validate="one_to_one")
    values = joined[["R", "T", "R_comsol", "T_comsol"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Missing/nonfinite COMSOL or RCWA powers")
    joined["delta_R"] = joined.R_comsol - joined.R
    joined["delta_T"] = joined.T_comsol - joined["T"]
    external_energy = np.abs(joined.R_comsol + joined.T_comsol - 1)
    tolerance = config["comsol"]
    max_error = float(np.max(np.abs(joined[["delta_R", "delta_T"]].values)))
    passed = bool((joined.status == "ok").all() and
                  values.min() >= -tolerance["power_tolerance"] and
                  values.max() <= 1 + tolerance["power_tolerance"] and
                  external_energy.max() <= tolerance["energy_tolerance"] and
                  max_error <= tolerance["absolute_tolerance"])
    output.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output / "comsol_comparison.csv", index=False)
    summary = {"passed": passed, "count": len(joined), "max_absolute_difference": max_error,
               "rmse_R": float(np.sqrt(np.mean(joined.delta_R ** 2))),
               "max_comsol_energy_error": float(external_energy.max()),
               "physical_signature": physical_signature(config),
               "request_hash": fingerprint(expected.drop(columns=["R_comsol", "T_comsol"]).to_dict("records")),
               "result_file": str(result_path.resolve())}
    # Blank request powers are excluded from hashing to preserve strict JSON.
    save_json(output / "comsol_summary.json", summary)
    return summary


def run_validation(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(config["_root"]) / "validation"
    root.mkdir(parents=True, exist_ok=True)
    settings = SolverConfig.from_mapping(config["rcwa"]["base"])
    reference = SolverConfig.from_mapping(config["rcwa"]["reference"])
    refined = SolverConfig.from_mapping(config["rcwa"]["reference_check"])
    grid = OpticalGrid.from_config(config["grid"])
    probe = OpticalGrid.from_config(config["validation"]["probe_grid"])
    example = Geometry.from_array(config["validation"]["example_geometry"])
    if feasibility(example, config):
        raise ValueError("Validation example is not feasible")
    single = pd.DataFrame(solve_geometry(example, grid, settings))
    single.to_csv(root / "single_geometry.csv", index=False)
    flat = replace(example, height_nm=0.0)
    flat_data = pd.DataFrame(solve_geometry(flat, probe, settings))
    analytic = np.array([fresnel(r.wavelength_nm, r.angle_deg, r.polarization)
                         for r in flat_data.itertuples(index=False)])
    flat_data["R_fresnel"], flat_data["T_fresnel"] = analytic.T
    flat_data["error_R"] = flat_data.R - analytic[:, 0]
    flat_data["error_T"] = flat_data["T"] - analytic[:, 1]
    flat_data.to_csv(root / "fresnel.csv", index=False)
    normal = flat_data[flat_data.angle_deg == 0].pivot(index="wavelength_nm", columns="polarization", values="R")
    if not {"TE", "TM"} <= set(normal.columns):
        raise ValueError("Validation probe must include 0 degrees and both polarizations")
    polarization_error = float(np.max(np.abs(normal.TE - normal.TM)))
    rng = np.random.default_rng(config["seed"])
    bounds = domain_bounds(config)
    geometries = [example]
    for _ in range(config["sampling"]["max_candidates"]):
        if len(geometries) >= config["validation"]["convergence_geometries"]:
            break
        g = Geometry.from_array(rng.uniform(bounds[:, 0], bounds[:, 1]))
        if not feasibility(g, config) and g.id not in {v.id for v in geometries}:
            geometries.append(g)
    if len(geometries) < config["validation"]["convergence_geometries"]:
        raise RuntimeError("Could not sample convergence geometries")
    rows = []
    for g in geometries:
        LOGGER.info("Convergence geometry %s", g.id)
        ref = pd.DataFrame(solve_geometry(g, probe, reference))
        runs = [("base", 0, settings), ("reference_check", 0, refined)]
        for axis in ("harmonics", "nz", "nx"):
            runs.extend((axis, value, replace(settings, **{axis: int(value)}))
                        for value in config["validation"]["sweeps"][axis])
        for axis, value, cfg in runs:
            measured = pd.DataFrame(solve_geometry(g, probe, cfg))
            delta = measured[["R", "T"]].values - ref[["R", "T"]].values
            ok = bool((measured.status == "ok").all() and (ref.status == "ok").all())
            rows.append({"geometry_id": g.id, "axis": axis, "value": value,
                         **asdict(cfg), "max_error": float(np.max(np.abs(delta))),
                         "rmse": float(np.sqrt(np.mean(delta ** 2))), "qc_ok": ok,
                         "elapsed_s": float(measured.elapsed_s.sum())})
    convergence = pd.DataFrame(rows)
    convergence.to_csv(root / "convergence.csv", index=False)
    threshold = config["validation"]["convergence_tolerance"]
    relevant = convergence[convergence.axis.isin(["base", "reference_check"])]
    analytic_error = float(np.max(np.abs(flat_data[["error_R", "error_T"]].values)))
    passed = bool((single.status == "ok").all() and (flat_data.status == "ok").all() and
                  analytic_error <= config["validation"]["fresnel_tolerance"] and
                  polarization_error <= config["validation"]["fresnel_tolerance"] and
                  relevant.qc_ok.all() and np.isfinite(relevant.max_error).all() and
                  (relevant.max_error <= threshold).all())
    comsol_grid = OpticalGrid.from_config(config["comsol"]["grid"])
    comsol_geometries = [flat] + geometries[:config["comsol"]["initial_geometries"]]
    export_comsol(comsol_geometries, comsol_grid, root / "comsol_request.csv")
    pd.DataFrame([row for g in comsol_geometries
                  for row in solve_geometry(g, comsol_grid, reference)]).to_csv(
                      root / "comsol_rcwa_reference.csv", index=False)
    summary = {"passed": passed, "physical_signature": physical_signature(config),
               "single_geometry_states": len(single), "fresnel_max_error": analytic_error,
               "normal_TE_TM_error": polarization_error,
               "convergence_max_error": float(relevant.max_error.max()),
               "comsol_status": "pending_external_results"}
    save_json(root / "physics_summary.json", summary)
    return summary


def require_validation(config: Mapping[str, Any]) -> None:
    """Require fresh certificates; changing physical settings invalidates old gates."""
    root = Path(config["_root"]) / "validation"
    signature = physical_signature(config)
    certificates = ["physics_summary.json"]
    if config["dataset"]["require_initial_comsol"]:
        certificates.append("comsol_summary.json")
    for name in certificates:
        path = root / name
        if not path.exists():
            raise RuntimeError(f"Missing validation certificate: {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not summary.get("passed") or summary.get("physical_signature") != signature:
            raise RuntimeError(f"Failed or stale physical validation: {path}")
