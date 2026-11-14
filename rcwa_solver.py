"""The ONLY module importing grcwa. No surrogate/analytic fallback creates labels.

Uses grcwa's public obj/Add_Layer*/Init_Setup/GridLayer_geteps/RT_Solve API.
The 2-D lattice adapter enforces Gy=0 to represent the requested 1-D problem.
Both fluxes are obtained from Maxwell solutions, never T=1-R.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from importlib.metadata import version
from time import perf_counter
from typing import Any, Mapping
import json
import logging

import numpy as np

from optics import (Geometry, OpticalGrid, MATERIAL_VERSION, fingerprint,
                    layer_duties, physical_errors, silica_index)

LOGGER = logging.getLogger(__name__)
ADAPTER_VERSION = "grcwa_1d_flux_adapter_v1"


@dataclass(frozen=True)
class SolverConfig:
    harmonics: int = 10
    nz: int = 16
    nx: int = 512
    energy_tolerance: float = 1e-6
    power_tolerance: float = 1e-8
    order_sum_tolerance: float = 1e-9

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> SolverConfig:
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown RCWA settings: {unknown}")
        obj = cls(**values)
        if obj.harmonics < 1 or obj.nz < 1 or obj.nx < 4 * obj.harmonics + 1:
            raise ValueError("Require harmonics>=1, nz>=1, nx>=4*harmonics+1")
        return obj

    @property
    def id(self) -> str:
        return fingerprint(asdict(self))


def solver_identity() -> dict[str, str]:
    """Fail early on missing dependency, rather than creating empty 'successful' data."""
    try:
        import grcwa  # noqa: F401; external dependency isolated to this module
        installed = version("grcwa")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("Install grcwa in the simulation environment; no synthetic fallback exists") from exc
    return {"solver": "grcwa", "solver_version": installed,
            "adapter_version": ADAPTER_VERSION, "material_version": MATERIAL_VERSION}


def qc_errors(row: Mapping[str, Any], settings: SolverConfig) -> list[str]:
    errors: list[str] = []
    r, t = float(row.get("R", np.nan)), float(row.get("T", np.nan))
    if not np.all(np.isfinite([r, t])):
        return ["nonfinite_power"]
    if min(r, t) < -settings.power_tolerance or max(r, t) > 1 + settings.power_tolerance:
        errors.append("power_bounds")
    if abs(r + t - 1) > settings.energy_tolerance:
        errors.append("energy_closure")
    return errors


def _build(g: Geometry, wavelength_nm: float, angle_deg: float,
           settings: SolverConfig) -> tuple[Any, np.ndarray]:
    import grcwa
    grcwa.set_backend("numpy")
    # grcwa uses c=1. All internal distances are micrometres, f=1/lambda_um.
    period = g.period_nm / 1000.0
    target_modes = 2 * settings.harmonics + 1
    # Small artificial y period pushes every nonzero Gy outside circular truncation.
    ly = period / (4 * settings.harmonics + 4)
    obj = grcwa.obj(target_modes + 1, [period, 0.0], [0.0, ly],
                    1000.0 / wavelength_nm, np.radians(angle_deg), 0.0, verbose=0)
    obj.Add_LayerUniform(0.0, 1.0)
    if g.height_nm > 0:
        for _ in range(settings.nz):
            obj.Add_LayerGrid(g.height_nm / 1000.0 / settings.nz, settings.nx, 1)
    eps = float(silica_index(wavelength_nm)) ** 2
    obj.Add_LayerUniform(0.0, eps)
    obj.Init_Setup(Gmethod=0)
    orders = np.asarray(obj.G, dtype=int)
    expected = set(range(-settings.harmonics, settings.harmonics + 1))
    if np.any(orders[:, 1] != 0) or set(orders[:, 0]) != expected:
        raise RuntimeError(f"Library truncation incompatible with 1-D adapter: G={orders.tolist()}")
    if g.height_nm > 0:
        centers = (np.arange(settings.nx) + 0.5) / settings.nx - 0.5
        grids = [np.where(np.abs(centers) < duty / 2, eps, 1.0)
                 for duty in layer_duties(g, settings.nz)]
        obj.GridLayer_geteps(np.concatenate(grids))
    return obj, orders


def solve_state(g: Geometry, wavelength_nm: float, angle_deg: float,
                polarization: str, settings: SolverConfig) -> dict[str, Any]:
    """Compute raw total/order power and diagnostic status; exceptions are recorded."""
    identity = solver_identity()
    start = perf_counter()
    row: dict[str, Any] = {**g.record(), "wavelength_nm": float(wavelength_nm),
        "angle_deg": float(angle_deg), "polarization": polarization, **identity,
        "solver_config_hash": settings.id, "solver_config_json": json.dumps(asdict(settings)),
        "harmonics": settings.harmonics, "nz": settings.nz, "nx": settings.nx,
        "lattice_y_nm": g.period_nm / (4 * settings.harmonics + 4),
        "R": np.nan, "T": np.nan, "R_zero": np.nan, "T_zero": np.nan,
        "energy_error": np.nan, "actual_mode_count": 0, "orders_json": "[]",
        "status": "failed", "error": "", "matrix_setups": 0}
    try:
        invalid = physical_errors(g, allow_flat=True)
        if invalid:
            raise ValueError(str(invalid))
        if polarization not in ("TE", "TM") or not -90 < angle_deg < 90:
            raise ValueError("Invalid polarization or angle")
        obj, orders = _build(g, wavelength_nm, angle_deg, settings)
        row["matrix_setups"] = 1
        zero_index = int(np.flatnonzero(np.all(orders == 0, axis=1))[0])
        obj.MakeExcitationPlanewave(p_amp=float(polarization == "TM"), p_phase=0.0,
                                   s_amp=float(polarization == "TE"), s_phase=0.0,
                                   order=zero_index)
        # Standard normalization divides by incident flux, NOT a posteriori R+T.
        r_orders, t_orders = obj.RT_Solve(normalize=1, byorder=1)
        r_orders, t_orders = np.asarray(r_orders, dtype=float), np.asarray(t_orders, dtype=float)
        r, t = float(np.sum(r_orders)), float(np.sum(t_orders))
        n_sub = float(silica_index(wavelength_nm))
        kx_normalized = np.sin(np.radians(angle_deg)) + orders[:, 0] * wavelength_nm / g.period_nm
        power_orders = [{"m": int(m), "R": float(rm), "T": float(tm),
                         "propagating_air": bool(abs(kx) < 1),
                         "propagating_substrate": bool(abs(kx) < n_sub)}
                        for m, rm, tm, kx in zip(orders[:, 0], r_orders, t_orders, kx_normalized)]
        row.update(R=r, T=t, R_zero=float(r_orders[zero_index]), T_zero=float(t_orders[zero_index]),
                   energy_error=abs(r + t - 1), actual_mode_count=len(orders),
                   orders_json=json.dumps(power_orders))
        errors = qc_errors(row, settings)
        if not np.all(np.isfinite(r_orders)) or not np.all(np.isfinite(t_orders)):
            errors.append("nonfinite_order_power")
        if min(float(r_orders.min()), float(t_orders.min())) < -settings.power_tolerance:
            errors.append("negative_order_power")
        row.update(status="ok" if not errors else "qc_failed", error=";".join(errors))
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        LOGGER.debug("RCWA failed for %s: %s", g.id, exc)
    row["elapsed_s"] = perf_counter() - start
    return row


def solve_geometry(g: Geometry, grid: OpticalGrid, settings: SolverConfig) -> list[dict[str, Any]]:
    """Deterministic geometry-major scan. Each state is solved independently."""
    return [solve_state(g, wavelength, angle, polarization, settings)
            for wavelength, angle, polarization in grid.states()]
