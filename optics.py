"""Shared physical definitions. Coordinates: x periodic, y invariant, z downward.

All public lengths are nm and angles degrees. No measured fabrication accuracy is
implied by configured manufacturing constraints. Utilities here keep provenance
and physical conventions identical throughout the ten-module project.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any, Mapping
import json
import os

import numpy as np

GEOMETRY_COLUMNS = ["period_nm", "height_nm", "top_duty", "bottom_duty"]
STATE_COLUMNS = ["wavelength_nm", "angle_deg", "polarization"]
MATERIAL_VERSION = "Malitson1965_fused_silica_20C_lossless_v1"


def fingerprint(value: Any) -> str:
    """Stable hash; full floating point precision is preserved, never rounded."""
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                             separators=(",", ":"), default=str).encode()).hexdigest()


def save_json(path: str | Path, value: Any) -> None:
    """Atomically write strict JSON, mapping nonfinite results to null."""
    def clean(x: Any) -> Any:
        if isinstance(x, Mapping):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, np.ndarray)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, Path):
            return str(x)
        return x
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class Geometry:
    period_nm: float
    height_nm: float
    top_duty: float
    bottom_duty: float

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> Geometry:
        return cls(**{k: float(row[k]) for k in GEOMETRY_COLUMNS})

    @classmethod
    def from_array(cls, values: Any) -> Geometry:
        return cls(*map(float, values))

    def array(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in GEOMETRY_COLUMNS], dtype=float)

    def record(self) -> dict[str, Any]:
        return {**asdict(self), "geometry_id": self.id, "sidewall_deg": self.sidewall_deg}

    @property
    def id(self) -> str:
        return fingerprint(dict(zip(GEOMETRY_COLUMNS, map(float, self.array()))))[:24]

    @property
    def sidewall_deg(self) -> float:
        if self.height_nm == 0:
            return 0.0
        return float(np.degrees(np.arctan2(
            self.period_nm * (self.bottom_duty - self.top_duty), 2 * self.height_nm)))


@dataclass(frozen=True)
class OpticalGrid:
    wavelengths_nm: tuple[float, ...]
    angles_deg: tuple[float, ...]
    polarizations: tuple[str, ...] = ("TE", "TM")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> OpticalGrid:
        def expand(v: Any) -> tuple[float, ...]:
            if isinstance(v, Mapping):
                start, stop, step = (float(v[k]) for k in ("start", "stop", "step"))
                if step <= 0 or stop < start:
                    raise ValueError("Invalid grid range")
                n = int(round((stop - start) / step))
                if not np.isclose(start + n * step, stop):
                    raise ValueError("Grid stop must be reachable by its step")
                return tuple(float(start + i * step) for i in range(n + 1))
            return tuple(map(float, v))
        obj = cls(expand(config["wavelengths_nm"]), expand(config["angles_deg"]),
                  tuple(config.get("polarizations", ["TE", "TM"])))
        if not obj.wavelengths_nm or not obj.angles_deg or not obj.polarizations:
            raise ValueError("Empty optical grid")
        if len(set(obj.states())) != len(obj.states()):
            raise ValueError("Duplicate optical states")
        if min(obj.wavelengths_nm) <= 0 or max(map(abs, obj.angles_deg)) >= 90:
            raise ValueError("Invalid wavelength or incidence angle")
        if not set(obj.polarizations) <= {"TE", "TM"}:
            raise ValueError("Polarizations must be TE/TM")
        return obj

    def states(self) -> list[tuple[float, float, str]]:
        return list(product(self.wavelengths_nm, self.angles_deg, self.polarizations))

    def record(self) -> dict[str, Any]:
        return asdict(self)


def silica_index(wavelength_nm: Any) -> np.ndarray:
    """Malitson Sellmeier relation; vacuum wavelength in nm, transparent silica."""
    w = np.asarray(wavelength_nm, dtype=float) / 1000.0
    if np.any(~np.isfinite(w)) or np.any((w < 0.21) | (w > 3.71)):
        raise ValueError("Material model restricted to 210--3710 nm")
    b = np.array([0.6961663, 0.4079426, 0.8974794])
    poles = np.array([0.0684043, 0.1162414, 9.896161])
    return np.sqrt(1 + np.sum(b * w[..., None] ** 2 /
                             (w[..., None] ** 2 - poles ** 2), axis=-1))


def fresnel(wavelength_nm: float, angle_deg: float, polarization: str,
            n_incident: float = 1.0) -> tuple[float, float]:
    """Independent lossless interface R and T from field coefficients and flux."""
    n2 = float(silica_index(wavelength_nm))
    ci = np.cos(np.radians(angle_deg))
    st = n_incident * np.sin(np.radians(angle_deg)) / n2
    if ci <= 0 or abs(st) >= 1:
        raise ValueError("This interface benchmark excludes grazing/TIR incidence")
    ct = np.sqrt(1 - st * st)
    if polarization == "TE":
        denominator = n_incident * ci + n2 * ct
        r, t = (n_incident * ci - n2 * ct) / denominator, 2 * n_incident * ci / denominator
    elif polarization == "TM":
        denominator = n2 * ci + n_incident * ct
        r, t = (n2 * ci - n_incident * ct) / denominator, 2 * n_incident * ci / denominator
    else:
        raise ValueError("Unknown polarization")
    return float(abs(r) ** 2), float(n2 * ct / (n_incident * ci) * abs(t) ** 2)


def layer_duties(geometry: Geometry, nz: int) -> np.ndarray:
    """Top-to-bottom midpoint staircase; RCWA convergence must be demonstrated."""
    if nz < 1:
        raise ValueError("nz must be positive")
    return geometry.top_duty + (geometry.bottom_duty - geometry.top_duty) * (
        np.arange(nz) + 0.5) / nz


def domain_bounds(config: Mapping[str, Any], domain: str = "cover") -> np.ndarray:
    return np.asarray([config["geometry"][domain][k] for k in GEOMETRY_COLUMNS], dtype=float)


def physical_errors(g: Geometry, allow_flat: bool = False) -> list[str]:
    errors = []
    if not np.all(np.isfinite(g.array())):
        return ["nonfinite_geometry"]
    if g.period_nm <= 0 or g.height_nm < 0 or (g.height_nm == 0 and not allow_flat):
        errors.append("nonpositive_size")
    if not 0 < g.top_duty <= g.bottom_duty < 1:
        errors.append("invalid_duty")
    return errors


@lru_cache(maxsize=128)
def _diffraction_limit(grid_json: str, angle_deg: float) -> float:
    grid = OpticalGrid.from_config(json.loads(grid_json))
    waves = np.array(grid.wavelengths_nm)
    return float(np.min(waves / (silica_index(waves) + np.sin(np.radians(angle_deg)))))


def feasibility(g: Geometry, config: Mapping[str, Any], domain: str | None = "cover",
                manufacturing: bool = True) -> list[str]:
    """Return reasons; no clipping, repair, or implicit rejection resampling."""
    errors = physical_errors(g)
    if errors:
        return errors
    if domain is not None:
        bounds = domain_bounds(config, domain)
        if np.any(g.array() < bounds[:, 0]) or np.any(g.array() > bounds[:, 1]):
            errors.append("outside_" + domain)
    constraints = config["geometry"]["constraints"]
    if manufacturing:
        if g.period_nm * g.top_duty < constraints["min_top_nm"]:
            errors.append("top_width")
        if g.period_nm * (1 - g.bottom_duty) < constraints["min_gap_nm"]:
            errors.append("bottom_gap")
        if g.sidewall_deg > constraints["max_sidewall_deg"]:
            errors.append("sidewall")
    if constraints.get("require_zero_order", True):
        limit = _diffraction_limit(json.dumps(config["grid"], sort_keys=True),
                                   float(constraints["diffraction_angle_deg"]))
        if g.period_nm >= limit * (1 - constraints.get("diffraction_margin", 0.0)):
            errors.append("propagating_higher_order")
    return errors


def perturbations(g: Geometry, standardized: np.ndarray,
                  tolerance: Mapping[str, Any]) -> list[Geometry]:
    """Apply supplied random draws; period/height relative, duties absolute."""
    scales = np.array([g.period_nm * tolerance["period_relative"],
                       g.height_nm * tolerance["height_relative"],
                       tolerance["top_absolute"], tolerance["bottom_absolute"]])
    return [Geometry.from_array(v) for v in g.array() + np.asarray(standardized) * scales]


def tolerance_draws(count: int, seed: int, tolerance: Mapping[str, Any]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if tolerance["distribution"] == "uniform":
        return rng.uniform(-1, 1, (count, 4))
    if tolerance["distribution"] == "gaussian":
        # Scales are standard deviations; unbounded draws are NOT clipped.
        return rng.normal(size=(count, 4))
    raise ValueError("Tolerance distribution must be uniform or gaussian")


def robust_feasible(g: Geometry, config: Mapping[str, Any]) -> bool:
    if feasibility(g, config, "design"):
        return False
    tolerance = config["tolerance"]
    if tolerance["distribution"] != "uniform":
        raise ValueError("Robust search uses bounded uniform tolerances; gaussian is validation-only")
    corners = np.array(list(product([-1.0, 1.0], repeat=4)))
    return all(not feasibility(v, config, "cover") for v in perturbations(g, corners, tolerance))


def feature_matrix(geometries: list[Geometry], grid: OpticalGrid) -> np.ndarray:
    """Geometry-major/state-minor order; raw lambda/theta retained; TE/TM one-hot."""
    return np.asarray([list(g.array()) + [w, a, float(p == "TE"), float(p == "TM")]
                       for g in geometries for w, a, p in grid.states()], dtype=np.float64)
