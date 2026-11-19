"""Compact manuscript Figures 2--5, using existing raster-v2 result CSVs.

Run from the repository root:
    python tools/plot_paper_figures_pier.py
    python tools/plot_paper_figures_pier.py --figure5-angular --formats png pdf svg

Requires numpy, pandas, matplotlib; no PyTorch or simulation is required.
Default canvas width: 110 mm. Outputs: RESULTS/paper_figures_pier.
Figure 2: three convergence sweeps and COMSOL residuals (2 x 2).
Figure 3: learning curves and combined IID/OOD parity (all states).
Figure 4: separate nominal/robust objective histories including model-data cost.
Figure 5: TE/TM spectra and manufacturing ECDF; optional angular-response panel.
All inputs and numerical validation follow the original plot_paper_figures.py.
Input files are read-only. The original plotting script is not required.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from hashlib import sha256
import importlib.util
import json
import logging
from pathlib import Path
import sys
from typing import Any, Iterable

try:
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, ScalarFormatter
except ImportError as exc:
    raise SystemExit("Missing plotting dependency. Run: python -m pip install numpy pandas matplotlib") from exc


LOG = logging.getLogger("paper_figures")
COLORS = {"random": "#777777", "uncertainty": "#D55E00", "task_diverse": "#0072B2",
          "direct": "#009E73", "bo": "#AA4499", "surrogate": "#0072B2",
          "nominal": "#0072B2", "robust": "#D55E00", "flat": "#666666"}
STRATEGIES = {"random": "Random", "uncertainty": "Uncertainty", "task_diverse": "Task-diverse"}
METHODS = {"random": "Sobol search", "direct": "Direct RCWA (DE)",
           "bo": "GP-based BO", "surrogate": "Surrogate-assisted"}
STATE_KEYS = ["geometry_id", "wavelength_nm", "angle_deg", "polarization"]
MM = 1 / 25.4


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def numeric(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    """Reject missing/nonfinite values rather than dropping or clipping rows."""
    for col in columns:
        require(col in frame, f"{context}: missing column {col}")
        require(np.isfinite(frame[col].to_numpy(dtype=float)).all(),
                f"{context}: nonfinite {col}; check the upstream results")


def truth(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.lower().isin(["true", "1"]).to_numpy()


def unique_states(frame: pd.DataFrame, context: str) -> None:
    require(not frame.duplicated(STATE_KEYS).any(), f"{context}: duplicate optical states")


def grid_size(grid: dict[str, Any]) -> int:
    def size(value: Any) -> int:
        if isinstance(value, dict):
            intervals = (value["stop"] - value["start"]) / value["step"]
            require(np.isclose(intervals, round(intervals)), "Grid endpoint is not reachable")
            return int(round(intervals)) + 1
        return len(value)
    return size(grid["wavelengths_nm"]) * size(grid["angles_deg"]) * len(grid["polarizations"])


def style() -> None:
    """Typography specified at final 110-mm physical width."""
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
        "axes.linewidth": .6, "lines.linewidth": 1, "lines.markersize": 2.5,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.labelpad": 2.5, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.handlelength": 1.5,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.unicode_minus": True, "savefig.facecolor": "white"})


def panel(ax: Any, letter: str, title: str) -> None:
    ax.set_title(f"({letter}) {title}", loc="left", pad=5, fontweight="bold")


def log_percent_axis(ax: Any) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 3, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


@dataclass
class PlotContext:
    project: Path
    root: Path
    output: Path
    formats: tuple[str, ...]
    dpi: int
    config: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    captions: dict[int, str] = field(default_factory=dict)
    products: list[str] = field(default_factory=list)
    model: Path | None = None
    figure5_angular: bool = False

    def track(self, path: Path) -> None:
        path = path.resolve()
        key = str(path)
        if key not in self.sources:
            digest = sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            self.sources[key] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}

    def read_json(self, relative: str | Path) -> dict[str, Any]:
        path = self.root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required result metadata not found: {path}")
        self.track(path)
        return json.loads(path.read_text(encoding="utf-8-sig"))

    def read_csv(self, relative: str | Path, columns: list[str]) -> pd.DataFrame:
        path = self.root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required result CSV not found: {path}")
        self.track(path)
        # usecols prevents the large per-order JSON columns from entering memory.
        data = pd.read_csv(path, usecols=columns, float_precision="round_trip")
        require(not data.empty, f"Empty input: {path}")
        return data

    def table(self, name: str, frame: pd.DataFrame) -> None:
        directory = self.output / "plot_data"
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_csv(directory / name, index=False)

    def save(self, fig: Any, name: str) -> None:
        # Fixed physical canvas; do not crop with bbox_inches='tight'.
        fig.canvas.draw()
        self.metrics.setdefault("canvas", {})[name] = {
            "width_mm": float(fig.get_figwidth() * 25.4),
            "height_mm": float(fig.get_figheight() * 25.4)}
        require(fig.get_figwidth() * 25.4 <= 110.0001, "Figure exceeds 110 mm")
        for fmt in self.formats:
            path = self.output / f"{name}.{fmt}"
            fig.savefig(path, dpi=self.dpi, format=fmt)
            self.products.append(path.name)
            LOG.info("Saved %s", path)
        plt.close(fig)

    @property
    def model_prefix(self) -> str:
        require(self.model is not None, "No selected model directory")
        return "_".join(self.model.relative_to(self.root).parts)


def figure2(ctx: PlotContext) -> None:
    """Three grouped panels: three convergence sweeps, R/T parity, residuals."""
    conv = ctx.read_csv("validation/convergence.csv",
                       ["geometry_id", "axis", "value", "max_error", "qc_ok"])
    numeric(conv, ["value", "max_error"], "Convergence")
    require(truth(conv.qc_ok).all(), "Convergence contains failed solver QC")
    require((conv.max_error >= 0).all(), "Negative convergence error")
    cmp = ctx.read_csv("final_validation/comsol_comparison.csv",
        ["condition_id", *STATE_KEYS, "R", "T", "R_comsol", "T_comsol", "status"])
    numeric(cmp, ["R", "T", "R_comsol", "T_comsol"], "COMSOL comparison")
    unique_states(cmp, "COMSOL comparison")
    require(cmp.condition_id.is_unique and cmp.status.eq("ok").all(), "Invalid COMSOL matching/QC")
    require(set(cmp.polarization) == {"TE", "TM"}, "COMSOL comparison needs TE and TM")
    cmp = cmp.sort_values(STATE_KEYS).reset_index(drop=True)
    cmp["condition_index"] = np.arange(1, len(cmp) + 1)
    cmp["delta_R"] = cmp.R_comsol - cmp.R
    cmp["delta_T"] = cmp.T_comsol - cmp["T"]
    tol = float(ctx.config["comsol"]["absolute_tolerance"])
    ct = float(ctx.config["validation"]["convergence_tolerance"])
    stats = conv.groupby(["axis", "value"], as_index=False).agg(
        maximum_error=("max_error", "max"), median_error=("max_error", "median"),
        geometries=("geometry_id", "nunique"))
    ctx.table("figure2_convergence.csv", stats)
    ctx.table("figure2_comsol.csv", cmp)
    fig, axes = plt.subplots(2, 2, figsize=(110 * MM, 98 * MM))
    fig.subplots_adjust(left=.14, right=.975, bottom=.105, top=.88, hspace=.65, wspace=.43)
    specs = [("harmonics", r"Fourier order $M$"), ("nz", r"Layers $N_z$"),
             ("nx", r"Lateral samples $N_x$")]
    used = stats[stats.axis.isin([s[0] for s in specs])]
    require((used.maximum_error > 0).all() and (used.median_error > 0).all(),
            "Log convergence plots require positive errors; no artificial floor is applied")
    limits = (min(used.median_error.min() * .65, ct / 8), max(used.maximum_error.max() * 1.4, ct * 1.4))
    for index, (axis, label) in enumerate(specs):
        ax = axes.flat[index]
        sub = stats[stats.axis.eq(axis)].sort_values("value")
        require(not sub.empty, f"Missing convergence sweep: {axis}")
        x = np.arange(len(sub))
        ax.semilogy(x, sub.maximum_error, "o-", color=COLORS["surrogate"], label="Maximum")
        ax.semilogy(x, sub.median_error, "s--", color=".5", label="Median")
        ax.axhline(ct, color=".25", ls=":", lw=.8, label="Tolerance")
        ax.set_xticks(x, [str(int(v)) for v in sub.value])
        ax.set(xlabel=label, ylabel="Absolute power error", ylim=limits)
        ax.grid(axis="y", which="major", color=".9", lw=.5)
        panel(ax, "abc"[index], ["Fourier convergence", "Layer convergence", "Lateral convergence"][index])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, 1), ncol=3,
               columnspacing=1.2)
    ax = axes[1, 1]
    for pol, color in [("TE", COLORS["nominal"]), ("TM", COLORS["robust"])]:
        d = cmp[cmp.polarization.eq(pol)]
        for power, marker in [("R", "o"), ("T", "x")]:
            ax.scatter(d.condition_index, d[f"delta_{power}"] / 1e-4, s=5,
                       marker=marker, color=color, linewidths=.45, alpha=.65, label=f"{power}, {pol}")
    for sign in [-1, 1]:
        ax.axhline(sign * tol / 1e-4, color=".35", ls="--", lw=.7)
    ax.axhline(0, color=".7", lw=.5)
    limit = max(tol, float(cmp[["delta_R", "delta_T"]].abs().to_numpy().max())) / 1e-4 * 1.4
    ax.set(ylim=(-limit, limit), xlabel="Matched-condition index",
           ylabel=r"COMSOL $-$ RCWA ($10^{-4}$)")
    ax.legend(loc="upper center", ncol=2, fontsize=7.5, columnspacing=.6, handletextpad=.2,
              frameon=True, facecolor="white", edgecolor="none", framealpha=1, borderpad=.15)
    panel(ax, "d", "Cross-solver residuals")
    ctx.metrics["figure2"] = {"matched_conditions": len(cmp), "max_absolute_R_T_difference":
        float(cmp[["delta_R", "delta_T"]].abs().to_numpy().max()), "rmse_R": float(np.sqrt(np.mean(cmp.delta_R ** 2))),
        "comsol_tolerance": tol, "qualification_geometry_count": int(conv.geometry_id.nunique())}
    ref = ctx.config["rcwa"]["reference"]
    ctx.captions[2] = (
        "Numerical convergence and cross-solver validation. (a-c) Maximum and median geometry-wise "
        "maximum absolute R/T errors for Fourier order, layer count, and lateral sampling sweeps, "
        f"relative to (M, Nz, Nx) = ({ref['harmonics']}, {ref['nz']}, {ref['nx']}); dotted lines mark the "
        f"convergence tolerance. (d) Signed RCWA-COMSOL differences for {len(cmp)} matched conditions; "
        "dashed lines mark acceptance limits. Conditions are ordered by geometry, wavelength, angle, and polarization.")
    ctx.save(fig, "Figure_2_Numerical_validation")


def figure3(ctx: PlotContext) -> None:
    """Validation learning curves and all-state confirmation parity; no re-inference."""
    notice = ctx.read_json("reports/test_evaluation_notice.json")
    require(notice.get("evaluation_set") == "confirmation" and notice.get("tests_opened") is True,
            "Figure 3 requires existing frozen confirmation-test predictions, not development tests")
    curves = ctx.read_csv("reports/al_learning_curves.csv",
                         ["strategy", "round", "training_geometries", "validation_R_rmse"])
    numeric(curves, ["round", "training_geometries", "validation_R_rmse"], "Learning curves")
    official = ctx.read_csv("reports/test_metrics.csv", ["model", "role", "evaluation_set", "RMSE", "geometries"])
    predictions: dict[str, pd.DataFrame] = {}
    rows = []
    for role in ["iid", "ood"]:
        name = f"{ctx.model_prefix}_{role}"
        d = ctx.read_csv(f"reports/{name}_predictions.csv", [*STATE_KEYS, "R", "R_pred", "role"])
        numeric(d, ["R", "R_pred"], f"{role.upper()} predictions")
        unique_states(d, role)
        require(d.role.eq(role).all(), f"Unexpected role in {name}")
        require(d.groupby("geometry_id").size().eq(grid_size(ctx.config["grid"])).all(), f"Incomplete states in {name}")
        rmse = float(np.sqrt(np.mean((d.R_pred - d.R) ** 2)))
        check = official[official.model.eq(name) & official.role.eq(role) & official.evaluation_set.eq("confirmation")]
        require(len(check) == 1 and np.isclose(rmse, check.iloc[0].RMSE, rtol=1e-8, atol=1e-12),
                f"Prediction CSV does not match the official confirmation RMSE: {name}")
        require(d.geometry_id.nunique() == check.iloc[0].geometries, f"Incomplete confirmation geometries: {name}")
        rows.append({"role": role, "states": len(d), "geometries": d.geometry_id.nunique(), "RMSE": rmse,
                     "MAE": float(np.mean(abs(d.R_pred - d.R))), "MaxAE": float(np.max(abs(d.R_pred - d.R)))})
        predictions[role] = d
    require(not set(predictions["iid"].geometry_id) & set(predictions["ood"].geometry_id), "IID/OOD geometry overlap")
    fig, axes = plt.subplots(1, 2, figsize=(110 * MM, 59 * MM))
    fig.subplots_adjust(left=.14, right=.975, bottom=.22, top=.80, wspace=.47)
    for strategy, label in STRATEGIES.items():
        d = curves[curves.strategy.eq(strategy)].sort_values("round")
        require(not d.empty and d["round"].is_unique, f"Missing or duplicate rounds: {strategy}")
        axes[0].plot(d.training_geometries, d.validation_R_rmse, marker="o", label=label, color=COLORS[strategy])
    axes[0].set(xlabel="Training geometries", ylabel="Validation R RMSE")
    axes[0].set_xticks(sorted(curves.training_geometries.unique()))
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    axes[0].set_box_aspect(1)
    axes[0].legend(loc="upper right", fontsize=7.7, labelspacing=.2, handlelength=1.1,
                   handletextpad=.3, borderaxespad=.2)
    panel(axes[0], "a", "Active learning")
    low = min(0.0, min(float(d[["R", "R_pred"]].min().min()) for d in predictions.values()) * 100)
    high = max(float(d[["R", "R_pred"]].max().max()) for d in predictions.values()) * 100 * 1.06
    ax = axes[1]
    # Plot every state with identical size and alpha. Draw OOD first so the dense IID
    # diagonal is visible; display order and both sample counts are recorded below.
    for role in ["ood", "iid"]:
        d = predictions[role]
        ax.scatter(d.R * 100, d.R_pred * 100, s=1, alpha=.22,
                   color=COLORS["robust"] if role == "ood" else COLORS["nominal"],
                   linewidths=0, rasterized=True)
    ax.plot([low, high], [low, high], "--", color=".25", lw=.7)
    ax.set(xlim=(low, high), ylim=(low, high), xlabel="RCWA R (%)", ylabel="Predicted R (%)")
    ax.set_aspect("equal", adjustable="box")
    for index, record in enumerate(rows):
        role = record["role"]
        ax.text(.04, .97 - index * .12, f"{role.upper()} RMSE {record['RMSE']:.2e}",
                color=COLORS["nominal"] if role == "iid" else COLORS["robust"],
                transform=ax.transAxes, ha="left", va="top", fontsize=7.8,
                bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none", "pad": .5})
    panel(ax, "b", "IID / OOD prediction")
    ctx.table("figure3_learning_curves.csv", curves)
    ctx.table("figure3_prediction_metrics.csv", pd.DataFrame(rows))
    ctx.metrics["figure3"] = {"selected_model": ctx.model_prefix, "confirmation": rows,
                              "all_states_drawn": True, "scatter_order": ["ood", "iid"], "RMSE_unit": "reflectance fraction"}
    ctx.captions[3] = (
        "Active learning and surrogate accuracy. (a) Validation RMSE versus training-geometry count. "
        f"(b) All saved predictions on fresh IID ({rows[0]['geometries']} geometries, blue) and OOD "
        f"({rows[1]['geometries']} geometries, orange) confirmation sets; the dashed line indicates equality. "
        "Reflectance axes use percentages; RMSE uses reflectance fractions. Both sets are evaluated with the frozen selected ensemble.")
    ctx.save(fig, "Figure_3_Learning_and_prediction")


def figure4(ctx: PlotContext) -> None:
    """Best independently solved objectives, including the surrogate data cost."""
    history = ctx.read_csv("reports/optimization_histories.csv", ["method_objective", "objective_kind",
        "logical_state_solves", "true_objective", "best_true_objective", "geometry_id"])
    numeric(history, ["logical_state_solves", "true_objective", "best_true_objective"], "Optimization history")
    metadata = ctx.read_json(ctx.model / "ensemble.json")
    train_count = len(set(metadata["train_geometry_ids"]))
    counts = ctx.config["dataset"]["counts"]
    overhead = (train_count + counts["validation"] + counts["calibration"]) * grid_size(ctx.config["grid"])
    fig, axes = plt.subplots(1, 2, figsize=(110 * MM, 65 * MM))
    fig.subplots_adjust(left=.13, right=.975, bottom=.32, top=.87, wspace=.26)
    plotted, summaries = [], []
    for ax, kind, letter in zip(axes, ["nominal", "robust"], "ab"):
        for method, label in METHODS.items():
            d = history[history.method_objective.eq(f"{method}_{kind}")].copy()
            require(not d.empty, f"Missing optimization history: {method}_{kind}")
            require(d.objective_kind.eq(kind).all(), "Mixed objective kinds")
            require((np.diff(d.logical_state_solves) > 0).all(), "History budgets must increase in saved evaluation order")
            best = np.minimum.accumulate(d.true_objective.to_numpy())
            require(np.allclose(best, d.best_true_objective, rtol=1e-9, atol=1e-12), "Inconsistent saved best-objective history")
            d["accounted_state_solves"] = d.logical_state_solves + (overhead if method == "surrogate" else 0)
            ax.step(d.accounted_state_solves / 1e5, best * 100, where="post", color=COLORS[method], label=label,
                    linewidth=1.5 if method == "surrogate" else 1.1)
            ax.plot(d.accounted_state_solves.iloc[-1] / 1e5, best[-1] * 100, "o", color=COLORS[method], ms=3)
            plotted.append(d)
            summaries.append({"method": method, "objective_kind": kind, "evaluations": len(d),
                "search_states": int(d.logical_state_solves.iloc[-1]), "model_data_states": overhead if method == "surrogate" else 0,
                "accounted_states": int(d.accounted_state_solves.iloc[-1]), "best_true_objective": float(best[-1])})
        ax.set_xlabel(r"RCWA states ($10^5$)")
        if kind == "nominal":
            ax.set_ylabel("Best verified objective (%)")
        ax.set_xlim(left=0)
        log_percent_axis(ax)
        panel(ax, letter, kind.capitalize())
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.53, .005), ncol=2, columnspacing=1.2, fontsize=8)
    ctx.table("figure4_accounted_histories.csv", pd.concat(plotted, ignore_index=True))
    ctx.table("figure4_budget_summary.csv", pd.DataFrame(summaries))
    ctx.metrics["figure4"] = {"surrogate_training_geometries": train_count, "shared_model_data_states": overhead,
        "routes": summaries, "budget_unit": "logical optical states, not wall time", "final_validation_excluded": True}
    ctx.captions[4] = (
        "RCWA-verified optimization performance versus accounted evaluation states. (a) Nominal mean reflectance. "
        "(b) Robust mean-plus-SD objective. Values use the search grid and a logarithmic vertical scale. "
        f"Surrogate curves include {overhead:,} training, validation, and calibration states per standalone comparison; "
        "the shared model cost is counted once when combining objectives. Curves end at actual consumption. "
        "Final verification is excluded. The nonadaptive baseline uses scrambled-Sobol sampling.")
    ctx.save(fig, "Figure_4_Optimization_budget")


def load_fresnel(ctx: PlotContext) -> Any:
    """Reuse the project's analytical optics without importing a model or solver."""
    path = ctx.project / "optics.py"
    if not path.is_file():
        raise FileNotFoundError(f"Figure 5 requires the project's optics.py: {path}")
    ctx.track(path)
    name = "_paper_figure_optics"
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.fresnel


def figure5(ctx: PlotContext) -> None:
    """Verified spectra and empirical manufacturing distributions, without jitter."""
    summary = ctx.read_csv("final_validation/design_summary.csv", ["geometry_id", "method", "objective_kind",
        "mean_R", "mc_total", "mc_feasible", "robust_mean_R", "robust_std_R"])
    spectra = ctx.read_csv("final_validation/nominal_spectra.csv", [*STATE_KEYS, "R", "T", "status"])
    samples = ctx.read_csv("final_validation/manufacturing_samples.csv", ["family_id", "sample", "feasible", "mean_R"])
    numeric(spectra, ["R", "T"], "Final spectra")
    unique_states(spectra, "Final spectra")
    require(spectra.status.eq("ok").all(), "Final spectra include states that failed QC")
    designs, selected, records = {}, {}, {}
    for kind in ["nominal", "robust"]:
        row = summary[summary.method.eq("surrogate") & summary.objective_kind.eq(kind)]
        require(len(row) == 1, f"Expected one surrogate {kind} design")
        record = row.iloc[0]
        designs[kind] = spectra[spectra.geometry_id.eq(record.geometry_id)].sort_values(STATE_KEYS).copy()
        d = designs[kind]
        require(len(d) == grid_size(ctx.config["final_validation"]["grid"]), "Incomplete dense verification grid")
        require(np.isclose(d.R.mean(), record.mean_R, rtol=1e-9), "Dense-grid mean disagrees with design summary")
        m = samples[samples.family_id.eq(record.geometry_id)].sort_values("sample").copy()
        require(m["sample"].is_unique and len(m) == record.mc_total, "Missing or duplicate manufacturing draws")
        require(truth(m.feasible).all() and record.mc_total == record.mc_feasible,
                "This figure expects all manufacturing draws feasible; inspect failed draws before plotting")
        numeric(m, ["mean_R"], "Manufacturing validation")
        require(np.isclose(m.mean_R.mean(), record.robust_mean_R, rtol=1e-9) and
                np.isclose(m.mean_R.std(ddof=1), record.robust_std_R, rtol=1e-9), "Manufacturing statistics disagree")
        selected[kind], records[kind] = m, record.to_dict()
    keys = STATE_KEYS[1:]
    require(designs["nominal"][keys].reset_index(drop=True).equals(designs["robust"][keys].reset_index(drop=True)),
            "Nominal and robust verification grids differ")
    fresnel = load_fresnel(ctx)
    flat = designs["nominal"][keys].copy()
    rt = [fresnel(float(row.wavelength_nm), float(row.angle_deg), str(row.polarization))
          for row in flat.itertuples(index=False)]
    flat["R"], flat["T"] = np.asarray(rt).T
    designs["flat"] = flat
    require(all((d.R > 0).all() for d in designs.values()), "Nonpositive reflectance cannot be shown on this log scale")
    fig = plt.figure(figsize=(110 * MM, (108 if ctx.figure5_angular else 100) * MM))
    gs = fig.add_gridspec(2, 2, left=.14, right=.975, bottom=.11, top=.82,
                          hspace=.63, wspace=.43, height_ratios=[1, .85])
    spectrum_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    angle_ax = fig.add_subplot(gs[1, 0]) if ctx.figure5_angular else None
    ecdf_ax = fig.add_subplot(gs[1, 1] if ctx.figure5_angular else gs[1, :])
    for ax, pol, letter in zip(spectrum_axes, ["TE", "TM"], "ab"):
        for kind in ["flat", "nominal", "robust"]:
            for angle, linestyle in [(0, "-"), (40, "--")]:
                d = designs[kind]
                line = d[d.polarization.eq(pol) & d.angle_deg.eq(angle)].sort_values("wavelength_nm")
                require(not line.empty, f"Missing {pol} spectrum at {angle} degrees")
                ax.plot(line.wavelength_nm, line.R * 100, color=COLORS[kind], linestyle=linestyle,
                        linewidth=1.5 if kind == "nominal" else 1.05,
                        marker="o" if kind == "robust" else None, markevery=10, ms=2.8, mfc="white", mew=.65)
        ax.set(xlabel="Wavelength (nm)", ylabel="Reflectance (%)", xlim=(flat.wavelength_nm.min(), flat.wavelength_nm.max()))
        log_percent_axis(ax)
        ax.tick_params(axis="y", labelsize=6)
        panel(ax, letter, f"{pol} spectra")
    kind_handles = [Line2D([], [], color=COLORS[k], lw=1.5, label=label)
                    for k, label in [("flat", "Flat silica"), ("nominal", "Nominal design"), ("robust", "Robust design")]]
    fig.legend(handles=kind_handles, loc="upper center", bbox_to_anchor=(.53, .995), ncol=3, fontsize=8, columnspacing=.9)
    fig.legend(handles=[Line2D([], [], color=".25", ls=ls, label=f"{angle}° incidence")
                        for angle, ls in [(0, "-"), (40, "--")]], loc="upper center",
               bbox_to_anchor=(.53, .945), ncol=2, fontsize=8)
    angle_rows = []
    for kind, d in designs.items():
        grouped = d.groupby(["angle_deg", "polarization"], as_index=False).R.mean()
        grouped["design"] = kind
        angle_rows.append(grouped)
        if angle_ax is not None:
            for pol, ls in [("TE", "-"), ("TM", "--")]:
                line = grouped[grouped.polarization.eq(pol)]
                angle_ax.plot(line.angle_deg, line.R * 100, color=COLORS[kind], ls=ls,
                    lw=1.2 if kind == "nominal" else .9, marker="o" if kind == "robust" else None,
                    ms=2, mfc="white", mew=.5)
    if angle_ax is not None:
        angle_ax.set(xlabel="Incidence angle (°)", ylabel="Mean R (%)")
        log_percent_axis(angle_ax)
        angle_ax.legend(handles=[Line2D([], [], color=".25", ls=ls, label=pol)
                                for pol, ls in [("TE", "-"), ("TM", "--")]],
                        loc="center right", fontsize=8)
        panel(angle_ax, "c", "Angular response")
    for kind, m in selected.items():
        values = np.sort(m.mean_R.to_numpy()) * 100
        probabilities = np.arange(1, len(values) + 1) / len(values)
        ecdf_ax.step(np.r_[values[0], values], np.r_[0, probabilities], where="post", color=COLORS[kind],
            ls="-" if kind == "nominal" else "--", label=f"{kind.capitalize()} (n = {len(values)})",
            lw=1.7 if kind == "nominal" else 1.1)
    ecdf_ax.set(xlabel="Perturbed-design mean R (%)", ylabel="Cumulative probability", ylim=(0, 1.02))
    ecdf_ax.legend(loc="lower right", fontsize=8.8)
    panel(ecdf_ax, "d" if ctx.figure5_angular else "c", "Manufacturing variation")
    ctx.table("figure5_spectra.csv", pd.concat([d.assign(design=k) for k, d in designs.items()], ignore_index=True))
    ctx.table("figure5_angle_response.csv", pd.concat(angle_rows, ignore_index=True))
    ctx.table("figure5_manufacturing_samples.csv", pd.concat([d.assign(design=k) for k, d in selected.items()], ignore_index=True))
    ctx.metrics["figure5"] = {"designs": records, "baseline_mean_R": float(flat.R.mean()),
        "baseline_source": "optics.fresnel", "curves_shifted_or_smoothed": False}
    ctx.metrics["figure5"]["angular_panel"] = ctx.figure5_angular
    ctx.captions[5] = (
        "Optical response and manufacturing sensitivity. (a, b) TE/TM spectra at 0° (solid) and 40° "
        "(dashed), with a planar-silica Fresnel baseline. Colors identify designs; open markers distinguish "
        "the robust design. Reflectance uses logarithmic axes. "
        + ("(c) Wavelength-averaged angular response: solid TE and dashed TM. " if ctx.figure5_angular else "")
        + ("(d) " if ctx.figure5_angular else "(c) ")
        + "Empirical cumulative distributions of grid-averaged reflectance under independent final manufacturing "
        "perturbations, using matched standardized draws across designs.")
    ctx.save(fig, "Figure_5_Final_performance")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Directory containing optics.py (default: repository root)")
    parser.add_argument("--results-dir", type=Path, help="Completed repeat directory; relative paths use project root")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: RESULTS/paper_figures_pier)")
    parser.add_argument("--figures", type=int, nargs="+", choices=[2, 3, 4, 5], default=[2, 3, 4, 5])
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=600, help="PNG and dense scatter rasterization resolution")
    parser.add_argument("--figure5-angular", action="store_true",
                        help="Include the angular-response panel (four-panel Figure 5)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
    project = args.project_root.resolve()
    root = (project / (args.results_dir or Path("outputs/silica_grating_v2_trainingfix/repeat_000"))).resolve()
    output = ((project / args.output_dir) if args.output_dir else root / "paper_figures_pier").resolve()
    try:
        require(args.dpi >= 100, "Use --dpi 100 or higher; 600 is recommended for submission")
        require(root.is_dir(), f"Result directory not found: {root}; specify --results-dir")
        ctx = PlotContext(project, root, output, tuple(dict.fromkeys(args.formats)), args.dpi)
        ctx.figure5_angular = args.figure5_angular
        ctx.config = ctx.read_json("logs/optimize_invocation.json")["config"]
        configured = Path(ctx.config["optimization"]["model_directory"])
        require(not configured.is_absolute(), "Expected a relocatable model_directory relative to the repeat directory")
        ctx.model = (root / configured).resolve()
        output.mkdir(parents=True, exist_ok=True)
        style()
        builders = {2: figure2, 3: figure3, 4: figure4, 5: figure5}
        for number in dict.fromkeys(args.figures):
            LOG.info("Building Figure %d from completed results", number)
            builders[number](ctx)
        (output / "figure_metrics.json").write_text(json.dumps(ctx.metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        captions = "# Figure captions\n\n" + "\n\n".join(f"**Figure {n}.** {text}" for n, text in ctx.captions.items())
        (output / "figure_captions.md").write_text(captions + "\n", encoding="utf-8")
        ctx.track(Path(__file__))
        manifest = {"project": str(project), "results": str(root), "figures": args.figures, "products": ctx.products,
                    "sources": ctx.sources, "python": sys.version.split()[0], "numpy": np.__version__,
                    "pandas": pd.__version__, "matplotlib": matplotlib.__version__, "dpi": args.dpi, "canvas_width_mm": 110, "figure5_angular": args.figure5_angular}
        (output / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        LOG.info("Finished. Figures, source tables, metrics and captions: %s", output)
        return 0
    except Exception:
        LOG.exception("Plotting failed. Original experiment files were not modified; no data were fabricated.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
