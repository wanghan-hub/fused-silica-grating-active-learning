"""Reproducible tables/figures. Final test evaluation is explicit and offline."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json

import numpy as np
import pandas as pd

from optics import save_json


def bootstrap_interval(values: np.ndarray, samples: int, confidence: float,
                       seed: int, statistic: str = "mean") -> tuple[float, float]:
    """Resample supplied independent units (geometries/runs/MC draws, never pixels)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all() or samples < 2 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap inputs or fewer than two independent units")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        draw = rng.choice(values, size=len(values), replace=True)
        estimates.append(float(np.sqrt(draw.mean())) if statistic == "rmse" else float(draw.mean()))
    alpha = (1 - confidence) / 2
    low, high = np.quantile(estimates, [alpha, 1 - alpha])
    return float(low), float(high)


def prediction_metrics(frame: pd.DataFrame, config: Mapping[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = frame.copy()
    frame["absolute_error"] = abs(frame.R_pred - frame.R)
    frame["squared_error"] = (frame.R_pred - frame.R) ** 2
    grouped = frame.groupby("geometry_id").agg(
        MAE=("absolute_error", "mean"), MSE=("squared_error", "mean"),
        MaxAE=("absolute_error", "max"), point_coverage=("covered", "mean"),
        geometry_covered=("covered", "all"))
    grouped["RMSE"] = np.sqrt(grouped.MSE)
    report = config["report"]
    result = {"MAE": float(frame.absolute_error.mean()), "RMSE": float(np.sqrt(frame.squared_error.mean())),
              "MaxAE": float(frame.absolute_error.max()), "geometries": len(grouped),
              "point_coverage": float(frame.covered.mean()),
              "simultaneous_geometry_coverage": float(grouped.geometry_covered.mean()),
              "mean_interval_width": float((frame.R_upper - frame.R_lower).mean()),
              "mean_conservation_error": float(abs(frame.R_pred + frame.T_pred - 1).mean())}
    if len(grouped) >= 2:
        result["geometry_bootstrap_MAE_CI"] = bootstrap_interval(grouped.MAE.values,
            report["bootstrap_samples"], report["confidence"], config["seed"])
        result["geometry_bootstrap_RMSE_CI"] = bootstrap_interval(grouped.MSE.values,
            report["bootstrap_samples"], report["confidence"], config["seed"], "rmse")
        result["geometry_bootstrap_coverage_CI"] = bootstrap_interval(grouped.geometry_covered.values.astype(float),
            report["bootstrap_samples"], report["confidence"], config["seed"])
    return result, grouped.reset_index()


def _save_figure(fig: Any, path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def run_report(config: Mapping[str, Any], evaluate_tests: bool = False) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(config["_root"])
    output = root / "reports"
    output.mkdir(parents=True, exist_ok=True)
    dpi = config["report"]["dpi"]
    metrics = []
    model_paths = sorted((root / "models").glob("*/ensemble.json"))
    model_paths += sorted((root / "active_learning").glob("*/round_*/model/ensemble.json"))
    if evaluate_tests:
        if not model_paths:
            raise RuntimeError("No model checkpoints available for final test evaluation")
        from dataset import audit_dataset, load_split, load_confirmation_split
        from surrogate import Ensemble, prediction_table
        audit_dataset(config)
        use_confirmation = config.get("confirmation_tests", {}).get("enabled", False)
        read_test = load_confirmation_split if use_confirmation else load_split
        tests = {role: read_test(config, role) for role in ("iid", "ood")}
        for metadata_path in model_paths:
            model = Ensemble.load(metadata_path.parent, config["model"]["device"],
                                  config["model"]["predict_batch_size"])
            for role, frame in tests.items():
                if set(frame.geometry_id) & set(model.metadata["train_geometry_ids"]):
                    raise ValueError("Model contains a final test geometry")
                predictions = prediction_table(model, frame)
                name = str(metadata_path.parent.relative_to(root)).replace("/", "_").replace("\\", "_") + "_" + role
                predictions.to_csv(output / f"{name}_predictions.csv", index=False)
                result, geometry_errors = prediction_metrics(predictions, config)
                geometry_errors.to_csv(output / f"{name}_geometry_errors.csv", index=False)
                metrics.append({"model": name, "role": role,
                                "evaluation_set": "confirmation" if use_confirmation else "original", **result})
                fig, ax = plt.subplots(figsize=(5, 4))
                step = max(1, len(predictions) // config["report"]["scatter_max_points"])
                selected = predictions.iloc[::step]
                ax.scatter(selected.R, selected.R_pred, s=4, alpha=0.25)
                limits = [float(min(selected.R.min(), selected.R_pred.min())),
                          float(max(selected.R.max(), selected.R_pred.max()))]
                ax.plot(limits, limits, "k--", linewidth=1)
                ax.set(xlabel="RCWA reflectance", ylabel="Predicted reflectance", title=name)
                _save_figure(fig, output / f"{name}_parity", dpi)
                calibration = predictions.assign(abs_error=abs(predictions.R - predictions.R_pred))
                calibration["uncertainty_bin"] = pd.qcut(calibration.R_std, q=config["report"]["calibration_bins"],
                                                         duplicates="drop")
                bins = calibration.groupby("uncertainty_bin", observed=True).agg(
                    uncertainty=("R_std", "mean"), empirical_MAE=("abs_error", "mean"), count=("R", "size"))
                bins.to_csv(output / f"{name}_calibration_bins.csv")
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(bins.uncertainty, bins.empirical_MAE, "o-")
                ax.set(xlabel="Ensemble standard deviation", ylabel="Empirical MAE", title="Uncertainty reliability")
                _save_figure(fig, output / f"{name}_calibration", dpi)
        pd.DataFrame(metrics).to_csv(output / "test_metrics.csv", index=False)
        save_json(output / "test_metrics.json", metrics)
        save_json(output / "test_evaluation_notice.json", {
            "evaluation_set": "confirmation" if use_confirmation else "original",
            "tests_opened": True, "instruction": "Do not use these results to retune and still call these sets untouched tests."})
    curves = []
    for path in sorted((root / "active_learning").glob("*/learning_curve.csv")):
        curves.append(pd.read_csv(path))
    if curves:
        combined = pd.concat(curves, ignore_index=True)
        combined.to_csv(output / "al_learning_curves.csv", index=False)
        for field, label in (("validation_R_rmse", "Validation reflectance RMSE"),
                             ("best_true_mean_R", "Best RCWA-verified mean reflectance")):
            fig, ax = plt.subplots(figsize=(6, 4))
            for strategy, group in combined.groupby("strategy"):
                ax.plot(group.logical_state_solves, group[field], "o-", label=strategy)
            ax.set(xlabel="Cumulative logical RCWA state solves", ylabel=label)
            ax.legend()
            _save_figure(fig, output / field, dpi)
    histories = []
    for path in sorted((root / "optimization").glob("*/true_history.csv")):
        frame = pd.read_csv(path)
        frame["method_objective"] = path.parent.name
        frame["best_true_objective"] = frame.true_objective.cummin()
        histories.append(frame)
    if histories:
        combined = pd.concat(histories, ignore_index=True)
        combined.to_csv(output / "optimization_histories.csv", index=False)
        # Attribute exactly the training labels consumed by the selected model,
        # not every unrelated ablation/strategy performed during research.
        from optimize import resolve_model_directory
        selected_model = resolve_model_directory(config)
        selected_training = selected_model.parent / "training.parquet"
        if not selected_training.exists():
            selected_training = root / "data" / "train_initial.parquet"
        label_overhead = None
        shared_evaluation_cost = 0
        for role in ("iid", "ood"):
            path = root / "data" / f"{role}.parquet"
            if path.exists():
                shared_evaluation_cost += int((pd.read_parquet(path).attempt + 1).sum())
        if selected_training.exists():
            consumed = pd.read_parquet(selected_training)
            model_metadata = selected_model / "ensemble.json"
            expected_ids = set(json.loads(model_metadata.read_text())["train_geometry_ids"]) if model_metadata.exists() else set()
            if set(consumed.geometry_id) == expected_ids:
                label_overhead = int((consumed.attempt + 1).sum())
                for role in ("validation", "calibration"):
                    path = root / "data" / f"{role}.parquet"
                    if path.exists():
                        label_overhead += int((pd.read_parquet(path).attempt + 1).sum())
        combined["training_validation_calibration_state_solves"] = [
            (label_overhead if str(name).startswith("surrogate_") else 0)
            for name in combined.method_objective]
        combined["end_to_end_state_solves_before_final_validation"] = (
            combined.logical_state_solves + combined.training_validation_calibration_state_solves)
        combined["shared_final_test_state_solves_excluded"] = shared_evaluation_cost
        combined.to_csv(output / "end_to_end_design_costs.csv", index=False)
        for kind, group in combined.groupby("objective_kind"):
            fig, ax = plt.subplots(figsize=(6, 4))
            for method, curve in group.groupby("method_objective"):
                ax.step(curve.logical_state_solves, curve.best_true_objective, where="post", label=method)
            ax.set(xlabel="Search RCWA state solves (training excluded)", ylabel=f"Best verified {kind} objective")
            ax.legend()
            _save_figure(fig, output / f"optimization_{kind}", dpi)
            fig, ax = plt.subplots(figsize=(6, 4))
            for method, curve in group.groupby("method_objective"):
                valid = curve.dropna(subset=["end_to_end_state_solves_before_final_validation"])
                ax.step(valid.end_to_end_state_solves_before_final_validation,
                        valid.best_true_objective, where="post", label=method)
            ax.set(xlabel="RCWA states: model data + design queries (final verification excluded)",
                   ylabel=f"Best verified {kind} objective")
            ax.legend()
            _save_figure(fig, output / f"end_to_end_optimization_{kind}", dpi)
    spectrum_path = root / "final_validation" / "nominal_spectra.parquet"
    if spectrum_path.exists():
        frame = pd.read_parquet(spectrum_path)
        for gid, group in frame.groupby("geometry_id"):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for (angle, pol), values in group.groupby(["angle_deg", "polarization"]):
                axes[0].plot(values.wavelength_nm, values.R, label=f"{pol} {angle:g} deg", alpha=0.7)
            for (wave, pol), values in group.groupby(["wavelength_nm", "polarization"]):
                if wave in config["report"]["angular_wavelengths_nm"]:
                    axes[1].plot(values.angle_deg, values.R, label=f"{pol} {wave:g} nm")
            axes[0].set(xlabel="Wavelength (nm)", ylabel="Reflectance", title=gid)
            axes[1].set(xlabel="Incidence angle (deg)", ylabel="Reflectance")
            axes[1].legend(fontsize=7)
            _save_figure(fig, output / f"spectrum_angles_{gid}", dpi)
    manufacturing = root / "final_validation" / "manufacturing_samples.csv"
    if manufacturing.exists():
        frame = pd.read_csv(manufacturing)
        fig, ax = plt.subplots(figsize=(7, 4))
        for family, group in frame.groupby("family_id"):
            valid = group.loc[group.feasible, "mean_R"].dropna()
            if len(valid):
                ax.hist(valid, bins=config["report"]["histogram_bins"], alpha=0.4, label=family[:8])
        ax.set(xlabel="RCWA mean reflectance under perturbation", ylabel="Realizations")
        ax.legend()
        _save_figure(fig, output / "manufacturing_distributions", dpi)
    designs_path = root / "final_validation" / "design_summary.csv"
    if designs_path.exists():
        designs = pd.read_csv(designs_path)
        designs.to_csv(output / "final_design_comparison.csv", index=False)
        fig, ax = plt.subplots(figsize=(max(6, len(designs) * 0.9), 4))
        positions = np.arange(len(designs))
        ax.bar(positions - 0.18, designs.mean_R, width=0.36, label="Nominal RCWA")
        if "robust_mean_R" in designs:
            ax.bar(positions + 0.18, designs.robust_mean_R, width=0.36,
                   yerr=designs.robust_std_R, label="Perturbed mean +/- std")
        ax.set_xticks(positions, [f"{r.method}\n{r.objective_kind}" for r in designs.itertuples()])
        ax.set(ylabel="Mean reflectance", title="Final physical design comparison")
        ax.legend()
        _save_figure(fig, output / "final_design_comparison", dpi)
    for scope in ("validation", "final_validation"):
        path = root / scope / "comsol_comparison.csv"
        if path.exists():
            frame = pd.read_csv(path)
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            for ax, key in zip(axes, ("R", "T")):
                ax.scatter(frame[key], frame[key + "_comsol"], s=12)
                lo, hi = min(frame[key].min(), frame[key + "_comsol"].min()), max(frame[key].max(), frame[key + "_comsol"].max())
                ax.plot([lo, hi], [lo, hi], "k--")
                ax.set(xlabel=f"RCWA {key}", ylabel=f"COMSOL {key}")
            _save_figure(fig, output / f"{scope}_comsol", dpi)
    convergence = root / "validation" / "convergence.csv"
    if convergence.exists():
        frame = pd.read_csv(convergence)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, axis in zip(axes, ("harmonics", "nz", "nx")):
            for _, group in frame[frame.axis == axis].groupby("geometry_id"):
                ax.plot(group.value, group.max_error, "o-", alpha=0.5)
            ax.set(xlabel=axis, ylabel="Max absolute difference from reference")
        _save_figure(fig, output / "rcwa_convergence", dpi)
    costs = []
    for path in sorted(root.rglob("*.json")):
        if "reports" in path.parts or "shards" in path.parts:
            continue
        content = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(content, dict):
            for key in ("wall_time_s", "training_time_s", "logical_state_solves", "solver_elapsed_s"):
                if key in content:
                    costs.append({"source": str(path.relative_to(root)), "metric": key, "value": content[key]})
    pd.DataFrame(costs, columns=["source", "metric", "value"]).to_csv(output / "cost_ledger.csv", index=False)
    result = {"test_evaluation_requested": evaluate_tests, "test_model_results": len(metrics),
              "png_count": len(list(output.glob("*.png"))),
              "cost_note": "Nested/overlapping ledger entries must not be summed blindly; search plots exclude training."}
    save_json(output / "report_summary.json", result)
    return result


def aggregate_repeats(experiment_root: Path) -> None:
    """Summarize independent run variation, keeping ensemble members out of run counts."""
    frames = [pd.read_csv(p) for p in experiment_root.glob("repeat_*/reports/al_learning_curves.csv")]
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    summary = combined.groupby(["strategy", "round"]).agg(
        independent_runs=("repeat_seed", "nunique"), mean_validation_rmse=("validation_R_rmse", "mean"),
        std_validation_rmse=("validation_R_rmse", "std"), mean_best_true_R=("best_true_mean_R", "mean"),
        std_best_true_R=("best_true_mean_R", "std"))
    summary.to_csv(experiment_root / "independent_run_summary.csv")
