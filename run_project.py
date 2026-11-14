"""Unified CLI; --stage check and --help work without PyTorch or scientific packages."""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
import logging
import os
from pathlib import Path
import platform
import re
import sys
from datetime import datetime, timezone

FILES = ["optics.py", "rcwa_solver.py", "validate_physics.py", "dataset.py", "surrogate.py",
         "active_learning.py", "optimize.py", "validate_designs.py", "report_results.py", "run_project.py",
         "revision_cli.py", "revision_inputs.py", "revision_learning.py", "revision_benchmark.py"]


def code_check(directory: Path) -> dict:
    """Compile in memory; inspect local imports without importing optional packages."""
    errors = []
    defined = {}
    trees = {}
    for name in FILES:
        path = directory / name
        if not path.exists():
            errors.append(f"Missing {name}")
            continue
        try:
            source = path.read_text(encoding="utf-8")
            trees[name[:-3]] = ast.parse(source, filename=str(path))
            compile(source, str(path), "exec")
            defined[name[:-3]] = {node.name for node in ast.walk(trees[name[:-3]])
                                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
            for node in ast.walk(trees[name[:-3]]):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    defined[name[:-3]].update(t.id for t in targets if isinstance(t, ast.Name))
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    defined[name[:-3]].update(a.asname or a.name.split(".")[0] for a in node.names)
        except (SyntaxError, OSError) as exc:
            errors.append(f"{name}: {exc}")
    for module, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in defined:
                for alias in node.names:
                    if alias.name not in defined[node.module]:
                        errors.append(f"{module}: unresolved local import {node.module}.{alias.name}")
            if module != "rcwa_solver" and isinstance(node, (ast.Import, ast.ImportFrom)):
                external = node.module if isinstance(node, ast.ImportFrom) else " ".join(a.name for a in node.names)
                if external and "grcwa" in external:
                    errors.append(f"RCWA API leaked into {module}")
    actual = {p.name for p in directory.glob("*.py")}
    if actual != set(FILES):
        errors.append(f"Project file list mismatch: missing={set(FILES)-actual}, extra={actual-set(FILES)}")
    dependencies = {name: importlib.util.find_spec(name) is not None for name in
                    ("numpy", "scipy", "pandas", "pyarrow", "yaml", "grcwa", "threadpoolctl",
                     "torch", "sklearn", "matplotlib")}
    physical_checks = "not_run_numpy_unavailable"
    if not errors and dependencies["numpy"]:
        try:
            import numpy as np
            from optics import Geometry, OpticalGrid, fresnel, layer_duties, silica_index
            grid = OpticalGrid.from_config({"wavelengths_nm": {"start": 450, "stop": 850, "step": 10},
                                            "angles_deg": [0, 10, 20, 30, 40]})
            assert len(grid.states()) == 410
            for wave in grid.wavelengths_nm:
                for angle in grid.angles_deg:
                    for pol in grid.polarizations:
                        r, t = fresnel(wave, angle, pol)
                        assert abs(r + t - 1) < 1e-12
                assert np.allclose(fresnel(wave, 0, "TE"), fresnel(wave, 0, "TM"), atol=1e-12)
            assert 1.45 < float(silica_index(550)) < 1.47
            duties = layer_duties(Geometry(180, 300, 0.25, 0.75), 16)
            assert len(duties) == 16 and np.all(np.diff(duties) > 0)
            physical_checks = "passed_analytic_and_geometry_checks_not_RCWA"
        except Exception as exc:
            errors.append(f"Analytic check: {type(exc).__name__}: {exc}")
    result = {"passed": not errors, "python": platform.python_version(), "errors": errors,
              "dependencies_available": dependencies, "physical_checks": physical_checks,
              "not_executed": ["RCWA", "PyTorch training", "COMSOL", "full research pipeline"]}
    (directory / "code_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML to run stages. --stage check works without it.") from exc
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"experiment", "seed", "grid", "geometry", "rcwa", "validation", "dataset",
                "sampling", "model", "calibration", "active_learning", "optimization",
                "tolerance", "final_validation", "metrics", "report", "paths", "comsol"}
    if not isinstance(config, dict) or not required <= set(config):
        raise ValueError(f"Missing configuration sections: {required - set(config or {})}")
    expected = {"train_initial", "validation", "calibration", "iid", "ood"}
    if set(config["dataset"]["counts"]) != expected:
        raise ValueError("Dataset must define train_initial/validation/calibration/iid/ood")
    if not all(isinstance(v, int) and v > 0 for v in config["dataset"]["counts"].values()):
        raise ValueError("Dataset counts must be positive integers")
    from optics import OpticalGrid, domain_bounds
    OpticalGrid.from_config(config["grid"])
    for domain in ("design", "cover", "ood"):
        bounds = domain_bounds(config, domain)
        if bounds.shape != (4, 2) or (bounds[:, 1] <= bounds[:, 0]).any():
            raise ValueError(f"Invalid {domain} bounds")
    if config["model"]["primary_variant"] not in config["model"]["variants"]:
        raise ValueError("Primary variant must be included in model.variants")
    if "reuse" in config:
        candidates = []
        for name in config["reuse"]["source_candidates"]:
            candidate = Path(name)
            candidates.append((candidate if candidate.is_absolute() else path.resolve().parent/candidate).resolve())
        config["reuse"]["source_repeat_root"] = str(next((p for p in candidates if (p/"data/manifest.json").is_file()), candidates[0]))
    quality = config["model"].get("quality")
    if quality is not None and (not 0 < quality["max_baseline_rmse_ratio"] < 1 or quality["max_member_median_ratio"] < 1):
        raise ValueError("Invalid training-quality thresholds")
    return config


def dispatch(stage: str, config: dict, args: argparse.Namespace) -> dict:
    if stage == "prepare_v2":
        from dataset import prepare_v2
        return prepare_v2(config)
    if stage == "training_check":
        from surrogate import run_training_check
        return run_training_check(config)
    if stage == "confirmation_tests":
        from dataset import confirmation_tests
        return confirmation_tests(config)
    if stage == "validate":
        from validate_physics import run_validation
        return run_validation(config)
    if stage == "dataset":
        from dataset import run_dataset
        return run_dataset(config)
    if stage == "audit":
        from dataset import audit_dataset
        return audit_dataset(config)
    if stage == "train":
        from surrogate import run_training
        return run_training(config)
    if stage == "calibrate":
        from dataset import load_split
        from surrogate import Ensemble, calibrate
        result = {}
        for variant in config["model"]["variants"]:
            directory = Path(config["_root"]) / "models" / variant
            model = Ensemble.load(directory, config["model"]["device"], config["model"]["predict_batch_size"])
            result[variant] = calibrate(model, load_split(config, "calibration"), config, directory)
        return result
    if stage == "active_learning":
        from active_learning import run_active_learning
        return run_active_learning(config)
    if stage == "optimize":
        from optimize import run_optimization
        return run_optimization(config)
    if stage == "final_validation":
        progress = Path(config["_root"]) / "optimization" / "progress.json"
        if progress.exists() and not json.loads(progress.read_text(encoding="utf-8")).get("complete"):
            raise RuntimeError("Optimization is incomplete. Resume --stage optimize before final_validation")
        from validate_designs import run_final_validation
        return run_final_validation(config)
    if stage == "comsol":
        if args.comsol_results is None:
            raise ValueError("--comsol-results is required for --stage comsol")
        import pandas as pd
        from validate_physics import compare_comsol
        directory = Path(config["_root"]) / ("validation" if args.scope == "initial" else "final_validation")
        result = compare_comsol(directory / "comsol_request.csv", Path(args.comsol_results),
                                pd.read_csv(directory / "comsol_rcwa_reference.csv", float_precision="round_trip"), config, directory)
        from optics import save_json
        summary_path = directory / ("physics_summary.json" if args.scope == "initial" else "summary.json")
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["cross_solver_status"] = "passed" if result["passed"] else "failed"
            summary["complete"] = bool(summary.get("passed") and result["passed"])
            save_json(summary_path, summary)
        return result
    if stage == "report":
        from report_results import run_report
        return run_report(config, evaluate_tests=args.evaluate_tests)
    raise ValueError(f"Unknown stage {stage}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["check", "validate", "comsol", "dataset", "audit",
                        "prepare_v2", "training_check", "confirmation_tests",
                        "train", "calibrate", "active_learning", "optimize", "final_validation", "report"])
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--experiment-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", help="Override runtime.rcwa_workers (positive integer or auto); no physical changes")
    parser.add_argument("--stop-after-subtask", metavar="METHOD_OBJECTIVE",
                        help="Optimize only: stop cleanly after e.g. direct_nominal; omit when resuming all remaining tasks")
    repeats = parser.add_mutually_exclusive_group()
    repeats.add_argument("--repeat-index", type=int, default=0)
    repeats.add_argument("--all-repeats", action="store_true")
    parser.add_argument("--comsol-results", type=Path)
    parser.add_argument("--scope", choices=["initial", "final"], default="initial")
    parser.add_argument("--evaluate-tests", action="store_true",
                        help="Explicit offline final IID/OOD evaluation; do not retune on these results")
    args = parser.parse_args(argv)
    if args.stop_after_subtask and args.stage != "optimize":
        parser.error("--stop-after-subtask is only supported by --stage optimize")
    if args.stage == "check":
        result = code_check(Path(__file__).resolve().parent)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 1
    # Must be set before any stage imports torch/CUDA.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        base = load_config(args.config.resolve())
        if args.workers is not None:
            base.setdefault("runtime", {})["rcwa_workers"] = args.workers
        if args.stop_after_subtask is not None:
            base.setdefault("runtime", {})["stop_after_subtask"] = args.stop_after_subtask
        from dataset import ExecutionSettings
        execution = ExecutionSettings.from_config(base)
        identifier = args.experiment_id or base["experiment"]["id"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", identifier) or identifier in (".", ".."):
            raise ValueError("experiment_id must be a simple directory name")
        output = Path(base["paths"]["output_root"])
        if not output.is_absolute():
            output = args.config.resolve().parent / output
        experiment_root = output / identifier
        offsets = base["experiment"]["repeat_seed_offsets"]
        indices = list(range(len(offsets))) if args.all_repeats else [args.repeat_index]
        if any(i < 0 or i >= len(offsets) for i in indices):
            raise ValueError("repeat-index outside configured repeat_seed_offsets")
        if args.stage == "comsol" and len(indices) > 1:
            raise ValueError("Import each repeat's distinct COMSOL CSV with --repeat-index")
        from optics import save_json
        for index in indices:
            config = copy.deepcopy(base)
            config["seed"] = int((args.seed if args.seed is not None else base["seed"]) + offsets[index])
            root = experiment_root / f"repeat_{index:03d}"
            root.mkdir(parents=True, exist_ok=True)
            config["_root"] = str(root.resolve())
            log_path = root / "logs"
            log_path.mkdir(exist_ok=True)
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                                handlers=[logging.StreamHandler(), logging.FileHandler(log_path / f"{args.stage}.log", encoding="utf-8")],
                                force=True)
            run = {"stage": args.stage, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "repeat_index": index, "config": config, "argv": sys.argv if argv is None else argv}
            save_json(log_path / f"{args.stage}_invocation.json", run)
            logging.info("Running %s in %s", args.stage, root)
            logging.info("RCWA scheduling: %d CPU workers x %d BLAS thread(s), %d states/chunk; logical CPUs=%s",
                         execution.workers, execution.blas_threads, execution.chunk_size, os.cpu_count())
            result = dispatch(args.stage, config, args)
            save_json(log_path / f"{args.stage}_result.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
            if result.get("passed") is False:
                return 2
        if args.stage == "report" and args.all_repeats:
            from report_results import aggregate_repeats
            aggregate_repeats(experiment_root)
        return 0
    except KeyboardInterrupt:
        logging.warning("Interrupted. Completed RCWA shards/chunks and optimization history are preserved; rerun the same stage to resume.")
        return 130
    except Exception:
        logging.exception("Stage failed; no synthetic results or silent fallbacks were substituted")
        return 1
    finally:
        module = sys.modules.get("dataset")
        if module is not None:
            module.close_rcwa_pool()


if __name__ == "__main__":
    from revision_cli import main as revision_main
    raise SystemExit(revision_main())
