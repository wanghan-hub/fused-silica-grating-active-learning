"""Revision experiments on fixed v2 geometries; the public entry is run_project.py."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import copy
import csv
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
from statistics import mean, stdev
import sys

ORIGINAL = ("random", "uncertainty", "task_diverse")
ABLATIONS = ("no_performance", "no_disagreement", "no_diversity")
METRICS = ("rmse_R", "mae_R", "geometry_coverage", "point_coverage", "mean_interval_width", "objective_rmse")


def strategies(index: int, full: bool, evaluating: bool) -> tuple[str, ...]:
    """Seed zero reuses the three submitted originals; only ablations need fitting."""
    if index == 0:
        return ORIGINAL + ABLATIONS if evaluating else ABLATIONS
    return ORIGINAL + ABLATIONS if full else ORIGINAL


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def simple_name(s: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", s):
        raise argparse.ArgumentTypeError("Expected one simple file/directory name")
    return s


@contextmanager
def experiment_lock(root: Path):
    """OS advisory lock: no stale lock after Ctrl+C or a process crash."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".revision.lock").open("a+b") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another v3 command is running for {root}; run stages sequentially") from exc
        yield


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Empty results for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def summarize(root: Path, seeds: list[int], full: bool) -> dict:
    """Aggregate complete evaluation records; sample SD is undefined for one seed."""
    records, bykey = [], {}
    for index, seed in enumerate(seeds):
        for s in strategies(index, full, True):
            p = root / f"repeat_{index:03d}" / "revision_evaluation" / f"{s}.json"
            if not p.is_file():
                raise RuntimeError(f"Missing {p}; finish al and evaluate before summary")
            result = read(p)
            if result["seed"] != seed or result["strategy"] != s:
                raise ValueError(f"Unexpected seed or strategy: {p}")
            metrics = result["metrics"]
            if sorted(m["split"] for m in metrics) != ["iid", "ood"]:
                raise ValueError(f"Incomplete splits: {p}")
            for m in metrics:
                row = {"seed": seed, "strategy": s, "split": m["split"],
                       "geometries": m["n_geometries"], "states": m["n_states"],
                       **{k: m[k] for k in METRICS}}
                records.append(row)
                bykey[seed, s, m["split"]] = row
    groups = defaultdict(list)
    pairs = defaultdict(list)
    raw_pairs = []
    for m in records:
        groups[m["strategy"], m["split"]].append(m)
        if m["strategy"] == "task_diverse":
            continue
        f = bykey[m["seed"], "task_diverse", m["split"]]
        delta = f["rmse_R"] - m["rmse_R"]
        pairs[m["strategy"], m["split"]].append(delta)
        raw_pairs.append({"seed": m["seed"], "split": m["split"], "comparison": m["strategy"],
                          "rmse_full_minus_comparison": delta,
                          "relative_reduction_percent": 100 * (-delta) / m["rmse_R"] if m["rmse_R"] else None})
    def stats(xs):
        return mean(xs), (stdev(xs) if len(xs) > 1 else None)
    summary = []
    for (s, split), members in groups.items():
        row = {"strategy": s, "split": split, "independent_runs": len(members)}
        for k in METRICS:
            row[k+"_mean"], row[k+"_sample_sd"] = stats([m[k] for m in members])
        summary.append(row)
    out = root / ("summary_full" if full else "summary_minimum")
    _csv(out/"per_seed.csv", records)
    _csv(out/"strategies.csv", summary)
    _csv(out/"paired_per_seed.csv", raw_pairs)
    _csv(out/"paired_summary.csv", [
        {"comparison": s, "split": split, "paired_seeds": len(xs), "mean_full_minus_comparison": stats(xs)[0],
         "sample_sd": stats(xs)[1]} for (s,split),xs in pairs.items()])
    return {"passed": True, "directory": str(out), "rows": len(records), "seeds": seeds}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_v3.yaml"))
    parser.add_argument("--stage", required=True, choices=("check","plan","prepare","al","evaluate","benchmark","summary"))
    rep = parser.add_mutually_exclusive_group()
    rep.add_argument("--repeat-index", type=int)
    rep.add_argument("--all-repeats", action="store_true")
    parser.add_argument("--strategy", choices=ORIGINAL+ABLATIONS, help="Just one AL/evaluation task")
    parser.add_argument("--full-matrix", action="store_true", help="Also ablate on both additional seeds")
    parser.add_argument("--workers", type=int, help="CPU RCWA workers, not GPU count")
    parser.add_argument("--optimizer-seed", type=int, help="One of the configured paired optimization seeds; default: all")
    parser.add_argument("--campaign", type=simple_name, default="primary", help="Isolated benchmark campaign")
    parser.add_argument("--source-v2-run", type=Path, help="Prepare only, if bundled snapshot is absent")
    a = parser.parse_args(argv)
    if a.strategy and a.stage not in ("al","evaluate"):
        parser.error("--strategy is only valid for al/evaluate")
    if a.stage == "benchmark" and (a.all_repeats or a.full_matrix):
        parser.error("Benchmark repeats the optimizer seed and uses one fixed AL seed; omit --all-repeats/--full-matrix")
    if a.stage != "benchmark" and (a.optimizer_seed is not None or a.campaign != "primary"):
        parser.error("--optimizer-seed/--campaign are for benchmark")
    if a.source_v2_run and a.stage != "prepare":
        parser.error("--source-v2-run is for prepare")
    if a.stage == "summary" and a.repeat_index is not None:
        parser.error("summary aggregates all configured seeds")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    from run_project import code_check, load_config
    if a.stage == "check":
        ans = code_check(Path(__file__).resolve().parent)
        ans["missing_packages"] = [k for k,v in ans["dependencies_available"].items() if not v]
        ok = ans["passed"] and not ans["missing_packages"]
        ans["passed"] = bool(ok)
        print(json.dumps(ans, indent=2))
        return 0 if ok else 1
    try:
        configfile = a.config.resolve()
        cfg = load_config(configfile)
        rev = cfg["revision"]
        offsets = cfg["experiment"]["repeat_seed_offsets"]
        seeds = [cfg["seed"] + off for off in offsets]
        if not seeds or offsets[0] != 0 or len(set(seeds)) != len(seeds):
            raise ValueError("Expected unique AL seeds with original offset zero first")
        if a.workers is not None:
            cfg["runtime"]["rcwa_workers"] = a.workers
        from dataset import ExecutionSettings
        execution = ExecutionSettings.from_config(cfg)
        base = (configfile.parent / rev["baseline"]).resolve()
        out = (configfile.parent / cfg["paths"]["output_root"] / simple_name(cfg["experiment"]["id"])).resolve()
        source = (a.source_v2_run or (configfile.parent / rev["source_run"])).resolve()
        for frozen in (base, source):
            if out == frozen or frozen in out.parents or out in frozen.parents:
                raise ValueError("Revision outputs must be separate from originals")
        bench_index = int(rev["benchmark_model_repeat"])
        opt_seeds = list(map(int, rev["optimizer_seeds"]))
        if not opt_seeds or len(set(opt_seeds)) != len(opt_seeds):
            raise ValueError("Expected unique optimizer seeds")
        idx = a.repeat_index if a.repeat_index is not None else (bench_index if a.stage == "benchmark" else 0)
        indices = list(range(len(seeds))) if a.all_repeats else [idx]
        if any(i not in range(len(seeds)) for i in indices):
            parser.error("repeat-index out of bounds")
        if a.stage == "benchmark" and (idx != bench_index or a.optimizer_seed is not None and a.optimizer_seed not in opt_seeds):
            parser.error("Use the configured benchmark model repeat and optimizer seeds")
        if a.stage == "plan":
            print(json.dumps({"baseline": str(base), "output": str(out), "workers": execution.workers,
                              "training_schedule": {s: strategies(i,a.full_matrix,False) for i,s in enumerate(seeds)},
                              "paired_optimizer_seeds": opt_seeds, "frozen_model_build_seed": seeds[bench_index]}, indent=2))
            return 0
        from revision_inputs import import_inputs, prepare_repeat, verify_snapshot
        from optics import save_json
        with experiment_lock(out):
            if a.stage == "prepare" and (a.source_v2_run or not (base/"v3_input_snapshot.json").is_file()):
                import_inputs(source, base)
            snap = verify_snapshot(base)
            if snap["base_seed"] != seeds[0] or snap["data_seed"] != cfg["reuse"]["data_seed"]:
                raise ValueError("Configured seeds do not match frozen reference")
            historical = read(base/"logs/active_learning_invocation.json")["config"]
            for k in ("grid","geometry","sampling","rcwa","dataset","validation","comsol","calibration",
                      "tolerance","metrics","active_learning","optimization","final_validation","model","confirmation_tests"):
                old, new = historical[k], cfg[k]
                if k == "model":
                    old, new = ({n:v for n,v in entry.items() if n not in ("device","cpu_threads")}
                                for entry in (old,new))
                if old != new:
                    raise ValueError(f"Setting {k} differs from submitted v2. For this paired experiment keep the fixed protocol.")
            # Bind results to all scientific settings and the snapshot. Hardware is recorded separately.
            from optics import fingerprint
            signature = fingerprint({k:v for k,v in cfg.items() if k not in ("paths","runtime","reuse","revision","experiment")} |
                                    {"snapshot":snap["signature"],"seeds":seeds,"optimizer_seeds":opt_seeds,
                                     "benchmark_model_repeat":bench_index})
            protocol_path = out/"revision_protocol.json"
            if protocol_path.exists() and read(protocol_path)["signature"] != signature:
                raise ValueError("Revision protocol changed: use a new experiment id")
            save_json(protocol_path, {"signature":signature,"snapshot":snap["signature"],"seeds":seeds,"optimizer_seeds":opt_seeds})
            if a.stage == "summary":
                print(json.dumps(summarize(out,seeds,a.full_matrix), indent=2))
                return 0
            for i in indices:
                run = copy.deepcopy(cfg)
                work = out / f"repeat_{i:03d}"
                run.update(_root=str(work), _baseline_root=str(base), seed=seeds[i])
                # Use bundled inputs only, never the live v2 tree.
                run["reuse"].update(source_repeat_root=str(base), sibling_repeat_cache=False)
                logdir = work/"logs"
                logdir.mkdir(parents=True,exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                                    handlers=[logging.StreamHandler(),logging.FileHandler(logdir/f"{a.stage}.log",encoding="utf-8")],force=True)
                logging.info("%s: AL seed %d, %d RCWA workers. Output: %s",a.stage,seeds[i],execution.workers,work)
                save_json(logdir/f"{stamp}_{a.stage}.json", {"stage":a.stage,"args":vars(a),"config":run,"protocol":signature})
                prepare_repeat(run)
                if a.stage == "prepare":
                    print(f"Prepared seed {seeds[i]} at {work}")
                elif a.stage in ("al","evaluate"):
                    from revision_learning import evaluate_revision, run_revision_learning
                    tasks = (a.strategy,) if a.strategy else strategies(i,a.full_matrix,a.stage == "evaluate")
                    if a.stage == "al" and i == 0 and any(s in ORIGINAL for s in tasks):
                        raise ValueError("Seed-zero original models are frozen; use evaluate, not al")
                    if a.stage == "al":
                        run_revision_learning(run,list(tasks))
                    else:
                        for origin, names in ((base,[s for s in tasks if i == 0 and s in ORIGINAL]),
                                              (work,[s for s in tasks if i != 0 or s not in ORIGINAL])):
                            if names:
                                evaluate_revision(run,names,model_root=origin)
                elif a.stage == "benchmark":
                    from revision_benchmark import run_benchmark
                    for opt_seed in (opt_seeds if a.optimizer_seed is None else [a.optimizer_seed]):
                        run_benchmark(run,opt_seed,campaign=a.campaign)
        return 0
    except KeyboardInterrupt:
        logging.warning("Interrupted. Re-run the same command; incomplete cold timings are not evidence.")
        return 130
    except Exception:
        logging.exception("Revision stage failed")
        return 1
    finally:
        if "dataset" in sys.modules:
            sys.modules["dataset"].close_rcwa_pool()


if __name__ == "__main__":
    raise SystemExit(main())
