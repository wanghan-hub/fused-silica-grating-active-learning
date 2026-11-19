"""Reproduce the retrospective design and interval-coverage diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revision_inputs import ORIGINAL, verify_snapshot
from run_project import load_config


def archive_analysis(repeat: Path, config: dict) -> pd.DataFrame:
    histories = [
        repeat / "optimization" / f"{method}_{objective}" / "true_history.csv"
        for method in config["optimization"]["methods"]
        for objective in config["optimization"]["objectives"]
    ]
    columns = ["geometry_id", "period_nm", "height_nm", "top_duty", "bottom_duty",
               "mean_R", "max_R"]
    archive = pd.concat(
        (pd.read_csv(path, usecols=columns, float_precision="round_trip") for path in histories),
        ignore_index=True,
    )
    archive = archive.drop_duplicates("geometry_id").sort_values("geometry_id")
    tolerance = config["tolerance"]
    minimum_period = archive.period_nm * (1 - tolerance["period_relative"])
    worst_width = minimum_period * (archive.top_duty - tolerance["top_absolute"])
    worst_gap = minimum_period * (1 - archive.bottom_duty - tolerance["bottom_absolute"])

    def best(label: str, candidates: pd.DataFrame, metric: str) -> dict:
        row = candidates.loc[candidates[metric].idxmin()]
        return {"criterion": label, "geometries": len(candidates),
                "geometry_id": row.geometry_id, "mean_R": row.mean_R, "max_R": row.max_R}

    rows = [best("lowest mean", archive, "mean_R"), best("lowest peak", archive, "max_R")]
    for height in (500, 475, 450):
        rows.append(best(f"height <= {height} nm", archive[archive.height_nm <= height], "mean_R"))
    for feature in (22, 24):
        subset = archive[(worst_width >= feature) & (worst_gap >= feature)]
        rows.append(best(f"corner feature >= {feature} nm", subset, "mean_R"))
    return pd.DataFrame(rows)


def ood_recalibration(baseline: Path, config: dict) -> pd.DataFrame:
    import torch
    from surrogate import Ensemble, table_features

    verify_snapshot(baseline)
    torch.set_num_threads(config["model"]["cpu_threads"])
    calibration = pd.read_parquet(baseline / "data/ood.parquet")
    confirmation = pd.read_parquet(baseline / "confirmation_tests/ood.parquet")
    calibration_ids = set(calibration.geometry_id)
    confirmation_ids = set(confirmation.geometry_id)
    if calibration_ids & confirmation_ids:
        raise ValueError("Calibration and confirmation overlap")
    rounds = config["active_learning"]["rounds"]
    coverage = config["calibration"]["coverage"]
    scope = config["geometry"]["ood"]["height_nm"]
    if any(not f.height_nm.between(*scope).all() for f in (calibration, confirmation)):
        raise ValueError("Expected height-OOD calibration and confirmation geometries")

    def infer(ensemble: Ensemble, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        prediction = ensemble.predict(table_features(data))
        spread = np.maximum(prediction.std[:, 0], ensemble.sigma_floor)
        residual = data.R.to_numpy() - prediction.mean[:, 0]
        scores = pd.Series(abs(residual) / spread).groupby(data.geometry_id.to_numpy()).max()
        return residual, spread, scores.to_numpy()

    rows = []
    for strategy in ORIGINAL:
        directory = baseline / "active_learning" / strategy / f"round_{rounds:03d}" / "model"
        model = Ensemble.load(directory, "cpu")
        if set(model.metadata["train_geometry_ids"]) & (calibration_ids | confirmation_ids):
            raise ValueError(f"OOD data overlaps training for {strategy}")
        _, _, calibration_scores = infer(model, calibration)
        residual, spread, test_scores = infer(model, confirmation)
        rank = int(np.ceil((len(calibration_scores) + 1) * coverage))
        for name, q in (("original IID", model.calibration_factor),
                        ("additional OOD", np.sort(calibration_scores)[rank - 1])):
            count = int((test_scores <= q).sum())
            rows.append({"strategy": strategy, "calibration": name, "rank": rank, "q": q,
                         "covered": count, "test_geometries": len(test_scores),
                         "coverage": count / len(test_scores), "mean_full_width": 2 * q * spread.mean(),
                         "test_rmse": np.sqrt(np.mean(residual ** 2))})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", choices=["archive", "ood"])
    parser.add_argument("source", type=Path, help="Baseline repeat for archive, frozen snapshot for ood")
    parser.add_argument("--output", type=Path, required=True, help="Path to the result CSV")
    parser.add_argument("--config", type=Path, default=ROOT / "config_v3.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    analyze = archive_analysis if args.analysis == "archive" else ood_recalibration
    result = analyze(args.source, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
