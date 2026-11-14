"""Train-only normalization, four optical heads, deep ensembles and calibration.

PyTorch is optional at import time. Physics/data/CLI checks never require it.
Calibration estimates empirical same-distribution coverage, not an OOD guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping
import copy
import json
import random
import logging

import numpy as np
import pandas as pd

from optics import GEOMETRY_COLUMNS, fingerprint, save_json

LOGGER = logging.getLogger(__name__)

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("This stage requires PyTorch on the training server. Physics stages do not.")


def table_features(frame: pd.DataFrame) -> np.ndarray:
    if not set(frame.polarization.unique()) <= {"TE", "TM"}:
        raise ValueError("Unexpected polarization labels")
    return np.column_stack([frame[GEOMETRY_COLUMNS + ["wavelength_nm", "angle_deg"]].to_numpy(float),
                            (frame.polarization == "TE").astype(float),
                            (frame.polarization == "TM").astype(float)])


def data_signature(frame: pd.DataFrame) -> str:
    columns = ["geometry_id", *GEOMETRY_COLUMNS, "wavelength_nm", "angle_deg", "polarization", "R", "T"]
    return sha256(pd.util.hash_pandas_object(frame[columns], index=False).values.tobytes()).hexdigest()


def select_device(name: str) -> str:
    require_torch()
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose model.device=cpu explicitly")
    return name


class OpticalMLP(nn.Module if nn is not None else object):
    """Raw continuous inputs preserved. Hard conservation applies to predictions only."""
    def __init__(self, width: int, layers: int, variant: str):
        require_torch()
        super().__init__()
        if variant not in {"r_only", "independent", "soft", "hard"}:
            raise ValueError("Unknown output variant")
        self.variant = variant
        modules = []
        for index in range(layers):
            modules.extend([nn.Linear(8 if index == 0 else width, width), nn.GELU()])
        modules.append(nn.Linear(width, 1 if variant in {"r_only", "hard"} else 2))
        self.network = nn.Sequential(*modules)

    def forward(self, inputs: Any) -> Any:
        output = torch.sigmoid(self.network(inputs))
        if self.variant in {"r_only", "hard"}:
            return torch.cat([output, 1 - output], dim=1)
        return output


@dataclass
class EnsemblePrediction:
    mean: np.ndarray
    std: np.ndarray
    members: np.ndarray


class Ensemble:
    def __init__(self, models: list[Any], mean: np.ndarray, scale: np.ndarray,
                 device: str, metadata: dict[str, Any], batch_size: int):
        self.models, self.mean, self.scale = models, mean, scale
        self.device, self.metadata, self.batch_size = device, metadata, batch_size
        self.calibration_factor = 1.0
        self.sigma_floor = 1e-6

    def predict(self, features: np.ndarray) -> EnsemblePrediction:
        require_torch()
        x = (np.asarray(features, dtype=np.float32) - self.mean) / self.scale
        if x.ndim != 2 or x.shape[1] != 8 or not np.isfinite(x).all() or not len(x):
            raise ValueError("Expected nonempty finite [N,8] inputs")
        outputs = []
        with torch.no_grad():
            for model in self.models:
                batches = [model(torch.as_tensor(x[i:i + self.batch_size], dtype=torch.float32,
                                                 device=self.device)).cpu().numpy()
                           for i in range(0, len(x), self.batch_size)]
                outputs.append(np.concatenate(batches))
        members = np.stack(outputs).astype(np.float64)
        return EnsemblePrediction(members.mean(axis=0), members.std(axis=0, ddof=1), members)

    @classmethod
    def load(cls, directory: Path, device: str = "auto", batch_size: int = 16384) -> Ensemble:
        require_torch()
        metadata = json.loads((directory / "ensemble.json").read_text(encoding="utf-8"))
        if not metadata.get("training_qc", {}).get("passed", False):
            raise RuntimeError(f"Model has no passing v2 training QC: {directory}. Keep v1 models archived; retrain in a fresh directory.")
        resolved = select_device(device)
        models = []
        for filename in metadata["members"]:
            checkpoint = torch.load(directory / filename, map_location="cpu", weights_only=True)
            model = OpticalMLP(metadata["width"], metadata["layers"], metadata["variant"])
            model.load_state_dict(checkpoint["state_dict"])
            model.to(resolved).eval()
            models.append(model)
        obj = cls(models, np.asarray(metadata["feature_mean"]), np.asarray(metadata["feature_scale"]),
                  resolved, metadata, batch_size)
        calibration_path = directory / "calibration.json"
        if calibration_path.exists():
            calibration = json.loads(calibration_path.read_text())
            if calibration["model_signature"] != metadata["training_signature"]:
                raise RuntimeError("Stale calibration metadata")
            obj.calibration_factor = calibration["factor"]
            obj.sigma_floor = calibration["sigma_floor"]
        return obj


def _validate_training_tables(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    if set(train.geometry_id) & set(validation.geometry_id):
        raise ValueError("Training/validation geometry leakage")
    if "family_id" in train and "family_id" in validation and set(train.family_id) & set(validation.family_id):
        raise ValueError("Training/validation perturbation-family leakage")
    for frame in (train, validation):
        if frame.empty or not (frame.status == "ok").all():
            raise ValueError("Only nonempty audited solver data may be used for training")
        if not np.isfinite(frame[["R", "T"]].values).all():
            raise ValueError("Nonfinite labels")
    if "role" in train and set(train.role) & {"validation", "calibration", "iid", "ood"}:
        raise ValueError("Protected evaluation data passed as training labels")


def train_ensemble(train: pd.DataFrame, validation: pd.DataFrame, config: Mapping[str, Any],
                   directory: Path, variant: str | None = None) -> Ensemble:
    require_torch()
    _validate_training_tables(train, validation)
    options = config["model"]
    variant = variant or options["primary_variant"]
    if int(options["ensemble_size"]) < 2:
        raise ValueError("Uncertainty estimation requires at least two ensemble members")
    directory.mkdir(parents=True, exist_ok=True)
    signature = fingerprint({"train": data_signature(train), "validation": data_signature(validation),
                             "options": options, "seed": config["seed"], "variant": variant,
                             "source": Path(__file__).read_text(encoding="utf-8")})
    existing = directory / "ensemble.json"
    if existing.exists():
        metadata = json.loads(existing.read_text())
        if metadata["training_signature"] != signature:
            raise RuntimeError(f"Model directory contains different experiment: {directory}")
        return Ensemble.load(directory, options["device"], options["predict_batch_size"])
    x, xv = table_features(train), table_features(validation)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < options["normalization_floor"]] = 1.0
    x = ((x - mean) / scale).astype(np.float32)
    xv = ((xv - mean) / scale).astype(np.float32)
    y, yv = train[["R", "T"]].to_numpy(np.float32), validation[["R", "T"]].to_numpy(np.float32)
    device = select_device(options["device"])
    torch.set_num_threads(int(options["cpu_threads"]))
    torch.use_deterministic_algorithms(bool(options["deterministic"]), warn_only=False)
    groups = train.groupby("geometry_id", sort=True).indices
    ids = sorted(groups)
    history, models, filenames = [], [], []
    started = perf_counter()
    quality = options["quality"]
    target_mean = y.mean(axis=0, dtype=np.float64)
    if np.any((target_mean <= 0) | (target_mean >= 1)):
        raise ValueError("Output-bias initialization requires strictly interior training mean R/T")
    # This changes initialization, not labels or the physical output mapping.
    # A common scalar rescales the complete loss, preserving soft-penalty weighting.
    loss_scale = max(float(np.var(y[:, 0], dtype=np.float64)), options["loss_scale_floor"])
    baseline_rmse = float(np.sqrt(np.mean((yv[:, 0]-target_mean[0])**2)))
    if baseline_rmse <= 0:
        raise ValueError("Degenerate validation baseline")
    qc: dict[str, Any] = {"passed": False, "baseline": "constant training-set mean R",
        "baseline_validation_rmse": baseline_rmse, "training_target_mean": target_mean.tolist(),
        "loss_scale": loss_scale, "members": [], "policy": dict(quality)}

    def loss_value(prediction: Any, target: Any) -> Any:
        if variant == "r_only":
            return torch.mean((prediction[:, 0] - target[:, 0]) ** 2) / loss_scale
        loss = torch.mean((prediction - target) ** 2)
        if variant == "soft":
            loss = loss + options["energy_weight"] * torch.mean((prediction.sum(dim=1) - 1) ** 2)
        return loss / loss_scale

    for member in range(int(options["ensemble_size"])):
        seed = int(config["seed"] + member * 1009)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        selected = rng.choice(ids, size=len(ids), replace=True) if options["bootstrap"] else ids
        indices = np.concatenate([groups[g] for g in selected])
        model = OpticalMLP(options["width"], options["layers"], variant).to(device)
        with torch.no_grad():
            head = model.network[-1]
            head.weight.mul_(float(options["output_weight_scale"]))
            means = target_mean[:1] if variant in {"r_only", "hard"} else target_mean
            head.bias.copy_(torch.as_tensor(np.log(means/(1-means)), device=device, dtype=head.bias.dtype))
        optimizer = torch.optim.AdamW(model.parameters(), lr=options["learning_rate"],
                                       weight_decay=options["weight_decay"])
        generator = torch.Generator().manual_seed(seed)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
            torch.from_numpy(x[indices]), torch.from_numpy(y[indices])),
            batch_size=options["batch_size"], shuffle=True, generator=generator, num_workers=0)
        best, wait, best_state = float("inf"), 0, None
        for epoch in range(int(options["epochs"])):
            model.train()
            running, examples = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_value(model(xb), yb)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), options["gradient_clip"])
                optimizer.step()
                running += float(loss.detach()) * len(xb)
                examples += len(xb)
            model.eval()
            squared, count = 0.0, 0
            with torch.no_grad():
                for index in range(0, len(xv), options["predict_batch_size"]):
                    xb = torch.as_tensor(xv[index:index + options["predict_batch_size"]], device=device)
                    target = torch.as_tensor(yv[index:index + options["predict_batch_size"]], device=device)
                    # R validation error is common to ALL head variants.
                    squared += float(torch.sum((model(xb)[:, 0] - target[:, 0]) ** 2))
                    count += len(xb)
            score = squared / count
            history.append({"member": member, "seed": seed, "epoch": epoch,
                            "train_loss": running / examples, "validation_R_mse": score})
            if score < best - options["min_delta"]:
                best, wait = score, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
            if epoch == 0 or (epoch+1) % int(options["log_every_epochs"]) == 0:
                LOGGER.info("%s member %d/%d epoch %d/%d validation RMSE %.6g (constant baseline %.6g)",
                            variant, member+1, options["ensemble_size"], epoch+1, options["epochs"],
                            np.sqrt(score), baseline_rmse)
            if epoch+1 >= int(options["min_epochs"]) and wait >= options["patience"]:
                break
        if best_state is None:
            raise RuntimeError("Training did not produce a finite checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        # Evaluate the accepted checkpoint directly without an artificial one-member SD.
        with torch.no_grad():
            estimates = np.concatenate([model(torch.as_tensor(xv[i:i+options["predict_batch_size"]],
                device=device)).cpu().numpy()[:, 0] for i in range(0, len(xv), options["predict_batch_size"])])
        error = float(np.sqrt(np.mean((estimates-yv[:, 0])**2)))
        member_qc = {"member": member, "seed": seed, "epochs_completed": epoch+1,
                     "best_epoch": min((h for h in history if h["member"] == member),
                                       key=lambda h: h["validation_R_mse"])["epoch"],
                     "validation_R_rmse": error, "baseline_rmse_ratio": error/baseline_rmse,
                     "prediction_mean_R": float(estimates.mean()),
                     "prediction_min_R": float(estimates.min()), "prediction_max_R": float(estimates.max()),
                     "passed": bool(np.isfinite(estimates).all() and
                         error <= float(quality["max_baseline_rmse_ratio"])*baseline_rmse)}
        qc["members"].append(member_qc)
        pd.DataFrame(history).to_csv(directory / "training_history.csv", index=False)
        save_json(directory / "training_qc.json", qc)
        if not member_qc["passed"]:
            raise RuntimeError(f"Training member {member} failed QC: {directory / 'training_qc.json'}. "
                               "No accepted ensemble was saved; investigate training using validation only.")
        filename = f"member_{member}.pt"
        temporary = directory / (filename + ".tmp")
        torch.save({"state_dict": best_state, "seed": seed}, temporary)
        temporary.replace(directory / filename)
        filenames.append(filename)
        models.append(model.eval())
    errors = np.array([m["validation_R_rmse"] for m in qc["members"]])
    qc["worst_to_median_member_rmse"] = float(errors.max()/max(float(np.median(errors)), 1e-15))
    qc["passed"] = bool(qc["worst_to_median_member_rmse"] <= quality["max_member_median_ratio"])
    save_json(directory / "training_qc.json", qc)
    if not qc["passed"]:
        raise RuntimeError(f"Unbalanced ensemble members; see {directory / 'training_qc.json'}")
    metadata = {"variant": variant, "width": options["width"], "layers": options["layers"],
                "members": filenames, "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
                "training_signature": signature, "train_geometry_ids": ids,
                "train_feature_min": table_features(train).min(axis=0).tolist(),
                "train_feature_max": table_features(train).max(axis=0).tolist(),
                "training_time_s": perf_counter() - started,
                "training_qc": qc, "training_protocol": "train_mean_bias_common_loss_scale_v2",
                "r_only_hard_equivalence": "For lossless labels these heads/losses are algebraically equivalent up to numerical residuals; not an independent physics-gain ablation.",
                "R_only_T_is_derived_prediction": variant == "r_only"}
    pd.DataFrame(history).to_csv(directory / "training_history.csv", index=False)
    save_json(existing, metadata)
    return Ensemble(models, mean, scale, device, metadata, options["predict_batch_size"])


def calibrate(ensemble: Ensemble, calibration: pd.DataFrame, config: Mapping[str, Any],
              directory: Path) -> dict[str, Any]:
    if set(calibration.geometry_id) & set(ensemble.metadata["train_geometry_ids"]):
        raise ValueError("Calibration/training geometry leakage")
    if calibration.empty or not (calibration.status == "ok").all():
        raise ValueError("Calibration data must pass QC")
    options = config["calibration"]
    prediction = ensemble.predict(table_features(calibration))
    sigma = np.maximum(prediction.std[:, 0], options["sigma_floor"])
    normalized_error = abs(calibration.R.to_numpy() - prediction.mean[:, 0]) / sigma
    # Each geometry contributes one maximum score: simultaneous coverage over its grid.
    scores = pd.DataFrame({"geometry_id": calibration.geometry_id.to_numpy(),
                           "score": normalized_error}).groupby("geometry_id").score.max().to_numpy()
    rank = int(np.ceil((len(scores) + 1) * options["coverage"]))
    if rank > len(scores):
        raise ValueError("Too few independent calibration geometries for requested coverage")
    factor = float(np.sort(scores)[rank - 1])
    ensemble.calibration_factor, ensemble.sigma_floor = factor, options["sigma_floor"]
    summary = {"model_signature": ensemble.metadata["training_signature"], "factor": factor,
               "sigma_floor": options["sigma_floor"], "target_coverage": options["coverage"],
               "score_definition": "maximum standardized R residual per geometry over fixed grid",
               "calibration_geometries": len(scores),
               "empirical_calibration_geometry_coverage": float(np.mean(scores <= factor)),
               "calibration_signature": data_signature(calibration)}
    save_json(directory / "calibration.json", summary)
    return summary


def prediction_table(ensemble: Ensemble, frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    prediction = ensemble.predict(table_features(frame))
    result["R_pred"], result["T_pred"] = prediction.mean.T
    result["R_std"], result["T_std"] = prediction.std.T
    width = ensemble.calibration_factor * np.maximum(result.R_std.values, ensemble.sigma_floor)
    result["R_lower"], result["R_upper"] = result.R_pred - width, result.R_pred + width
    result["covered"] = (result.R >= result.R_lower) & (result.R <= result.R_upper)
    return result


def run_training_check(config: Mapping[str, Any]) -> dict[str, Any]:
    """Full configured initial primary-ensemble training/QC, reusable by --stage train."""
    from dataset import audit_dataset, load_split
    audit_dataset(config)
    variant = config["model"]["primary_variant"]
    directory = Path(config["_root"]) / "models" / variant
    model = train_ensemble(load_split(config, "train_initial"), load_split(config, "validation"),
                           config, directory, variant)
    prediction_table(model, load_split(config, "validation")).to_csv(directory / "validation_predictions.csv", index=False)
    return {"passed": model.metadata["training_qc"]["passed"], "variant": variant,
            "device": model.device, "training_qc": model.metadata["training_qc"],
            "note": "Validation only; initial primary model reused by train. Other AL sets still need their own QC."}


def run_training(config: Mapping[str, Any]) -> dict[str, Any]:
    from dataset import audit_dataset, load_split
    audit_dataset(config)
    train, val, calibration = [load_split(config, role) for role in
                                ("train_initial", "validation", "calibration")]
    result = {}
    for variant in config["model"]["variants"]:
        directory = Path(config["_root"]) / "models" / variant
        ensemble = train_ensemble(train, val, config, directory, variant)
        result[variant] = calibrate(ensemble, calibration, config, directory)
        prediction_table(ensemble, val).to_csv(directory / "validation_predictions.csv", index=False)
    return result
