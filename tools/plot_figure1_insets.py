
"""Draw the two small spectral plots for manuscript Figure 1.

Run without arguments for explicitly labelled SCHEMATIC illustrations:
    python plot_figure1_insets_final.py

Use existing CSV results, without PyTorch or another simulation:
    python plot_figure1_insets_final.py --mode results

Dependencies: numpy, matplotlib. Each plot is saved as transparent PNG (600 dpi),
editable SVG, and PDF; the plotted numbers and provenance are also exported.
Shading always means mean +/- ONE ensemble standard deviation, NOT a calibrated
confidence interval. No clipping, smoothing, or silent synthetic fallback is used
when reading real results. Schematic curves are illustrative analytic functions.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, MaxNLocator


BASE = Path(__file__).resolve().parent
BLUE = "#087FF5"
COLORS = (BLUE, "#F08020", "#149C88")
LOG = logging.getLogger("figure1_insets")


@dataclass
class Spectrum:
    """One geometry, one angle, one polarization; wavelength is the only scan."""

    wavelength: np.ndarray
    reflectance: np.ndarray
    label: str
    geometry_id: str
    std: np.ndarray | None = None


def schematic_spectra() -> tuple[Spectrum, list[Spectrum]]:
    """Deliberately synthetic illustrations, not RCWA labels or fitted results."""
    wavelength = np.linspace(450.0, 850.0, 81)
    x = (wavelength - 450.0) / 400.0
    base = 0.003 + 0.055 * (2 * x - 1) ** 4 + 0.030 * np.exp(-12 * x)
    offsets = np.array([-0.22, -0.10, 0.0, 0.11, 0.20])[:, None]
    members = base[None, :] * np.exp(
        offsets * (0.9 + 0.25 * np.sin(2 * np.pi * x))
    )
    prediction = Spectrum(
        wavelength,
        members.mean(axis=0),
        "Mean prediction",
        "schematic_prediction",
        members.std(axis=0, ddof=1),
    )
    curves = (
        0.004 + 0.050 * (2 * x - 1) ** 4 + 0.060 * np.exp(-14 * x),
        0.003 + 0.028 * (2 * x - 0.7) ** 2 + 0.025 * np.exp(-17 * x),
        0.002 + 0.018 * x ** 2 + 0.020 * np.exp(-16 * x),
    )
    dense = [
        Spectrum(wavelength, y, f"Example {i}", f"schematic_{i}")
        for i, y in enumerate(curves, start=1)
    ]
    return prediction, dense


def read_states(
    path: Path,
    angle: float,
    polarization: str,
    values: tuple[str, ...],
) -> dict[str, list[dict[str, str]]]:
    """Read this project's CSV schema, selecting a fixed optical condition."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing results file: {path}. No schematic fallback."
        )

    groups: dict[str, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "geometry_id",
            "wavelength_nm",
            "angle_deg",
            "polarization",
            *values,
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")

        for row in reader:
            if (
                row["polarization"].strip().upper() != polarization
                or not np.isclose(
                    float(row["angle_deg"]), angle, rtol=0, atol=1e-8
                )
            ):
                continue

            if "status" in row and row["status"].strip() != "ok":
                raise ValueError(
                    f"{path}: non-ok simulation state in selected condition"
                )

            groups.setdefault(row["geometry_id"], []).append(row)

    if not groups:
        raise ValueError(
            f"{path}: no rows for angle={angle}, polarization={polarization}"
        )

    return groups


def to_spectrum(
    rows: list[dict[str, str]],
    gid: str,
    value: str,
    label: str,
    spread: str | None = None,
) -> Spectrum:
    rows = sorted(rows, key=lambda row: float(row["wavelength_nm"]))
    wavelength = np.array(
        [float(row["wavelength_nm"]) for row in rows]
    )
    reflectance = np.array([float(row[value]) for row in rows])
    std = (
        None
        if spread is None
        else np.array([float(row[spread]) for row in rows])
    )

    arrays = [wavelength, reflectance] + (
        [] if std is None else [std]
    )

    if len(wavelength) < 2 or not all(
        np.isfinite(a).all() for a in arrays
    ):
        raise ValueError(
            f"{gid}: nonfinite values or fewer than two wavelength points"
        )

    if np.any(np.diff(wavelength) <= 0):
        raise ValueError(
            f"{gid}: duplicate wavelength states; do not mix conditions"
        )

    if std is not None and (std < 0).any():
        raise ValueError(
            f"{gid}: ensemble standard deviations must be nonnegative"
        )

    return Spectrum(wavelength, reflectance, label, gid, std)


def result_spectra(
    args: argparse.Namespace,
) -> tuple[Spectrum, list[Spectrum], dict]:
    root = args.root.resolve()

    prediction_path = (
        args.predictions_csv
        or root
        / "active_learning"
        / "task_diverse"
        / "round_004"
        / "validation_predictions.csv"
    ).resolve()

    rcwa_path = (
        args.rcwa_csv
        or root / "final_validation" / "nominal_spectra.csv"
    ).resolve()

    predictions = read_states(
        prediction_path,
        args.angle,
        args.polarization,
        ("R_pred", "R_std"),
    )

    gid = args.geometry_id or sorted(predictions)[0]
    if gid not in predictions:
        raise ValueError(f"Prediction geometry not found: {gid}")

    prediction = to_spectrum(
        predictions[gid],
        gid,
        "R_pred",
        "Mean prediction",
        "R_std",
    )

    states = read_states(
        rcwa_path,
        args.angle,
        args.polarization,
        ("R",),
    )

    ids = args.design_ids or sorted(states)[:3]

    if len(set(ids)) != len(ids) or not 1 <= len(ids) <= 3:
        raise ValueError(
            "Choose one to three distinct --design-ids for the small RCWA inset"
        )

    if set(ids) - set(states):
        raise ValueError(
            f"RCWA geometries not found: {set(ids) - set(states)}"
        )

    dense = [
        to_spectrum(states[g], g, "R", f"D{i}")
        for i, g in enumerate(ids, start=1)
    ]

    return prediction, dense, {
        "prediction_file": str(prediction_path),
        "rcwa_file": str(rcwa_path),
        "selection": (
            "Explicit IDs, or sorted geometry IDs; "
            "no ranking by prediction error"
        ),
    }


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "svg.fonttype": "none",
        "svg.hashsalt": "figure1-insets",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def choose_scale(curves: list[Spectrum], requested: str) -> str:
    """Never hide a nonpositive mean-minus-std bound on a logarithmic axis."""
    lower = [
        s.reflectance if s.std is None else s.reflectance - s.std
        for s in curves
    ]
    positive = all((array > 0).all() for array in lower)

    if requested == "log" and not positive:
        raise ValueError(
            "Log axis requires all values AND lower spread bounds > 0. "
            "Use --yscale linear or auto; values will not be clipped."
        )

    if requested == "auto":
        if not positive:
            LOG.info(
                "Using a linear axis to preserve nonpositive values/spread bounds"
            )
        return "log" if positive else "linear"

    return requested


def draw_plot(
    curves: list[Spectrum],
    prediction: bool,
    args: argparse.Namespace,
    output: Path,
) -> dict:
    """Draw one spectrum plot; keep the compact RCWA inset free of a title."""

    # Surrogate spectrum keeps its original dimensions and font sizes.
    # Dense-grid RCWA uses a 16:6 aspect ratio and smaller text.
    if prediction:
        width_mm = args.width_mm
        height_mm = args.height_mm
        font_scale = 1.0
    else:
        width_mm = args.width_mm * (2 / 3)  # 48 mm by default
        height_mm = width_mm * (6 / 16)     # 18 mm: exact 16:6 ratio
        font_scale = 0.75

    fig, ax = plt.subplots(
        figsize=(width_mm / 25.4, height_mm / 25.4),
        layout="constrained",
    )

    scale = choose_scale(curves, args.yscale)

    for i, spectrum in enumerate(curves):
        color = COLORS[i % len(COLORS)]

        ax.plot(
            spectrum.wavelength,
            spectrum.reflectance,
            color=color,
            linewidth=1.45 * font_scale,
            label=spectrum.label,
            zorder=3,
        )

        if spectrum.std is not None:
            ax.fill_between(
                spectrum.wavelength,
                spectrum.reflectance - spectrum.std,
                spectrum.reflectance + spectrum.std,
                color="#81BCFF",
                alpha=0.45,
                linewidth=0,
                label=r"Ensemble spread ($\pm1$ SD)",
                zorder=2,
            )

    ax.set_yscale(scale)

    ax.set_xlabel(
        r"$\lambda$ (nm)",
        labelpad=2 * font_scale,
        fontsize=9 * font_scale,
    )

    ax.set_ylabel(
        r"$\hat{R}$" if prediction else r"$R$",
        labelpad=2 * font_scale,
        fontsize=9 * font_scale,
    )

    xmin = min(float(s.wavelength.min()) for s in curves)
    xmax = max(float(s.wavelength.max()) for s in curves)
    ax.set_xlim(xmin, xmax)

    if np.isclose(xmin, 450) and np.isclose(xmax, 850):
        ax.set_xticks([450, 550, 650, 750, 850])
    else:
        ax.xaxis.set_major_locator(MaxNLocator(5))

    if scale == "log":
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
        ax.yaxis.set_minor_formatter(NullFormatter())

    # Reserve space for the legend without clipping or changing any data.
    lower = min(
        float(
            np.min(
                s.reflectance
                if s.std is None
                else s.reflectance - s.std
            )
        )
        for s in curves
    )

    upper = max(
        float(
            np.max(
                s.reflectance
                if s.std is None
                else s.reflectance + s.std
            )
        )
        for s in curves
    )

    if scale == "log":
        low, high = np.log10(lower), np.log10(upper)
        span = max(high - low, 0.15)
        ax.set_ylim(
            10 ** (low - 0.05 * span),
            10 ** (high + 0.45 * span),
        )
    else:
        span = max(upper - lower, abs(upper) * 0.1, 1e-6)
        ax.set_ylim(
            lower - 0.05 * span,
            upper + 0.45 * span,
        )

    # Scale tick labels and tick marks for the smaller RCWA figure.
    ax.tick_params(
        which="major",
        length=3 * font_scale,
        width=0.75 * font_scale,
        pad=2 * font_scale,
        labelsize=7.5 * font_scale,
    )

    ax.tick_params(
        which="minor",
        length=1.5 * font_scale,
        width=0.5 * font_scale,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8 * font_scale)

    # IMPORTANT:
    # Only the Surrogate spectrum gets a title and subtitle.
    # The two lines above Dense-grid RCWA are intentionally removed.
    if prediction:
        if args.mode == "schematic":
            title = "Schematic spectrum"
            subtitle = r"Fixed geometry, $\theta$, pol."
        else:
            title = "Surrogate spectrum"
            subtitle = rf"$\theta={args.angle:g}^\circ$, {args.polarization}"

        ax.set_title(
            title + "\n" + subtitle,
            pad=6,
            fontsize=9,
        )

    # Keep the legend, but reduce its font and spacing for Dense-grid RCWA.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        borderaxespad=0.2 * font_scale,
        ncol=1 if prediction else len(curves),
        fontsize=7 * font_scale,
        handlelength=1.4 * font_scale,
        handletextpad=0.4 * font_scale,
        columnspacing=0.65 * font_scale,
        labelspacing=0.25 * font_scale,
    )

    stem = (
        ("ensemble_spectrum" if prediction else "dense_rcwa_spectra")
        + "_"
        + args.mode
    )

    filenames = []

    for suffix in ("png", "svg", "pdf"):
        destination = output / f"{stem}.{suffix}"
        fig.savefig(
            destination,
            dpi=args.dpi,
            transparent=not args.white_background,
        )
        filenames.append(str(destination))

    plt.close(fig)

    # Export the plotted data without modifying the original values.
    with (
        output / f"{stem}_data.csv"
    ).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "mode",
            "geometry_id",
            "curve",
            "wavelength_nm",
            "R",
            "R_std",
        ])

        for s in curves:
            for j, wavelength in enumerate(s.wavelength):
                writer.writerow([
                    args.mode,
                    s.geometry_id,
                    s.label,
                    wavelength,
                    s.reflectance[j],
                    "" if s.std is None else s.std[j],
                ])

    return {
        "files": filenames,
        "yscale": scale,
        "figure_size_mm": {
            "width": width_mm,
            "height": height_mm,
        },
        "title_shown": prediction,
        "geometries": [
            {
                "label": s.label,
                "geometry_id": s.geometry_id,
                "wavelength_points": len(s.wavelength),
            }
            for s in curves
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=("schematic", "results"),
        default="schematic",
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=BASE / "outputs" / "silica_grating_v2_trainingfix" / "repeat_000",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=BASE / "figure1_insets",
    )

    parser.add_argument("--predictions-csv", type=Path)
    parser.add_argument("--rcwa-csv", type=Path)

    parser.add_argument(
        "--geometry-id",
        help="Prediction example geometry; default: first sorted ID",
    )

    parser.add_argument(
        "--design-ids",
        nargs="+",
        help="One to three RCWA geometry IDs",
    )

    parser.add_argument("--angle", type=float, default=0.0)

    parser.add_argument(
        "--polarization",
        choices=("TE", "TM"),
        default="TE",
    )

    parser.add_argument(
        "--yscale",
        choices=("auto", "log", "linear"),
        default="auto",
    )

    parser.add_argument("--width-mm", type=float, default=72.0)
    parser.add_argument("--height-mm", type=float, default=52.0)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--white-background", action="store_true")

    args = parser.parse_args()

    if args.width_mm <= 0 or args.height_mm <= 0 or args.dpi <= 0:
        parser.error("Figure dimensions and DPI must be positive")

    if not np.isfinite([
        args.width_mm,
        args.height_mm,
        args.angle,
    ]).all():
        parser.error("Dimensions and angle must be finite")

    if args.mode == "schematic" and any((
        args.predictions_csv,
        args.rcwa_csv,
        args.geometry_id,
        args.design_ids,
    )):
        parser.error("CSV files and geometry IDs require --mode results")

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    LOG.setLevel(logging.INFO)

    configure_style()

    if args.mode == "schematic":
        prediction, dense = schematic_spectra()
        sources = {
            "source": (
                "Analytic illustrative functions; "
                "no solver or trained model used"
            )
        }
        LOG.warning(
            "SCHEMATIC MODE: these illustrations are not numerical research results"
        )
    else:
        prediction, dense, sources = result_spectra(args)

    # Validate both figures before writing either one.
    choose_scale([prediction], args.yscale)
    choose_scale(dense, args.yscale)

    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)

    provenance = {
        "mode": args.mode,
        **sources,
        "angle_deg": args.angle if args.mode == "results" else None,
        "polarization": (
            args.polarization if args.mode == "results" else "unspecified"
        ),
        "band": (
            "Ensemble mean +/- 1 sample standard deviation; "
            "not a confidence interval"
        ),
        "prediction": draw_plot([prediction], True, args, output),
        "rcwa": draw_plot(dense, False, args, output),
    }

    manifest = output / f"plot_manifest_{args.mode}.json"
    manifest.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    LOG.info(
        "Saved two plots in PNG/SVG/PDF and plotted-data CSVs to %s",
        output,
    )
    LOG.info(
        "Geometry selections and data sources: %s",
        manifest,
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
