# Active Learning and Surrogate-Assisted Optimization of Low-Reflectance Fused-Silica Gratings

Research code for geometry-level active learning and nominal and robust design of one-dimensional trapezoidal fused-silica gratings. The pipeline uses rigorous coupled-wave analysis (RCWA) to generate reflectance and transmittance labels. It trains a neural ensemble, checks selected designs on a denser RCWA grid, and compares matched conditions with independent COMSOL calculations.

## Setup

Use Python 3.10 and run the commands from the repository root.

```sh
python -m pip install -r requirements.txt
```

Install PyTorch 2.7.1 for training, inference, and surrogate-assisted search. Choose the appropriate CPU or CUDA build using the [PyTorch installation guide](https://pytorch.org/get-started/previous-versions/#v271). RCWA runs on CPU workers. The nominal timing benchmark used PyTorch 2.7.1 with CUDA 11.8 for ensemble training.

```sh
python run_project.py --baseline --stage check
python -m unittest discover -s tests
```

## Data and archived results

Download `raster_reproduction_data.zip` from the [repository Releases](https://github.com/wanghan-hub/fused-silica-grating-active-learning/releases) and extract it at the repository root.

| Directory | Contents |
|---|---|
| `inputs/v2_frozen/` | Original Parquet splits and geometry manifests, independent IID/OOD confirmation labels, and the three original-seed final ensembles. The snapshot records per-file SHA-256 hashes. |
| `inputs/comsol/` | Independent COMSOL exports for the published initial and final comparison conditions. |
| `inputs/paper_run/` | Archived prediction, convergence, search, and verification tables for Figures 2–5 and the retrospective design analyses. |
| `inputs/paper_results/` | Archived active-learning, ablation, interval calibration, and paired timing summaries. |

The archive retains the numerical inputs, predictions, and checkpoints used for the supplied analyses. A local path in the initial COMSOL certificate is stored as a relative reference in the release snapshot. The commands below regenerate simulation labels and optimization outputs.

## Generate RCWA labels

The baseline settings are in `config_v2.yaml`. Each geometry has 410 states (450–850 nm in 10 nm steps, five angles from 0° to 40°, and TE/TM). The initial train, validation, calibration, IID, and height-OOD splits contain 128, 64, 64, 128, and 64 geometries, respectively.

```sh
python run_project.py --baseline --experiment-id silica_grating_label_build --stage validate
python run_project.py --baseline --experiment-id silica_grating_label_build --stage comsol --scope initial --comsol-results inputs/comsol/initial_results.csv
python run_project.py --baseline --experiment-id silica_grating_label_build --stage dataset
python run_project.py --baseline --experiment-id silica_grating_label_build --stage audit
```

Validation writes a request CSV under `outputs/silica_grating_label_build/repeat_000/validation/`. For a new COMSOL calculation, use that request and preserve the condition IDs and parameters when adding the independent `R_comsol` and `T_comsol` columns. Windows scripts and Java sources for COMSOL 6.1 are in `comsol/`. The initial script runs the archived 90 conditions. The final script `comsol/final/Prepare-FinalComsol.ps1` creates a batch from a new `--stage final_validation` request. Pass your local `-ComsolRoot` to the resulting `Run-Comsol.ps1`. Compare archived final designs with their matching CSV only.

## Reproduce the original workflow

Start with the generated split directory above or the downloaded frozen split.

```sh
python run_project.py --baseline --stage prepare_v2
python run_project.py --baseline --stage train
python run_project.py --baseline --stage calibrate
python run_project.py --baseline --stage active_learning
python run_project.py --baseline --stage optimize
python run_project.py --baseline --stage final_validation
python run_project.py --baseline --stage confirmation_tests
python run_project.py --baseline --stage report --evaluate-tests
```

The results directory is `outputs/silica_grating_v2_trainingfix/repeat_000/`. Import newly solved final COMSOL results with `python run_project.py --baseline --stage comsol --scope final --comsol-results PATH_TO_MATCHED_CSV` before plotting the new comparison.

## Reproduce the revision experiments

The default CLI reads `config_v3.yaml` and verifies the frozen snapshot. The three primary acquisition strategies share fixed test geometries across seeds. The component ablations use the original seed. The timing experiment builds one surrogate from fresh labels, then repeats paired nominal searches with three optimizer seeds.

```sh
python run_project.py --stage plan
python run_project.py --stage prepare --all-repeats
python run_project.py --stage al --all-repeats
python run_project.py --stage evaluate --all-repeats
python run_project.py --stage summary
python run_project.py --stage benchmark --workers 16
```

Set `--workers` to the available CPU resources. Keep the worker count and hardware fixed within a timing campaign. Results are written under `outputs/silica_grating_v3_revision/`. The standalone surrogate timing charges the full measured model build to each search. Both routes time dense-grid verification separately.

## Plot and analyze the archived results

```sh
python tools/plot_paper_figures_pier.py --results-dir inputs/paper_run --output-dir outputs/paper_figures --formats pdf
python tools/analyze_paper.py archive inputs/paper_run --output outputs/archive_analysis.csv
python tools/analyze_paper.py ood inputs/v2_frozen --output outputs/posthoc_recalibration.csv
```

The archive analysis reranks previously evaluated designs. The OOD recalibration uses a separate 64-geometry height-OOD calibration split and keeps the original trained models fixed.
