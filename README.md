# DynamGeom

MATLAB code for Stoll, Valluru & Rudebeck (2026) Dynamic geometry remapping of neural activity within frontal and subcortical areas during decision-making.

The behavioral and neurophysiological data are available on Zenodo ([doi:10.5281/zenodo.17524410](https://doi.org/10.5281/zenodo.17524410)) and are described in London et al., Scientific Data (2026).

## Requirements

- MATLAB R2024a (or newer) with the Statistics and Machine Learning Toolbox.
- Estimated marginal means for mixed-effects models (emmeans) toolbox from John Hartman ([available on Github](https://github.com/jackatta/estimated-marginal-means)), bundled in `scripts/emmeans/`

## Folder layout

- `data_final/`: per-session data (`*_spk.mat`, `*_EOG.mat`) and the pooled `*_pool.mat` files built by `main_000_create_dataset.m`.
- `processed/`: intermediate results that scripts pass to each other (see the table below).
- `scripts/`: `main_*.m`, `utils/` (helper functions), `emmeans/`.
- `report_XXX/`: figures and a statistics log (`output_XXX.txt`) written by the corresponding script.

## Pipeline

Scripts are numbered in the order they must be run. Each one reuses its cached `processed/` files when present; set the `overwrite*` flags at the top of a script to recompute them.

| Script | Analysis | Figures | Report folder | Reads (`processed/`) | Writes (`processed/`) |
|---|---|---|---|---|---|
| `main_000_create_dataset.m` | Pools raw spikes into `data_final/*_pool.mat` | – | – | – | – |
| `main_000_behav.m` | Choice behavior, saccades, spiking statistics | 1B–F, S3 | `report_000` | – | `behav_pref.mat`, `saccadeCounts_*.mat`, `spiking_info.mat` |
| `main_001_anova_lda.m` | Single-neuron ANOVAs, within-task LDA, task-general subspace | S1, S2 | `report_001` | – | `anova_lda.mat`, `flavor_1fc.mat`, `side_1fc.mat` |
| `main_002_states.m` | Cross-task probability states, gaze-linked posteriors, area ablation | 2B–F, 2H, S5A–D | `report_002` | `behav_pref.mat`, `saccadeCounts_*.mat` | `states_2afc_final.mat` |
| `main_003_crossdecoding.m` | 2x2 cross-state decoding of chosen flavor and response side | 3C–D, S6–S8 | `report_003` | `states_2afc_final.mat`, `behav_pref.mat` | `states_2afc_fr.mat`, `states_2afc_ccgp.mat` |
| `main_004_unitstability.m` | Single-neuron tuning across probability states | S11 | `report_004` | `states_2afc_final.mat`, `flavor_1fc.mat`, `side_1fc.mat`, `states_2afc_ccgp.mat` | – |
| `main_005_thr_sensitivity.m` | Robustness to state threshold and duration | S5E, S9 | `report_005` (one subfolder per threshold) | `states_2afc_final.mat`, `saccadeCounts_reduced.mat`, `behav_pref.mat` | `states_2afc_fr_<thr>.mat`, `states_2afc_ccgp_<thr>.mat`, `states_2afc_thr_summary.mat` |
| `main_006_lda_vs_linear.m` | Categorical (LDA) vs continuous (linear) probability decoders | S4 | `report_006` | `states_2afc_final.mat` | – |
| `main_007_statespace.m` | Cross-validated 2D projections on chosen/unchosen-state LDA axes | S10 | `report_007` | `states_2afc_fr.mat`, `behav_pref.mat` | `states_2afc_cvplane.mat` |

`states_2afc_final.mat` holds the probability-state decoding of every session (built by `main_002_states.m` using the function `scripts\utils\utils_decoding_crosstask_rmvarea.m`). Every later file in the table derives from it: after rebuilding it, delete `states_2afc_fr*.mat` and `states_2afc_ccgp*.mat` (or set `overwrite_fr` / `overwrite_decoding` in `main_003`) so that `main_003` and `main_005` recompute them.

## Usage

1. Place the data in `data_final/` and run `scripts/main_000_create_dataset.m` once if the `*_pool.mat` files are missing.
2. In MATLAB, run `run('scripts/run_all.m')` from the project root.

`run_all.m` deletes the report folders of the `main_*` scripts before running them. Report files are overwritten on each run.

`run_all.m` also resets the random number generator (`rng(seed, 'twister')`, with `seed = 55555` set at the top of `run_all.m`) before each script, so rebuilding the `processed/` files reproduces the same pseudo-populations, cross-validation folds, trial subsamples and bootstraps. When running a script on its own, call `rng(55555, 'twister')` first to get the same results.

Scripts can also be run individually or section by section: each one resolves the project root from its own location (`mfilename('fullpath')`), or from the current folder when run section by section, so they do not depend on the MATLAB working directory.
