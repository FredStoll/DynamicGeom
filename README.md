# PAV2INS Pipeline

This repository contains MATLAB scripts to build pooled datasets and run the analysis pipeline.

## Main Scripts

- `scripts/run_all.m`: runs all main scripts except dataset creation.
- `scripts/main_000_create_dataset.m`: builds `data_final/*_pool.mat` from raw `*_spk.mat` files.
- `scripts/main_000_behav.m`: behavioral preference/saccade analysis, report and figures in `report_000`.
- `scripts/main_001_anova_lda.m`: ANOVA + LDA outputs in `processed` and report in `report_001`.
- `scripts/main_002_states.m`: state decoding outputs and report in `report_002`.
- `scripts/main_003_crossdecoding.m`: CCGP/shCCGP analyses and report in `report_003`.
- `scripts/main_004_unitstability.m`: unit stability figures/statistics and report in `report_004`.

## Path Handling

Most scripts use this pattern:

- `f = mfilename('fullpath')`
- if empty, fallback to `pwd`
- otherwise compute project root from script location
- build project-relative paths from that root

This means scripts do not require a specific MATLAB working directory.

## Expected Folder Layout

At project root:

- `data_final/` (generated pooled `*_pool.mat` files)
- `processed/` (analysis outputs)
- `scripts/` (pipeline scripts)
- `report_000/` ... `report_004/` (generated reports/figures)

## Typical Usage

1. Open MATLAB and set current folder to project root (`PAV2INS`).
2. Run full analysis pipeline:
   - `run('scripts/run_all.m')`

## Notes

- `scripts/run_all.m` removes existing report folders before running scripts.
- Report files are overwritten (`'w'` mode) on each run.
- If you run individual scripts, each script creates its own report folder if needed.
