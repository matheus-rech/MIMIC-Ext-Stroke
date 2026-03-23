# CLAUDE.md — Stroke Digital Twin Pipeline

## What This Is
Stroke digital twin pipeline executing against MIMIC-IV 3.1.
Implements the full Grieves triad: physical entity (MIMIC-IV stroke patients), virtual entity (BN+GAN generative models), and bidirectional connection layer (UQ, drift detection, feedback loops).

## Critical Constraints
- **Never modify** source MIMIC-IV or eICU data on the external drive
- **PHI compliance** — all outputs must be PHI-safe
- Python venv at `.venv/` (Python 3.12)

## Pipeline Status (updated 2026-03-23)

| Component | Status | Key Output |
|-----------|--------|------------|
| Cohort extraction | **Re-running** | ~10,000 patients (ICD codes corrected) |
| Static features | **Done** | Demographics, comorbidities, labs, subtype |
| Timeseries | **Done** | 11 channels, 72h window |
| Preprocessing | **Done** | Train/val/test split, imputation (train-fitted), normalization |
| Bayesian Network | **Done** | Static profile generation |
| DGAN (BCE + WGAN-GP) | **Done** | Temporal trajectory generation |
| CTGAN | **Done** | Static baseline |
| TVAE | **Done** | Collapsed — needs more epochs |
| Evaluation | **Done** | 12-metric framework |
| Connection layer | **Done** | TwinState, UQ, drift detection |
| eICU validation | **Ready** | Script + SQL ready, data downloaded |

## ICD Phenotype Definition

### Stroke (corrected 2026-03-23)
```
ICD-9:  430 (SAH), 431 (ICH), 432 (other ICH), 433 (precerebral occlusion),
        434 (cerebral occlusion), 435 (TIA), 436 (acute CVD)
ICD-10: I60 (SAH), I61 (ICH), I62 (other nontraumatic ICH),
        I63 (cerebral infarction), I64 (stroke unspecified),
        I65 (precerebral occlusion), I66 (cerebral occlusion),
        I67 (other CVD), G45 (TIA)
```

### Subtypes
- ischemic (I63, 433-434), ich (I61-I62, 431-432), sah (I60, 430), tia (G45, 435), other (I64, 436)

### Comorbidities
- hypertension (I10, 401), diabetes (E11, 250), afib (I48, 4273)
- dyslipidemia (E78, 272), CKD (N18, 585), CAD (I21/I25, 410/412)

## Data Paths
- **MIMIC-IV**: `/Volumes/VMDrive/Databases/` (external drive, 31 files, 10 GB)
- **eICU-CRD**: `/Volumes/VMDrive/Databases2/` (external drive, 31 files, 5.7 GB)

## eICU External Validation
```bash
python scripts/run_eicu_validation.py \
  --eicu-path /Volumes/VMDrive/Databases2/ \
  --mimic-cohort ./outputs/cohort/stroke_cohort.parquet
```

## src/ Package Structure
```
src/
  connection/   # Grieves triad: state, UQ, drift
  data/         # extract, features, preprocess, eicu_validation
  models/       # bayesian_net, dgan_model (BCE+WGAN-GP), ctgan_baseline, hybrid
  evaluation/   # fidelity, temporal, privacy, utility, clinical_rules, rubins_rules
  simulation/   # scenario_simulator, counterfactual
sql/            # 4 DuckDB SQL queries (cohort, static, timeseries, eICU)
scripts/        # run_eicu_validation, optimize_dgan, generate_all_outputs
notebooks/      # 4 notebooks (EDA, cohort summary, evaluation, demo)
tests/          # 12 test files
```

## Audit Fixes Applied (2026-03-23)
- Removed `last_careunit` (absent in MIMIC-IV 3.1)
- Added missing ICD codes: ICD-9 430-432, ICD-10 I62/I64
- Added non-invasive BP itemids (220179/220180/220181)
- Fixed eICU short patterns (`%sah%`/`%tia%`) → full diagnostic strings
- Fixed SQL paths to use `Path(__file__)` resolution
- Removed dead `use_sigmoid` parameter from Discriminator
- Documented GP metadata interpolation design decision
- Hybrid model forwards WGAN-GP params to DGAN
- Config updated with external drive paths
- Preprocessing: fit imputation on train, apply to all splits
- `.gitignore`: fixed `data/` → `/data/` to not block `src/data/`

## Sister Pipeline
- **TBI**: `~/mimic-tbi/` (github.com/matheus-rech/mimic-tbi)
- **Skill docs**: `~/.claude/skills/digital-twin-architect/references/`
