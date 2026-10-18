# Data Dictionary

Contraction responses in *Stentor coeruleus*, scored from video. Each row records whether a single cell contracted to a single stimulus presentation.

---

## Main experiment

### Files

ws_single.csv

control.csv

#### Data structure

| Column | Type | Description |
|---|---|---|
| `fold_name` | text | Folder ID (`control.csv` only) |
| `condition` | text | Experimental condition (see condition key below) |
| `run` | integer | Replicate run within a condition (`control.csv` includes both `fold_name` and `run`; `ws_single.csv` uses `run` only) |
| `isi` | integer | Inter-stimulus interval between weak and strong tap (seconds) |
| `iti` | integer | Inter-trial interval between strong tap and next weak-strong pair (seconds) |
| `trial` | integer | Trial number within a run (`control.csv` only) |
| `stimulus` | integer | Stimulus number within a run (1–60) |
| `cell_number` | integer | Individual cell index within a run |
| `contract` | binary | Whether the cell contracted (1.0) or not (0.0). NaN where cell contracts prior to the stimulus |

### Experimental conditions

Default parameters: ISI = 1s, ITI = 59s unless specified in the condition name, ISI, and ITI columns.

#### Core conditions (`ws_single.csv`)

| Prefix | Name | Description |
|---|---|---|
| `ws_*` | Weak-Strong | Primary experimental condition: weak tap (CS) paired with strong tap (US) with a specific ISI and ITI |
| `ww_*` | Weak-Weak | Weak tap (CS) paired with another weak tap (US) |
| `hab_ws_*` | Habituation then Weak-Strong | Tests necessity of period of habituation before pairing |
| `w_*` | Weak-only | CS presented without US |
| `s_*` | Strong-only | US presented alone |

#### Modified timing conditions (`ws_single.csv`)

| Condition | ISI | ITI | Notes |
|---|---|---|---|
| `ws_ISI1_ITI45` | 1 s | 45 s | |
| `ws_ISI1_ITI59` | 1 s | 59 s | Default timing |
| `ws_ISI10_ITI45` | 10 s | 45 s | |
| `ws_ISI10_ITI59` | 10 s | 59 s | |
| `ws_ISI10_ITI590` | 10 s | 590 s | |


#### Controls (`control.csv`)

| Condition | Description |
|---|---|
| `s_ISI1_ITI45` | Strong-only control |
| `wswww_ISI1_ITI45` | One strong interleaved with weak stimuli (arousal control) |
| `wsssw_ISI1_ITI45` | A series of three strong taps followed by weak (sensitization control) |

Three additional conditions (`s_ISI1_ITI59`, `w_ISI20_ITI45` and `hab_ws_ISI20_ITI45`, `ww_ISI1_ITI59`) are not part of the main analysis and do not change the findings we presented but are included to support future reanalysis.

---

## Followup: common-test paradigm

Data from follow-up experiments designed to distinguish associative learning from sensitization using a common-test design. Each experiment has three phases: pre-habituation (weak taps only), training (weak-strong pairs for experimental group, weak + strong taps for control group), and a single weak-tap test. If the two groups respond differently at the common test, the difference reflects the pairing during training rather than sensitization.

### Files

dataset_followup_1.csv
dataset_followup_2.csv

Similar structure to control.csv and ws_single.csv, with added columns:

| Column | Type | Description |
|---|---|---|
| `phase` | text | Experimental phase: `hab` (pre-habituation), `train` (training), or `test` (common test) |
| `break12` | float | Delay between habituation and training (seconds) |
| `break23` | float | Delay between training and test (seconds) |

### Conditions

| Condition | Training protocol |
|---|---|
| `experimental` | 5 weak-strong pairs (CS-US pairing) |
| `control` | 1 weak tap followed by 4 strong taps (sensitization control) |

---

## Data selection criteria

- Only included cells that remained anchored throughout experiments.