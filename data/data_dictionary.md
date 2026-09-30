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
| `fold_name` | text | Folder ID  |
| `condition` | text | Experimental condition (see condition key below) |
| `run` | integer | Replicate run within a condition |
| `isi` | integer | Inter-stimulus interval between weak and strong tap ( in seconds) |
| `iti` | integer | Inter-trial interval between strong tap and next weak-strong pair (seconds) |
| `trial` | integer | Trial number within a run (`control.csv` only) |
| `stimulus` | integer | Stimulus number within a run (1–60) |
| `cell_number` | integer | Individual cell index within a run |
| `contract` | binary | Whether the cell contracted (1.0) or not (0.0) |

### Experimental conditions

Default parameters: ISI = 1 s, ITI = 59 s unless specified in the condition name.

#### Core conditions (`ws_single.csv`)

| Prefix | Name | Description |
|---|---|---|
| `ws_*` | Weak-Strong | Primary experimental condition: weak tap (CS) paired with strong tap (US) with a specific ISI and ITI |
| `ww_*` | Weak-Weak | Weak tap (CS) paired with another weak tap (US)|
| `hab_ws_*` | Habituation then Weak-Strong | Tests necessity of prehabituation for sensitization |
| `w_*` | Weak-only | CS presented without US (60 s between taps) |
| `s_*` | Strong-only | Strength of the US alone (60 s between taps) |

#### Modified timing conditions (`ws_single.csv`)

| Condition | ISI | ITI | Notes |
|---|---|---|---|
| `ws_ISI1_ITI45` | 1 s | 45 s | |
| `ws_ISI1_ITI59` | 1 s | 59 s | Default timing |
| `ws_ISI10_ITI45` | 10 s | 45 s | |
| `ws_ISI10_ITI59` | 10 s | 59 s | |
| `ws_ISI10_ITI590` | 10 s | 590 s | Scaled ×10, 20 stimuli |

#### Controls (`control.csv`)

| Condition | Description |
|---|---|
| `s_ISI1_ITI45` | Strong-only control |
| `wswww_ISI1_ITI45` | One strong interleaved with weak stimuli (arousal control) |
| `wsssw_ISI1_ITI45` | Multiple strong followed by weak (sensitization control) |

---

## Followup: common-test paradigm

Data from followup experiments designed to distinguish associative learning from sensitization using a common-test design. Each experiment has three phases: pre-habituation (weak taps only), training (weak-strong pairs for experimental group, weak + strong taps for control group), and a single weak-tap test. If the two groups respond differently at the common test, the difference reflects the pairing during training rather than non-associative sensitization.

### Files

dataset_followup_1.csv
dataset_followup_2.csv

Similar structure to control.csv and ws_single.csv, with added columns:

| Column | Type | Description |
|---|---|---|
| `folder` | text | Experiment timestamp |
| `phase` | text | Experimental phase: `hab` (pre-habituation), `train` (training), or `test` (common test) |
| `break12` | float | Delay between habituation and training (in seconds) |
| `break23` | float | Delay between training and test (in seconds) |

### Conditions

| Condition | Training protocol |
|---|---|
| `experimental` | 5 weak-strong pairs (CS-US pairing) |
| `control` | 1 weak tap followed by 4 strong taps (sensitization control) |

---

## Data selection criteria

- Only included cells that remained anchored throughout experiments.
- Excluded datasets where cells showed atypical responses (weak tap response > strong tap response) and significant inconsistency between stages despite similar experimental conditions.
