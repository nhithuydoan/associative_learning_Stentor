# Data Dictionary

Contraction responses in *Stentor coeruleus*, scored from video. Each row records whether a single cell contracted to a single stimulus presentation. The data covers associative sensitization experiments using weak-strong tap pairings at various temporal parameters, plus arousal and sensitization controls.

---

## Files at a glance

| File | Grain (one row =) | Use it for |
|---|---|---|
| `ws_single.csv` | one cell × one stimulus | Weak-strong pairings and timing conditions |
| `control.csv` | one cell × one stimulus | Arousal and sensitization controls |

---

## Column definitions

The two files share a common core schema. `control.csv` adds two extra columns (`fold_name` and `trial`).

### Session and protocol

| Column | Type | Description |
|---|---|---|
| `fold_name` | text | Folder ID / experiment timestamp (`control.csv` only) |
| `condition` | text | Experimental condition (see condition key below) |
| `run` | integer | Replicate run within a condition |
| `isi` | integer | Inter-stimulus interval between weak and strong tap (seconds) |
| `iti` | integer | Inter-trial interval between strong tap and next weak-strong pair (seconds) |
| `trial` | integer | Trial number within a run (`control.csv` only) |
| `stimulus` | integer | Stimulus number within a run (1–60) |

### Cell

| Column | Type | Description |
|---|---|---|
| `cell_number` | integer | Individual cell index within a run |

### Response

| Column | Type | Description |
|---|---|---|
| `contract` | binary | Whether the cell contracted (1.0) or not (0.0) |

---

## Experimental conditions

Default parameters: ISI = 1 s, ITI = 59 s unless specified in the condition name.

### Core conditions (`ws_single.csv`)

| Prefix | Name | Description |
|---|---|---|
| `ws_*` | Weak-Strong | Primary experimental condition — weak tap (CS) paired with strong tap (US) |
| `ww_*` | Weak-Weak | Tests US strength and CS responses |
| `hab_ws_*` | Habituation then Weak-Strong | Tests necessity of prehabituation for sensitization |
| `w_*` | Weak-only | CS presented without US (60 s between taps) |
| `s_*` | Strong-only | Strength of the US alone (60 s between taps) |

### Modified timing conditions (`ws_single.csv`)

| Condition | ISI | ITI | Notes |
|---|---|---|---|
| `ws_ISI1_ITI45` | 1 s | 45 s | |
| `ws_ISI1_ITI59` | 1 s | 59 s | Default timing |
| `ws_ISI10_ITI45` | 10 s | 45 s | |
| `ws_ISI10_ITI59` | 10 s | 59 s | |
| `ws_ISI10_ITI590` | 10 s | 590 s | Scaled ×10, 20 stimuli |
| `ws_ISI20_ITI45` | 20 s | 45 s | |

### Controls (`control.csv`)

| Condition | Description |
|---|---|
| `s_ISI1_ITI45` | Strong-only control |
| `wswww_ISI1_ITI45` | One strong interleaved with weak stimuli (arousal control) |
| `wsssw_ISI1_ITI45` | Multiple strong followed by weak (sensitization control) |

---

## Data selection criteria

- Only included cells that remained anchored throughout experiments.
- Excluded datasets where cells showed atypical responses (weak tap response > strong tap response) and significant inconsistency between stages despite similar experimental conditions.

---

## The files

### `ws_single.csv` — weak-strong pairings

Single-cell contraction responses across all core experimental conditions and timing variants. 216,740 rows, 7 columns.

### `control.csv` — arousal and sensitization controls

Single-cell contraction responses for patterned stimulus controls that test whether the observed sensitization can be explained by arousal or sensitization alone. 61,380 rows, 9 columns.
