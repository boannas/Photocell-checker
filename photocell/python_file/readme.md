# Photocell Record for Monitor Frequency Check

This project is designed to **validate the refresh rate of a monitor** for **SSVEP (Steady-State Visual Evoked Potentials)** experiments. By using a **photocell sensor** alongside **LSL markers**, it records and analyzes the screen refresh stability.

---

## Project Structure

The project contains:
- **Jupyter Notebooks** for recording, processing, and visualization
- **Python Scripts** for modular processing and recording
- **Data folders** organized by subject ID

---

## Workflow Overview

### 1. Recording (`photocell_record.ipynb`)
- Records **photocell values** and **LSL markers** from MATLAB-based stimuli.
- Saves output as `.csv` in the `raw_data` folder.

### 2. Processing (`process_raw.ipynb`)
- Aligns timestamps between **LDR signals** and **LSL markers**.
- Outputs aligned data to the `process_data` folder.

### 3. Visualization (`plot.ipynb`)
- Plots and analyzes the **monitor’s frequency stability** using the processed data.

---

## Python Script Functions

### `photocell.py`
- Records:
  - Photocell sensor values
  - LSL markers
- Outputs raw `.csv` files to the `raw_data` folder

### `process_freq.py`
- Handles:
  - Preprocessing
  - Frequency computation
  - Timestamp alignment
  - Visualization utilities

---

## Folder Structure

```plaintext
raw_data/
   └── sXX/
       └── sXX_block_N_raw.csv

process_data/
   └── sXX/
       └── sXX_block_N_process.csv

functions/
   ├── photocell.py
   └── process_freq.py

photocell_record.ipynb
process_raw.ipynb
plot.ipynb
```

- `sXX`: Subject folder (e.g., `s1`, `s2`)
- `block_N`: Data file for a specific block number (e.g., `block_1`, `block_2`)

---