# Water Quality Prediction using Bayesian Belief Network
## Assignment 2 - PS10

### Overview
This solution implements a Bayesian Belief Network (BBN) for water quality prediction, addressing all four questions in the assignment:
1. Construct a Bayesian Belief Network for water quality data
2. Predict water potability for given attribute values
3. Infer probability for given attribute values including potability
4. Find conditional probability of water quality being good under specific conditions

### Files Included
- `water_quality_bbn.py` — Main Python implementation
- `water_quality_bbn.ipynb` — Jupyter Notebook version of the same implementation
- `water_potability.csv` — Water quality dataset
- `inputPS10.txt` — Input file for testing (Q2 attribute values)
- `outputPS10.txt` — Generated output file (created on run)
- `designPS10_group90.md` — Design document
- `requirements.txt` — Python dependencies
- `README.md` — This file

### Prerequisites
- Python 3.8 or higher
- Required packages (install with `pip install -r requirements.txt`):
  - pandas
  - numpy
  - pgmpy

### Data File
The dataset `water_potability.csv` must be present in the same directory. It should contain the following columns:
- ph
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic_carbon
- Trihalomethanes
- Turbidity
- Potability (0 = Not potable, 1 = Potable)

Rows with missing values are automatically dropped during preprocessing.

### Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data:**
   - Ensure `water_potability.csv` is in the same directory
   - Ensure the CSV has all required columns

3. **Run the program:**
   ```bash
   python water_quality_bbn.py
   ```

4. **Input file format (`inputPS10.txt`):**
   One attribute per line — key and value separated by a space. Lines starting with `#` are treated as comments.
   ```
   # Question 2 input
   ph 3.72
   Hardness 204.89
   Solids 20791.32
   Chloramines 7.3
   Sulfate 368.5
   Conductivity 564.30
   Turbidity 2.96
   ```
   The program reads Q2 attribute values from this file. Q3 reuses the same attribute values and Q4 uses fixed categorical conditions (low ph, high hardness, high solids).

5. **Output file (`outputPS10.txt`):**
   - Automatically **deleted and recreated** on every run to prevent stale data.
   - Contains clearly labelled results for all three questions (Q2, Q3, Q4).

### Implementation Details

**Bayesian Belief Network Structure:**
- All water quality attributes (ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity) are parent nodes
- Potability is the child node influenced by all attributes
- This structure captures the influence of each water quality parameter on potability

**Data Discretization:**
- Continuous variables are discretized into three categories: low, medium, high
- Uses quantile-based binning (33rd and 66th percentiles)
- Necessary for Bayesian Network learning with discrete variables

**Parameter Learning:**
- Uses Bayesian Estimator with Dirichlet priors for robust parameter learning
- Falls back to Maximum Likelihood Estimation if needed
- Handles missing data gracefully

**Inference:**
- Uses Variable Elimination algorithm for exact inference
- Supports both prediction and probability queries
- Handles conditional probability queries

### Key Features
- Modular and well-documented code
- Handles missing values (NaN rows dropped after discretization)
- Output file deleted before each run — no stale results
- All three questions (Q2, Q3, Q4) are run and written in a single execution
- Graceful fallback from Bayesian Estimator to MLE if fitting fails

### Notes
- The actual evaluation will use different input files — no values are hardcoded.
- Ensure `water_potability.csv` and `inputPS10.txt` are in the same directory as the script.
- The Jupyter Notebook (`water_quality_bbn.ipynb`) mirrors the `.py` implementation and can be used for interactive exploration.


# Changelog, Aug 6

| # | Issue / Task | Fix / Action |
|---|-------------|-------------|
| 1 | **Wrong data file name** — `water_quality.csv` hardcoded in `main()` | Changed to `water_potability.csv` to match the actual file in the workspace |
| 2 | **Quantile scale bug** — `percentiles = [33.33, 66.67]` passed to `quantile()` which expects values in [0, 1] | Fixed to `[0.3333, 0.6667]` |
| 3 | **NaN values crash** — `water_potability.csv` has missing values (~15%) that cause pgmpy `fit()` to fail | Added `dropna()` + `reset_index(drop=True)` after discretization in `discretize_data()` |
| 4 | **Only one question was run** — `main()` used `if/elif/else` to run only Q2, Q3 *or* Q4 based on input structure | Restructured to always run all three questions (Q2 → Q3 → Q4) sequentially and write all results to `outputPS10.txt` |
| 5 | **Q2 output missing summary sentence** | Added percentage line matching the sample output format |
| 6 | **Output helper refactor** | Added `append_output_file()` and `_write_query_result()` to cleanly separate write vs. append logic; replaced the single overwriting `write_output_file()` |
| 7 | **Stale output file across runs** | Added `import os` and `os.remove(output_file)` at the start of every run so the output file is deleted and fully recreated — no leftover data from previous executions |
| 8 | **Notebook out of sync with .py** | Synced all of the above fixes (items 1–7) into the corresponding notebook cells in `water_quality_bbn.ipynb` |

---

| # | Issue / Task | Fix / Action |
|---|-------------|-------------|
| 9 | **`BayesianNetwork` deprecated** — pgmpy raised a deprecation error that was swallowed by the broad `except` in `main()`, printing a misleading "file not found" message | Replaced `BayesianNetwork` with `DiscreteBayesianNetwork` in both import and `construct_bbn()` |
| 10 | **`fit()` API changed** — new pgmpy no longer accepts `prior_type` as a keyword on `fit()`, and requires an initialized estimator instance; `DiscreteMLE` not present in installed version | Replaced `fit()` calls with `BayesianEstimator(model, data).get_parameters()` + `model.add_cpds()` as the primary path (version-agnostic); added three further fallbacks in cascade order |
| 11 | **Potability dtype drift** — `dropna()` can silently convert integer columns to `float64` when NaNs were present elsewhere in the row, breaking pgmpy's discrete CPD checks | Added `.astype(int)` cast on `Potability` after `dropna()` in `discretize_data()` |
| 12 | **Dummy outputs — wrong network direction** — `attributes → Potability` topology creates a joint CPD with 3⁹ × 2 = 39,366 parameters; with ~2011 samples almost every cell is empty and Dirichlet smoothing forces all predictions to ~0.5 | Flipped to generative Naïve Bayes direction `Potability → attributes`: 56 parameters total, fully data-driven, non-uniform probabilities |
| 13 | **Bin edge inconsistency** — `get_category()` recomputed 33rd/67th percentiles from the raw data (different row set than post-dropna training data), causing boundary drift between train and query time | Added `self._bin_edges` dict; `pd.qcut` called with `retbins=True` to capture exact bin edges; `get_category()` uses stored edges with raw-percentile fallback |
| 14 | **No model validation** — a silently broken CPD table would produce wrong inference results with no warning | Added `model.check_model()` after `add_cpds()` to verify CPD consistency with the graph structure |

---

## Dead Code Removal

| # | Removed | Reason |
|---|---------|--------|
| 15 | `import numpy as np` | `np` was never referenced anywhere in the file |
| 16 | `write_output_file()` function | Defined but never called — all writes go through `append_output_file()` + inline `open('w')` |
| 17 | `input_context` parameter on `append_output_file()` | Received but never read inside the function body |
| 18 | `'prediction'` key in `predict_potability()` return dict | `_write_query_result()` only reads `prob_0` / `prob_1`; the key was computed and silently discarded |
