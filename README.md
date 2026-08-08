# Water Quality Prediction using Bayesian Belief Network
## Assignment 2 - PS10

### Overview
This solution implements a Bayesian Belief Network (BBN) for water quality prediction, addressing all four questions in the assignment:
1. Construct a Bayesian Belief Network for water quality data
2. Predict water potability for given attribute values
3. Infer probability for given attribute values including potability
4. Find conditional probability of water quality being good under specific conditions

### Files Included
- `water_quality_bbn.py` - Main Python implementation
- `inputPS10.txt` - Sample input file for testing
- `requirements.txt` - Python dependencies
- `README.md` - This file

### Prerequisites
- Python 3.8 or higher
- Required packages (install with `pip install -r requirements.txt`):
  - pandas
  - numpy
  - pgmpy

### Data File Required
You need to provide the water quality dataset as `water_quality.csv` in the same directory. The CSV file should contain the following columns:
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

### Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your data:**
   - Place your water quality dataset as `water_potability.csv` in the same directory
   - Ensure the CSV has all required columns

3. **Run the program:**
   ```bash
   python water_quality_bbn.py
   ```

4. **Input file format (inputPS10.txt):**
   - For Question 2 (Prediction):
     ```
     ph 3.72
     Hardness 204.89
     Solids 20791.32
     Chloramines 7.3
     Sulfate 368.5
     Conductivity 564.30
     Turbidity 2.96
     ```
   
   - For Question 3 (Inference with Potability):
     ```
     Hardness 248.0
     Solids 28749
     Chloramines 7.5
     Sulfate 393
     Conductivity 283
     Organic_carbon 13.78
     Trihalomethanes 84.6
     Turbidity 2.67
     Potability 1
     ```

5. **Output file (outputPS10.txt):**
   - The program will generate `outputPS10.txt` with the results
   - Format matches the sample output specified in the assignment

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
- Error handling for missing files and invalid data
- Flexible input/output file handling
- Supports all three query types from the assignment
- Automatic detection of query type based on input

### Notes
- The actual evaluation will use different input files
- Do not hardcode any values in the code
- Ensure the input file format matches the specifications
- The program automatically detects the query type based on input content
