## Water Quality

*Read through this entire document very carefully before you start!*

## Problem Statement

Water quality data is provided along with this file and contains the following attributes

**Attributes Information:**

1. ph: pH of water (0 to 14).
2. Hardness: Capacity of water to precipitate soap in mg/L.
3. Solids: Total dissolved solids in ppm.
4. Chloramines: Amount of Chloramines in ppm.
5. Sulfate: Amount of Sulfates dissolved in mg/L.
6. Conductivity: Electrical conductivity of water in μS/cm.
7. Organic_carbon: Amount of organic carbon in ppm.
8. Trihalomethanes: Amount of Trihalomethanes in μg/L.
9. Turbidity: Measure of light emitting property of water in NTU.
10. Potability: Indicates if water is safe for human consumption. Potable - 1 and Not potable - 0

You are required to do the following:

### Question 1: Python

1. Construct a Bayesian Belief Network for the given data.

   Use appropriate methods to predict the following:

2. Predict the water quality for the following data:

   | ph | Hardness | Solids | Chloramines | Sulfate | Conductivity | Turbidity |
   |------|----------|----------|-------------|---------|---------------|-----------|
   | 3.72 | 204.89 | 20791.32 | 7.3 | 368.5 | 564.30 | 2.96 |

3. Infer the probability for the data with the following properties:

   | Hardness | Solids | Chloramines | Sulfate | Conductivity | Organic carbon | Trihalomethanes | Turbidity | Potability |
   |----------|--------|-------------|---------|---------------|-----------------|------------------|-----------|------------|
   | 248. | 28749 | 7.5 | 393 | 283 | 13.78 | 84.6 | 2.67 | 1 |

4. Find the probability of the quality of water being good and the attributes take the following values: low ph, high in hardness, with high presence of solids, and other chemicals.

### Sample Input

1. Predict the water quality for the following data:

   | ph | Hardness | Solids | Chloramines | Sulfate | Conductivity | Turbidity |
   |------|----------|----------|-------------|---------|---------------|-----------|
   | 3.72 | 204.89 | 20791.32 | 7.3 | 368.5 | 564.30 | 2.96 |

*Note that the input/output data shown here is only for understanding and testing, the actual file used for evaluation will be different.*

### Sample Output

| Potability | 0 | 1 |
|---|---|---|
| Probability | 0.571838 | 0.428162 |

Probability of the water being good is 42.81% when considered low ph value and remaining variables high value from given dataset

*Note that the input/output data shown here is only for understanding and testing, the actual file used for evaluation will be different.*

**Display the output in outputPSXX.txt.**

### Sample Input 2

1. Infer the probability for the data with the following properties:

   | Hardness | Solids | Chloramines | Sulfate | Conductivity | Organic carbon | Trihalomethanes | Turbidity | Potability |
   |----------|--------|-------------|---------|---------------|-----------------|------------------|-----------|------------|
   | 248. | 28749 | 7.5 | 393 | 283 | 13.78 | 84.6 | 2.67 | 1 |

*Note that the input/output data shown here is only for understanding and testing, the actual file used for evaluation will be different.*

### Sample Output 2

| Potability | 0 | 1 |
|---|---|---|
| Probability | 0.65359 | 0.34751 |

Probability of the water being good is 34.75% when the given attribute values are given from the dataset

*Note that the input/output data shown here is only for understanding and testing, the actual file used for evaluation will be different.*

## Deliverables

- PDF document `designPSXX_<group id>.pdf` detailing your solution.
- `inputPSXX.txt` file used for testing
- `outputPSXX.txt` file generated while testing
- `.py` file containing the python code. Do not fragment your code into multiple files


## Instructions

- It is compulsory to make use of the data structure(s) / algorithms mentioned in the problem statement.
- Ensure that all data structure insert and delete operations throw appropriate messages when their capacity is empty or full. Also ensure basic error handling is implemented.
- Ensure that the input, prompt and output file guidelines are adhered to. 
- The input, prompt and output samples shown here are only a representation of the syntax to be used. Actual files used to evaluate the submissions will be different. Hence, do not hard code any values into the code.
- Please note that the design document must include:
  - One alternate way of modeling the problem with the performance implications.
  - Writing a good technical report and well documented code is an art. Your report cannot exceed 4 pages. Your code must be modular and quite well documented.

## Evaluation

- Grading will depend on:
  - Fully executable code with all functionality working as expected
    - Well-structured and commented code
    - Accuracy of the design document.
    - Every bug in the functionality will have negative marking.
    - Marks will be deducted if your program fails to read the input file used for evaluation due to change / deviation from the required syntax.
