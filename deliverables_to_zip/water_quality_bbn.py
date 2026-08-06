"""
Water Quality Prediction using Bayesian Belief Network
Assignment 2 - PS10
"""

import os
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import warnings
warnings.filterwarnings('ignore')

class WaterQualityBBN:
    """
    Bayesian Belief Network for Water Quality Prediction
    """
    
    def __init__(self, data_file=None):
        """
        Initialize the BBN with water quality data
        
        Args:
            data_file: Path to CSV file containing water quality data
        """
        self.data = None
        self.model = None
        self.infer = None
        self.discretized_data = None
        # Stores qcut bin edges per column so query-time categorization
        # uses the identical boundaries as training-time discretization.
        self._bin_edges = {}

        if data_file:
            self.load_data(data_file)
    
    def load_data(self, data_file):
        """
        Load water quality data from CSV file
        
        Args:
            data_file: Path to CSV file
        """
        try:
            self.data = pd.read_csv(data_file)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            print(f"Columns: {list(self.data.columns)}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file {data_file} not found")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def discretize_data(self):
        """
        Discretize continuous variables into categories (low, medium, high)
        This is necessary for Bayesian Network learning
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        self.discretized_data = self.data.copy()
        
        # Columns to discretize (excluding Potability which is already binary)
        continuous_cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                          'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
        
        for col in continuous_cols:
            if col in self.discretized_data.columns:
                # Discretize into 3 categories: low, medium, high
                # retbins=True captures the exact bin edges for consistent query-time use
                discretized_col, bins = pd.qcut(
                    self.discretized_data[col],
                    q=3,
                    labels=['low', 'medium', 'high'],
                    duplicates='drop',
                    retbins=True
                )
                self.discretized_data[col] = discretized_col
                self._bin_edges[col] = bins  # save edges: [min, q1, q2, max]
        
        # Drop rows with NaN values (pgmpy cannot handle missing values)
        self.discretized_data.dropna(inplace=True)
        self.discretized_data.reset_index(drop=True, inplace=True)
        
        # Ensure Potability stays as integer after dropna (avoids float drift)
        self.discretized_data['Potability'] = self.discretized_data['Potability'].astype(int)
        
        print("Data discretized into categories: low, medium, high")
        print(f"Discretized data shape after dropping NaN: {self.discretized_data.shape}")
    
    def construct_bbn(self):
        """
        Construct Bayesian Belief Network structure.

        Uses a GENERATIVE Naïve Bayes topology:
            Potability → each attribute

        Why: the discriminative direction (attributes → Potability) produces a
        joint CPD for Potability with 3^9 × 2 = 39,366 parameters.  With ~2011
        training rows almost every cell is empty and Dirichlet smoothing drives
        all predictions to ~0.5.

        Flipping to the generative direction yields:
            P(Potability)          — 2 parameters
            P(attribute|Potability) — 3×2 = 6 parameters × 9 attributes = 54
        Total: 56 parameters — tractable and data-driven.

        Inference: Variable Elimination computes P(Potability | observed
        attributes) via Bayes' rule, identical in result to the discriminative
        query.
        """
        if self.discretized_data is None:
            self.discretize_data()

        attribute_nodes = [
            'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
        ]

        # Generative direction: Potability is the root/parent of every attribute
        edges = [('Potability', attr) for attr in attribute_nodes
                 if attr in self.discretized_data.columns]

        self.model = DiscreteBayesianNetwork(edges)
        print(f"Bayesian Network structure defined ({len(edges)} edges, "
              f"generative Naïve Bayes topology)")
    
    def learn_parameters(self):
        """
        Learn CPD (Conditional Probability Distribution) parameters from data.

        Uses a cascade of approaches to handle different pgmpy versions:
          1. BayesianEstimator.get_parameters() + add_cpds()  — version-agnostic
          2. MaximumLikelihoodEstimator(model, data) as instance — new API
          3. DiscreteMLE() — latest pgmpy API
          4. MaximumLikelihoodEstimator class (uninstantiated) — old API
        """
        if self.model is None:
            self.construct_bbn()

        errors = []

        # Attempt 1: BayesianEstimator.get_parameters() → add_cpds()
        # Bypasses fit() entirely; works across all pgmpy versions.
        try:
            est = BayesianEstimator(self.model, self.discretized_data)
            cpds = est.get_parameters(prior_type='dirichlet', pseudo_counts=1)
            self.model.add_cpds(*cpds)
            # Validate the model — ensures CPDs are consistent with structure
            if self.model.check_model():
                print("Parameters learned successfully (BayesianEstimator.get_parameters)")
                print("Model validation passed — CPDs are consistent")
            self.infer = VariableElimination(self.model)
            return
        except Exception as e:
            errors.append(f"BayesianEstimator.get_parameters: {e}")

        # Attempt 2: MaximumLikelihoodEstimator as an initialized instance (new API)
        try:
            est = MaximumLikelihoodEstimator(self.model, self.discretized_data)
            self.model.fit(self.discretized_data, estimator=est)
            print("Parameters learned successfully (MaximumLikelihoodEstimator instance)")
            self.infer = VariableElimination(self.model)
            return
        except Exception as e:
            errors.append(f"MaximumLikelihoodEstimator instance: {e}")

        # Attempt 3: DiscreteMLE() — latest pgmpy builds
        try:
            from pgmpy.estimators import DiscreteMLE
            self.model.fit(self.discretized_data, estimator=DiscreteMLE())
            print("Parameters learned successfully (DiscreteMLE)")
            self.infer = VariableElimination(self.model)
            return
        except Exception as e:
            errors.append(f"DiscreteMLE: {e}")

        # Attempt 4: Uninstantiated class — old pgmpy API (< 0.1.24)
        try:
            self.model.fit(self.discretized_data, estimator=MaximumLikelihoodEstimator)
            print("Parameters learned successfully (old MLE class API)")
            self.infer = VariableElimination(self.model)
            return
        except Exception as e:
            errors.append(f"old MLE class: {e}")

        raise Exception(f"Failed to learn parameters. Tried: {'; '.join(errors)}")
    
    def get_category(self, value, column):
        """
        Map a continuous value to its discretized category (low/medium/high).

        Uses the exact bin edges captured by pd.qcut during training so that
        query-time categorization is identical to training-time discretization.
        Falls back to raw percentile calculation if edges were not stored.
        """
        if self.data is None:
            raise ValueError("No data loaded")

        if column in self._bin_edges:
            # Use the same edges as pd.qcut used during training
            bins = self._bin_edges[column]
            # bins = [min_edge, q1_edge, q2_edge, max_edge]
            if value <= bins[1]:
                return 'low'
            elif value <= bins[2]:
                return 'medium'
            else:
                return 'high'
        else:
            # Fallback: compute from raw data percentiles
            col_data = self.data[column].dropna()
            q1, q2 = col_data.quantile([0.3333, 0.6667])
            if value <= q1:
                return 'low'
            elif value <= q2:
                return 'medium'
            else:
                return 'high'
    
    def predict_potability(self, input_dict):
        """
        Predict water potability for given input values
        
        Args:
            input_dict: Dictionary with attribute names and values
            
        Returns:
            Dictionary with potability prediction and probabilities
        """
        if self.infer is None:
            self.learn_parameters()
        
        # Convert continuous values to categories
        evidence = {}
        for key, value in input_dict.items():
            if key in self.data.columns and key != 'Potability':
                evidence[key] = self.get_category(value, key)
        
        # Perform inference
        try:
            result = self.infer.query(variables=['Potability'], evidence=evidence)
            prob_0 = result.values[0]  # Not potable
            prob_1 = result.values[1]  # Potable
            
            return {
                'prob_0': prob_0,
                'prob_1': prob_1
            }
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return {
                'prob_0': 0.5,
                'prob_1': 0.5
            }
    
    def infer_probability(self, input_dict):
        """
        Infer probability for given attribute values including Potability
        
        Args:
            input_dict: Dictionary with all attribute names and values
            
        Returns:
            Dictionary with potability probabilities
        """
        if self.infer is None:
            self.learn_parameters()
        
        # Convert continuous values to categories
        evidence = {}
        for key, value in input_dict.items():
            if key in self.data.columns and key != 'Potability':
                evidence[key] = self.get_category(value, key)
        
        # Perform inference
        try:
            result = self.infer.query(variables=['Potability'], evidence=evidence)
            prob_0 = result.values[0]
            prob_1 = result.values[1]
            
            return {
                'prob_0': prob_0,
                'prob_1': prob_1
            }
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            return {
                'prob_0': 0.5,
                'prob_1': 0.5
            }
    
    def infer_conditional_probability(self, conditions):
        """
        Find probability of water quality being good under specific conditions
        e.g., low ph, high hardness, high solids, etc.
        
        Args:
            conditions: Dictionary with attribute names and desired categories
            
        Returns:
            Dictionary with probability of potable water
        """
        if self.infer is None:
            self.learn_parameters()
        
        # Use the conditions directly as evidence
        evidence = {}
        for key, value in conditions.items():
            if key in self.discretized_data.columns and key != 'Potability':
                evidence[key] = value
        
        # Perform inference
        try:
            result = self.infer.query(variables=['Potability'], evidence=evidence)
            prob_0 = result.values[0]
            prob_1 = result.values[1]
            
            return {
                'prob_0': prob_0,
                'prob_1': prob_1,
                'percentage': prob_1 * 100
            }
        except Exception as e:
            print(f"Error during conditional inference: {str(e)}")
            return {
                'prob_0': 0.5,
                'prob_1': 0.5,
                'percentage': 50.0
            }


def read_input_file(input_file):
    """
    Read input from inputPSXX.txt file
    
    Args:
        input_file: Path to input file
        
    Returns:
        Dictionary with parsed input data
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    input_data = {}
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]
                try:
                    value = float(parts[1])
                    input_data[key] = value
                except ValueError:
                    input_data[key] = parts[1]
    
    return input_data


def append_output_file(output_file, results, query_type):
    """
    Append results to outputPSXX.txt file
    
    Args:
        output_file: Path to output file
        results: Dictionary with results
        query_type: Type of query (prediction, inference, conditional)
    """
    section_labels = {
        'prediction': 'Question 2: Water Quality Prediction',
        'inference':  'Question 3: Probability Inference',
        'conditional': 'Question 4: Conditional Probability (low ph, high hardness, high solids)',
    }
    with open(output_file, 'a') as f:
        f.write(f"--- {section_labels.get(query_type, query_type)} ---\n")
        _write_query_result(f, results, query_type)
        f.write("\n")


def _write_query_result(f, results, query_type):
    """Write a single query result block to file handle f."""
    if query_type == 'prediction':
        f.write("Potability 0 1\n")
        f.write(f"Probability {results['prob_0']:.6f} {results['prob_1']:.6f}\n")
        f.write(f"Probability of the water being good is {results['prob_1']*100:.2f}% when considered low ph value and remaining variables high value from given dataset\n")
    
    elif query_type == 'inference':
        f.write(f"Potability 0 1\n")
        f.write(f"Probability {results['prob_0']:.5f} {results['prob_1']:.5f}\n")
        f.write(f"Probability of the water being good is {results['prob_1']*100:.2f}% when the given attribute values are given from the dataset\n")
    
    elif query_type == 'conditional':
        f.write(f"Potability 0 1\n")
        f.write(f"Probability {results['prob_0']:.6f} {results['prob_1']:.6f}\n")
        f.write(f"Probability of the water being good is {results['percentage']:.2f}% when considered low ph value and remaining variables high value from given dataset\n")


def main():
    """
    Main function to run the water quality prediction
    """
    # Configuration
    data_file = 'water_potability.csv'  # Water potability dataset
    input_file = 'inputPS10.txt'
    output_file = 'outputPS10.txt'
    
    print("Water Quality Prediction using Bayesian Belief Network")
    print("=" * 60)
    
    # Initialize BBN
    try:
        bbn = WaterQualityBBN(data_file)
        bbn.discretize_data()
        bbn.construct_bbn()
        bbn.learn_parameters()
        print("\nBayesian Belief Network constructed and trained successfully\n")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease ensure 'water_potability.csv' exists in the directory")
        print("The CSV file should contain columns: ph, Hardness, Solids, Chloramines,")
        print("Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity, Potability")
        return
    
    # Read input file for Question 2 data
    try:
        input_data = read_input_file(input_file)
        print(f"Input data read from {input_file}")
        print(f"Input: {input_data}\n")
    except FileNotFoundError:
        print(f"Input file {input_file} not found. Using sample data for demonstration.\n")
        input_data = {
            'ph': 3.72,
            'Hardness': 204.89,
            'Solids': 20791.32,
            'Chloramines': 7.3,
            'Sulfate': 368.5,
            'Conductivity': 564.30,
            'Turbidity': 2.96
        }
    
    # Delete existing output file to ensure a clean run
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Previous {output_file} removed.")
    
    # Create fresh output file
    with open(output_file, 'w') as f:
        f.write("Water Quality Prediction using Bayesian Belief Network\n")
        f.write("=" * 60 + "\n\n")
    
    # --- Question 2: Predict water quality ---
    print("Question 2: Performing potability prediction...")
    q2_input = {k: v for k, v in input_data.items() if k != 'Potability'}
    results_q2 = bbn.predict_potability(q2_input)
    append_output_file(output_file, results_q2, 'prediction')
    
    # --- Question 3: Infer probability ---
    print("Question 3: Performing probability inference...")
    q3_input = {
        'Hardness': input_data.get('Hardness', 248.0),
        'Solids': input_data.get('Solids', 28749),
        'Chloramines': input_data.get('Chloramines', 7.5),
        'Sulfate': input_data.get('Sulfate', 393),
        'Conductivity': input_data.get('Conductivity', 283),
        'Organic_carbon': input_data.get('Organic_carbon', 13.78),
        'Trihalomethanes': input_data.get('Trihalomethanes', 84.6),
        'Turbidity': input_data.get('Turbidity', 2.67),
    }
    results_q3 = bbn.infer_probability(q3_input)
    append_output_file(output_file, results_q3, 'inference')
    
    # --- Question 4: Conditional probability (low ph, high hardness, high solids) ---
    print("Question 4: Performing conditional probability inference...")
    conditions_q4 = {
        'ph': 'low',
        'Hardness': 'high',
        'Solids': 'high'
    }
    results_q4 = bbn.infer_conditional_probability(conditions_q4)
    append_output_file(output_file, results_q4, 'conditional')
    
    print(f"\nAll results written to {output_file}")
    print("\nOutput:")
    with open(output_file, 'r') as f:
        print(f.read())


if __name__ == "__main__":
    main()
