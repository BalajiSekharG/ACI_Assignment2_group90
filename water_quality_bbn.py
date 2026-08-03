"""
Water Quality Prediction using Bayesian Belief Network
Assignment 2 - PS10
"""

import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
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
                # Discretize into 3 categories: low (0), medium (1), high (2)
                self.discretized_data[col] = pd.qcut(
                    self.discretized_data[col], 
                    q=3, 
                    labels=['low', 'medium', 'high'],
                    duplicates='drop'
                )
        
        print("Data discretized into categories: low, medium, high")
    
    def construct_bbn(self):
        """
        Construct Bayesian Belief Network structure
        Using a simplified structure where all attributes influence Potability
        """
        if self.discretized_data is None:
            self.discretize_data()
        
        # Define network structure
        # All water quality attributes influence Potability
        self.model = BayesianNetwork([
            ('ph', 'Potability'),
            ('Hardness', 'Potability'),
            ('Solids', 'Potability'),
            ('Chloramines', 'Potability'),
            ('Sulfate', 'Potability'),
            ('Conductivity', 'Potability'),
            ('Organic_carbon', 'Potability'),
            ('Trihalomethanes', 'Potability'),
            ('Turbidity', 'Potability')
        ])
        
        print("Bayesian Network structure defined")
    
    def learn_parameters(self):
        """
        Learn CPD (Conditional Probability Distribution) parameters from data
        """
        if self.model is None:
            self.construct_bbn()
        
        try:
            # Use Bayesian Estimator with Dirichlet priors
            self.model.fit(
                self.discretized_data, 
                estimator=BayesianEstimator,
                prior_type='dirichlet',
                pseudo_counts=[1, 1, 1]
            )
            print("Parameters learned successfully")
            
            # Initialize inference object
            self.infer = VariableElimination(self.model)
            
        except Exception as e:
            print(f"Error learning parameters: {str(e)}")
            # Fallback to Maximum Likelihood Estimation
            try:
                self.model.fit(
                    self.discretized_data,
                    estimator=MaximumLikelihoodEstimator
                )
                print("Parameters learned using MLE fallback")
                self.infer = VariableElimination(self.model)
            except Exception as e2:
                raise Exception(f"Failed to learn parameters: {str(e2)}")
    
    def get_category(self, value, column):
        """
        Get the category (low/medium/high) for a continuous value based on data distribution
        
        Args:
            value: Continuous value to categorize
            column: Column name to use for distribution
            
        Returns:
            Category string ('low', 'medium', 'high')
        """
        if self.data is None:
            raise ValueError("No data loaded")
        
        col_data = self.data[column].dropna()
        percentiles = [33.33, 66.67]
        q1, q2 = col_data.quantile(percentiles)
        
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
            
            prediction = 1 if prob_1 > prob_0 else 0
            
            return {
                'prediction': prediction,
                'prob_0': prob_0,
                'prob_1': prob_1
            }
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            # Return default if inference fails
            return {
                'prediction': 0,
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


def write_output_file(output_file, results, query_type):
    """
    Write results to outputPSXX.txt file
    
    Args:
        output_file: Path to output file
        results: Dictionary with results
        query_type: Type of query (prediction, inference, conditional)
    """
    with open(output_file, 'w') as f:
        if query_type == 'prediction':
            f.write(f"Potability 0 1\n")
            f.write(f"Probability {results['prob_0']:.6f} {results['prob_1']:.6f}\n")
            f.write(f"Predicted Potability: {results['prediction']}\n")
        
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
    data_file = 'water_quality.csv'  # Update with actual data file path
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
        print("\nPlease ensure 'water_quality.csv' file exists in the directory")
        print("The CSV file should contain columns: ph, Hardness, Solids, Chloramines,")
        print("Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity, Potability")
        return
    
    # Read input file
    try:
        input_data = read_input_file(input_file)
        print(f"Input data read from {input_file}")
        print(f"Input: {input_data}\n")
    except FileNotFoundError:
        print(f"Input file {input_file} not found. Using sample data for demonstration.\n")
        # Sample data for Question 2
        input_data = {
            'ph': 3.72,
            'Hardness': 204.89,
            'Solids': 20791.32,
            'Chloramines': 7.3,
            'Sulfate': 368.5,
            'Conductivity': 564.30,
            'Turbidity': 2.96
        }
    
    # Determine query type based on input
    if 'Potability' in input_data:
        # Question 3: Inference with given potability
        print("Performing probability inference...")
        results = bbn.infer_probability(input_data)
        write_output_file(output_file, results, 'inference')
    elif all(k in input_data for k in ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Turbidity']):
        # Question 2: Prediction
        print("Performing potability prediction...")
        results = bbn.predict_potability(input_data)
        write_output_file(output_file, results, 'prediction')
    else:
        # Question 4: Conditional probability
        print("Performing conditional probability inference...")
        conditions = {
            'ph': 'low',
            'Hardness': 'high',
            'Solids': 'high'
        }
        results = bbn.infer_conditional_probability(conditions)
        write_output_file(output_file, results, 'conditional')
    
    print(f"\nResults written to {output_file}")
    print("\nOutput:")
    with open(output_file, 'r') as f:
        print(f.read())


if __name__ == "__main__":
    main()
