import pandas as pd
import os
import numpy as np

def process_pickle_files(directory):
    max_vals = {}
    min_vals = {}

    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            filepath = os.path.join(directory, filename)
            try:
                df = pd.read_pickle(filepath)
                # Iterate through each column in the DataFrame
                for column in df.columns:
                    # Update the max values
                    if column in max_vals:
                        max_vals[column] = max(max_vals[column], df[column].max())
                    else:
                        max_vals[column] = df[column].max()
                    # Update the min values
                    if column in min_vals:
                        min_vals[column] = min(min_vals[column], df[column].min())
                    else:
                        min_vals[column] = df[column].min()
            except Exception as e:
                print(f"Could not process {filename}: {e}")

    return max_vals, min_vals

directory = 'data/NumericalData_processed'
max_vals, min_vals = process_pickle_files(directory)
print(f"Max values across all DataFrames for {directory}:", max_vals)
print(f"Min values across all DataFrames for {directory}:", min_vals)

directory = '../NumericalData_processed'
max_vals, min_vals = process_pickle_files(directory)
print(f"Max values across all DataFrames for {directory}:", max_vals)
print(f"Min values across all DataFrames for {directory}:", min_vals)

