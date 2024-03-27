import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths to the directories containing the pkl files
dir_refined_v1 = '../liam_test/NumericalData_refined_v1'
dir_processed = '../liam_test/NumericalData_processed'

# Retrieve the list of file names in each directory
files_refined_v1 = sorted([f for f in os.listdir(dir_refined_v1) if f.endswith('.pkl')])
files_processed = sorted([f for f in os.listdir(dir_processed) if f.endswith('.pkl')])

# Initialize lists to hold comparison data
means_refined_v1, variances_refined_v1 = [], []
means_processed, variances_processed = [], []

# Iterate over the files assuming each pair of files across folders corresponds to each other
for file_name in files_refined_v1:
    if file_name in files_processed:
        # Load the datasets
        refined_v1 = pd.read_pickle(os.path.join(dir_refined_v1, file_name))
        processed = pd.read_pickle(os.path.join(dir_processed, file_name))

        # Calculate and store mean and variance for each dataset
        means_refined_v1.append(refined_v1.mean())
        variances_refined_v1.append(refined_v1.var())

        means_processed.append(processed.mean())
        variances_processed.append(processed.var())

# Assuming the first file's columns represent all files' columns accurately
comparison_df = pd.DataFrame({
    'Column': refined_v1.columns,
    'Mean_Refined_v1': pd.concat(means_refined_v1, axis=1).mean(axis=1).values,
    'Variance_Refined_v1': pd.concat(variances_refined_v1, axis=1).mean(axis=1).values,
    'Mean_Processed': pd.concat(means_processed, axis=1).mean(axis=1).values,
    'Variance_Processed': pd.concat(variances_processed, axis=1).mean(axis=1).values
})

# Visualization: Scatter plot for means
plt.figure(figsize=(10, 5))
sns.scatterplot(data=comparison_df, x='Mean_Refined_v1', y='Mean_Processed', hue='Column', legend=False)
plt.title('Aggregate Comparison of Means')
plt.xlabel('Mean of Refined_v1')
plt.ylabel('Mean of Processed')
plt.grid(True)
plt.show()

# Scatter plot for variances
plt.figure(figsize=(10, 5))
sns.scatterplot(data=comparison_df, x='Variance_Refined_v1', y='Variance_Processed', hue='Column', legend=False)
plt.title('Aggregate Comparison of Variances')
plt.xlabel('Variance of Refined_v1')
plt.ylabel('Variance of Processed')
plt.grid(True)
plt.show()