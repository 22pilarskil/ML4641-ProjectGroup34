# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # Paths to the directories containing the pkl files
# dir_refined_v1 = '../data_vis_task/NumericalData_refined_v1'
# dir_processed = '../data_vis_task/NumericalData_processed'

# # Retrieve the list of file names in each directory
# files_refined_v1 = sorted([f for f in os.listdir(dir_refined_v1) if f.endswith('.pkl')])
# files_processed = sorted([f for f in os.listdir(dir_processed) if f.endswith('.pkl')])

# # Initialize lists to hold comparison data
# means_refined_v1, variances_refined_v1 = [], []
# means_processed, variances_processed = [], []

# # Iterate over the files assuming each pair of files across folders corresponds to each other
# for file_name in files_refined_v1:
#     if file_name in files_processed:
#         # Load the datasets
#         refined_v1 = pd.read_pickle(os.path.join(dir_refined_v1, file_name))
#         processed = pd.read_pickle(os.path.join(dir_processed, file_name))

#         # Calculate and store mean and variance for each dataset
#         means_refined_v1.append(refined_v1.mean())
#         variances_refined_v1.append(refined_v1.var())

#         means_processed.append(processed.mean())
#         variances_processed.append(processed.var())

# # Assuming the first file's columns represent all files' columns accurately
# comparison_df = pd.DataFrame({
#     'Column': refined_v1.columns,
#     'Mean_Refined_v1': pd.concat(means_refined_v1, axis=1).mean(axis=1).values,
#     'Variance_Refined_v1': pd.concat(variances_refined_v1, axis=1).mean(axis=1).values,
#     'Mean_Processed': pd.concat(means_processed, axis=1).mean(axis=1).values,
#     'Variance_Processed': pd.concat(variances_processed, axis=1).mean(axis=1).values
# })

# # Visualization: Scatter plot for means
# plt.figure(figsize=(10, 5))
# sns.scatterplot(data=comparison_df, x='Mean_Refined_v1', y='Mean_Processed', hue='Column', legend=False)
# plt.title('Aggregate Comparison of Means')
# plt.xlabel('Mean of Refined_v1')
# plt.ylabel('Mean of Processed')
# plt.grid(True)
# plt.show()

# # Scatter plot for variances
# plt.figure(figsize=(10, 5))
# sns.scatterplot(data=comparison_df, x='Variance_Refined_v1', y='Variance_Processed', hue='Column', legend=False)
# plt.title('Aggregate Comparison of Variances')
# plt.xlabel('Variance of Refined_v1')
# plt.ylabel('Variance of Processed')
# plt.grid(True)
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import os

# Directories containing the pkl files for each dataset
dir_refined_v1 = '../data_vis_task/NumericalData_refined_v3'
dir_processed = '../data_vis_task/NumericalData_processed'

# Function to calculate max and min values for each feature for each stock
def calculate_stock_extremes(directory):
    stock_max_values = {}
    stock_min_values = {}

    # Retrieve the list of file names in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.pkl')]

    # Iterate over the files
    for file_name in files:
        file_path = os.path.join(directory, file_name)
        data = pd.read_pickle(file_path)

        # Iterate over each column (measurement) in the dataframe
        for column in data.columns:
            if column not in stock_max_values:
                stock_max_values[column] = []
                stock_min_values[column] = []

            # Append max and min values for each stock
            stock_max_values[column].append(data[column].max())
            stock_min_values[column].append(data[column].min())

    return stock_max_values, stock_min_values

# Calculate max and min for both datasets
stock_max_values_refined, stock_min_values_refined = calculate_stock_extremes(dir_refined_v1)
stock_max_values_processed, stock_min_values_processed = calculate_stock_extremes(dir_processed)

# Function to plot histograms of max and min values
def plot_stock_extremes_histogram(stock_max_values, stock_min_values, dataset_name):
    for measurement in stock_max_values.keys():
        plt.figure(figsize=(12, 6))

        # Max value histogram for each measurement
        plt.subplot(1, 2, 1)
        plt.hist(stock_max_values[measurement], bins=50, alpha=0.7)
        plt.title(f'Frequency of Max Values of {measurement} in {dataset_name}')
        plt.xlabel('Max ' + measurement)
        plt.ylabel('Frequency')

        # Min value histogram for each measurement
        plt.subplot(1, 2, 2)
        plt.hist(stock_min_values[measurement], bins=50, alpha=0.7)
        plt.title(f'Frequency of Min Values of {measurement} in {dataset_name}')
        plt.xlabel('Min ' + measurement)
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

# Plotting for both datasets
plot_stock_extremes_histogram(stock_max_values_refined, stock_min_values_refined, 'Refined_v1')
plot_stock_extremes_histogram(stock_max_values_processed, stock_min_values_processed, 'Processed')
