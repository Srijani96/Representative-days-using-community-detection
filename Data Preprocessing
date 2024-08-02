import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure inline plotting for Jupyter notebooks
%matplotlib inline

# Function to rename files in a directory
def rename_files(directory, new_name):
    for count, filename in enumerate(os.listdir(directory)):
        file_extension = os.path.splitext(filename)[1]
        new_filename = f"{new_name}_{count}{file_extension}"
        source = os.path.join(directory, filename)
        destination = os.path.join(directory, new_filename)
        os.rename(source, destination)
    print(f"Files in {directory} have been renamed to {new_name}_<number>")

# Example usage
rename_files('path/to/your/directory', 'input')

# Load your datasets
dfT = pd.read_csv('path/to/your/dataset_T.csv')
dfQ = pd.read_csv('path/to/your/dataset_Q.csv')

# Perform data analysis
dfT2 = dfT.describe().T
dfQ2 = dfQ.describe().T

# Combine the dataframes
df_compare = pd.concat([dfT2, dfQ2], axis=1)

# Save the comparison to a CSV file
output_path = "Documents/Clustering_for_image_analysis/meteo_mat/data/data_compare.csv"
df_compare.to_csv(output_path, index=False)

print("Data comparison saved to", output_path)
