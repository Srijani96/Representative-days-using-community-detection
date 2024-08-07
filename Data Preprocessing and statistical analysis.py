# Data Analysis and Preprocessing Notebook

## Importing Libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

%matplotlib inline
plt.style.use('seaborn')

## Data Loading and Initial Exploration

def load_data(file_path):
    """Load data from CSV file and perform initial exploration."""
    df = pd.read_csv(file_path)
    print("Dataset shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    return df

# Load datasets
dfT = load_data('path/to/your/dataset_T.csv')
dfQ = load_data('path/to/your/dataset_Q.csv')

## Data Preprocessing

def preprocess_data(df, dataset_name):
    """Preprocess the dataset."""
    print(f"\nPreprocessing {dataset_name} dataset:")
    
    # Handle missing values
    imputer = IterativeImputer(random_state=42)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    print("Missing values after imputation:", df_imputed.isnull().sum().sum())
    
    # Remove duplicates
    df_cleaned = df_imputed.drop_duplicates()
    print("Shape after removing duplicates:", df_cleaned.shape)
    
    # Handle outliers (using IQR method)
    Q1 = df_cleaned.quantile(0.25)
    Q3 = df_cleaned.quantile(0.75)
    IQR = Q3 - Q1
    df_no_outliers = df_cleaned[~((df_cleaned < (Q1 - 1.5 * IQR)) | (df_cleaned > (Q3 + 1.5 * IQR))).any(axis=1)]
    print("Shape after removing outliers:", df_no_outliers.shape)
    
    return df_no_outliers

dfT_cleaned = preprocess_data(dfT, "Temperature")
dfQ_cleaned = preprocess_data(dfQ, "Other Variable")

## Exploratory Data Analysis (EDA)

def perform_eda(df, dataset_name):
    """Perform Exploratory Data Analysis on the dataset."""
    print(f"\nEDA for {dataset_name} dataset:")
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title(f"Correlation Matrix for {dataset_name}")
    plt.show()
    
    # Distribution plots
    plt.figure(figsize=(12, 4 * len(df.columns)))
    for i, column in enumerate(df.columns):
        plt.subplot(len(df.columns), 1, i+1)
        sns.histplot(df[column], kde=True)
        plt.title(f"Distribution of {column}")
    plt.tight_layout()
    plt.show()
    
    # Box plots
    plt.figure(figsize=(12, 6))
    df.boxplot()
    plt.title(f"Box Plots for {dataset_name}")
    plt.xticks(rotation=45)
    plt.show()

perform_eda(dfT_cleaned, "Temperature")
perform_eda(dfQ_cleaned, "Other Variable")

## Statistical Analysis

def perform_statistical_analysis(df1, df2, name1, name2):
    """Perform statistical analysis comparing two datasets."""
    print(f"\nStatistical Analysis comparing {name1} and {name2}:")
    
    # Combine datasets for comparison
    df_combined = pd.concat([df1.add_prefix(f'{name1}_'), df2.add_prefix(f'{name2}_')], axis=1)
    
    # Descriptive statistics
    print("\nDescriptive Statistics:")
    print(df_combined.describe())
    
    # T-tests to compare means
    for column in df1.columns:
        t_stat, p_value = stats.ttest_ind(df1[column], df2[column])
        print(f"\nt-test for {column}:")
        print(f"t-statistic: {t_stat}")
        print(f"p-value: {p_value}")
    
    # Visualize comparison of means
    plt.figure(figsize=(12, 6))
    df_combined.filter(regex='mean$').plot(kind='bar')
    plt.title("Comparison of Means")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    return df_combined

df_compare = perform_statistical_analysis(dfT_cleaned, dfQ_cleaned, "Temp", "Other")

## Principal Component Analysis (PCA)

def perform_pca(df1, df2):
    """Perform PCA on the combined dataset."""
    print("\nPrincipal Component Analysis:")
    
    # Combine and scale the data
    df_combined = pd.concat([df1, df2], axis=1)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df_combined)
    
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(data_scaled)
    
    # Plot explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Analysis')
    plt.show()
    
    # Print the explained variance ratio
    print("Explained Variance Ratio:")
    print(pca.explained_variance_ratio_)

perform_pca(dfT_cleaned, dfQ_cleaned)

## Save Processed Data

output_path = "Documents/Clustering_for_image_analysis/meteo_mat/data/data_compare.csv"
df_compare.to_csv(output_path, index=True)
print(f"\nProcessed and compared data saved to {output_path}")
