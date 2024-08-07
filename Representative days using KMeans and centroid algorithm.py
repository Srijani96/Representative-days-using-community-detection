import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

def load_data(file_path, features):
    """
    Load data from CSV and select specified features.
    """
    try:
        data = pd.read_csv(file_path)
        return data[features]
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except KeyError as e:
        print(f"Error: One or more specified features not found in the dataset. {e}")
        return None

def standardize_data(df, scaler_type='MinMax'):
    """
    Standardize the data using the specified scaler.
    """
    scalers = {
        'Standard': StandardScaler(),
        'MinMax': MinMaxScaler(),
        'Robust': RobustScaler(),
        'MaxAbs': MaxAbsScaler(),
        'Quantile': QuantileTransformer(output_distribution='uniform')
    }
    
    if scaler_type not in scalers:
        print(f"Error: Unknown scaler type '{scaler_type}'. Using MinMaxScaler.")
        scaler_type = 'MinMax'
    
    scaler = scalers[scaler_type]
    return scaler.fit_transform(df)

def apply_kmeans(df_standardized, k):
    """
    Apply KMeans clustering on the standardized DataFrame.
    """
    kmeans = KMeans(n_clusters=k, random_state=seed)
    return kmeans.fit_predict(df_standardized)

def find_closest_to_center(df, kmeans, k):
    """
    Find the index closest to each cluster center.
    """
    closest_indices = {}
    for cluster_num in range(k):
        cluster_indices = df.index[df['Cluster'] == cluster_num]
        cluster_center = kmeans.cluster_centers_[cluster_num]
        distances = cdist(df.loc[cluster_indices, df.columns[:-1]], [cluster_center])
        closest_index = cluster_indices[np.argmin(distances)]
        closest_indices[cluster_num] = closest_index
    return closest_indices

def main(file_path, features, k=10, scaler_type='MinMax'):
    # Load and preprocess data
    df = load_data(file_path, features)
    if df is None:
        return

    # Standardize data
    df_standardized = standardize_data(df, scaler_type)

    # Apply KMeans clustering
    df['Cluster'] = apply_kmeans(df_standardized, k)

    # Find closest indices to cluster centers
    kmeans = KMeans(n_clusters=k, random_state=42).fit(df_standardized)
    closest_indices = find_closest_to_center(df, kmeans, k)

    # Display results
    for cluster_num, closest_index in closest_indices.items():
        print(f"Cluster {cluster_num} - Closest Index: {closest_index}")

    return df, closest_indices

if __name__ == "__main__":
    file_path = 'path_to_your_data.csv'  # Replace with your actual file path
    features = ['meanT', 'meanQ']  # Replace with your actual feature names
    k = 15  # Number of clusters
    scaler_type = 'MinMax'  # Scaler type to use

    df, closest_indices = main(file_path, features, k, scaler_type)
