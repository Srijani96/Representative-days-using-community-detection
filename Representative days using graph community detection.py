import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer
from sklearn.metrics.pairwise import euclidean_distances
from community import community_louvain

def load_and_preprocess_data(file_path, features):
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

def apply_scaling(df, scaler):
    """
    Apply specified scaling method to the dataframe.
    """
    return scaler.fit_transform(df)

def calculate_distance_matrix(normalized_data):
    """
    Calculate the Euclidean distance matrix for normalized data.
    """
    return euclidean_distances(normalized_data, normalized_data)

def find_nearest_neighbors(distance_matrix, k):
    """
    Find k nearest neighbors for each data point.
    """
    return np.argsort(distance_matrix, axis=1)[:, :(k+1)]

def create_graph_from_neighbors(neighbor_table):
    """
    Create a graph from the neighbor table.
    """
    G = nx.Graph()
    for row in neighbor_table.itertuples(index=False):
        day = row[0]
        neighbors = [int(neighbor) for neighbor in row[1:] if not pd.isna(neighbor)]
        G.add_node(day)
        for neighbor in neighbors:
            G.add_edge(day, neighbor)
    return G

def detect_communities(G, resolution=0.85):
    """
    Detect communities using Louvain algorithm.
    """
    return community_louvain.best_partition(G, resolution=resolution)

def visualize_communities(G, partition):
    """
    Visualize the communities.
    """
    pos = nx.spring_layout(G, seed=seed)
    cluster_indices = list(partition.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw(G, pos, with_labels=False, node_size=50, node_color=cluster_indices, cmap=plt.get_cmap('viridis'), ax=ax)
    plt.title("Communities with quantile transformer normalization")
    plt.show()

def calculate_pagerank(communities, G):
    """
    Calculate PageRank for each community and find the date with maximum PageRank.
    """
    community_pageranks = {}
    max_pagerank_dates = {}
    
    for community_index, community in communities.items():
        subgraph = G.subgraph(community)
        pagerank = nx.pagerank(subgraph)
        community_pageranks[community_index] = pagerank
        max_pagerank_dates[community_index] = max(pagerank, key=pagerank.get)
    
    return community_pageranks, max_pagerank_dates

def main():
    # Load and preprocess data
    df = load_and_preprocess_data('path_to_your_data.csv', ['meanT', 'meanQ'])
    if df is None:
        return

    # Apply different scaling methods
    scalers = {
        'Standard': StandardScaler(),
        'MinMax': MinMaxScaler(feature_range=(0, 1)),
        'Robust': RobustScaler(),
        'MaxAbs': MaxAbsScaler(),
        'Quantile': QuantileTransformer(output_distribution='uniform')
    }

    distance_matrices = {}
    for name, scaler in scalers.items():
        normalized_data = apply_scaling(df, scaler)
        distance_matrices[name] = calculate_distance_matrix(normalized_data)

    # Find nearest neighbors
    k = 10
    nearest_neighbors = find_nearest_neighbors(distance_matrices['MinMax'], k)
    df_neighbor = pd.DataFrame(nearest_neighbors)

    # Create graph and detect communities
    G = create_graph_from_neighbors(df_neighbor)
    partition = detect_communities(G)

    # Visualize communities
    visualize_communities(G, partition)

    # Calculate PageRank
    communities = {index: [node for node, comm in partition.items() if comm == index] for index in set(partition.values())}
    community_pageranks, max_pagerank_dates = calculate_pagerank(communities, G)

    # Create DataFrame with community information
    community_data = []
    for community_index, pagerank in community_pageranks.items():
        for node, score in pagerank.items():
            community_data.append({
                'Community': community_index,
                'Date': node,
                'PageRank': score
            })
    df_community = pd.DataFrame(community_data)

    print(df_community)

if __name__ == "__main__":
    main()
