# clustering_utils.py
import numpy as np
import hdbscan
from sklearn.cluster import DBSCAN, MeanShift, AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def apply_dbscan(point_cloud, eps=0.8, min_samples=5):
    """
    Apply DBSCAN clustering algorithm.

    Args:
        point_cloud (numpy.ndarray): A (N, 3) array where N is the number of points in the point cloud,
                                     and 3 represents the (x, y, z) coordinates.
        eps (float): The maximum distance between two points to be considered as neighbors (default: 0.8).
        min_samples (int): The minimum number of points required to form a core point (default: 5).

    Returns:
        labels (numpy.ndarray): An array of cluster labels for each point (-1 indicates noise).
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(point_cloud)
    return labels

def apply_hdbscan(point_cloud, min_samples=5):
    """
    Apply HDBSCAN clustering algorithm.

    Args:
        point_cloud (numpy.ndarray): A (N, 3) array representing the 3D point cloud (x, y, z coordinates).
        min_samples (int): The minimum number of points required to form a core point (default: 5).

    Returns:
        labels (numpy.ndarray): An array of cluster labels for each point (-1 indicates noise).
    """
    clusterer = hdbscan.HDBSCAN(min_samples=min_samples)
    labels = clusterer.fit_predict(point_cloud)
    return labels

def apply_meanshift(point_cloud):
    """
    Apply Mean Shift clustering algorithm.

    Args:
        point_cloud (numpy.ndarray): A (N, 3) array representing the 3D point cloud (x, y, z coordinates).

    Returns:
        labels (numpy.ndarray): An array of cluster labels for each point.
    """
    meanshift = MeanShift()
    labels = meanshift.fit_predict(point_cloud)
    return labels

def apply_graph_based_clustering(point_cloud, n_clusters=5):
    """
    Apply graph-based clustering (Agglomerative Clustering) algorithm.

    Args:
        point_cloud (numpy.ndarray): A (N, 3) array representing the 3D point cloud (x, y, z coordinates).
        n_clusters (int): The number of clusters to find (default: 5).

    Returns:
        labels (numpy.ndarray): An array of cluster labels for each point.
    """
    # Convert the point cloud into a distance matrix
    distance_matrix = squareform(pdist(point_cloud))
    # Use agglomerative clustering on the distance matrix
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
    labels = agglomerative.fit_predict(distance_matrix)
    return labels

def clustering_algorithm(algorithm, point_cloud, **kwargs):
    """
    Utility function to apply different clustering algorithms to a point cloud.

    Args:
        algorithm (str): The clustering algorithm to apply. Choices are:
            - 'dbscan': Use DBSCAN clustering.
            - 'hdbscan': Use HDBSCAN clustering.
            - 'meanshift': Use Mean Shift clustering.
            - 'graph': Use graph-based (Agglomerative Clustering) clustering.
        point_cloud (numpy.ndarray): A (N, 3) array representing the 3D point cloud (x, y, z coordinates).
        **kwargs: Additional parameters to pass to the selected clustering algorithm.
            - For 'dbscan': Pass 'eps' (float) and 'min_samples' (int).
            - For 'hdbscan': Pass 'min_samples' (int).
            - For 'meanshift': No additional parameters needed.
            - For 'graph': Pass 'n_clusters' (int).

    Returns:
        labels (numpy.ndarray): An array of cluster labels for each point.
    """
    if algorithm == 'dbscan':
        return apply_dbscan(point_cloud, **kwargs)
    elif algorithm == 'hdbscan':
        return apply_hdbscan(point_cloud, **kwargs)
    elif algorithm == 'meanshift':
        return apply_meanshift(point_cloud)
    elif algorithm == 'graph':
        return apply_graph_based_clustering(point_cloud, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


####################### Noise Reductions #######################

def apply_noise_reduction(point_cloud_noisy, nb_neighbors=20, std_ratio=1.0):
    """
    Apply statistical outlier removal to reduce noise in a point cloud.

    Args:
        point_cloud_noisy (numpy.ndarray): The noisy point cloud data (N, 3).
        nb_neighbors (int): Number of neighbors to consider for noise reduction.
        std_ratio (float): Standard deviation ratio for the noise filter.

    Returns:
        numpy.ndarray: The denoised point cloud data.
    """
    # Convert numpy array to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_noisy)

    # Apply statistical outlier removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd_clean = pcd.select_by_index(ind)

    # Extract the cleaned point cloud as a numpy array
    point_cloud_filtered = np.asarray(pcd_clean.points)
    return point_cloud_filtered

def visualize_denoising(point_cloud_noisy, point_cloud_filtered):
    """
    Visualize the point cloud before and after denoising.

    Args:
        point_cloud_noisy (numpy.ndarray): The noisy point cloud data (N, 3).
        point_cloud_filtered (numpy.ndarray): The denoised point cloud data (M, 3).
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the noisy data (before denoising)
    ax.scatter(point_cloud_noisy[:, 0], point_cloud_noisy[:, 1], point_cloud_noisy[:, 2],
               color='lightgrey', label='Noisy Data', alpha=0.5)

    # Plot the denoised data (after denoising)
    ax.scatter(point_cloud_filtered[:, 0], point_cloud_filtered[:, 1], point_cloud_filtered[:, 2],
               color='blue', label='Denoised Data')

    ax.set_title('Point Cloud Before and After Denoising')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()