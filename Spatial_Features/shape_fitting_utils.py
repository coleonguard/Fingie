# shape_fitting_utils.py

import numpy as np
from sklearn.decomposition import PCA
from pyransac3d import Plane, Sphere, Cylinder, Line
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def fit_shapes_ransac(cluster_data, thresholds):
    """
    Try fitting different shapes using RANSAC and select the best one based on inliers.

    Args:
        cluster_data (numpy.ndarray): The point cloud data for the cluster (N, 3).
        thresholds (dict): Thresholds for each shape.

    Returns:
        dict: Information about the best-fitting shape.
    """
    X = cluster_data  # (n_points, 3)
    max_inliers = 0
    best_model = None
    best_model_name = None
    best_inliers = None

    shapes = ['line', 'plane', 'sphere', 'cylinder']
    for shape in shapes:
        try:
            if shape == 'line':
                model = Line()
                threshold = thresholds.get('line', 0.05)
                [point, direction], inliers = model.fit(X, thresh=threshold)
                num_inliers = len(inliers)
                if num_inliers > max_inliers:
                    max_inliers = num_inliers
                    best_model = {'point': point.flatten(), 'direction': direction.flatten()}
                    best_model_name = 'line'
                    best_inliers = X[inliers]
            elif shape == 'plane':
                model = Plane()
                threshold = thresholds.get('plane', 0.05)
                plane_eq, inliers = model.fit(X, thresh=threshold)
                num_inliers = len(inliers)
                if num_inliers > max_inliers:
                    max_inliers = num_inliers
                    best_model = {'plane_eq': plane_eq}
                    best_model_name = 'plane'
                    best_inliers = X[inliers]
            elif shape == 'sphere':
                model = Sphere()
                threshold = thresholds.get('sphere', 0.05)
                center, radius, inliers = model.fit(X, thresh=threshold)
                num_inliers = len(inliers)
                if num_inliers > max_inliers:
                    max_inliers = num_inliers
                    best_model = {'center': center.flatten(), 'radius': radius}
                    best_model_name = 'sphere'
                    best_inliers = X[inliers]
            elif shape == 'cylinder':
                model = Cylinder()
                threshold = thresholds.get('cylinder', 0.05)
                axis_start, axis_end, radius, inliers = model.fit(X, thresh=threshold)
                num_inliers = len(inliers)
                if num_inliers > max_inliers:
                    max_inliers = num_inliers
                    best_model = {
                        'axis_start': axis_start.flatten(),
                        'axis_end': axis_end.flatten(),
                        'radius': radius
                    }
                    best_model_name = 'cylinder'
                    best_inliers = X[inliers]
        except Exception as e:
            # Handle exceptions (e.g., fitting failed)
            pass

    return {
        'model': best_model,
        'model_name': best_model_name,
        'inliers': best_inliers
    }

def fit_shapes_pca_ransac(cluster_data, thresholds):
    """
    Use PCA to decide which shapes to fit, then apply RANSAC.

    Args:
        cluster_data (numpy.ndarray): The point cloud data for the cluster (N, 3).
        thresholds (dict): Thresholds for each shape.

    Returns:
        dict: Information about the best-fitting shape.
    """
    X = cluster_data  # (n_points, 3)
    pca = PCA(n_components=3)
    pca.fit(X)
    explained_variance = pca.explained_variance_ratio_

    # Decide shapes to fit based on PCA
    if explained_variance[0] > 0.8:
        shapes_to_fit = ['line', 'cylinder']
    elif explained_variance[0] + explained_variance[1] > 0.95:
        shapes_to_fit = ['plane']
    else:
        shapes_to_fit = ['sphere']

    max_inliers = 0
    best_model = None
    best_model_name = None
    best_inliers = None

    for shape in shapes_to_fit:
        try:
            if shape == 'line':
                model = Line()
                threshold = thresholds.get('line', 0.05)
                [point, direction], inliers = model.fit(X, thresh=threshold)
                num_inliers = len(inliers)
                if num_inliers > max_inliers:
                    max_inliers = num_inliers
                    best_model = {'point': point.flatten(), 'direction': direction.flatten()}
                    best_model_name = 'line'
                    best_inliers = X[inliers]
            elif shape == 'plane':
                model = Plane()
                threshold = thresholds.get('plane', 0.05)
                plane_eq, inliers = model.fit(X, thresh=threshold)
                num_inliers = len(inliers)
                if num_inliers > max_inliers:
                    max_inliers = num_inliers
                    best_model = {'plane_eq': plane_eq}
                    best_model_name = 'plane'
                    best_inliers = X[inliers]
            elif shape == 'sphere':
                model = Sphere()
                threshold = thresholds.get('sphere', 0.05)
                center, radius, inliers = model.fit(X, thresh=threshold)
                num_inliers = len(inliers)
                if num_inliers > max_inliers:
                    max_inliers = num_inliers
                    best_model = {'center': center.flatten(), 'radius': radius}
                    best_model_name = 'sphere'
                    best_inliers = X[inliers]
            elif shape == 'cylinder':
                model = Cylinder()
                threshold = thresholds.get('cylinder', 0.05)
                axis_start, axis_end, radius, inliers = model.fit(X, thresh=threshold)
                num_inliers = len(inliers)
                if num_inliers > max_inliers:
                    max_inliers = num_inliers
                    best_model = {
                        'axis_start': axis_start.flatten(),
                        'axis_end': axis_end.flatten(),
                        'radius': radius
                    }
                    best_model_name = 'cylinder'
                    best_inliers = X[inliers]
        except Exception as e:
            pass

    return {
        'model': best_model,
        'model_name': best_model_name,
        'inliers': best_inliers,
        'explained_variance': explained_variance
    }

def pca_analysis(cluster_data):
    """
    Perform PCA analysis on the cluster data without shape fitting.

    Args:
        cluster_data (numpy.ndarray): The point cloud data for the cluster (N, 3).

    Returns:
        dict: Contains the principal components, explained variance, and mean.
    """
    X = cluster_data  # (n_points, 3)
    pca = PCA(n_components=3)
    pca.fit(X)
    components = pca.components_
    explained_variance = pca.explained_variance_ratio_
    mean = pca.mean_
    return {
        'components': components,
        'explained_variance': explained_variance,
        'mean': mean
    }

def normal_estimation(cluster_data, k_neighbors=30):
    """
    Estimate normals for the cluster data.

    Args:
        cluster_data (numpy.ndarray): The point cloud data for the cluster (N, 3).
        k_neighbors (int): Number of nearest neighbors to use for normal estimation.

    Returns:
        dict: Contains the normals for each point in the cluster.
    """
    # Convert to open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cluster_data)
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))
    normals = np.asarray(pcd.normals)
    return {
        'normals': normals
    }

def shape_fitting_algorithm(method, cluster_data, thresholds=None):
    """
    General function to select the shape fitting approach.

    Args:
        method (str): The shape fitting method to apply. Choices are:
            - 'ransac_inlier': Try all shapes and select based on inliers.
            - 'pca_ransac': Use PCA to select shapes before RANSAC.
            - 'pca_analysis': Perform PCA analysis without shape fitting.
            - 'normal_estimation': Estimate normals for the cluster data.
        cluster_data (numpy.ndarray): The point cloud data for the cluster (N, 3).
        thresholds (dict, optional): Thresholds for each shape (used for shape fitting methods).

    Returns:
        dict: Information about the result, depends on the method.
    """
    if method == 'ransac_inlier':
        return fit_shapes_ransac(cluster_data, thresholds)
    elif method == 'pca_ransac':
        return fit_shapes_pca_ransac(cluster_data, thresholds)
    elif method == 'pca_analysis':
        return pca_analysis(cluster_data)
    elif method == 'normal_estimation':
        return normal_estimation(cluster_data)
    else:
        raise ValueError(f"Unknown method: {method}")

def visualize_clusters_and_shapes(point_cloud_filtered, labels, cluster_models, method):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    unique_labels = set(labels)
    colors = plt.cm.get_cmap("tab20", len(unique_labels))

    for k in unique_labels:
        class_member_mask = (labels == k)
        cluster_data = point_cloud_filtered[class_member_mask]

        if k == -1:
            color = 'k'  # Black for noise
            label = 'Noise'
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2],
                       color=color, label=label)
            continue

        color = colors(k)
        label = f'Cluster {k}'

        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2],
                   color=color, label=label, alpha=0.6)

        result = cluster_models[k]

        if method in ['ransac_inlier', 'pca_ransac']:
            best_model = result['model']
            best_model_name = result['model_name']

            if best_model is None:
                continue

            if best_model_name == 'line':
                # Plot the line
                point = best_model['point']
                direction = best_model['direction']
                t = np.linspace(-50, 50, 100)
                line_points = point + np.outer(t, direction)
                ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2],
                        color=color, linewidth=2)
            elif best_model_name == 'plane':
                # Plot the plane
                a, b, c, d = best_model['plane_eq']
                # Create grid to plot plane
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 10),
                                     np.linspace(ylim[0], ylim[1], 10))
                zz = (-a * xx - b * yy - d) / c
                ax.plot_surface(xx, yy, zz, color=color, alpha=0.5)
            elif best_model_name == 'sphere':
                # Plot the sphere
                center = best_model['center']
                radius = best_model['radius']
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = center[0] + radius * np.cos(u) * np.sin(v)
                y = center[1] + radius * np.sin(u) * np.sin(v)
                z = center[2] + radius * np.cos(v)
                ax.plot_surface(x, y, z, color=color, alpha=0.5)
            elif best_model_name == 'cylinder':
                # Plot the cylinder
                axis_start = best_model['axis_start']
                axis_end = best_model['axis_end']
                radius = best_model['radius']
                # Generate cylinder data
                v = axis_end - axis_start
                mag = np.linalg.norm(v)
                v = v / mag
                # Create orthogonal vectors
                not_v = np.array([1, 0, 0])
                if np.allclose(v, not_v):
                    not_v = np.array([0, 1, 0])
                n1 = np.cross(v, not_v)
                n1 /= np.linalg.norm(n1)
                n2 = np.cross(v, n1)
                # Generate the points
                t = np.linspace(0, mag, 50)
                theta = np.linspace(0, 2 * np.pi, 50)
                t, theta = np.meshgrid(t, theta)
                Xc = axis_start[0] + v[0] * t + radius * np.cos(theta) * n1[0] + radius * np.sin(theta) * n2[0]
                Yc = axis_start[1] + v[1] * t + radius * np.cos(theta) * n1[1] + radius * np.sin(theta) * n2[1]
                Zc = axis_start[2] + v[2] * t + radius * np.cos(theta) * n1[2] + radius * np.sin(theta) * n2[2]
                ax.plot_surface(Xc, Yc, Zc, color=color, alpha=0.5)
        elif method == 'pca_analysis':
            # Visualize PCA components
            components = result['components']
            mean = result['mean']
            # Plot the principal components as arrows
            for i in range(3):
                vec = components[i]
                start = mean
                end = mean + vec * 2  # Scale for visualization
                ax.quiver(start[0], start[1], start[2],
                          vec[0], vec[1], vec[2],
                          color=['r', 'g', 'b'][i], length=2, normalize=True)
        elif method == 'normal_estimation':
            # Visualize normals
            normals = result['normals']
            # Subsample points for visualization (e.g., every 10th point)
            step = max(1, len(cluster_data) // 100)
            ax.quiver(cluster_data[::step, 0], cluster_data[::step, 1], cluster_data[::step, 2],
                      normals[::step, 0], normals[::step, 1], normals[::step, 2],
                      length=0.5, color='k', normalize=True)
        else:
            pass  # Other methods

    ax.set_title('3D Point Cloud with Clusters and Results')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()