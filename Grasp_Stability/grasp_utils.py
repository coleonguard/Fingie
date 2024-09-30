# grasp_utils.py

import numpy as np
from scipy.spatial import KDTree, ConvexHull
import open3d as o3d
from sklearn.decomposition import PCA

def grasp_evaluation(normal_method, gws_method, closure_method, point_cloud,
                     joint_positions, translation, **kwargs):
    """
    Evaluate grasp using surface normal estimation, GWS, and force closure.

    Args:
        normal_method (str): Method to estimate surface normals ('pca' or
                             'normal_estimation').
        gws_method (str): Method for Grasp Wrench Space ('convex_hull').
        closure_method (str): Method for Force Closure ('friction_cone').
        point_cloud (numpy.ndarray): (N, 3) array representing the 3D point cloud.
        joint_positions (numpy.ndarray): (M, 3) array of joint positions.
        translation (numpy.ndarray): (3,) array representing object translation.
        **kwargs: Additional parameters for specific methods.

    Returns:
        gws_quality (float): GWS score based on the selected method.
        force_closure (bool): Whether force closure is achieved.
    """
    # 1. Translate point cloud by the provided translation
    point_cloud_translated = point_cloud + translation

    # 2. Estimate surface normals
    normals = estimate_normals(normal_method, point_cloud_translated, **kwargs)

    # 3. Find closest contact points
    contact_points, contact_normals = find_contact_points(
        point_cloud_translated, normals, joint_positions)

    # 4. Evaluate GWS using convex hull
    gws_quality = evaluate_gws(gws_method, contact_points, contact_normals)

    # 5. Evaluate Force Closure using friction cone analysis
    force_closure = evaluate_force_closure(
        closure_method, contact_points, contact_normals, **kwargs)

    return gws_quality, force_closure

def estimate_normals(method, point_cloud, **kwargs):
    """
    Estimate surface normals for a point cloud.

    Args:
        method (str): Method for normal estimation ('pca' or 'normal_estimation').
        point_cloud (numpy.ndarray): (N, 3) array representing the 3D point cloud.
        **kwargs: Additional parameters for specific methods.

    Returns:
        normals (numpy.ndarray): Surface normals for the point cloud.
    """
    if method == 'pca':
        return estimate_normals_pca(point_cloud)
    elif method == 'normal_estimation':
        k_neighbors = kwargs.get('k_neighbors', 30)
        return estimate_normals_open3d(point_cloud, k_neighbors)
    else:
        raise ValueError(f"Unknown normal estimation method: {method}")

def estimate_normals_pca(point_cloud):
    """
    Estimate normals using PCA on local neighborhoods.

    Args:
        point_cloud (numpy.ndarray): The point cloud (N, 3).

    Returns:
        normals (numpy.ndarray): Estimated normals for each point.
    """
    normals = []
    tree = KDTree(point_cloud)
    for point in point_cloud:
        idx = tree.query(point, k=10)[1]
        neighbors = point_cloud[idx]
        pca = PCA(n_components=3)
        pca.fit(neighbors)
        normal = pca.components_[-1]
        normals.append(normal)
    return np.array(normals)

def estimate_normals_open3d(point_cloud, k_neighbors=30):
    """
    Estimate normals using Open3D normal estimation.

    Args:
        point_cloud (numpy.ndarray): The point cloud (N, 3).
        k_neighbors (int): Number of nearest neighbors for normal estimation.

    Returns:
        normals (numpy.ndarray): Estimated normals for each point.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))
    normals = np.asarray(pcd.normals)
    return normals

def find_contact_points(point_cloud, normals, joint_positions):
    """
    Find the closest points in the point cloud to the grasping joints.

    Args:
        point_cloud (numpy.ndarray): (N, 3) point cloud.
        normals (numpy.ndarray): Normals corresponding to the point cloud.
        joint_positions (numpy.ndarray): (M, 3) array of joint positions.

    Returns:
        contact_points (numpy.ndarray): Closest points to the gripper's joints.
        contact_normals (numpy.ndarray): Normals at the contact points.
    """
    tree = KDTree(point_cloud)
    contact_points = []
    contact_normals = []
    for joint in joint_positions:
        dist, idx = tree.query(joint)
        contact_points.append(point_cloud[idx])
        contact_normals.append(normals[idx])
    return np.array(contact_points), np.array(contact_normals)

def evaluate_gws(method, contact_points, contact_normals):
    """
    Evaluate Grasp Wrench Space (GWS) using the convex hull method.

    Args:
        method (str): Method to compute GWS ('convex_hull').
        contact_points (numpy.ndarray): Contact points on the object.
        contact_normals (numpy.ndarray): Surface normals at the contact points.

    Returns:
        gws_quality (float): Convex hull volume of the grasp wrench space.
    """
    if method == 'convex_hull':
        return evaluate_gws_convex_hull(contact_points, contact_normals)
    else:
        raise ValueError(f"Unknown GWS method: {method}")

def evaluate_gws_convex_hull(contact_points, contact_normals):
    """
    Compute the convex hull of the contact wrenches (force and torque).

    Args:
        contact_points (numpy.ndarray): Points of contact.
        contact_normals (numpy.ndarray): Normals at the contact points.

    Returns:
        gws_quality (float): Convex hull volume of the grasp wrench space.
    """
    wrenches = []
    for p, n in zip(contact_points, contact_normals):
        torque = np.cross(p, n)
        wrench = np.hstack((n, torque))
        wrenches.append(wrench)
    wrenches = np.array(wrenches)
    hull = ConvexHull(wrenches)
    return hull.volume

def evaluate_force_closure(method, contact_points, contact_normals,
                           friction_coefficient=0.5):
    """
    Evaluate force closure using friction cones.

    Args:
        method (str): Method to compute force closure ('friction_cone').
        contact_points (numpy.ndarray): Points of contact.
        contact_normals (numpy.ndarray): Normals at contact points.
        friction_coefficient (float): Coefficient of friction.

    Returns:
        force_closure (bool): Whether force closure is achieved.
    """
    if method == 'friction_cone':
        return evaluate_force_closure_friction_cone(
            contact_points, contact_normals, friction_coefficient)
    else:
        raise ValueError(f"Unknown force closure method: {method}")

def evaluate_force_closure_friction_cone(contact_points, contact_normals,
                                         friction_coefficient):
    """
    Evaluate force closure using friction cones at contact points.

    Args:
        contact_points (numpy.ndarray): Contact points.
        contact_normals (numpy.ndarray): Normals at contact points.
        friction_coefficient (float): Friction coefficient.

    Returns:
        force_closure (bool): Whether force closure is achieved.
    """
    num_contacts = len(contact_normals)
    F = []
    for normal in contact_normals:
        f = generate_friction_cone_vectors(normal, friction_coefficient)
        F.extend(f)
    F = np.array(F)
    hull = ConvexHull(F)
    return np.all(hull.equations[:, -1] < 0)

def generate_friction_cone_vectors(normal, mu, num_vectors=8):
    """
    Generate vectors within the friction cone around a normal vector.

    Args:
        normal (numpy.ndarray): Surface normal vector.
        mu (float): Coefficient of friction.
        num_vectors (int): Number of vectors to sample within the cone.

    Returns:
        vectors (list): List of vectors within the friction cone.
    """
    vectors = []
    normal = normal / np.linalg.norm(normal)
    for i in range(num_vectors):
        theta = 2 * np.pi * i / num_vectors
        tangent = np.array([np.cos(theta), np.sin(theta), 0])
        tangent = tangent - tangent.dot(normal) * normal
        tangent = tangent / np.linalg.norm(tangent)
        f = normal + mu * tangent
        vectors.append(f / np.linalg.norm(f))
    return vectors
