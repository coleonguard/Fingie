# joint.py

import numpy as np
import numpy as np
import matplotlib.pyplot as plt
# from hand import Hand
# from visualization import visualize_hand

class Joint:
    def __init__(self, name, parent=None, length=1.0, rotation_axes=[np.array([0, 0, 1])], offset=np.zeros(3)):
        """
        Initialize a joint.

        Args:
            name (str): Name of the joint.
            parent (Joint): Parent joint.
            length (float): Length of the joint segment.
            rotation_axes (list of np.array): Axes of rotation for the joint (one or two axes).
            offset (np.array): Offset from the parent joint.
        """
        self.name = name
        self.parent = parent
        self.length = length
        self.rotation_axes = [axis / np.linalg.norm(axis) for axis in rotation_axes]
        self.offset = offset  # Local offset from parent joint
        self.angles = [0.0] * len(rotation_axes)  # Current joint angles (in radians)
        self.position = np.zeros(3)  # Global position
        self.orientation = np.eye(3)  # Global orientation matrix
        self.children = []

        if parent:
            parent.children.append(self)

def compute_forward_kinematics(joint):
    """
    Recursively compute the global position and orientation of the joint.
    """
    if joint.parent is None:
        joint.position = joint.offset
        joint.orientation = np.eye(3)
    else:
        # The parent joint has already been computed
        joint.orientation = joint.parent.orientation.copy()
        for axis, angle in zip(joint.rotation_axes, joint.angles):
            R = rotation_matrix(axis, angle)
            joint.orientation = joint.orientation @ R
        # Compute joint's position
        joint.position = joint.parent.position + joint.parent.orientation @ joint.offset
    # Recursively compute for child joints
    for child in joint.children:
        compute_forward_kinematics(child)

def rotation_matrix(axis, angle):
    """
    Compute rotation matrix using Rodrigues' rotation formula.
    """
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R