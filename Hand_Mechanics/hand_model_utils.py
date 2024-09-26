# hand_model_utils.py

import numpy as np

class Joint:
    def __init__(self, name, parent=None, length=1.0, rotation_axis=np.array([0, 0, 1]), offset=None):
        """
        Initialize a joint.

        Args:
            name (str): Name of the joint.
            parent (Joint): Parent joint.
            length (float): Length of the joint segment.
            rotation_axis (np.array): Axis of rotation for the joint.
            offset (np.array): Offset from the parent joint.
        """
        self.name = name
        self.parent = parent
        self.length = length
        self.rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        if offset is None:
            # Default offset is along the parent's local z-axis by parent's length
            if parent is not None:
                self.offset = np.array([0, 0, parent.length])
            else:
                self.offset = np.array([0, 0, 0])  # Root joint
        else:
            self.offset = offset  # Local offset from parent joint
        self.angle = 0.0  # Current joint angle (in radians)
        self.position = np.zeros(3)  # Global position
        self.orientation = np.eye(3)  # Global orientation matrix
        self.children = []

        if parent:
            parent.children.append(self)

class ProximitySensor:
    def __init__(self, name, parent_joint, sensor_offset=None, sensor_orientation=np.eye(3)):
        """
        Initialize a proximity sensor.

        Args:
            name (str): Name of the sensor.
            parent_joint (Joint): The joint to which the sensor is attached.
            sensor_offset (float): Distance from the start of the joint segment (default is 0.5 * parent_joint.length).
            sensor_orientation (np.array): Local orientation matrix of the sensor.
        """
        self.name = name
        self.parent_joint = parent_joint
        if sensor_offset is None:
            self.sensor_offset = 0.5 * parent_joint.length  # Default to mid-way through the joint
        else:
            self.sensor_offset = sensor_offset  # Distance along the joint in the same units as joint length
        self.sensor_orientation = sensor_orientation  # Local orientation
        self.global_position = np.zeros(3)
        self.global_orientation = np.eye(3)
        self.reading = None  # Distance measurement

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

def compute_forward_kinematics(joint):
    """
    Recursively compute the global position and orientation of the joint.
    """
    if joint.parent is None:
        joint.position = joint.offset
        joint.orientation = np.eye(3)
    else:
        # The parent joint has already been computed
        R = rotation_matrix(joint.rotation_axis, joint.angle)
        joint.orientation = joint.parent.orientation @ R
        # Compute joint's position
        joint.position = joint.parent.position + joint.parent.orientation @ joint.offset
    # Recursively compute for child joints
    for child in joint.children:
        compute_forward_kinematics(child)

def update_sensors(sensors):
    """
    Update global position and orientation of each sensor.
    """
    for sensor in sensors:
        joint = sensor.parent_joint
        # Compute sensor position along the joint segment
        sensor_local_position = np.array([0, 0, sensor.sensor_offset])
        sensor.global_orientation = joint.orientation @ sensor.sensor_orientation
        sensor.global_position = joint.position + joint.orientation @ sensor_local_position

def compute_detected_points(sensors):
    """
    Compute the detected points based on sensor readings.

    Returns:
        np.array: Array of detected points (n_points, 3).
    """
    detected_points = []
    for sensor in sensors:
        if sensor.reading is not None:
            # Assuming sensor measures along its local z-axis
            direction = sensor.global_orientation[:, 2]  # Third column of orientation matrix
            point = sensor.global_position + direction * sensor.reading
            detected_points.append(point)
    return np.array(detected_points)
