# finger.py

import numpy as np
import matplotlib.pyplot as plt
from joint import Joint, rotation_matrix, compute_forward_kinematics
# from hand import Hand
# from visualization import visualize_hand

class Finger:
    def __init__(self, name, hand, base_offset, joint_lengths, mcp_rotation_axes):
        """
        Initialize a finger.

        Args:
            name (str): Name of the finger.
            hand (Hand): The hand object this finger belongs to.
            base_offset (np.array): Offset from the hand's base to the MCP joint.
            joint_lengths (dict): Lengths of the finger's joints.
            mcp_rotation_axes (list of np.array): Rotation axes for the MCP joint.
        """
        self.name = name
        self.joints = []
        self.hand = hand

        # MCP joint
        mcp = Joint(
            f'{name}_mcp',
            parent=hand.base_joint,
            length=joint_lengths['MCP'],
            rotation_axes=mcp_rotation_axes,
            offset=base_offset
        )
        self.joints.append(mcp)

        # PIP joint (flexion/extension)
        pip = Joint(
            f'{name}_pip',
            parent=mcp,
            length=joint_lengths['PIP'],
            rotation_axes=[np.array([0, 0, 1])]
        )
        self.joints.append(pip)

        # DIP joint (flexion/extension)
        dip = Joint(
            f'{name}_dip',
            parent=pip,
            length=joint_lengths['DIP'],
            rotation_axes=[np.array([0, 0, 1])]
        )
        self.joints.append(dip)

        # TIP joint (no rotation)
        tip = Joint(
            f'{name}_tip',
            parent=dip,
            length=joint_lengths['TIP'],
            rotation_axes=[]
        )
        self.joints.append(tip)

    def set_joint_angles(self, joint_angles):
        """
        Set the joint angles for this finger.

        Args:
            joint_angles (dict): Dictionary mapping joint names to angle values (in degrees).
        """
        for joint in self.joints:
            if joint.name in joint_angles:
                angles_in_radians = [np.deg2rad(angle) for angle in joint_angles[joint.name]]
                joint.angles = angles_in_radians

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

# hand.py
import numpy as np
import matplotlib.pyplot as plt
from joint import Joint, rotation_matrix, compute_forward_kinematics
from finger import Finger
# from visualization import visualize_hand

class Hand:
    def __init__(self):
        """
        Initialize the hand model.
        """
        self.base_joint = Joint('wrist', length=0.0)
        self.fingers = {}

        # Define joint lengths
        self.joint_lengths = {
            'MCP': 5.0,
            'PIP': 3.5,
            'DIP': 2.5,
            'TIP': 2.0
        }

        # Define finger base offsets (from wrist)
        finger_spacing = 1.0  # Spacing between fingers
        starting_x = -1.5  # Starting position along X-axis
        self.finger_names = ['index', 'middle', 'ring', 'pinky']
        for i, name in enumerate(self.finger_names):
            base_offset = np.array([starting_x + i * finger_spacing, 0, 0])
            mcp_rotation_axes = [
                np.array([0, 0, 1]),  # Flexion/extension
                np.array([0, 1, 0])   # Abduction/adduction
            ]
            finger = Finger(
                name,
                self,
                base_offset,
                self.joint_lengths,
                mcp_rotation_axes
            )
            self.fingers[name] = finger

        # Add thumb
        self.add_thumb()

    def add_thumb(self):
        # Thumb joint lengths
        thumb_joint_lengths = {
            'CMC': 4.0,
            'MCP': 2.5,
            'IP': 2.0,
            'TIP': 1.5
        }

        # Offset and rotation for thumb CMC joint
        thumb_offset = np.array([2.0, -1.0, 0.0])  # Adjust as needed
        thumb_rotation_axes = [
            np.array([0, 0, 1]),  # Abduction/adduction
            np.array([0, 1, 0])   # Flexion/extension
        ]

        # CMC Joint
        thumb_cmc = Joint(
            'thumb_cmc',
            parent=self.base_joint,
            length=thumb_joint_lengths['CMC'],
            rotation_axes=thumb_rotation_axes,
            offset=thumb_offset
        )

        # MCP Joint
        thumb_mcp = Joint(
            'thumb_mcp',
            parent=thumb_cmc,
            length=thumb_joint_lengths['MCP'],
            rotation_axes=[np.array([0, 1, 0])]  # Flexion/extension
        )

        # IP Joint
        thumb_ip = Joint(
            'thumb_ip',
            parent=thumb_mcp,
            length=thumb_joint_lengths['IP'],
            rotation_axes=[np.array([0, 1, 0])]  # Flexion/extension
        )

        # TIP Joint
        thumb_tip = Joint(
            'thumb_tip',
            parent=thumb_ip,
            length=thumb_joint_lengths['TIP'],
            rotation_axes=[]
        )

        # Add thumb joints to the hand
        self.thumb = [thumb_cmc, thumb_mcp, thumb_ip, thumb_tip]

    def set_joint_angles(self, joint_angles):
        """
        Set the joint angles for the entire hand.

        Args:
            joint_angles (dict): Dictionary mapping joint names to list of angle values (in degrees).
        """
        # Set angles for fingers
        for finger in self.fingers.values():
            finger.set_joint_angles(joint_angles)

        # Set angles for thumb
        for joint in self.thumb:
            if joint.name in joint_angles:
                angles_in_radians = [np.deg2rad(angle) for angle in joint_angles[joint.name]]
                joint.angles = angles_in_radians

    def compute_forward_kinematics(self):
        """
        Compute the forward kinematics for the entire hand.
        """
        compute_forward_kinematics(self.base_joint)

    def get_all_joints(self):
        """
        Get a list of all joints in the hand.

        Returns:
            list: All joints in the hand.
        """
        joints = [self.base_joint]
        for finger in self.fingers.values():
            joints.extend(finger.joints)
        joints.extend(self.thumb)
        return joints

# visualization.py
import numpy as np
import matplotlib.pyplot as plt
from joint import Joint, rotation_matrix, compute_forward_kinematics
from hand import Hand

def visualize_hand(hand):
    joints = hand.get_all_joints()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot joints and segments
    for joint in joints:
        if joint.parent is not None:
            # Draw line between joint and parent joint
            xs = [joint.parent.position[0], joint.position[0]]
            ys = [joint.parent.position[1], joint.position[1]]
            zs = [joint.parent.position[2], joint.position[2]]
            ax.plot(xs, ys, zs, 'k-', linewidth=2)

    # Plot joints
    for joint in joints:
        ax.scatter(joint.position[0], joint.position[1], joint.position[2], c='b', marker='o', s=50)
        # Optionally, annotate joint names
        # ax.text(joint.position[0], joint.position[1], joint.position[2], joint.name, fontsize=8)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Hand Model Visualization')
    ax.set_box_aspect([1,1,1])  # Equal aspect ratio

    # Set limits for better visualization
    ax.set_xlim(-5, 10)
    ax.set_ylim(-10, 5)
    ax.set_zlim(0, 20)

    plt.show()
