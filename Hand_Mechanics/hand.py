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
