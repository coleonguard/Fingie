# hand_mechanics.py
import numpy as np

def rotation_matrix(axis, theta):
    """
    Create a rotation matrix given an axis ('x', 'y', 'z') and an angle in degrees.
    """
    theta = np.radians(theta)
    if axis == 'x':
        R = np.array([[1, 0, 0],
                      [0, np.cos(theta), -np.sin(theta)],
                      [0, np.sin(theta),  np.cos(theta)]])
    elif axis == 'y':
        R = np.array([[ np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])
    elif axis == 'z':
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta),  np.cos(theta), 0],
                      [0, 0, 1]])
    else:
        raise ValueError("Invalid axis. Axis must be 'x', 'y', or 'z'.")
    return R

class Joint:
    def __init__(self, name, parent=None, position=np.zeros(3), axes=[], angles=[], children=None):
        self.name = name
        self.parent = parent  # Parent joint
        self.position = position  # Relative position from parent joint
        self.axes = axes  # Axes of rotation ['x', 'y', 'z']
        self.angles = angles  # Joint angles in degrees
        self.children = children if children is not None else []  # Child joints
        self.rotation = np.eye(3)  # Local rotation matrix
        self.global_rotation = np.eye(3)  # Global rotation matrix
        self.global_position = np.zeros(3)  # Global position

        if self.parent:
            self.parent.children.append(self)

    def update_rotation(self):
        self.rotation = np.eye(3)
        for axis, angle in zip(self.axes, self.angles):
            R = rotation_matrix(axis, angle)
            self.rotation = self.rotation @ R

def compute_forward_kinematics(joint, parent_position=np.zeros(3), parent_rotation=np.eye(3)):
    """
    Recursive function to compute the position of each joint.
    """
    joint.update_rotation()
    joint.global_rotation = parent_rotation @ joint.rotation
    joint.global_position = parent_position + parent_rotation @ joint.position

    for child in joint.children:
        compute_forward_kinematics(child, joint.global_position, joint.global_rotation)

class Hand:
    def __init__(self, finger_spacing=None, finger_lengths=None, thumb_length=None):
        """
        Initialize the Hand object with customizable individual-dependent features.

        Parameters:
        - finger_spacing: List of Y positions for each finger's MCP joint relative to the wrist.
          Default: [-1.5, -0.5, 0.5, 1.5] (from pinky to index finger)
        - finger_lengths: Dictionary specifying lengths for each finger segment.
          Default: Lengths for 'index', 'middle', 'ring', 'pinky' fingers.
        - thumb_length: Dictionary specifying lengths for the thumb segments.
          Default: {'cmc_to_mcp': 1.0, 'mcp_to_ip': 1.0, 'ip_to_tip': 0.8}
        """
        # Default finger spacing (from pinky to index finger)
        if finger_spacing is None:
            finger_spacing = [-1.5, -0.5, 0.5, 1.5]  # Y positions relative to the wrist
        self.finger_spacing = finger_spacing

        # Default finger lengths
        if finger_lengths is None:
            finger_lengths = {
                'index': {'mcp_to_pip': 3.0, 'pip_to_dip': 2.0, 'dip_to_tip': 1.5},
                'middle': {'mcp_to_pip': 3.5, 'pip_to_dip': 2.5, 'dip_to_tip': 1.5},
                'ring': {'mcp_to_pip': 3.0, 'pip_to_dip': 2.0, 'dip_to_tip': 1.5},
                'pinky': {'mcp_to_pip': 2.5, 'pip_to_dip': 1.5, 'dip_to_tip': 1.0},
            }
        self.finger_lengths = finger_lengths

        # Default thumb lengths
        if thumb_length is None:
            thumb_length = {'cmc_to_mcp': 3.2, 'mcp_to_ip': 2.5, 'ip_to_tip': 1.5}
        self.thumb_length = thumb_length

        # Adjust the wrist position to be below the fingers (along Z-axis)
        self.wrist = Joint('wrist', position=np.array([0, 0, -5]))
        # Define fingers
        self.fingers = []
        self.define_fingers()

    def define_fingers(self):
        # Define finger joints with 'tip' included
        # Each finger has joints: mcp, pip, dip, tip

        # Unpack finger lengths
        fl = self.finger_lengths
        fs = self.finger_spacing

        # Pinky finger
        pinky_mcp = Joint('pinky_mcp', parent=self.wrist, position=np.array([fs[0], 0, 5]), axes=['z', 'x'], angles=[0, 0])
        pinky_pip = Joint('pinky_pip', parent=pinky_mcp, position=np.array([0, 0, fl['pinky']['mcp_to_pip']]), axes=['x'], angles=[0])
        pinky_dip = Joint('pinky_dip', parent=pinky_pip, position=np.array([0, 0, fl['pinky']['pip_to_dip']]), axes=['x'], angles=[0])
        pinky_tip = Joint('pinky_tip', parent=pinky_dip, position=np.array([0, 0, fl['pinky']['dip_to_tip']]), axes=[], angles=[])
        self.fingers.append([pinky_mcp, pinky_pip, pinky_dip, pinky_tip])

        # Ring finger
        ring_mcp = Joint('ring_mcp', parent=self.wrist, position=np.array([fs[1], 0, 5]), axes=['z', 'x'], angles=[0, 0])
        ring_pip = Joint('ring_pip', parent=ring_mcp, position=np.array([0, 0, fl['ring']['mcp_to_pip']]), axes=['x'], angles=[0])
        ring_dip = Joint('ring_dip', parent=ring_pip, position=np.array([0, 0, fl['ring']['pip_to_dip']]), axes=['x'], angles=[0])
        ring_tip = Joint('ring_tip', parent=ring_dip, position=np.array([0, 0, fl['ring']['dip_to_tip']]), axes=[], angles=[])
        self.fingers.append([ring_mcp, ring_pip, ring_dip, ring_tip])

        # Middle finger
        middle_mcp = Joint('middle_mcp', parent=self.wrist, position=np.array([fs[2], 0, 5]), axes=['z', 'x'], angles=[0, 0])
        middle_pip = Joint('middle_pip', parent=middle_mcp, position=np.array([0, 0, fl['middle']['mcp_to_pip']]), axes=['x'], angles=[0])
        middle_dip = Joint('middle_dip', parent=middle_pip, position=np.array([0, 0, fl['middle']['pip_to_dip']]), axes=['x'], angles=[0])
        middle_tip = Joint('middle_tip', parent=middle_dip, position=np.array([0, 0, fl['middle']['dip_to_tip']]), axes=[], angles=[])
        self.fingers.append([middle_mcp, middle_pip, middle_dip, middle_tip])

        # Index finger
        index_mcp = Joint('index_mcp', parent=self.wrist, position=np.array([fs[3], 0, 5]), axes=['z', 'x'], angles=[0, 0])
        index_pip = Joint('index_pip', parent=index_mcp, position=np.array([0, 0, fl['index']['mcp_to_pip']]), axes=['x'], angles=[0])
        index_dip = Joint('index_dip', parent=index_pip, position=np.array([0, 0, fl['index']['pip_to_dip']]), axes=['x'], angles=[0])
        index_tip = Joint('index_tip', parent=index_dip, position=np.array([0, 0, fl['index']['dip_to_tip']]), axes=[], angles=[])
        self.fingers.append([index_mcp, index_pip, index_dip, index_tip])

        # Thumb
        tl = self.thumb_length
        thumb_cmc = Joint('thumb_cmc', parent=self.wrist, position=np.array([2.0, -1.0, 0]), axes=['z', 'x'], angles=[0, 0])
        thumb_mcp = Joint('thumb_mcp', parent=thumb_cmc, position=np.array([0, 0, tl['cmc_to_mcp']]), axes=['x'], angles=[0])
        thumb_ip = Joint('thumb_ip', parent=thumb_mcp, position=np.array([0, 0, tl['mcp_to_ip']]), axes=['x'], angles=[0])
        thumb_tip = Joint('thumb_tip', parent=thumb_ip, position=np.array([0, 0, tl['ip_to_tip']]), axes=[], angles=[])
        self.fingers.append([thumb_cmc, thumb_mcp, thumb_ip, thumb_tip])

    def set_joint_angles(self, joint_angles):
        """
        Set the joint angles given a dictionary of joint names and angles.
        """
        all_joints = self.get_all_joints()
        for joint_name, angles in joint_angles.items():
            for joint in all_joints:
                if joint.name == joint_name:
                    joint.angles = angles

    def compute_forward_kinematics(self):
        compute_forward_kinematics(self.wrist)

    def get_all_joints(self):
        """
        Return a list of all joints in the hand.
        """
        joints = []
        def traverse(joint):
            joints.append(joint)
            for child in joint.children:
                traverse(child)
        traverse(self.wrist)
        return joints

def visualize_hand(hand, use_plotly=True):
    """
    Visualize the hand model using Matplotlib or Plotly.

    Parameters:
    - hand: Hand object to visualize.
    - use_plotly (bool): If True, use Plotly for interactive visualization. If False, use Matplotlib.
    """
    joints = hand.get_all_joints()

    # Extract joint positions
    lines = []
    x_coords = []
    y_coords = []
    z_coords = []
    joint_names = []
    for joint in joints:
        if joint.parent is not None:
            x_line = [joint.parent.global_position[0], joint.global_position[0]]
            y_line = [joint.parent.global_position[1], joint.global_position[1]]
            z_line = [joint.parent.global_position[2], joint.global_position[2]]
            lines.append((x_line, y_line, z_line))

        x_coords.append(joint.global_position[0])
        y_coords.append(joint.global_position[1])
        z_coords.append(joint.global_position[2])
        joint_names.append(joint.name)

    if use_plotly:
        import plotly.graph_objects as go

        fig = go.Figure()

        # Add lines (bones)
        for line in lines:
            fig.add_trace(go.Scatter3d(
                x=line[0],
                y=line[1],
                z=line[2],
                mode='lines',
                line=dict(color='black', width=5)
            ))

        # Add markers (joints)
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers+text',
            marker=dict(size=5, color='blue'),
            text=joint_names,
            textposition="top center"
        ))

        # Set layout
        fig.update_layout(
            title='Hand Model Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data',
                xaxis=dict(range=[-5, 5]),
                yaxis=dict(range=[-5, 5]),
                zaxis=dict(range=[-5, 15])
            )
        )

        fig.show()
    else:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot joints and segments
        for line in lines:
            ax.plot(line[0], line[1], line[2], 'k-', linewidth=2)

        # Plot joints
        ax.scatter(x_coords, y_coords, z_coords, c='b', marker='o', s=50)

        # Optionally, annotate joint names
        for joint in joints:
            ax.text(joint.global_position[0], joint.global_position[1], joint.global_position[2],
                    joint.name, fontsize=8)

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Hand Model Visualization')
        ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

        # Set limits for better visualization
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-5, 15)

        plt.show()
