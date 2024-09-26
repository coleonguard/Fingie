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
