import torch
import torch.nn as nn


class GraspingLoss(nn.Module):
    """
    Loss function for the grasping project.
    Uses Mean Squared Error between the predicted and actual biomechanical data.
    """

    def __init__(self):
        super(GraspingLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, outputs, targets, data):
        # outputs and targets shape: (batch_size, seq_length, num_labels)
        return self.mse(outputs, targets)
