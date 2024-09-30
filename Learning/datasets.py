# datasets.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import networkx as nx
from torch_geometric.utils import from_networkx
from filtering import apply_filter  # Import the filtering module

class GraspingDataset(Dataset):
    """
    Dataset class for the grasping project.
    Loads proximity sensor data as inputs and biomechanical data as labels.
    """

    def __init__(self, data_path, use_graph=False, filter_method=None, filter_params={}):
        self.use_graph = use_graph
        self.filter_method = filter_method
        self.filter_params = filter_params

        # Load data from files (Replace with your actual data loading logic)
        num_samples = 1000
        sequence_length = 50
        num_features = 10  # Number of features from proximity sensors
        num_labels = 5     # Number of biomechanical outputs

        self.inputs = np.random.randn(num_samples, sequence_length, num_features)
        self.labels = np.random.randn(num_samples, sequence_length, num_labels)

        # Apply filtering if specified
        if self.filter_method is not None:
            for i in range(num_samples):
                for j in range(num_features):
                    self.inputs[i, :, j] = apply_filter(
                        self.filter_method,
                        self.inputs[i, :, j],
                        **self.filter_params
                    )

        # If using graphs, construct adjacency matrices or edge lists
        if self.use_graph:
            self.graphs = []
            for _ in range(num_samples):
                G = nx.complete_graph(num_features)
                edge_index = from_networkx(G).edge_index
                self.graphs.append(torch.tensor(edge_index, dtype=torch.long))

        # Convert numpy arrays to PyTorch tensors
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if self.use_graph:
            return self.inputs[idx], self.labels[idx], self.graphs[idx]
        else:
            return self.inputs[idx], self.labels[idx]


def create_data_loaders(dataset, batch_size, validation_split):
    """
    Splits the dataset into training and validation sets and creates data loaders.
    """
    # Calculate split indices
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    # Shuffle indices
    np.random.shuffle(indices)

    # Split indices
    train_indices, val_indices = indices[split:], indices[:split]

    # Create subsets
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Define a function to load data from files
def load_data(input_file, label_file, file_type='npy'):
    """
    Load proximity sensor data (inputs) and biomechanical data (labels) from files.
    
    Args:
        input_file (str): Path to the file containing input data (proximity sensor features).
        label_file (str): Path to the file containing label data (biomechanical data).
        file_type (str): Type of file ('npy', 'csv', 'pickle', etc.).

    Returns:
        inputs (torch.Tensor): Loaded input data as a PyTorch tensor.
        labels (torch.Tensor): Loaded label data as a PyTorch tensor.
    """
    if file_type == 'npy':
        inputs = np.load(input_file)
        labels = np.load(label_file)
    elif file_type == 'csv':
        inputs = np.genfromtxt(input_file, delimiter=',')
        labels = np.genfromtxt(label_file, delimiter=',')
    elif file_type == 'pickle':
        import pickle
        with open(input_file, 'rb') as f:
            inputs = pickle.load(f)
        with open(label_file, 'rb') as f:
            labels = pickle.load(f)
    else:
        raise ValueError("Unsupported file type: {}".format(file_type))
    
    # Convert the data to PyTorch tensors
    inputs = torch.tensor(inputs, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    
    return inputs, labels