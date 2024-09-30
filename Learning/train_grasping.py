# train_grasping.py

import os
import torch
import wandb
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import GraspingDataset, create_data_loaders
from models import initialize_model
from losses import GraspingLoss
import torch.optim as optim


class CheckPointManager:
    """Manages saving model checkpoints during training."""

    def __init__(self, metric_name="loss"):
        self.metric_name = metric_name
        self.best_loss = float("inf")
        self.ckpt_path = str(Path(__file__).parent / "models")
        os.makedirs(self.ckpt_path, exist_ok=True)

    def save(self, model, metric, epoch, step, model_name):
        self._model_save(model, model_name)
        artifact = wandb.Artifact(
            type="model",
            name=f"{wandb.run.id}_{model_name}",
            metadata={self.metric_name: metric, "epoch": epoch, "step": step},
        )

        artifact.add_dir(str(self.ckpt_path))

        aliases = ["latest"]

        if self.best_loss > metric:
            self.best_loss = metric
            aliases.append("best")

        wandb.run.log_artifact(artifact, aliases=aliases)

    def _model_save(self, model, model_name):
        torch.save(model.state_dict(), f"{self.ckpt_path}/{model_name}.pth")


def train_model(model, train_loader, val_loader, cfg, device):
    """
    Training loop for the model.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        cfg (dict): Configuration dictionary.
        device (torch.device): Device to run the training on.
    """
    loss_fn = GraspingLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.get('learning_rate', 0.001))
    num_epochs = cfg.get('num_epochs', 10)
    use_graph = cfg.get('model_type') == 'STGCN'
    model_name = cfg.get('model_type')
    ckpt_manager = CheckPointManager(metric_name="loss")

    step = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in pbar:
            step += 1
            optimizer.zero_grad()
            if use_graph:
                data, targets, edge_index = batch[0].to(device), batch[1].to(device), batch[2]
                # Prepare data for ST-GCN
                data = data.permute(0, 2, 1).unsqueeze(-1)
                outputs = model(data, edge_index[0].to(device))
                loss = loss_fn(outputs, targets, data)
            else:
                data, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(data)
                loss = loss_fn(outputs, targets, data)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Log loss
            wandb.log({"train_loss": loss.item()}, step=step)
            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = epoch_loss / len(train_loader)
        wandb.log({"avg_train_loss": avg_train_loss}, step=step)

        # Validation
        avg_val_loss = validate_model(model, val_loader, loss_fn, device, use_graph)
        wandb.log({"val_loss": avg_val_loss}, step=step)

        # Save checkpoint
        ckpt_manager.save(model, avg_val_loss, epoch=epoch, step=step, model_name=model_name)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")


def validate_model(model, val_loader, loss_fn, device, use_graph=False):
    """
    Validates the model on the validation set.

    Args:
        model (nn.Module): The model to validate.
        val_loader (DataLoader): DataLoader for validation data.
        loss_fn (nn.Module): Loss function.
        device (torch.device): Device to run validation on.
        use_graph (bool): Whether the model uses graph data.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            if use_graph:
                data, targets, edge_index = batch[0].to(device), batch[1].to(device), batch[2]
                data = data.permute(0, 2, 1).unsqueeze(-1)
                outputs = model(data, edge_index[0].to(device))
                loss = loss_fn(outputs, targets, data)
            else:
                data, targets = batch[0].to(device), batch[1].to(device)
                outputs = model(data)
                loss = loss_fn(outputs, targets, data)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss


def main(cfg):
    # Seed RNG
    torch.manual_seed(cfg.get('seed', 42))
    np.random.seed(cfg.get('seed', 42))

    # Set torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataset
    use_graph = cfg.get('model_type') == 'STGCN'
    dataset = GraspingDataset(data_path='path_to_your_data', use_graph=use_graph)
    input_dim = dataset.inputs.shape[2]
    output_dim = dataset.labels.shape[2]

    # Create data loaders
    batch_size = cfg.get('batch_size', 32)
    validation_split = cfg.get('validation_split', 0.2)
    train_loader, val_loader = create_data_loaders(dataset, batch_size, validation_split)

    # Initialize model
    model_type = cfg.get('model_type', 'TCN')
    model = initialize_model(model_type, input_dim, output_dim, cfg)
    model.to(device)

    # Initialize WandB
    experiment_name = cfg.get('experiment_name', 'Grasping_Project')
    wandb.init(project="Grasping_Project", entity="your_wandb_entity", name=experiment_name, config=cfg)

    # Train the model
    train_model(model, train_loader, val_loader, cfg, device)


if __name__ == "__main__":
    # Example configuration dictionary
    cfg = {
        'seed': 42,
        'batch_size': 32,
        'validation_split': 0.2,
        'learning_rate': 0.001,
        'num_epochs': 10,
        'model_type': 'STGCN',  # 'TCN', 'Transformer', 'STGCN', or 'RNNWithAttention'
        'kernel_size': 3,
        'dropout': 0.2,
        'hidden_size': 128,
        'num_layers': 2,
        'experiment_name': 'Grasping_with_STGCN',
        # Add more configuration parameters as needed
    }
    main(cfg)
