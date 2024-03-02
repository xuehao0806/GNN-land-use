import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch
from torch_geometric.loader import NeighborLoader, RandomNodeSampler

@torch.no_grad()
def test(model, loader, device):
    """
        Evaluate the model performance on a given dataset.

        Parameters:
        - model (torch.nn.Module): The trained model to be evaluated.
        - loader (DataLoader): The PyTorch DataLoader for loading the dataset.
        - device (torch.device): The device (CPU/GPU) on which the computation will be performed.

        Returns:
        - The average loss over the test dataset.
        """
    model.eval() # Set the model to evaluation mode
    total_loss = total_examples = 0
    loss_fn = torch.nn.MSELoss() # Define the loss function
    for data in loader: # Iterate over each batch from the loader
        data = data.to(device) # Move the data to the specified device
        out = model(data.x, data.edge_index) # Forward pass: compute the model output
        loss = loss_fn(out.squeeze(), data.y.float()) # Compute the loss
        total_loss += float(loss) * data.num_nodes  # Aggregate the loss
        total_examples += data.num_nodes  # Count the total number of examples
    return total_loss / total_examples  # Return the average loss


def train(model, train_loader, optimizer, device):
    """
        Train the model using the given training data.

        Parameters:
        - model (torch.nn.Module): The model to be trained.
        - train_loader (DataLoader): The PyTorch DataLoader for loading the training dataset.
        - optimizer (torch.optim.Optimizer): The optimizer used for model training.
        - device (torch.device): The device (CPU/GPU) on which the computation will be performed.

        Returns:
        - The average loss over the training dataset.
        """
    model.train()  # Set the model to training mode
    total_loss = total_examples = 0
    loss_fn = torch.nn.MSELoss()  # Define the loss function
    for data in train_loader:  # Iterate over each batch from the loader
        data = data.to(device)  # Move the data to the specified device
        optimizer.zero_grad()  # Clear the gradients
        out = model(data.x, data.edge_index)  # Forward pass: compute the model output
        loss = loss_fn(out.squeeze(), data.y.float())  # Compute the loss
        loss.backward()  # Backward pass: compute the gradient
        optimizer.step()  # Update the model parameters
        total_loss += float(loss) * data.num_nodes  # Aggregate the loss
        total_examples += data.num_nodes  # Count the total number of examples
    return total_loss / total_examples  # Return the average loss

# Define a function to initialize loaders
def get_loader(data, loader_name):
    """
    Prepares and returns the data loaders for training, validation, and testing datasets.

    Parameters:
    - data (torch_geometric.data.Data): The input graph data.
    - loader_name (str): The name of the loader to use. It should be either 'Neighbor' or 'RandomNode'.

    Returns:
    - train_loader: DataLoader for the training dataset.
    - val_loader: DataLoader for the validation dataset.
    - test_loader: DataLoader for the testing dataset.

    Raises:
    - ValueError: If the loader_name is not recognized.
    """
    # Preprocess the data by removing edges and nodes related to the test set to avoid data leakage
    data_for_train_val = remove_reserved_nodes_and_edges(data, data.test_mask)

    if loader_name == 'Neighbor':
        # Parameters for NeighborLoader
        kwargs = {'batch_size': 64, 'num_workers': 2, 'persistent_workers': True, 'subgraph_type': 'induced'}
        # Initialize NeighborLoader for training, validation, and testing
        # The loader fetches neighbors up to 1 layer deep with a maximum of 10 neighbors for each node
        train_loader = NeighborLoader(data_for_train_val, num_neighbors=[10] * 1, input_nodes=data.train_mask, **kwargs)
        val_loader = NeighborLoader(data, num_neighbors=[10] * 1, input_nodes=data.val_mask, **kwargs)
        test_loader = NeighborLoader(data, num_neighbors=[10] * 1, input_nodes=data.test_mask, **kwargs)

    elif loader_name == 'RandomNode':
        # Parameters for RandomNodeSampler
        kwargs = {'num_workers': 4, 'persistent_workers': True}
        # Initialize RandomNodeSampler for training, validation, and testing
        # The dataset is divided into parts (60 for training, 20 for validation, and 20 for testing)
        train_loader = RandomNodeSampler(data, num_parts=60, shuffle=True, **kwargs)
        val_loader = RandomNodeSampler(data, num_parts=20, shuffle=False, **kwargs)
        test_loader = RandomNodeSampler(data, num_parts=20, shuffle=False, **kwargs)

    else:
        # Raise an error if an unsupported loader name is provided
        raise ValueError(f"Unsupported loader name: {loader_name}")

    return train_loader, val_loader, test_loader


def remove_reserved_nodes_and_edges(data, reserved_node_mask):
    """
    Modifies the input graph data by removing the edges connected to reserved nodes
    and zeroing out features for these nodes to prevent data leakage during training.

    Parameters:
    - data (torch_geometric.data.Data): The input graph data.
    - reserved_node_mask (torch.Tensor): A boolean mask indicating which nodes are reserved (e.g., for testing).

    Returns:
    - data_sub (torch_geometric.data.Data): The modified graph data with reserved nodes and edges removed.
    """
    # Retrieve the indices of reserved nodes
    reserved_nodes = reserved_node_mask.nonzero().squeeze()
    # Clone the input data to avoid modifying the original graph
    data_sub = data.clone()
    # Initialize a mask to identify edges that should be retained
    edge_mask = torch.ones(data_sub.edge_index.size(1), dtype=torch.bool)

    # Iterate over reserved nodes to update the edge mask, marking edges connected to these nodes for removal
    for node in reserved_nodes:
        # Mark all outgoing and incoming edges of the reserved node as False (to be removed)
        edge_mask[data_sub.edge_index[0] == node] = False
        edge_mask[data_sub.edge_index[1] == node] = False

    # Apply the mask to filter out edges connected to reserved nodes
    data_sub.edge_index = data_sub.edge_index[:, edge_mask]

    # If edge attributes exist, filter them out as well
    if hasattr(data_sub, 'edge_attr'):
        data_sub.edge_attr = data_sub.edge_attr[edge_mask]

    # Zero out the features of reserved nodes to prevent their influence during training/validation
    data_sub.x[reserved_nodes] = 0
    # Update the masks to exclude reserved nodes from being considered in respective phases (training/validation/testing)
    data_sub.train_mask[reserved_nodes] = False
    data_sub.val_mask[reserved_nodes] = False
    data_sub.test_mask[reserved_nodes] = False

    return data_sub

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate the mean absolute percentage error (MAPE) between true and predicted values.

    Parameters:
    - y_true (numpy.array): The true values.
    - y_pred (numpy.array): The predicted values.

    Returns:
    - The MAPE value as a percentage.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero by masking out zero values in y_true
    non_zero_mask = y_true != 0
    # Compute MAPE only for non-zero true values and multiply by 100 to get a percentage
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

# 1. evaluation
def evaluation(model, loader, device):
    """
    Evaluate the model using the provided loader and compute various metrics.

    Parameters:
    - model (torch.nn.Module): The model to be evaluated.
    - loader (DataLoader): The data loader to provide input data.
    - device (torch.device): The device to perform computation on.

    Returns:
    - results_df (pandas.DataFrame): A DataFrame containing evaluation metrics for each indicator.
    """
    model.eval()  # Set the model to evaluation mode
    # Initialize dictionaries to store predictions and actual values for 6 assumed indicators
    predictions = {i: [] for i in range(6)}
    actuals = {i: [] for i in range(6)}

    # Iterate over the loader to collect model predictions and actual values
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)  # Get model predictions
        # Store predictions and actuals for each indicator
        for i in range(6):
            predictions[i].extend(out[:, i].cpu().detach().numpy())
            actuals[i].extend(data.y[:, i].cpu().numpy())

    # Define the indicators and metrics to be evaluated
    indicators = ['office', 'sustenance', 'transport', 'retail', 'leisure', 'residence']
    metrics = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R2']

    # Initialize a DataFrame to store the computed metrics
    results_df = pd.DataFrame(columns=metrics)

    # Compute and store each metric for every indicator
    for i, indicator in enumerate(indicators):
        mse = mean_squared_error(actuals[i], predictions[i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals[i], predictions[i])
        mape = mean_absolute_percentage_error(actuals[i], predictions[i])
        r2 = r2_score(actuals[i], predictions[i])

        # Populate the DataFrame with the results
        results_df.loc[indicator] = [round(mse, 3), round(rmse, 3), round(mae, 3), round(mape, 3), round(r2, 3)]

    return results_df