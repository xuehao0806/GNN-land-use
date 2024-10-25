
# Heterogeneous Graph Neural Networks with Post-hoc Explanations for Multi-modal and Explainable Land Use Inference

This repository contains the implementation of the paper **"Heterogeneous Graph Neural Networks with Post-hoc Explanations for Multi-modal and Explainable Land Use Inference."** The project focuses on land use inference using heterogeneous graph neural networks (HGT) and includes detailed post-hoc explanations through feature attribution and counterfactual analysis.

You can read the full paper on [arXiv](https://arxiv.org/abs/2406.13724#).

## Project Structure

- **Data Processing**
  - We process daily boarding/alighting data from various transportation modes (e.g., tube, bus) within London.
  - Features are standardized using z-score normalization, and edge features are constructed using walking routes between stations.
  - POI data is used to calculate land use density for each node (zone).

- **Graph Construction**
  - We build heterogeneous graphs using nodes (representing locations) and edges (representing travel connections). Node features are derived from mobility data, and edge features are based on transportation networks.
  
- **Model Training**
  - The heterogeneous graph neural network (HGT) is implemented using PyTorch Geometric (PyG). The model is trained on land use prediction tasks, with performance evaluated on multiple datasets.

- **Post-hoc Explanations**
  - We modify PyG’s Captum API to enable feature attribution for multi-target regression tasks using Integrated Gradients.
  - Counterfactual explanations are implemented to explore how changes in input data affect land use type predictions.

## Installation Instructions

### Prerequisites
To install the necessary dependencies, refer to the `requirements.txt` file. You can create the environment using the following command:

```bash
conda create --name <env_name> --file requirements.txt
```

### Modifying PyTorch Geometric for Multi-target Regression Explanations

We made modifications to the original PyTorch Geometric package to adapt it for multi-target regression tasks. These adjustments allow us to compute explanations (e.g., feature attribution) in this context. To install our custom version of PyG:

1. First, uninstall the original `torch_geometric` package:
   
   ```bash
   pip uninstall torch-geometric
   ```

2. Then, install the forked version from GitHub:

   ```bash
   git clone https://github.com/adamdejl/pytorch_geometric
   cd pytorch_geometric
   pip install -e .
   ```

   **Note:** The current implementation does not fully support feature attributions for RGCN models due to issues with the `edge_types` argument in the explanation logic. We recommend experimenting with homograph models or exploring fixes for this issue.

## Project Workflow

The overall workflow is visualized in the following figure, which outlines the entire pipeline from data preprocessing to model training and post-hoc explanation methods.

![Workflow Visualization](visualisation/Detailed_methodology_flow_charts.jpg)

### 1. Data Preparation
Scripts for preparing the data, including normalization and feature extraction:

- `processing_input_1_selected_inner_london.py`: Prepares the input mobility data for nodes.
- `processing_graph_1_road_types.py`: Processes edge features such as road types and walking distances.
- `data_preparation.py`: Standardizes features and prepares the dataset for graph construction.

### 2. Graph Construction
Scripts for building heterogeneous graphs:

- `data_preparation_hetero.py`: Constructs heterogeneous subgraphs.
- `main_hetero.py`: Defines the data loaders for training, validation, and testing.

### 3. Model Training
Scripts related to the model definition and training:

- `models.py`: Contains the HGT class with layers of HGT-Conv, residual connections, and an MLP for final prediction.
- `main_hetero.py`: Trains and tests the HGT model.

### 4. Explainability
Scripts for implementing post-hoc explanations:

- `pytorch_geometric.py`: Modified PyG's Captum API for multi-target regression.
- `explainability_FA.py`: Generates feature attribution explanations using Integrated Gradients.
- `counterfactual_explanation.py`: Provides counterfactual explanations compatible with HGT models.

## Notebooks
A Jupyter notebook demonstrating how to compute GNN feature attributions using the implemented methods is included. You can find this notebook in the repository for more detailed usage examples.
