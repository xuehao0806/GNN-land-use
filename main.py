import torch
from torch.optim import AdamW, Adam, SGD, RMSprop
from models import get_model
from utils import get_loader, train, test, evaluation
import argparse

## PARAMETERS
# Create an argument parser object
parser = argparse.ArgumentParser(description='Train a GNN for land use identification.')
# Add arguments to the parser.
parser.add_argument('--loader_name', type=str, default='Neighbor', choices=['Neighbor', 'RandomNode'], help='Data loader to use (Neighbor or RandomNode).')
parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the model.')
parser.add_argument('--model_name', type=str, default='GraphSAGE', choices=['RGCN', 'GCN', 'GAT','NN','GraphSAGE'], help='Model to use for training.')
parser.add_argument('--optimizer_name', type=str, default='Adam', choices=['Adam', 'AdamW', 'SGD', 'RMSprop'], help='Model to use for training.')

parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate for the optimizer.')
parser.add_argument('--sampling_ratio', type=float, default=1.0, help='Sampling ratio for the Neighbor loader.')
parser.add_argument('--noise_coefficient', type=float, default=0, help='Sampling ratio for the Neighbor loader.')
parser.add_argument('--epoch_num', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--data_path', type=str, default='data/processed/outer/data_homo.pt', help='Path to the data file.')
parser.add_argument('--model_save_path', type=str, default='models/', help='Path where the trained model should be saved.')

# Parse the command-line arguments
args = parser.parse_args()
# Extract the arguments to use them in the training script
loader_name = args.loader_name
print(f'the sampling way is {loader_name}')
hidden_size = args.hidden_size
model_name = args.model_name
num_relations = 5
print(f'the model is {model_name}')
learning_rate = args.learning_rate
epoch_num = args.epoch_num
data_path = args.data_path
sampling_ratio = args.sampling_ratio
noise_coefficient = args.noise_coefficient
model_save_path = args.model_save_path
optimizer_name = args.optimizer_name
# Set up the device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'the device is {device}')

## DATA LOADING AND SAMPLING

if __name__ == "__main__":
    data = torch.load(data_path)
    edge_type = data.edge_attr[:, 1:].argmax(dim=1)
    # print(f"HeteroData object: {data}")
    train_loader, val_loader, test_loader = get_loader(data, loader_name, sampling_ratio,
                                                       noise_coefficient)  # Initialise Dataloaders

    ## MODELLING
    num_features = data.x.shape[1]
    num_classes = data.y.shape[1]
    model = get_model(model_name, num_features, hidden_size, num_classes, device)
    # OPTIMISER
    optimizer_dict = {
        'Adam': Adam,
        'AdamW': AdamW,
        'SGD': SGD,
        'RMSprop': RMSprop
    }
    try:
        optimizer = optimizer_dict[optimizer_name](model.parameters(), lr=learning_rate)
    except KeyError:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}. Choose from {list(optimizer_dict.keys())}.")


    ## TRAINING
    for epoch in range(1, epoch_num + 1):
        loss = train(model_name, model, train_loader,optimizer, device)
        val_acc = test(model_name, model, val_loader, device)
        test_acc = test(model_name, model, test_loader, device)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, ',
              f'Val: {val_acc:.4f} Test: {test_acc:.4f}')

    ## EVALUATION & SAVING
    results_save_path = f'evaluation/hidden_size/{model_name}_{loader_name}_{hidden_size}.csv'
    # results_save_path = f'evaluation/optimizer/{model_name}_{loader_name}_{optimizer_name}.csv'
    # results_save_path = f'evaluation/sampling_ratio/{model_name}_{loader_name}_{sampling_ratio}.csv'
    # results_save_path = f'evaluation/noise_rate/{model_name}_{loader_name}_{noise_coefficient}.csv'
    results = evaluation(model_name, model, test_loader, device)
    results.to_csv(results_save_path, index=True)
    # torch.save(model, f'{model_save_path}{model_name}_{loader_name}.pth')
    print(results)