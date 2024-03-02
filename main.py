import torch
from torch.optim import AdamW
from models import get_model
from utils import get_loader, train, test, remove_reserved_nodes_and_edges, evaluation
import argparse



## PARAMETERS
# Create an argument parser object
parser = argparse.ArgumentParser(description='Train a GNN for land use identifcation.')
# Add arguments to the parser.
parser.add_argument('--loader_name', type=str, default='Neighbor', choices=['Neighbor', 'RandomNode'], help='Data loader to use (Neighbor or RandomNode).')
parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the model.')
parser.add_argument('--model_name', type=str, default='GraphSAGE', choices=['GraphSAGE', 'GCNModel', 'GATModel'], help='Model to use for training.')
parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate for the optimizer.')
parser.add_argument('--epoch_num', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--data_path', type=str, default='data/processed/data.pt', help='Path to the data file.')
parser.add_argument('--model_save_path', type=str, default='models/', help='Path where the trained model should be saved.')

# Parse the command-line arguments
args = parser.parse_args()
# Extract the arguments to use them in the training script
loader_name = args.loader_name
hidden_size = args.hidden_size
model_name = args.model_name
learning_rate = args.learning_rate
epoch_num = args.epoch_num
data_path = args.data_path
model_save_path = args.model_save_path
# Set up the device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## DATA LOADING AND SAMPLING
data = torch.load(data_path)
data_for_train_val = remove_reserved_nodes_and_edges(data, data.test_mask) # removed edges/nodes data in training including the test nodes
train_loader, val_loader, test_loader = get_loader(data, loader_name) # Initialise Dataloaders

## MODELLING
num_features = data.x.shape[1]
num_classes = data.y.shape[1]
model = get_model(model_name, num_features, hidden_size, num_classes, device)
optimizer = AdamW(model.parameters(), lr=learning_rate)

## TRAINING
loss_history = []
val_acc_history = []
test_acc_history = []
for epoch in range(1, epoch_num + 1):
    loss = train(model, train_loader,optimizer, device)
    val_acc = test(model, val_loader, device)
    test_acc = test(model, test_loader, device)
    loss_history.append(loss)
    val_acc_history.append(val_acc)
    test_acc_history.append(test_acc)
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, ',
          f'Val: {val_acc:.4f} Test: {test_acc:.4f}')

## EVALUATION & SAVING
results = evaluation(model, test_loader, device)
torch.save(model, f'{model_save_path}{model_name}_{loader_name}.pth')
print(results)