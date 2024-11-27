import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, num_node_features)
        self.conv2 = GCNConv(num_node_features, num_node_features)
        self.conv3 = GCNConv(num_node_features, num_node_features)
        self.L1 = Linear(num_node_features, 2)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.L1(x)

        return x

def GCNTrain(model, trainloader, optimizer, criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    lossTrain = 0
    for data in trainloader:  # Iterate in batches over the training dataset.
        optimizer.zero_grad()  # Clear gradients.
        data.to(device)
        out = model.forward(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        # statistic.
        lossTrain += loss.item() * len(data.y)

    lossTrain /= len(trainloader.dataset)

    return model, lossTrain

def GCNTest(model, loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    preds = []
    labels = []
    for data in loader:  # Iterate in batches over the training/test dataset.
        data.to(device)
        out = model.forward(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        # statistic.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        preds.extend(pred.int().tolist())
        labels.extend(data.y.int().tolist())

    acc = correct / len(loader.dataset)  # Derive ratio of correct predictions.

    return acc, preds, labels