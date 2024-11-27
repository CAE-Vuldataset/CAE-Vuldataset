import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GCN_twin(torch.nn.Module):
    def __init__(self, num_node_features):
        super(GCN_twin, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, num_node_features)
        self.conv2 = GCNConv(num_node_features, num_node_features)
        self.conv3 = GCNConv(num_node_features, num_node_features)
        self.L1 = Linear(num_node_features * 2, 2)

    def forward(self, x0, edge_attr0, edge_index0, x1, edge_attr1, edge_index1, batch0, batch1):
        # 1. Obtain node embeddings
        x0 = self.conv1(x0, edge_index0)
        x0 = x0.relu()
        x0 = self.conv2(x0, edge_index0)
        x0 = x0.relu()
        x0 = self.conv3(x0, edge_index0)
        # 2. Readout layer
        x0 = global_mean_pool(x0, batch0)  # [batch_size, hidden_channels]

        # 1. Obtain node embeddings
        x1 = self.conv1(x1, edge_index1)
        x1 = x1.relu()
        x1 = self.conv2(x1, edge_index1)
        x1 = x1.relu()
        x1 = self.conv3(x1, edge_index1)
        # 2. Readout layer
        x1 = global_mean_pool(x1, batch1)  # [batch_size, hidden_channels]

        x = torch.cat([x0, x1], dim=1)
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
        out = model.forward(data.x_s, data.edge_attr_s, data.edge_index_s, data.x_t, data.edge_attr_t, data.edge_index_t, data.x_s_batch, data.x_t_batch)  # Perform a single forward pass.
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
        out = model.forward(data.x_s, data.edge_attr_s, data.edge_index_s, data.x_t, data.edge_attr_t, data.edge_index_t, data.x_s_batch, data.x_t_batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        # statistic.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        preds.extend(pred.int().tolist())
        labels.extend(data.y.int().tolist())

    acc = correct / len(loader.dataset)  # Derive ratio of correct predictions.

    return acc, preds, labels