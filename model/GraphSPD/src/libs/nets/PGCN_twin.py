import torch
from torch.nn import Linear, Sequential, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling
from torch_geometric.nn import global_mean_pool, global_max_pool

class PGCN_twin(torch.nn.Module): # PGCN
    def __init__(self, num_node_features):
        super(PGCN_twin, self).__init__()
        torch.manual_seed(12345)
        self.conv1A = GCNConv(num_node_features, num_node_features)
        self.conv1B = GCNConv(num_node_features, num_node_features)
        self.conv1C = GCNConv(num_node_features, num_node_features)
        self.conv1D = GCNConv(num_node_features, num_node_features)
        self.conv2 = GCNConv(num_node_features * 4, num_node_features * 2)
        self.conv3 = GCNConv(num_node_features * 2, num_node_features)
        self.mlp = Sequential(
            Linear(num_node_features * 4, 8),
            ReLU(),
            Linear(8, 2)
        )

    def forward(self, x0, edge_attr0, edge_index0, x1, edge_attr1, edge_index1, batch0, batch1):
        # 1. Obtain graph embeddings.
        # Input 1
        x0A = self.conv1A(x0, edge_index0, edge_attr0[:, 0].contiguous()).relu()
        x0B = self.conv1B(x0, edge_index0, edge_attr0[:, 1].contiguous()).relu()
        x0C = self.conv1C(x0, edge_index0, edge_attr0[:, 2].contiguous()).relu()
        x0D = self.conv1D(x0, edge_index0, edge_attr0[:, 3].contiguous()).relu()
        # x0 = (x0A + x0B + x0C + x0D) / 4.0 # average
        x0 = torch.cat((x0A, x0B, x0C, x0D), dim=1)  # concat
        x0 = self.conv2(x0, edge_index0).relu()
        x0 = self.conv3(x0, edge_index0)
        # x0 = global_mean_pool(x0, batch0)
        x0 = torch.cat([global_mean_pool(x0, batch0), global_max_pool(x0, batch0)], dim=1)

        # Input 2
        x1A = self.conv1A(x1, edge_index1, edge_attr1[:, 0].contiguous()).relu()
        x1B = self.conv1B(x1, edge_index1, edge_attr1[:, 1].contiguous()).relu()
        x1C = self.conv1C(x1, edge_index1, edge_attr1[:, 2].contiguous()).relu()
        x1D = self.conv1D(x1, edge_index1, edge_attr1[:, 3].contiguous()).relu()
        # x1 = (x1A + x1B + x1C + x1D) / 4.0 # average
        x1 = torch.cat((x1A, x1B, x1C, x1D), dim=1)  # concat
        x1 = self.conv2(x1, edge_index1).relu()
        x1 = self.conv3(x1, edge_index1)
        # x1 = global_mean_pool(x1, batch1)
        x1 = torch.cat([global_mean_pool(x1, batch1), global_max_pool(x1, batch1)], dim=1)

        # 2. Readout layer
        x = torch.cat([x0, x1], dim=1)
        x = F.dropout(x, p=0.5, training=self.training)

        # 3. Apply a final classifier
        logits = self.mlp(x)

        return logits

class PGCN_g_twin(torch.nn.Module): # PGCN_global
    def __init__(self, num_node_features):
        super(PGCN_g_twin, self).__init__()
        torch.manual_seed(12345)
        self.conv1A = GCNConv(num_node_features, num_node_features)
        self.conv1B = GCNConv(num_node_features, num_node_features)
        self.conv1C = GCNConv(num_node_features, num_node_features)
        self.conv1D = GCNConv(num_node_features, num_node_features)
        self.conv2 = GCNConv(num_node_features, num_node_features)
        self.conv3 = GCNConv(num_node_features, num_node_features)
        self.pool = SAGPooling(num_node_features * 3, ratio=0.5)
        self.mlp = Sequential(
            Linear(num_node_features * 6, 32),
            ReLU(),
            Linear(32, 8),
            ReLU(),
            Linear(8, 2)
        )

    def forward(self, x0, edge_attr0, edge_index0, x1, edge_attr1, edge_index1, batch0, batch1):
        # 1. Obtain graph embeddings.
        # Layer 1
        x0A = self.conv1A(x0, edge_index0, edge_attr0[:, 0].contiguous()).relu()
        x0B = self.conv1B(x0, edge_index0, edge_attr0[:, 1].contiguous()).relu()
        x0C = self.conv1C(x0, edge_index0, edge_attr0[:, 2].contiguous()).relu()
        x0D = self.conv1D(x0, edge_index0, edge_attr0[:, 3].contiguous()).relu()
        gcn01 = (x0A + x0B + x0C + x0D) / 4.0 # average
        # gcn01 = torch.cat((x0A, x0B, x0C, x0D), dim=1) # concat
        gcn02 = self.conv2(gcn01, edge_index0).relu()
        gcn03 = self.conv3(gcn02, edge_index0)
        # 2. Readout layer
        gcn0 = torch.cat([gcn01, gcn02, gcn03], dim=1)
        pool0, pool_index0, _, pool_batch0, _, _ = self.pool(gcn0, edge_index0, batch=batch0)
        readout0 = global_mean_pool(pool0, pool_batch0)  # [batch_size, hidden_channels]

        # 1. Obtain graph embeddings.
        # Layer 1
        x1A = self.conv1A(x1, edge_index1, edge_attr1[:, 0].contiguous()).relu()
        x1B = self.conv1B(x1, edge_index1, edge_attr1[:, 1].contiguous()).relu()
        x1C = self.conv1C(x1, edge_index1, edge_attr1[:, 2].contiguous()).relu()
        x1D = self.conv1D(x1, edge_index1, edge_attr1[:, 3].contiguous()).relu()
        gcn11 = (x1A + x1B + x1C + x1D) / 4.0  # average
        # gcn11 = torch.cat((x1A, x1B, x1C, x1D), dim=1) # concat
        gcn12 = self.conv2(gcn11, edge_index1).relu()
        gcn13 = self.conv3(gcn12, edge_index1)
        # 2. Readout layer
        gcn1 = torch.cat([gcn11, gcn12, gcn13], dim=1)
        pool1, pool_index1, _, pool_batch1, _, _ = self.pool(gcn1, edge_index1, batch=batch1)
        readout1 = global_mean_pool(pool1, pool_batch1)  # [batch_size, hidden_channels]

        readout = torch.cat([readout0, readout1], dim=1)
        # 3. Apply a final classifier
        logits = self.mlp(readout)

        return logits

class PGCN_h_twin(torch.nn.Module): # PGCN_hierarchical
    def __init__(self, num_node_features):
        super(PGCN_h_twin, self).__init__()
        torch.manual_seed(12345)
        self.conv1A = GCNConv(num_node_features, num_node_features)
        self.conv1B = GCNConv(num_node_features, num_node_features)
        self.conv1C = GCNConv(num_node_features, num_node_features)
        self.conv1D = GCNConv(num_node_features, num_node_features)
        self.pool1 = SAGPooling(num_node_features, ratio=0.5)
        self.conv2 = GCNConv(num_node_features, num_node_features)
        self.pool2 = SAGPooling(num_node_features, ratio=0.5)
        self.conv3 = GCNConv(num_node_features, num_node_features)
        self.pool3 = SAGPooling(num_node_features, ratio=0.5)
        self.mlp = Sequential(
            Linear(num_node_features * 6, 32),
            ReLU(),
            Linear(32, 8),
            ReLU(),
            Linear(8, 2)
        )

    def forward(self, x0, edge_attr0, edge_index0, x1, edge_attr1, edge_index1, batch0, batch1):
        # 1. Obtain node embeddings.
        # Layer 1
        x0A = self.conv1A(x0, edge_index0, edge_attr0[:, 0].contiguous()).relu()
        x0B = self.conv1B(x0, edge_index0, edge_attr0[:, 1].contiguous()).relu()
        x0C = self.conv1C(x0, edge_index0, edge_attr0[:, 2].contiguous()).relu()
        x0D = self.conv1D(x0, edge_index0, edge_attr0[:, 3].contiguous()).relu()
        gcn01 = (x0A + x0B + x0C + x0D) / 4.0  # average
        pool01, edge_index01, _, batch01, _, _ = self.pool1(gcn01, edge_index0, batch=batch0)
        global_pool01 = global_mean_pool(pool01, batch01)
        # Layer 2
        gcn02 = self.conv2(pool01, edge_index01).relu()
        pool02, edge_index02, _, batch02, _, _ = self.pool2(gcn02, edge_index01, batch=batch01)
        global_pool02 = global_mean_pool(pool02, batch02)
        # Layer 3
        gcn03 = self.conv3(pool02, edge_index02)
        pool03, edge_index03, _, batch03, _, _ = self.pool3(gcn03, edge_index02, batch=batch02)
        global_pool03 = global_mean_pool(pool03, batch03)
        # 2. Readout layer
        readout0 = torch.cat([global_pool01, global_pool02, global_pool03], dim=1)  # [batch_size, hidden_channels]

        x1A = self.conv1A(x1, edge_index1, edge_attr1[:, 0].contiguous()).relu()
        x1B = self.conv1B(x1, edge_index1, edge_attr1[:, 1].contiguous()).relu()
        x1C = self.conv1C(x1, edge_index1, edge_attr1[:, 2].contiguous()).relu()
        x1D = self.conv1D(x1, edge_index1, edge_attr1[:, 3].contiguous()).relu()
        gcn11 = (x1A + x1B + x1C + x1D) / 4.0  # average
        pool11, edge_index11, _, batch11, _, _ = self.pool1(gcn11, edge_index1, batch=batch1)
        global_pool11 = global_mean_pool(pool11, batch11)
        # Layer 2
        gcn12 = self.conv2(pool11, edge_index11).relu()
        pool12, edge_index12, _, batch12, _, _ = self.pool2(gcn12, edge_index11, batch=batch11)
        global_pool12 = global_mean_pool(pool12, batch12)
        # Layer 3
        gcn13 = self.conv3(pool12, edge_index12)
        pool13, edge_index13, _, batch13, _, _ = self.pool3(gcn13, edge_index12, batch=batch12)
        global_pool13 = global_mean_pool(pool13, batch13)
        # 2. Readout layer
        readout1 = torch.cat([global_pool11, global_pool12, global_pool13], dim=1)  # [batch_size, hidden_channels]

        readout = torch.cat([readout0, readout1], dim=1)
        # 3. Apply a final classifier
        logits = self.mlp(readout)

        return logits

def PGCNTrain(model, trainloader, optimizer, criterion):
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

def PGCNTest(model, loader):
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