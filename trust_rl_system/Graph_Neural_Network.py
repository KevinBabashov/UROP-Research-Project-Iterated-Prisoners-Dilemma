import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TrustGNN(torch.nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16, output_dim=2):
        super(TrustGNN, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.gcn1(x, edge_index))
        x = self.lin(x)
        return x
