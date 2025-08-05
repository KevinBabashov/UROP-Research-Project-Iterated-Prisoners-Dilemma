from Graph_Neural_Network import TrustGNN
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
import torch.nn as nn

# Simulate or load your training data
def load_data():
    # Dummy data (replace this with your real data loading)
    x = torch.rand((5, 5))  # 5 nodes, 5 features
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 0],
                               [1, 0, 3, 2, 0, 4]], dtype=torch.long)
    y = torch.tensor([0, 1, 0, 1, 1])  # Dummy labels
    return Data(x=x, edge_index=edge_index, y=y)

def train_model():
    data = load_data()
    model = TrustGNN(input_dim=data.x.shape[1], hidden_dim=16, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "saved_models/trust_gnn.pth")
    print("✅ Model saved")

if __name__ == "__main__":
    train_model()
