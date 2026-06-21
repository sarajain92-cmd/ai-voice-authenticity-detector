import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from training.dataset import CSVDataset
from models.mlp_classifier import DeepfakeMLP
from torch.utils.data import random_split, DataLoader
from sklearn.preprocessing import StandardScaler

dataset = CSVDataset("data/KAGGLE/dataset-balanced.csv", fit_scaler=True)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

model = DeepfakeMLP(input_dim=26)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

best_val_acc = 0
losses = []

for epoch in range(10):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            out = model(x)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    val_acc = correct / total

    print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Val Acc = {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/mlp_best.pt")

plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.savefig("models/loss_curve.png")
plt.show()
import joblib

joblib.dump(model.state_dict(), "model.pkl")