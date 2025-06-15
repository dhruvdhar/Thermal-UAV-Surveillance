# MGO and evaluation functions
import numpy as np
import torch
import torch.nn as nn
from scripts.4_model_mgo_cnn import ThermalCNN

def evaluate(params, train_loader, val_loader, device):
    model = ThermalCNN(dropout=params[0], filters=int(params[1])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params[2])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(2):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total

def MGO(train_loader, val_loader, device, population_size=10, max_iter=10):
    pop = np.random.rand(population_size, 3)
    pop[:, 0] = np.random.uniform(0.1, 0.5, population_size)
    pop[:, 1] = np.round(np.random.uniform(16, 128, population_size))
    pop[:, 2] = np.random.uniform(0.0001, 0.01, population_size)

    best_score, best_params = 0, None
    for _ in range(max_iter):
        for p in pop:
            acc = evaluate(p, train_loader, val_loader, device)
            if acc > best_score:
                best_score, best_params = acc, p
        pop += np.random.normal(0, 0.01, pop.shape)
        pop[:, 0] = np.clip(pop[:, 0], 0.1, 0.5)
        pop[:, 1] = np.clip(np.round(pop[:, 1]), 16, 128)
        pop[:, 2] = np.clip(pop[:, 2], 0.0001, 0.01)
    return best_params
