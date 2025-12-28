import torch
import torch.nn as nn
import numpy as np

from model import MinePredictorCNN
from data_generator import generate_dataset


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, Y = generate_dataset(num_games=2000)

    X = torch.tensor(X, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)

    model = MinePredictorCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(10):
        optimizer.zero_grad()

        logits = model(X)

        #Mask out revealed cells
        hidden_mask = X[:, 0, :, :]  #channel 0 = hidden
        loss = loss_fn(logits * hidden_mask, Y * hidden_mask)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "mine_predictor.pt")
    print("Model saved to mine_predictor.pt")


if __name__ == "__main__":
    train()
