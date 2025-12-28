import torch
import numpy as np
from critic_model import CriticNet
from critic_data_generator import generate_critic_data

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, A, Y = generate_critic_data(num_games=500)

    X = torch.tensor(X, dtype=torch.float32).to(device)
    A = torch.tensor(A, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)
    Y = Y / (5 * 5)


    model = CriticNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(10):
        opt.zero_grad()
        pred = model(X, A)
        loss = loss_fn(pred, Y)
        loss.backward()
        opt.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "critic.pt")

if __name__ == "__main__":
    train()
