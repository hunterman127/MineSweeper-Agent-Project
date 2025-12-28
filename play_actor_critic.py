import torch
import numpy as np
from minesweeper_env import MinesweeperEnv
from critic_model import CriticNet
from data_generator import encode_board


def play(seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    height = width = 5
    critic = CriticNet(height, width).to(device)
    critic.load_state_dict(torch.load("critic.pt", map_location=device))
    critic.eval()

    env = MinesweeperEnv(height, width, 5, seed)
    env.open_cell(0, 0)

    MAX_STEPS = height * width * 2
    steps = 0

    while not env.game_over and steps < MAX_STEPS:
        steps += 1
        obs = env.get_observed_board()

        board = torch.tensor(
            encode_board(obs),
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        actions, coords = [], []
        for r in range(height):
            for c in range(width):
                if obs[r][c] is None:
                    actions.append([r / height, c / width])
                    coords.append((r, c))

        if not actions:
            break

        A = torch.tensor(actions, dtype=torch.float32).to(device)
        boards = board.repeat(len(A), 1, 1, 1)

        with torch.no_grad():
            values = critic(boards, A)

        idx = torch.argmax(values).item()
        r, c = coords[idx]

        res = env.open_cell(r, c)
        if res == -1:
            break

    return env.cells_revealed
