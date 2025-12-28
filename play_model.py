import torch
import numpy as np

from minesweeper_env import MinesweeperEnv
from model import MinePredictorCNN
from data_generator import encode_board


def play_game(height=5, width=5, mines=5, seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MinePredictorCNN().to(device)
    model.load_state_dict(torch.load("mine_predictor.pt", map_location=device))
    model.eval()

    env = MinesweeperEnv(height, width, mines, seed)

    #First move
    env.open_cell(0, 0)

    step = 0

    while not env.game_over:
        obs = env.get_observed_board()
        x = encode_board(obs)
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        #Mask revealed cells
        for r in range(height):
            for c in range(width):
                if obs[r][c] is not None:
                    probs[r][c] = 1.0  #never choose revealed

        #Choose safest cell
        flat = probs.flatten()
        k = min(5, len(flat))
        safe_indices = flat.argsort()[:k]
        choice = np.random.choice(safe_indices)
        r, c = np.unravel_index(choice, probs.shape)

        result = env.open_cell(r, c)

        #print(f"Step {step}: chose {(r, c)} → result {result}")
        step += 1

        if result == -1:
            #print(" Model hit a mine")
            break

    print("Cells revealed:", env.cells_revealed)
    print("Mines triggered:", env.mines_triggered)

    return env.cells_revealed, env.mines_triggered


if __name__ == "__main__":
    play_game()
