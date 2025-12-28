import numpy as np
from minesweeper_env import MinesweeperEnv
from logic_bot import LogicBot


def encode_board(obs):
    """
    Converts observed board into a 3-channel tensor:
    channel 0: hidden cells
    channel 1: revealed numbers (0–8)
    channel 2: revealed mask
    """
    h = len(obs)
    w = len(obs[0])

    hidden = np.zeros((h, w), dtype=np.float32)
    revealed_vals = np.zeros((h, w), dtype=np.float32)
    revealed_mask = np.zeros((h, w), dtype=np.float32)

    for r in range(h):
        for c in range(w):
            if obs[r][c] is None:
                hidden[r][c] = 1.0
            else:
                revealed_vals[r][c] = obs[r][c] / 8.0
                revealed_mask[r][c] = 1.0

    return np.stack([hidden, revealed_vals, revealed_mask])


def generate_game(height=5, width=5, mines=5, seed=None):
    env = MinesweeperEnv(height, width, mines, seed)
    bot = LogicBot(height, width)

    X = []
    Y = []

    # First move
    r, c = 0, 0
    clue = env.open_cell(r, c)
    bot.update(r, c, clue)

    while not env.game_over and bot.remaining:
        obs = env.get_observed_board()

        # Sync remaining cells
        for i in range(height):
            for j in range(width):
                if obs[i][j] is not None:
                    bot.remaining.discard((i, j))

        #Save training snapshot
        X.append(encode_board(obs))

        #True mine labels
        mine_map = np.zeros((height, width), dtype=np.float32)
        for (mr, mc) in env.mines:
            mine_map[mr][mc] = 1.0
        Y.append(mine_map)

        bot.infer()

        if not bot.remaining:
            break

        move = bot.choose_cell()
        result = env.open_cell(*move)

        if result == -1:
            break

        bot.update(move[0], move[1], result)

    return X, Y


def generate_dataset(num_games=100):
    all_X = []
    all_Y = []

    for i in range(num_games):
        X, Y = generate_game(seed=i)
        all_X.extend(X)
        all_Y.extend(Y)

    return np.array(all_X), np.array(all_Y)


if __name__ == "__main__":
    X, Y = generate_dataset(num_games=10)
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
