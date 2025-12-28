import numpy as np

from minesweeper_env import MinesweeperEnv
from logic_bot import LogicBot
from play_model import play_game as play_cnn
from play_actor_critic import play as play_actor_critic


#Logic Bot evaluation
def play_logic(seed, height=5, width=5, mines=5):
    env = MinesweeperEnv(height, width, mines, seed)
    bot = LogicBot(height, width)

    #First move (safe)
    env.open_cell(0, 0)

    while not env.game_over and bot.remaining:
        obs = env.get_observed_board()

        #sync remaining
        for r in range(height):
            for c in range(width):
                if obs[r][c] is not None:
                    bot.remaining.discard((r, c))

        bot.infer()
        if not bot.remaining:
            break

        r, c = bot.choose_cell()
        res = env.open_cell(r, c)

        if res == -1:
            break

        bot.update(r, c, res)

    return env.cells_revealed, env.mines_triggered

def play_actor_critic_wrapper(seed):
    cells = play_actor_critic(seed)
    return cells, 1  #actor stops on first mine

def play_cnn_wrapper(seed):
    return play_cnn(height=5, width=5, mines=5, seed=seed)


# Evaluation runner
def evaluate(bot_fn, name, N=20):
    cells = []
    mines = []

    for seed in range(N):
        revealed, triggered = bot_fn(seed)
        cells.append(revealed)
        mines.append(triggered)

    print(f"\n{name}")
    print("Avg cells revealed:", np.mean(cells))
    print("Avg mines triggered:", np.mean(mines))
    print("Min / Max cells:", min(cells), "/", max(cells))

# Main
if __name__ == "__main__":
    N = 20

    evaluate(play_logic, "Logic Bot", N)
    evaluate(play_cnn_wrapper, "CNN Bot (Task 1)", N)
    evaluate(play_actor_critic_wrapper, "Actor–Critic (Task 2)", N)
