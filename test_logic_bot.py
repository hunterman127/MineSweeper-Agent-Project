from minesweeper_env import MinesweeperEnv
from logic_bot import LogicBot

def print_board(obs):
    for row in obs:
        print(["." if x is None else x for x in row])
    print()

def run_test(height=5, width=5, mines=5, seed=42):
    print("=== STARTING TEST ===")

    env = MinesweeperEnv(height=height, width=width, num_mines=mines, seed=seed)
    bot = LogicBot(height, width)

    #First move (guaranteed safe)
    r, c = 0, 0
    clue = env.open_cell(r, c)
    print(f"First move at {(r, c)} -> clue {clue}")
    bot.update(r, c, clue)

    step = 1

    while not env.game_over and bot.remaining:
        print(f"--- Step {step} ---")

        #Sync bot remaining with environment (important!)
        obs = env.get_observed_board()
        for i in range(height):
            for j in range(width):
                if obs[i][j] is not None:
                    bot.remaining.discard((i, j))

        bot.infer()

        if not bot.remaining:
            break

        move = bot.choose_cell()
        result = env.open_cell(*move)

        print(f"Bot chooses {move} -> result {result}")

        if result is None:
            continue  #skip invalid move

        if result == -1:
            print(" Mine hit!")
            break

        bot.update(move[0], move[1], result)

        print_board(env.get_observed_board())

        step += 1

    print("=== TEST COMPLETE ===")
    print("Cells revealed:", env.cells_revealed)
    print("Mines triggered:", env.mines_triggered)


if __name__ == "__main__":
    run_test()
