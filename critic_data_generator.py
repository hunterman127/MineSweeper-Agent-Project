import numpy as np
from minesweeper_env import MinesweeperEnv
from logic_bot import LogicBot
from data_generator import encode_board


def generate_critic_data(num_games=500, height=5, width=5, mines=5):
    X = []   #board states
    A = []   #actions (r, c)
    Y = []   #survival lengths

    MAX_ROLLOUT = height * width * 2  #hard safety cap

    for seed in range(num_games):
        env = MinesweeperEnv(height, width, mines, seed)
        bot = LogicBot(height, width)

        #First move (safe by construction)
        env.open_cell(0, 0)

        while not env.game_over and bot.remaining:
            obs = env.get_observed_board()

            #Sync bot.remaining with revealed cells
            for r in range(height):
                for c in range(width):
                    if obs[r][c] is not None:
                        bot.remaining.discard((r, c))

            bot.infer()
            if not bot.remaining:
                break

            move = bot.choose_cell()

            #Save snapshot BEFORE move
            X.append(encode_board(obs))
            A.append([move[0] / height, move[1] / width])


            #Execute chosen move
            steps = 0
            result = env.open_cell(*move)

            #Immediate death
            if result == -1:
                Y.append(0)
                break

            bot.update(move[0], move[1], result)
            steps += 1

            #Rollout to estimate survival
            rollout_steps = 0
            while (
                not env.game_over
                and bot.remaining
                and rollout_steps < MAX_ROLLOUT
            ):
                rollout_steps += 1

                obs = env.get_observed_board()

                #Keep remaining consistent
                for r in range(height):
                    for c in range(width):
                        if obs[r][c] is not None:
                            bot.remaining.discard((r, c))

                bot.infer()
                if not bot.remaining:
                    break

                m = bot.choose_cell()
                r = env.open_cell(*m)

                if r == -1:
                    break

                bot.update(m[0], m[1], r)
                steps += 1

            # Record bounded survival length
            Y.append(steps)

    return np.array(X), np.array(A), np.array(Y)


if __name__ == "__main__":
    X, A, Y = generate_critic_data(num_games=50)
    print("X:", X.shape)
    print("A:", A.shape)
    print("Y:", Y.shape)
    print("Y stats → min:", Y.min(), "max:", Y.max(), "mean:", Y.mean())
