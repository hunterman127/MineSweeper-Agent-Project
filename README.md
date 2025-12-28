Minesweeper Neural Network Bots
==============================

This project explores different approaches to building bots that can play the game
of Minesweeper. The goal is to compare traditional logic-based decision making with
neural network and actor–critic methods, and see how learned models perform relative
to a rule-based baseline.

The project was built as a deep learning final project, but it is organized so that
anyone can run, test, and modify the bots.


------------------------------------------------------------
BOTS IMPLEMENTED
------------------------------------------------------------

1) Logic Bot (Baseline)
- Uses standard Minesweeper inference rules
- Marks cells as mines or safe when logically guaranteed
- When no inference is possible, it guesses randomly
- Serves as a strong baseline for comparison

2) CNN Bot (Task 1)
- Uses a convolutional neural network
- Predicts the probability that each hidden cell is a mine
- Chooses among the lowest-risk cells
- Learns local spatial patterns but does not reason long-term

3) Actor–Critic Bot (Task 2)
- Uses a critic network to predict expected future survival
- Evaluates all possible moves and chooses the one with the highest value
- Optimizes long-term progress instead of immediate safety
- Performs best overall in evaluation


------------------------------------------------------------
BOARD REPRESENTATION
------------------------------------------------------------

The Minesweeper board is encoded as a 3-channel tensor:

- Channel 1: Revealed vs hidden cells
- Channel 2: Clue values for revealed cells
- Channel 3: Unknown cell indicators

The neural networks never see the true mine locations.
This prevents cheating and keeps the task realistic.


------------------------------------------------------------
PROJECT FILES
------------------------------------------------------------

Core game and bots:
- minesweeper_env.py        : Minesweeper game environment
- logic_bot.py              : Rule-based Minesweeper bot
- model.py                  : CNN model for Task 1
- critic_model.py           : Critic network for Task 2

Data and encoding:
- data_generator.py         : Board encoding utilities
- critic_data_generator.py  : Generates training data for critic

Training:
- train.py                  : Trains the CNN bot
- train_critic.py           : Trains the critic network

Playing and evaluation:
- play_model.py             : Play a game using the CNN bot
- play_actor_critic.py      : Play a game using the actor–critic bot
- evaluate_model.py         : Evaluate CNN bot performance
- evaluate_all.py           : Compare all bots

Saved models:
- mine_predictor.pt         : Trained CNN weights
- critic.pt                 : Trained critic weights

Documentation:
- writeup.txt               : Project writeup


------------------------------------------------------------
REQUIREMENTS
------------------------------------------------------------

- Python 3.8+
- PyTorch
- NumPy

Install dependencies if needed:
pip install torch numpy


------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------

Play a single game with the CNN bot:
python play_model.py

Play a single game with the Actor–Critic bot:
python play_actor_critic.py

Evaluate all bots on the same boards:
python evaluate_all.py


------------------------------------------------------------
TRAINING (OPTIONAL)
------------------------------------------------------------

If you want to retrain the models from scratch:

Train CNN:
python train.py

Train Critic:
python train_critic.py

The trained weights will be saved as:
- mine_predictor.pt
- critic.pt


------------------------------------------------------------
RESULTS (5x5 board, 5 mines)
------------------------------------------------------------

Average performance over 20 games:

Logic Bot:
- Avg cells revealed: ~13.7

CNN Bot (Task 1):
- Avg cells revealed: ~14.6

Actor–Critic Bot (Task 2):
- Avg cells revealed: ~15.0

The actor–critic approach performs best by optimizing long-term survival
instead of local safety.


------------------------------------------------------------
FINAL NOTES
------------------------------------------------------------

- The logic bot performs well early but struggles when forced to guess
- The CNN bot learns useful spatial patterns but lacks long-term reasoning
- The actor–critic bot ranks guesses based on expected future progress

This project demonstrates that framing Minesweeper as a sequential decision
problem leads to better strategies than purely rule-based or local approaches.

Feel free to modify, extend, or experiment with the code.
