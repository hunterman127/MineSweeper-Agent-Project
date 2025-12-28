from play_model import play_game

def evaluate(num_games=20):
    total_revealed = 0
    total_mines = 0

    for i in range(num_games):
        print(f"\n=== GAME {i+1} ===")
        revealed, mines = play_game(seed=i)
        total_revealed += revealed
        total_mines += mines

    print("\n=== SUMMARY ===")
    print("Avg cells revealed:", total_revealed / num_games)
    print("Avg mines triggered:", total_mines / num_games)

if __name__ == "__main__":
    evaluate()
