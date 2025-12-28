import random
from collections import deque

class MinesweeperEnv:
    def __init__(self, height=22, width=22, num_mines=50, seed=None):
        self.height = height
        self.width = width
        self.num_mines = num_mines

        if seed is not None:
            random.seed(seed)

        #Board will be generated on first move
        self.board = None
        self.mines = None
        self.first_move = True

        self.revealed = [[False for _ in range(width)] for _ in range(height)]
        self.game_over = False
        self.mines_triggered = 0
        self.cells_revealed = 0

    def _neighbors(self, r, c):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    yield nr, nc

    def _generate_board_safe(self, safe_r, safe_c):
        self.board = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.mines = set()

        #Do not allow mines on first cell or its neighbors
        forbidden = {(safe_r, safe_c)}
        forbidden.update(self._neighbors(safe_r, safe_c))

        while len(self.mines) < self.num_mines:
            r = random.randrange(self.height)
            c = random.randrange(self.width)
            if (r, c) not in forbidden:
                self.mines.add((r, c))

        #Compute clues
        for r, c in self.mines:
            for nr, nc in self._neighbors(r, c):
                if (nr, nc) not in self.mines:
                    self.board[nr][nc] += 1

    def open_cell(self, r, c):
        """
        Opens a cell.
        Returns:
            -1 if mine
             0-8 otherwise
        """
        if self.game_over or self.revealed[r][c]:
            return None

        #FIRST MOVE SAFETY
        if self.first_move:
            self._generate_board_safe(r, c)
            self.first_move = False

        self.revealed[r][c] = True
        self.cells_revealed += 1

        if (r, c) in self.mines:
            self.game_over = True
            self.mines_triggered += 1
            return -1

        clue = self.board[r][c]

        if clue == 0:
            self._flood_fill(r, c)

        return clue

    def _flood_fill(self, r, c):
        queue = deque([(r, c)])
        while queue:
            cr, cc = queue.popleft()
            for nr, nc in self._neighbors(cr, cc):
                if not self.revealed[nr][nc] and (nr, nc) not in self.mines:
                    self.revealed[nr][nc] = True
                    self.cells_revealed += 1
                    if self.board[nr][nc] == 0:
                        queue.append((nr, nc))

    def get_observed_board(self):
        obs = [[None for _ in range(self.width)] for _ in range(self.height)]
        for r in range(self.height):
            for c in range(self.width):
                if self.revealed[r][c]:
                    obs[r][c] = self.board[r][c]
        return obs
