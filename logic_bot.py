import random

class LogicBot:
    def __init__(self, height, width):
        self.height = height
        self.width = width

        #Cells believed to be safe
        self.safe = set()

        #Cells believed to be mines
        self.mines = set()

        #Cells not yet revealed
        self.remaining = {(r, c) for r in range(height) for c in range(width)}

        #Revealed clues: (r, c) -> clue value
        self.clues = {}

    def neighbors(self, r, c):
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    yield nr, nc

    def choose_cell(self):
        """
        Choose a cell to open:
        Prefer known safe cells
        Otherwise choose randomly from remaining
        """
        while self.safe:
            cell = self.safe.pop()
            if cell in self.remaining:
                return cell

        return random.choice(list(self.remaining))

    def update(self, r, c, clue):
        """
        Update internal state after opening a cell.
        """
        if clue is None:
            return

        self.remaining.discard((r, c))
        self.clues[(r, c)] = clue

        #If clue is 0, all neighbors are safe
        if clue == 0:
            for n in self.neighbors(r, c):
                if n in self.remaining and n not in self.safe:
                    self.safe.add(n)

    def infer(self):
        """
        Apply logical inference rules until no new information is found.
        """
        changed = True
        passes = 0
        MAX_PASSES = 100

        while changed and passes < MAX_PASSES:
            changed = False
            passes += 1

            for (r, c), clue in list(self.clues.items()):
                if clue is None:
                    continue

                neighbors = list(self.neighbors(r, c))
                unrevealed = [n for n in neighbors if n in self.remaining]
                known_mines = [n for n in neighbors if n in self.mines]
                known_safe = [n for n in neighbors if n not in self.remaining]

                #Rule 1:
                #If remaining mines must all be in unrevealed neighbors
                if clue - len(known_mines) == len(unrevealed):
                    for n in unrevealed:
                        if n not in self.mines:
                            self.mines.add(n)
                            self.remaining.discard(n)
                            changed = True

                #Rule 2:
                #If remaining safe cells must all be unrevealed neighbors
                if (len(neighbors) - clue) - len(known_safe) == len(unrevealed):
                    for n in unrevealed:
                        if n not in self.safe:
                            self.safe.add(n)
                            changed = True

        if passes == MAX_PASSES:
            print("Infer loop capped to prevent infinite loop")
