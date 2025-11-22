import random
import itertools

# --- Utility functions ---

def print_board(board):
    symbols = {1: "X", -1: "O", 0: " "}
    print("\n".join([" | ".join(symbols[c] for c in board[i:i+3]) for i in range(0, 9, 3)]))
    print("-" * 9)

def check_winner(board):
    """Return +1 if MENACE wins, -1 if opponent wins, 0 if draw, None if ongoing."""
    lines = [
        (0,1,2), (3,4,5), (6,7,8),  # rows
        (0,3,6), (1,4,7), (2,5,8),  # cols
        (0,4,8), (2,4,6)            # diagonals
    ]
    for a,b,c in lines:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    if 0 not in board:
        return 0  # draw
    return None

def canonical_form(board):
    """Return a canonical board representation accounting for symmetries."""
    # All 8 rotations/reflections
    def rotate(b): return [b[i] for i in [6,3,0,7,4,1,8,5,2]]
    def reflect(b): return [b[i] for i in [2,1,0,5,4,3,8,7,6]]
    boards = []
    b = board
    for _ in range(4):
        boards.append(tuple(b))
        boards.append(tuple(reflect(b)))
        b = rotate(b)
    return min(boards)

# --- MENACE class ---

class Menace:
    def __init__(self, initial_beads=3):
        self.boxes = {}          # state -> [bead counts for moves 0-8]
        self.history = []        # record (state, move) pairs for the current game
        self.initial_beads = initial_beads

    def start_game(self):
        self.history = []

    def choose_move(self, board):
        """Select a move using weighted random choice."""
        state = canonical_form(board)
        if state not in self.boxes:
            # Initialize a new matchbox for this state
            counts = [0]*9
            for i,c in enumerate(state):
                if c == 0:
                    counts[i] = self.initial_beads
            self.boxes[state] = counts
        counts = self.boxes[state]
        legal_moves = [i for i,c in enumerate(state) if c==0]
        weights = [counts[i] for i in legal_moves]
        if sum(weights) == 0:  # fallback if no beads
            move = random.choice(legal_moves)
        else:
            move = random.choices(legal_moves, weights=weights, k=1)[0]
        self.history.append((state, move))
        return move

    def reinforce(self, result):
        """Adjust beads based on game result."""
        # Win +3, Draw +1, Loss -1 (cannot go below 1)
        delta = {1: 3, 0: 1, -1: -1}[result]
        for state, move in self.history:
            self.boxes[state][move] = max(1, self.boxes[state][move] + delta)
        self.history = []

# --- Game simulation ---

def play_game(menace, opponent_random=True, verbose=False):
    board = [0]*9
    menace.start_game()
    player = 1  # MENACE starts (X)
    while True:
        if player == 1:
            move = menace.choose_move(board)
        else:
            # Opponent (random)
            legal = [i for i,c in enumerate(board) if c==0]
            move = random.choice(legal)
        board[move] = player
        winner = check_winner(board)
        if verbose:
            print_board(board)
        if winner is not None:
            menace.reinforce(winner)
            return winner
        player *= -1  # switch turns

# --- Training loop ---

def train_menace(rounds=5000):
    menace = Menace()
    results = {1:0, 0:0, -1:0}
    for i in range(rounds):
        result = play_game(menace)
        results[result] += 1
        if (i+1) % 500 == 0:
            print(f"After {i+1} games: Wins={results[1]}, Draws={results[0]}, Losses={results[-1]}")
    return menace

# --- Run training ---
if __name__ == "__main__":
    trained_menace = train_menace(2000)
    print("\nTraining complete. MENACE learned from experience!")
