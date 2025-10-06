"""
Week 1 - Missionaries & Cannibals + Rabbit Leap
Solved using BFS and DFS
"""

from collections import deque

# ----------------------------
# Problem 1: Missionaries & Cannibals
# ----------------------------
class StateMC:
    def __init__(self, missionaries, cannibals, boat):
        self.m = missionaries
        self.c = cannibals
        self.boat = boat  # 1 = left, 0 = right

    def is_valid(self):
        if self.m < 0 or self.c < 0 or self.m > 3 or self.c > 3:
            return False
        if self.m > 0 and self.m < self.c:
            return False
        if (3 - self.m) > 0 and (3 - self.m) < (3 - self.c):
            return False
        return True

    def is_goal(self):
        return self.m == 0 and self.c == 0 and self.boat == 0

    def __eq__(self, other):
        return (self.m, self.c, self.boat) == (other.m, other.c, other.boat)

    def __hash__(self):
        return hash((self.m, self.c, self.boat))

    def __str__(self):
        return f"(M={self.m}, C={self.c}, Boat={'Left' if self.boat else 'Right'})"


def get_successors(state):
    successors = []
    moves = [(1,0),(2,0),(0,1),(0,2),(1,1)]
    for m, c in moves:
        if state.boat == 1:
            new_state = StateMC(state.m - m, state.c - c, 0)
        else:
            new_state = StateMC(state.m + m, state.c + c, 1)
        if new_state.is_valid():
            successors.append(new_state)
    return successors


def bfs_mc(start):
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        state, path = queue.popleft()
        if state.is_goal():
            return path
        for succ in get_successors(state):
            if succ not in visited:
                visited.add(succ)
                queue.append((succ, path + [succ]))
    return None


def dfs_mc(start):
    stack = [(start, [start])]
    visited = {start}
    while stack:
        state, path = stack.pop()
        if state.is_goal():
            return path
        for succ in get_successors(state):
            if succ not in visited:
                visited.add(succ)
                stack.append((succ, path + [succ]))
    return None


# ----------------------------
# Problem 2: Rabbit Leap
# ----------------------------
class StateRabbit:
    def __init__(self, positions):
        self.positions = positions  # list of 7 positions: E, W, or _

    def __eq__(self, other):
        return self.positions == other.positions

    def __hash__(self):
        return hash(tuple(self.positions))

    def __str__(self):
        return "".join(self.positions)

    def is_goal(self):
        return self.positions == ['W','W','W','_','E','E','E']


def get_successors_rabbit(state):
    successors = []
    positions = state.positions
    for i, r in enumerate(positions):
        if r == "E":
            # move right
            if i+1 < len(positions) and positions[i+1] == "_":
                new_pos = positions[:]
                new_pos[i], new_pos[i+1] = "_","E"
                successors.append(StateRabbit(new_pos))
            elif i+2 < len(positions) and positions[i+1] == "W" and positions[i+2] == "_":
                new_pos = positions[:]
                new_pos[i], new_pos[i+2] = "_","E"
                successors.append(StateRabbit(new_pos))
        elif r == "W":
            # move left
            if i-1 >= 0 and positions[i-1] == "_":
                new_pos = positions[:]
                new_pos[i], new_pos[i-1] = "_","W"
                successors.append(StateRabbit(new_pos))
            elif i-2 >= 0 and positions[i-1] == "E" and positions[i-2] == "_":
                new_pos = positions[:]
                new_pos[i], new_pos[i-2] = "_","W"
                successors.append(StateRabbit(new_pos))
    return successors


def bfs_rabbit(start):
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        state, path = queue.popleft()
        if state.is_goal():
            return path
        for succ in get_successors_rabbit(state):
            if succ not in visited:
                visited.add(succ)
                queue.append((succ, path + [succ]))
    return None


def dfs_rabbit(start):
    stack = [(start, [start])]
    visited = {start}
    while stack:
        state, path = stack.pop()
        if state.is_goal():
            return path
        for succ in get_successors_rabbit(state):
            if succ not in visited:
                visited.add(succ)
                stack.append((succ, path + [succ]))
    return None


# ----------------------------
# Run Week1
# ----------------------------
if __name__ == "__main__":
    print("Missionaries & Cannibals:")
    start = StateMC(3,3,1)
    print("BFS Solution:")
    for s in bfs_mc(start):
        print(s)
    print("\nDFS Solution:")
    for s in dfs_mc(start):
        print(s)

    print("\nRabbit Leap:")
    start_r = StateRabbit(["E","E","E","_","W","W","W"])
    print("BFS Solution:")
    for s in bfs_rabbit(start_r):
        print(s)
    print("\nDFS Solution:")
    for s in dfs_rabbit(start_r):
        print(s)
