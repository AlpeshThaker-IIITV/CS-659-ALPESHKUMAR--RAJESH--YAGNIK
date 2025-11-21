
import numpy as np
import random
import math
from copy import deepcopy

random.seed(1)
np.random.seed(1)


def to_bipolar(x):
    # x in {0,1} -> { -1, +1 }
    return 2 * x - 1


def to_binary(s):
    # s in {-1,+1} -> {0,1}
    return ((s + 1) // 2).astype(int)


def hamming(a, b):
    return np.sum(a != b)


# -------------------------
# Generic Hopfield (binary/bipolar)
# -------------------------
class Hopfield:
    def __init__(self, N):
        self.N = N
        self.W = np.zeros((N, N), dtype=float)

    def train_hebb(self, patterns):
        """
        patterns: array-like of shape (P, N) with values in {-1, +1}
        Uses normalized Hebb rule: W = sum p p^T / N, diagonal zeroed.
        """
        N = self.N
        self.W.fill(0.0)
        for p in patterns:
            p = p.reshape(N, 1)
            self.W += p @ p.T
        self.W /= N
        np.fill_diagonal(self.W, 0.0)

    def energy(self, s):
        # s is {-1,+1}
        s = s.reshape(self.N, 1)
        return -0.5 * float(s.T @ self.W @ s)

    def recall_async(self, s_init, steps=5000):
        s = s_init.copy()
        N = self.N
        for _ in range(steps):
            i = random.randrange(N)
            net = float(np.dot(self.W[i], s))
            s[i] = 1 if net >= 0 else -1
        return s

    def recall_sync(self, s_init, steps=100):
        s = s_init.copy()
        for _ in range(steps):
            net = self.W @ s
            s_new = np.where(net >= 0, 1, -1)
            if np.array_equal(s_new, s):
                return s
            s = s_new
        return s



# Problem 1: 1.	Implement 10x10 associative memory (binary) using Hopfield network.

def problem1_demo():

    N = 100
    P = 5

    patterns = []
    for i in range(P):
        v = np.random.choice([1, -1], N)
        patterns.append(v)
    patterns = np.array(patterns)

    net = Hopfield(N)
    net.train_hebb(patterns)

    print("Problem 1 — 10x10 associative demo")
    for idx, pat in enumerate(patterns):
        print(f"\nPattern {idx}")
        for noise in [0.0, 0.1, 0.2, 0.3]:
            # flip noise fraction bits
            noisy = pat.copy()
            nflip = int(round(noise * N))
            if nflip > 0:
                ix = np.random.choice(N, nflip, replace=False)
                noisy[ix] *= -1
            out = net.recall_async(noisy, steps=2000)
            ham = hamming(out, pat)
            print(f" noise {noise:.2f} -> hamming {ham}/{N}")
    print("done problem1\n")



# Problem 2: 2.	Find out the capacity of the hopfield network in terms of storage of distinct patterns.

def problem2_capacity_test(max_patterns=20, trials=10):

    N = 100
    noise_frac = 0.1
    results = []

    for P in range(1, max_patterns + 1):
        succ_count = 0
        total = 0
        for t in range(trials):
            pats = np.array([np.random.choice([1, -1], N) for _ in range(P)])
            net = Hopfield(N)
            net.train_hebb(pats)

            # test each stored pattern with one noisy start
            for p in range(P):
                patt = pats[p].copy()
                nflip = int(round(noise_frac * N))
                ix = np.random.choice(N, nflip, replace=False)
                noisy = patt.copy()
                noisy[ix] *= -1
                out = net.recall_async(noisy, steps=2000)
                if np.array_equal(out, patt):
                    succ_count += 1
                total += 1

        frac = succ_count / total
        results.append((P, frac))
        print(f"P={P:2d}  recovery_rate={frac:.3f}")

    print("\nInterpretation: recovery_rate close to 1 means good storage.")
    print("Theoretical rule of thumb: P_max ~ 0.138 * N -> ~13 for N=100")
    return results


# Problem 3 : 3.	What is the error correcting capability of your Hopfield network?

def problem3_error_basin(P=6, trials=50):

    N = 100
    pats = np.array([np.random.choice([1, -1], N) for _ in range(P)])
    net = Hopfield(N)
    net.train_hebb(pats)

    noise_fracs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    print(f"Problem 3 — Error-correcting for P={P}")
    for f in noise_fracs:
        succ = 0
        tot = 0
        flips = int(round(f * N))
        for p in pats:
            for _ in range(trials):
                noisy = p.copy()
                if flips > 0:
                    ix = np.random.choice(N, flips, replace=False)
                    noisy[ix] *= -1
                out = net.recall_async(noisy, steps=3000)
                if np.array_equal(out, p):
                    succ += 1
                tot += 1
        print(f" noise {f:.2f} -> success {(succ / tot):.3f} ({succ}/{tot})")
    print("done problem3\n")



#Problem 4: 4.	Setup the energy function for the Eight-rook problem and solve the same using Hopfield network.  Give reasons for choosing specific weights for the network.

def problem4_eight_rook(max_iter=2000, restarts=5, A=3.0, B=3.0):


    def single_run():

        X = np.zeros((8, 8), dtype=int)

        for i in range(8):
            if random.random() < 0.8:
                j = random.randrange(8)
                X[i, j] = 1

        for _ in range(max_iter):
            i = random.randrange(8)
            j = random.randrange(8)
            row_sum = X[i].sum()
            col_sum = X[:, j].sum()
            score = A * (1 - row_sum) + B * (1 - col_sum)
            # tie-breaker: if score==0, keep as is
            if score > 0:
                X[i, j] = 1
            else:
                X[i, j] = 0

        return X

    print("Problem 4 — Eight-rook solver (simple Hopfield-like)")
    for r in range(restarts):
        X = single_run()
        rows_ok = all(X.sum(axis=1) == 1)
        cols_ok = all(X.sum(axis=0) == 1)
        print(f" restart {r+1}: rows_ok={rows_ok}, cols_ok={cols_ok}")
        if rows_ok and cols_ok:
            print(" Solution (1 marks rook):")
            print(X)
            print("done problem4\n")
            return X


    print("No perfect solution from simple runs. Trying greedy fix.")
    X = single_run()

    for i in range(8):
        best_j = None
        best_score = -1e9
        for j in range(8):
            # measure how good making (i,j)=1 would be
            row_sum = X[i].sum()
            col_sum = X[:, j].sum()
            score = A * (1 - row_sum) + B * (1 - col_sum)
            if score > best_score:
                best_score = score
                best_j = j
        # set that position and clear others in row
        X[i, :] = 0
        X[i, best_j] = 1

    rows_ok = all(X.sum(axis=1) == 1)
    cols_ok = all(X.sum(axis=0) == 1)
    print(f" After greedy fix: rows_ok={rows_ok}, cols_ok={cols_ok}")
    print("Resulting board:")
    print(X)
    print("done problem4\n")
    return X



# Problem 5: 5.	Solve a TSP (traveling salesman problem) of 10 cities with a Hopfield network.  How many weights do you need for the network?

def problem5_tsp(cities=10, max_iter=5000, tries=8, A=500.0, B=500.0, D=1.0):

    N = cities

    d = np.random.rand(N, N)
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)

    def score_for(V, i, p):

        p_plus = (p + 1) % N
        p_minus = (p - 1) % N
        city_sum = V[i, :].sum()  # sum over positions
        pos_sum = V[:, p].sum()  # sum over cities

        dist_term = 0.0
        for j in range(N):
            dist_term += d[i, j] * (V[j, p_plus] + V[j, p_minus])
        return A * (1 - city_sum) + B * (1 - pos_sum) - D * dist_term

    best_tour = None
    best_cost = float('inf')

    print("Problem 5 — TSP (10 cities) via Hopfield-like updates")
    for attempt in range(tries):
        # random init: each row roughly has one 1
        V = np.zeros((N, N), dtype=int)
        for i in range(N):
            p0 = random.randrange(N)
            V[i, p0] = 1

        for it in range(max_iter):
            i = random.randrange(N)
            p = random.randrange(N)
            sc = score_for(V, i, p)
            if sc > 0:
                V[i, p] = 1
            else:
                V[i, p] = 0

        for i in range(N):
            row = V[i, :]
            if row.sum() == 1:
                continue
            # choose best p by score
            best_p = None
            best_s = -1e12
            for p in range(N):
                s = score_for(V, i, p)
                if s > best_s:
                    best_s = s
                    best_p = p
            V[i, :] = 0
            V[i, best_p] = 1

        rows_ok = all(V.sum(axis=1) == 1)
        cols_ok = all(V.sum(axis=0) == 1)

        if not (rows_ok and cols_ok):
            continue

        tour = [int(np.where(V[:, p] == 1)[0][0]) for p in range(N)]

        cost = 0.0
        for p in range(N):
            i = tour[p]
            j = tour[(p + 1) % N]
            cost += d[i, j]

        print(f" attempt {attempt+1}: valid tour found cost={cost:.4f}")
        if cost < best_cost:
            best_cost = cost
            best_tour = tour

    if best_tour is None:
        print("No valid tour found reliably in tries.")
    else:
        print("Best tour cost:", best_cost)
        print("Tour (city indices):", best_tour)
    print("done problem5\n")
    return best_tour, best_cost



problem1_demo()


print("Problem 2 — capacity quick test (this may take a while if you increase max_patterns)")
problem2_capacity_test(max_patterns=12, trials=5)

problem3_error_basin(P=6, trials=40)

problem4_eight_rook(max_iter=2000, restarts=6, A=4.0, B=4.0)

problem5_tsp(cities=10, max_iter=4000, tries=8, A=400.0, B=400.0, D=1.0)
