import os
import re
import heapq
import string

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    parts = re.split(r'[.!?]', text)
    return [p.strip().lower() for p in parts if p.strip()]


# -----------------------------
# Edit Distance
# -----------------------------
def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]


# -----------------------------
# A* Search for Alignment
# -----------------------------
def heuristic(i, j, n1, n2):
    return abs((n1 - i) - (n2 - j))


def astar_align(sents1, sents2):
    n1, n2 = len(sents1), len(sents2)
    pq = [(0, (0, 0, 0))]
    parent = {}
    visited = set()

    while pq:
        f_score, (i, j, cost) = heapq.heappop(pq)
        if (i, j) in visited:
            continue
        visited.add((i, j))

        if i == n1 and j == n2:
            path = []
            node = (i, j)
            while node in parent:
                prev, info = parent[node]
                path.append(info)
                node = prev
            return list(reversed(path))

        if i < n1 and j < n2:
            d = edit_distance(sents1[i], sents2[j])
            new_cost = cost + d
            parent[(i+1, j+1)] = ((i, j), (sents1[i], sents2[j], d))
            heapq.heappush(pq, (new_cost + heuristic(i+1, j+1, n1, n2), (i+1, j+1, new_cost)))

        if i < n1:
            parent[(i+1, j)] = ((i, j), (sents1[i], None, 1))
            heapq.heappush(pq, (cost + 1 + heuristic(i+1, j, n1, n2), (i+1, j, cost + 1)))

        if j < n2:
            parent[(i, j+1)] = ((i, j), (None, sents2[j], 1))
            heapq.heappush(pq, (cost + 1 + heuristic(i, j+1, n1, n2), (i, j+1, cost + 1)))

    return []


# -----------------------------
# Similarity Calculation
# -----------------------------
def calc_similarity(align):
    total = 0
    similar = 0
    for s1, s2, d in align:
        if s1 and s2:
            total += 1
            max_len = max(len(s1), len(s2))
            sim = (1 - d / max_len) * 100
            if sim >= 70:
                similar += 1
    return round((similar / total) * 100, 2) if total else 0


# -----------------------------
# Helpers
# -----------------------------
def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def run_case(f1, f2, label):
    print("\n" + "=" * 60)
    print(f"Running: {label}")
    print("=" * 60)
    t1, t2 = read_file(f1), read_file(f2)
    s1, s2 = preprocess(t1), preprocess(t2)

    align = astar_align(s1, s2)
    for a in align:
        print(a)

    score = calc_similarity(align)
    print(f"\nPlagiarism Similarity: {score}%")
    print("=" * 60)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    base = "test_docs"

    run_case(f"{base}/doc1_case1.txt", f"{base}/doc2_case1.txt", "Identical Documents")
    run_case(f"{base}/doc1_case2.txt", f"{base}/doc2_case2.txt", "Modified Documents")
    run_case(f"{base}/doc1_case3.txt", f"{base}/doc2_case3.txt", "Different Documents")
    run_case(f"{base}/doc1_case4.txt", f"{base}/doc2_case4.txt", "Partial Overlap")
