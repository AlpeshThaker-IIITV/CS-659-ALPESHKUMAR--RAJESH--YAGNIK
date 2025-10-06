#!/usr/bin/env python3
"""
Run Code using this Arguements

python Week 4 Puzzle.py --mat-path "scrambled_lena.mat" --grid-size 4 --band-width 8 --alpha-ncc 0.4 --alpha-mse 0.6 --use-smart-init --verbose --seed 42

"""

import numpy as np
import matplotlib.pyplot as plt
import random
import math
import argparse
import os
import time

# --- This global variable will be redefined in main() to use CLI arguments ---
_edge_cost_params = {'alpha_ncc': 0.5, 'alpha_mse': 0.5}


def load_octave_mat_text_or_scipy(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        lines = None
    if lines:
        dims, start_idx = None, None
        for i, line in enumerate(lines):
            s = line.strip()
            if not s or s.startswith('#'): continue
            parts = s.split()
            if len(parts) == 2:
                try:
                    dims = (int(parts[0]), int(parts[1]));
                    start_idx = i + 1;
                    break
                except ValueError:
                    pass
        if dims:
            tokens = []
            for line in lines[start_idx:]:
                line = line.strip()
                if not line or line.startswith('#'): continue
                tokens.extend(line.split())
            pixels = np.array(tokens, dtype=np.int32)
            expected = dims[0] * dims[1]
            if pixels.size >= expected:
                pixels = pixels[:expected]
                arr = pixels.reshape(dims, order='F').astype(np.float64)
                if arr.max() > 1.0: arr /= 255.0
                return arr
    try:
        from scipy.io import loadmat
        md = loadmat(filepath)
        candidates = [v for k, v in md.items() if
                      isinstance(v, np.ndarray) and v.ndim == 2 and np.issubdtype(v.dtype, np.number)]
        if candidates:
            arr = max(candidates, key=lambda x: x.size).astype(np.float64)
            if arr.max() > 1.0: arr /= 255.0
            return arr
    except Exception:
        pass
    raise ValueError("Could not parse .mat file.")


def create_puzzle(image, grid_size):
    h, w = image.shape
    crop = min(h, w) - (min(h, w) % grid_size)
    image = image[:crop, :crop].copy()
    piece_size = crop // grid_size
    pieces = [image[r * piece_size:(r + 1) * piece_size, c * piece_size:(c + 1) * piece_size].copy() for r in
              range(grid_size) for c in range(grid_size)]
    return pieces, list(range(len(pieces))), piece_size


def precompute_edges(pieces, band_width=4):
    n = len(pieces)
    left, right, top, bottom = [None] * n, [None] * n, [None] * n, [None] * n
    bw = max(1, band_width)
    for i, p in enumerate(pieces):
        h, w = p.shape;
        bw_h, bw_w = min(bw, h), min(bw, w)
        left[i] = p[:, :bw_w].copy();
        right[i] = p[:, -bw_w:].copy()
        top[i] = p[:bw_h, :].copy();
        bottom[i] = p[-bw_h:, :].copy()
    return {'left': left, 'right': right, 'top': top, 'bottom': bottom}


def edge_cost(A, B):
    A_flat, B_flat = A.ravel().astype(np.float64), B.ravel().astype(np.float64)
    A0, B0 = A_flat - A_flat.mean(), B_flat - B_flat.mean()
    denom = (np.linalg.norm(A0) * np.linalg.norm(B0) + 1e-10)
    ncc_diss = 1.0 - (np.dot(A0, B0) / denom)
    mse = np.mean((A_flat - B_flat) ** 2)
    return float(_edge_cost_params['alpha_ncc'] * ncc_diss + _edge_cost_params['alpha_mse'] * mse)


def calculate_energy_from_edges(state, edges, grid_size):
    total = 0.0
    for pos, pid in enumerate(state):
        row, col = divmod(pos, grid_size)
        if col < grid_size - 1: total += edge_cost(edges['right'][pid], edges['left'][state[pos + 1]])
        if row < grid_size - 1: total += edge_cost(edges['bottom'][pid], edges['top'][state[pos + grid_size]])
    return total


def reconstruct_image(state, pieces, grid_size, piece_size):
    img_size = grid_size * piece_size
    out = np.zeros((img_size, img_size), dtype=np.float64)
    for pos, pid in enumerate(state):
        r, c = divmod(pos, grid_size)
        out[r * piece_size:(r + 1) * piece_size, c * piece_size:(c + 1) * piece_size] = pieces[pid]
    return out


def greedy_initial_state(pieces, edges, grid_size):
    n = len(pieces)
    placed, state = {random.randrange(n)}, [-1] * n
    state[0] = list(placed)[0]
    for pos in range(1, n):
        row, col = divmod(pos, grid_size)
        best_pid, best_score = -1, float('inf')
        for pid in range(n):
            if pid in placed: continue
            score = 0.0
            if col > 0: score += edge_cost(edges['right'][state[pos - 1]], edges['left'][pid])
            if row > 0: score += edge_cost(edges['bottom'][state[pos - grid_size]], edges['top'][pid])
            if score < best_score: best_score, best_pid = score, pid
        state[pos] = best_pid
        placed.add(best_pid)
    return state


def local_refinement(state, edges, grid_size, max_iters=2000):
    n, it = len(state), 0
    current_energy = calculate_energy_from_edges(state, edges, grid_size)
    while it < max_iters:
        it += 1
        a, b = random.sample(range(n), 2)
        state[a], state[b] = state[b], state[a]
        new_energy = calculate_energy_from_edges(state, edges, grid_size)
        if new_energy < current_energy:
            current_energy = new_energy
        else:
            state[a], state[b] = state[b], state[a]  # Revert
    return state, current_energy


def solve_with_simulated_annealing(pieces, initial_state, grid_size, edges,
                                   initial_temp=800.0, cooling_rate=0.99998,
                                   max_iterations=100000, verbose=True, reheating=True):
    current_state, n = initial_state.copy(), len(initial_state)
    current_energy = calculate_energy_from_edges(current_state, edges, grid_size)
    best_state, best_energy = current_state.copy(), current_energy
    temperature = initial_temp
    history = [best_energy]
    last_improvement_iter = 0
    stagnation_window = max(2500, int(0.03 * max_iterations))

    for i in range(max_iterations):
        s = current_state.copy()
        move_type = random.random()
        if move_type < 0.85:
            a, b = random.sample(range(n), 2);
            s[a], s[b] = s[b], s[a]
        elif move_type < 0.98 and grid_size >= 2:
            r, c = random.randint(0, grid_size - 2), random.randint(0, grid_size - 2)
            idxs = [r * grid_size + c, r * grid_size + c + 1, (r + 1) * grid_size + c, (r + 1) * grid_size + c + 1]
            perm = idxs.copy();
            random.shuffle(perm)
            pieces_to_perm = [s[i] for i in idxs]
            for i, p_idx in enumerate(idxs): s[p_idx] = pieces_to_perm[perm.index(p_idx)]
        else:
            a, b, c = random.sample(range(n), 3);
            s[a], s[b], s[c] = s[c], s[a], s[b]

        neigh_energy = calculate_energy_from_edges(s, edges, grid_size)
        delta = neigh_energy - current_energy
        if delta < 0 or random.random() < math.exp(-delta / max(1e-12, temperature)):
            current_state, current_energy = s, neigh_energy
            if current_energy < best_energy:
                best_energy, best_state = current_energy, current_state.copy()
                last_improvement_iter = i

        temperature *= cooling_rate
        history.append(best_energy)

        if reheating and (i - last_improvement_iter) > stagnation_window and (i % 1000 == 0):
            if verbose: print(f"[reheat] @ iter {i}, T -> {temperature * 1.7:.2f}")
            temperature *= 1.7
            for _ in range(3):
                a, b = random.sample(range(n), 2);
                current_state[a], current_state[b] = current_state[b], current_state[a]
            current_energy = calculate_energy_from_edges(current_state, edges, grid_size)
            if current_energy < best_energy:
                best_energy, best_state = current_energy, current_state.copy()
                last_improvement_iter = i

        if verbose and (i + 1) % max(1, max_iterations // 10) == 0:
            print(f"Iter {i + 1}/{max_iterations} | Temp: {temperature:.2f} | Best E: {best_energy:.6f}")
    return best_state, best_energy, history


def main(args):
    if args.seed is not None: random.seed(args.seed); np.random.seed(args.seed)

    global _edge_cost_params
    _edge_cost_params['alpha_ncc'] = args.alpha_ncc
    _edge_cost_params['alpha_mse'] = args.alpha_mse

    img = load_octave_mat_text_or_scipy(args.mat_path)
    print(f"Loaded image {img.shape}, using GRID_SIZE={args.grid_size}")
    pieces, natural_state, piece_dim = create_puzzle(img, args.grid_size)
    print(f"Created {len(pieces)} pieces ({piece_dim}x{piece_dim})")
    edges = precompute_edges(pieces, band_width=args.band_width)

    if args.use_smart_init:
        print("Using greedy initialization...")
        start_state = greedy_initial_state(pieces, edges, args.grid_size)
    else:
        start_state = natural_state.copy();
        random.shuffle(start_state)

    print("\nStage 1: Coarse Annealing...")
    t = time.time()
    state1, energy1, _ = solve_with_simulated_annealing(pieces, start_state, args.grid_size, edges, args.stage1_temp,
                                                        args.stage1_cool, args.stage1_iters, args.verbose, True)
    print(f"Stage 1 finished in {time.time() - t:.1f}s | energy: {energy1:.6f}")

    print("\nStage 2: Fine-Tuning...")
    t = time.time()
    state2, energy2, history2 = solve_with_simulated_annealing(pieces, state1, args.grid_size, edges, args.stage2_temp,
                                                               args.stage2_cool, args.stage2_iters, args.verbose, True)
    print(f"Stage 2 finished in {time.time() - t:.1f}s | energy: {energy2:.6f}")

    print("\nApplying local greedy refinement...")
    state3, energy3 = local_refinement(state2, edges, args.grid_size, max_iters=args.local_refine_iters)
    print(f"Refinement finished | final energy: {energy3:.6f}")

    final_state, final_energy = state3, energy3
    print(f"\n✅ Finished. Final energy: {final_energy:.6f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes[0].imshow(reconstruct_image(natural_state, pieces, args.grid_size, piece_dim), cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Scrambled Puzzle (Input)');
    axes[0].axis('off')
    axes[1].imshow(reconstruct_image(final_state, pieces, args.grid_size, piece_dim), cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Solved Puzzle\nEnergy: {final_energy:.6f}');
    axes[1].axis('off')
    axes[2].plot(history2);
    axes[2].set_title('Best Energy over Iterations (stage 2)')
    axes[2].set_xlabel('Iteration');
    axes[2].set_ylabel('Energy (dissimilarity)');
    axes[2].grid(True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150);
    print(f"Saved result figure to: {args.output}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final improved puzzle solver.")
    parser.add_argument("--mat-path", type=str, default="scrambled_lena (1).mat")
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--band-width", type=int, default=8, help="Width of pixel band for edge comparison.")
    parser.add_argument("--alpha-ncc", type=float, default=0.4, help="Weight for NCC dissimilarity.")
    parser.add_argument("--alpha-mse", type=float, default=0.6, help="Weight for MSE.")
    parser.add_argument("--use-smart-init", action='store_true', help="Use greedy init.")
    parser.add_argument("--stage1-iters", type=int, default=250000)
    parser.add_argument("--stage2-iters", type=int, default=500000)
    parser.add_argument("--stage1-temp", type=float, default=900.0)
    parser.add_argument("--stage2-temp", type=float, default=100.0)
    parser.add_argument("--stage1-cool", type=float, default=0.999993)
    parser.add_argument("--stage2-cool", type=float, default=0.999996)
    parser.add_argument("--local-refine-iters", type=int, default=5000)
    parser.add_argument("--output", type=str, default="solution_final.png")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--seed", type=int, default=None)
    main(parser.parse_args())
