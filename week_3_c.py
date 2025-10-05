import random
import time
from itertools import combinations

def generate_random_3sat(num_clauses, num_vars):
    """
    Generate a random 3-SAT formula with the specified number of clauses and variables.
    Each clause has exactly 3 literals, variables are distinct within a clause,
    and literals can be negated randomly.
    """
    formula = []
    vars_list = list(range(1, num_vars + 1))

    for _ in range(num_clauses):
        chosen_vars = random.sample(vars_list, 3)
        clause = []
        for var in chosen_vars:
            # Randomly decide to negate or not
            literal = var if random.choice([True, False]) else -var
            clause.append(literal)
        formula.append(clause)
    return formula


def is_clause_satisfied(clause, assignment):
    """
    Check if a clause is satisfied by the current variable assignment.
    A clause is satisfied if at least one literal is true.
    """
    for literal in clause:
        variable = abs(literal)
        value = assignment.get(variable, None)
        if value is None:
            # Variable not assigned yet, skip
            continue
        # Literal is true if var assigned True and literal positive, or var False and literal negative
        if (literal > 0 and value) or (literal < 0 and not value):
            return True
    return False


def count_satisfied_clauses(formula, assignment):
    """
    Count how many clauses in the formula are satisfied by the given assignment.
    """
    satisfied_count = 0
    for clause in formula:
        if is_clause_satisfied(clause, assignment):
            satisfied_count += 1
    return satisfied_count


def create_random_assignment(num_vars):
    """
    Create a random truth assignment for all variables.
    """
    return {var: random.choice([True, False]) for var in range(1, num_vars + 1)}


def flip_var_value(assignment, variable):
    """
    Flip the boolean value assigned to a specific variable.
    """
    assignment[variable] = not assignment[variable]


def copy_assignment(assignment):
    """
    Make a copy of the current assignment dictionary.
    """
    return dict(assignment)


# Heuristics

def heuristic_unsatisfied_count(formula, assignment):
    """
    Heuristic function that returns the number of unsatisfied clauses.
    Lower value means better solution.
    """
    total_clauses = len(formula)
    satisfied = count_satisfied_clauses(formula, assignment)
    return total_clauses - satisfied


def heuristic_satisfied_count(formula, assignment):
    """
    Heuristic function that returns the number of satisfied clauses.
    Higher value means better solution.
    """
    return count_satisfied_clauses(formula, assignment)


# Hill Climbing Algorithm

def hill_climbing(formula, num_vars, heuristic_func, max_iterations=1000):
    """
    Basic hill climbing to maximize/minimize heuristic_func.
    Returns best assignment found and number of steps taken.
    """
    current_assignment = create_random_assignment(num_vars)
    current_score = heuristic_func(formula, current_assignment)

    for iteration in range(max_iterations):
        # If solution is perfect, stop
        if current_score == len(formula):
            return current_assignment, iteration

        improved = False

        for var in range(1, num_vars + 1):
            neighbor = copy_assignment(current_assignment)
            flip_var_value(neighbor, var)
            neighbor_score = heuristic_func(formula, neighbor)

            # Decide if neighbor is better
            if heuristic_func == heuristic_satisfied_count:
                better = neighbor_score > current_score
            else:
                better = neighbor_score < current_score

            if better:
                current_assignment = neighbor
                current_score = neighbor_score
                improved = True
                break  # move to the better neighbor immediately

        if not improved:
            # No better neighbor found; local optimum reached
            break

    return current_assignment, max_iterations


# Beam Search Algorithm

def beam_search(formula, num_vars, heuristic_func, beam_width=3, max_iterations=1000):
    """
    Beam search maintains a set (beam) of best assignments at each iteration.
    Expands neighbors and keeps top beam_width candidates.
    """
    beam = [create_random_assignment(num_vars)]
    best_assignment = beam[0]
    best_score = heuristic_func(formula, best_assignment)

    for iteration in range(max_iterations):
        if best_score == len(formula):
            return best_assignment, iteration

        all_candidates = []

        # Generate neighbors for all assignments in the beam
        for assignment in beam:
            for var in range(1, num_vars + 1):
                neighbor = copy_assignment(assignment)
                flip_var_value(neighbor, var)
                score = heuristic_func(formula, neighbor)
                all_candidates.append((score, neighbor))

        # Sort candidates according to heuristic
        reverse_sort = heuristic_func == heuristic_satisfied_count
        all_candidates.sort(key=lambda x: x[0], reverse=reverse_sort)

        # Keep top candidates for next beam
        beam = [candidate[1] for candidate in all_candidates[:beam_width]]

        top_score = all_candidates[0][0]
        if (reverse_sort and top_score > best_score) or (not reverse_sort and top_score < best_score):
            best_score = top_score
            best_assignment = all_candidates[0][1]

        if best_score == len(formula):
            return best_assignment, iteration

    return best_assignment, max_iterations


# Variable Neighborhood Descent (VND)

def generate_k_flip_neighbors(assignment, num_vars, k):
    """
    Generate neighbors by flipping exactly k variables.
    """
    neighbors = []
    variables = list(range(1, num_vars + 1))

    for var_combo in combinations(variables, k):
        neighbor = copy_assignment(assignment)
        for var in var_combo:
            flip_var_value(neighbor, var)
        neighbors.append(neighbor)

    return neighbors


def variable_neighborhood_descent(formula, num_vars, heuristic_func, max_iterations=1000):
    """
    VND explores neighborhoods of increasing size (flipping 1, 2, then 3 variables).
    Restarts neighborhood search when improvement is found.
    """
    current_assignment = create_random_assignment(num_vars)
    current_score = heuristic_func(formula, current_assignment)
    neighborhoods = [1, 2, 3]

    iteration = 0
    while iteration < max_iterations:
        improved = False

        for k in neighborhoods:
            neighbors = generate_k_flip_neighbors(current_assignment, num_vars, k)
            reverse_sort = heuristic_func == heuristic_satisfied_count

            # Evaluate neighbors and sort
            neighbors_with_scores = [(heuristic_func(formula, nb), nb) for nb in neighbors]
            neighbors_with_scores.sort(key=lambda x: x[0], reverse=reverse_sort)

            best_neighbor_score = neighbors_with_scores[0][0]

            if reverse_sort:
                better = best_neighbor_score > current_score
            else:
                better = best_neighbor_score < current_score

            if better:
                current_assignment = neighbors_with_scores[0][1]
                current_score = best_neighbor_score
                improved = True
                break  # restart with smallest neighborhood

        if not improved:
            # No improvement in any neighborhood
            break

        iteration += 1

    return current_assignment, iteration


# --- Main driver function to run experiments ---

def run_all_experiments():
    # Some sample problem sizes (m = #clauses, n = #variables)
    test_settings = [
        (20, 10),
        (40, 20),
        (60, 30),
    ]

    heuristics = [heuristic_unsatisfied_count, heuristic_satisfied_count]
    algorithms = {
        "Hill Climbing": hill_climbing,
        "Beam Search (width=3)": lambda f, n, h: beam_search(f, n, h, beam_width=3),
        "Beam Search (width=4)": lambda f, n, h: beam_search(f, n, h, beam_width=4),
        "Variable Neighborhood Descent": variable_neighborhood_descent
    }

    for m, n in test_settings:
        formula = generate_random_3sat(m, n)
        print(f"\nRunning experiments for m={m} clauses, n={n} variables")

        for heuristic_func in heuristics:
            print(f"\nUsing heuristic: {heuristic_func.__name__}")
            for algo_name, algo_func in algorithms.items():
                start_time = time.time()
                solution, steps_taken = algo_func(formula, n, heuristic_func)
                elapsed_time = time.time() - start_time

                satisfied_clauses = count_satisfied_clauses(formula, solution)
                print(f"{algo_name}: Satisfied {satisfied_clauses}/{m} clauses; Steps: {steps_taken}; Time: {elapsed_time:.3f} sec")


if __name__ == "__main__":
    run_all_experiments()
