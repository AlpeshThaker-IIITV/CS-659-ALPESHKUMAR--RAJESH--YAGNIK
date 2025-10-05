import random

def generate_k_sat(k, num_clauses, num_vars):
    """
    Creates a random k-SAT formula with the specified parameters.

    Parameters:
    k (int): Number of literals per clause
    num_clauses (int): Total number of clauses
    num_vars (int): Total number of distinct variables

    Returns:
    List of clauses, where each clause is a list of integers representing literals.
    Positive integer = variable, negative integer = negated variable.
    """
    if k > num_vars:
        raise ValueError("Each clause can't have more literals than total variables.")

    all_vars = list(range(1, num_vars + 1))  # List of variable indices
    formula = []

    for _ in range(num_clauses):
        # Randomly pick k unique variables for this clause
        chosen_vars = random.sample(all_vars, k)
        clause = []

        for var in chosen_vars:
            # Flip a coin to decide if the variable is negated
            if random.choice([True, False]):
                clause.append(var)
            else:
                clause.append(-var)

        formula.append(clause)

    return formula

def print_formula(formula):
    """
    Prints the formula in a readable way with variables and negations.
    """
    formatted_clauses = []

    for clause in formula:
        literals = []
        for lit in clause:
            if lit > 0:
                literals.append(f"x{lit}")
            else:
                literals.append(f"NOT x{abs(lit)}")
        formatted_clauses.append("(" + " OR ".join(literals) + ")")

    print(" AND\n".join(formatted_clauses))

if __name__ == "__main__":
    # Pick some random but reasonable values for variables and clauses
    num_vars = random.randint(4, 10)                # Between 4 and 10 variables
    k = random.randint(2, min(5, num_vars))         # Literals per clause, capped by num_vars
    num_clauses = random.randint(num_vars, num_vars * 2)  # Number of clauses between num_vars and twice that

    print(f"Randomly picked parameters:")
    print(f" - Literals per clause (k): {k}")
    print(f" - Number of clauses (m): {num_clauses}")
    print(f" - Number of variables (n): {num_vars}\n")

    formula = generate_k_sat(k, num_clauses, num_vars)

    print(f"Here's the generated {k}-SAT formula:")
    print_formula(formula)
