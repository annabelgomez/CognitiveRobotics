# Explaination: Python implementation of the bounds for a given belief b
# belief b: a list of tuples representing the set of particles {(x_i, w_i)}
# action a: the action taken
# observation z_k+1 : the next observation
# P(x_i | x_j, a): the transition proabbility function 
# P(z|x): the likelihood function 
# m: the maximum value of the transition probability 
# A*_k and A*_k+1 : the sets of particle indices for simplified beliefs


import numpy as np

def calculate_bounds(b, a, z_next, P_x_given_x_a, P_z_given_x, m, As_k, As_k_plus_1):
    """
    Calculates the lower and upper bounds for the belief.
    :param b: List of tuples [(x_i, w_i)], representing the belief
    :param a: Action taken
    :param z_next: Observation
    :param P_x_given_x_a: Function that gives P(x_i | x_j, a)
    :param P_z_given_x: Function that gives P(z | x)
    :param m: Maximum value of P(x_{k+1} | x_k, a)
    :param As_k: Set of particle indices for simplified belief at time k
    :param As_k_plus_1: Set of particle indices for simplified belief at time k+1
    :return: Lower and upper bounds for the belief
    """
    lower_bound, upper_bound = 0, 0
    # Lower Bound
    for i, (x_i, w_i) in enumerate(b):
        P_z_x_i = P_z_given_x(z_next, x_i)
        if i not in As_k_plus_1:
            lower_bound -= w_i * np.log(m * P_z_x_i)
        else:
            sum_term = sum(P_x_given_x_a(x_i, x_j, a) * w_j for j, (x_j, w_j) in enumerate(b))
            lower_bound -= w_i * np.log(P_z_x_i * sum_term)
    # Upper Bound
    for i, (x_i, w_i) in enumerate(b):
        sum_term = sum(P_z_given_x(z_next, x_i) * P_x_given_x_a(x_i, x_j, a) * w_j for j in As_k)
        upper_bound -= w_i * np.log(sum_term)
    return lower_bound, upper_bound

# Example POMDP setup
belief = [(np.array([1.0, 2.0]), 0.5), (np.array([3.0, 4.0]), 0.5)]
action = "move_forward"
z_next = np.array([2.5, 3.5])
P_x_given_x_a = lambda x_i, x_j, a: 1 / (np.linalg.norm(x_i - x_j) + 1)
P_z_given_x = lambda z, x: 1 / (np.linalg.norm(z - x) + 1)
m = 1.0
As_k = {0}
As_k_plus_1 = {1}

lower, upper = calculate_bounds(belief, action, z_next, P_x_given_x_a, P_z_given_x, m, As_k, As_k_plus_1)
print(f"Lower Bound: {lower}, Upper Bound: {upper}")
