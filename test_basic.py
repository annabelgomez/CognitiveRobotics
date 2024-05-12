import numpy as np
from belief_tree import BeliefTree
from adapt_simplification import find_optimal_policy
from graph_tree import visualize_belief_tree

pn = 100
initial_particles = np.random.randn(pn, 1)
initial_weights = np.ones(pn) / pn
print("initial_weights", initial_weights)

T = BeliefTree(2, initial_particles, initial_weights)
T.construct_belief_tree()

LB, UB, action_sequence = find_optimal_policy(T.head_node)
print('LB', LB)
print('UB', UB)
print('action_sequence', action_sequence)

print(T.head_node)

visualize_belief_tree(T.head_node)