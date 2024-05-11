import numpy as np
from scipy.stats import norm
from belief_tree import Node, BeliefTree, ParticleFilter

def find_optimal_policy(T):
    s = 5
    return adapt_simplification(T, s)

def adapt_simplification(T, s_i):
    pf_new = ParticleFilter(T.particle_filter.particles.copy(), T.particle_filter.weights.copy())
    x_s_new, w_s_new, indices = pf_new.simplify(s_i)
    if T.isleaf() == True:
        lb, ub = calculate_bounds(
            T.particles, T.weights, 
            indices, 
            T.prev_action, T.prev_observation, #observation = z_next
            ParticleFilter.transition_probability,
            ParticleFilter.observation_model,
            T.parent.particles,
            T.parent.weights
        )
        return lb, ub

    for action in T.actions:
        #need to get mean over the observations for this action
        lb_arr = []
        ub_arr = []
        for child in T.get_children_by_action(action):
            actions, LB, UB = adapt_simplification(child, s_i)
            lb_arr.append(LB)
            ub_arr.append(UB)
        mean_lb = sum(lb_arr)/len(lb_arr)
        mean_ub = sum(ub_arr)/len(ub_arr)



def calculate_bounds(x_new, w_new, indices, action, observation, transition_probability, observation_model, x_s_old, w_s_old):
    m = 1.0
    eps = 1e-10  # Small constant to prevent log(0)
    lower_bound, upper_bound = 0, 0
    for i, (x_i, w_i) in enumerate(zip(x_new, w_new)):
        P_z_x_i = observation_model(x_i, observation) #P(z_k+1 | x_i_k+1)
        if i not in indices:
            lower_bound -= w_i * np.log(m * P_z_x_i + eps) #w_i = w_i_k+1 
        else:
            #transition_probability(x_i, x_j) = P(x_i | x_j, a_k)
            sum_term = sum(transition_probability(x_i, x_j, action) * w_j for x_j, w_j in zip(x_s_old, w_s_old))
            lower_bound -= w_i * np.log(P_z_x_i * sum_term + eps)
    #here we need to sum over the PREVIOUS state k (j in As_k)
    for x_i, w_i in zip(x_new, w_new):
        sum_term = sum(observation_model(x_j, observation) * transition_probability(x_i, x_j, action) * w_j for x_j, w_j in zip(x_s_old, w_s_old))
        upper_bound -= w_i * np.log(sum_term + eps)

    a = np.log(sum(observation_model(x_i, observation) * w_i for x_i, w_i in zip(x_new, w_new)) + eps)

    return lower_bound + a, upper_bound + a