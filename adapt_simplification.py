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
            T.prev_action, T.prev_observation,
            ParticleFilter.transition_probability,
            ParticleFilter.observation_model,
            T.parent.particles,
            T.parent.weights
        )
        T.set_bounds(lb=lb, ub=ub)
        return lb, ub

    action_mean_bounds = compare_actions(T, s_i)
    best_action = choose_best_action(action_mean_bounds)

    if best_action == False:
        #increase s and repeat.
        adapt_simplification(T, s_i + 1)
    else:
        if T.is_head_node() == True:
            lb = 0
            ub = 0
        else:
        #get the bounds for this node 
        #reward of going from parent node to
        #this node via prev_action and prev_observation
            lb, ub = calculate_bounds(
                T.particles, T.weights, 
                indices, 
                T.prev_action, T.prev_observation,
                ParticleFilter.transition_probability,
                ParticleFilter.observation_model,
                T.parent.particles,
                T.parent.weights
            )
        LB = lb + LB
        UB = ub + UB
        T.set_bounds(LB, UB)
        return LB, UB, action

def compare_actions(T, s_i):
    action_mean_bounds = {}
    for action in T.actions:    
        #mean future bounds over possible observations for this action
        lb_arr = []
        ub_arr = []
        child_nodes = T.get_children_by_action(action)
        for child in child_nodes:
            LB, UB = adapt_simplification(child, s_i)
            lb_arr.append(LB)
            ub_arr.append(UB)
        mean_lb = sum(lb_arr)/len(lb_arr)
        mean_ub = sum(ub_arr)/len(ub_arr)
        action_mean_bounds[action] = {'lb': mean_lb, 'ub': mean_ub}
    return action_mean_bounds

def choose_best_action(action_mean_bounds):
    """
    action_mean_bounds[action] = {'lb': mean_lb, 'ub': mean_ub} 
    Prune branches
    Return two terms:
    1st term: Boolean; True if len is 1
    2nd term: new list of pruned_children
    """
    LB_star = max(action_mean_bounds[action]['lb'] for action in action_mean_bounds.keys())

    pruned_actions = []
    for action in action_mean_bounds.keys():
        if LB_star > action_mean_bounds[action]['ub']:
            print("Pruning child")
        else:
            pruned_actions.append(action_mean_bounds[action])
    if len(pruned_actions) == 1:
        return pruned_actions[0]
    else:
        return False

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