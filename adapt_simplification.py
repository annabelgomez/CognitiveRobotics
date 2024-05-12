import numpy as np
from scipy.stats import norm
from belief_tree import Node, BeliefTree, ParticleFilter

def find_optimal_policy(T):
    s = 10
    return adapt_simplification(T, s)

def adapt_simplification(T, s_i):
    print("Adapt Simplification Running", "s_i:", s_i)
    pf_new = ParticleFilter(T.particles.copy(), T.weights.copy())
    _, _, indices = pf_new.simplify(s_i)

    #if we are at the bottom of the tree (T is leaf), then return the bounds of leaf T.
    if T.is_leaf() == True:
        print("T.is_leaf() == True")
        lb, ub = calculate_bounds(
            T.particles, T.weights, 
            indices, 
            T.prev_action, T.prev_observation,
            T.parent.particles,
            T.parent.weights
        )
        T.set_bounds(lb=lb, ub=ub)
        return lb, ub, []
    print("T.is_leaf() == False")
    #Get here if T is a subtree and has children.
    child_nodes = compare_children(T, s_i)
    #prune_children prunes the tree if possible and returns the optimal action if
    #one is found.
    optimal_child, child_action_branch = prune_children(T)

    #optimal_child returns False if the bounds of the actions are overlapping.
    if optimal_child == False:
        #increase s and repeat.
        return adapt_simplification( T , s_i+10 )

    #if best_action returns the optimal action:
    else:
        LB = optimal_child.get_lower_bound()
        UB = optimal_child.get_upper_bound()

        #get optimal action for node T
        best_action = optimal_child.prev_action

        print("optimal_child", optimal_child)
        print("child_action_branch", child_action_branch)
        new_action_branch = [best_action] + child_action_branch
        T.set_optimal_action(new_action_branch)

        # if T is the head node then we're done and we can return the optimal 
        # action sequence and the bounds on the associated rewards.
        if T.is_head_node() == True:
            T.set_bounds(LB, UB)
            return LB, UB, new_action_branch
        else:
        #get the bounds for this node 
        #reward of going from parent node to
        #this node via prev_action and prev_observation
            lb, ub = calculate_bounds(
                T.particles, T.weights, 
                indices, 
                T.prev_action, T.prev_observation,
                T.parent.particles,
                T.parent.weights
            )
            LB = lb + LB
            UB = ub + UB
            T.set_bounds(LB, UB)
            print(LB, UB, new_action_branch)
            return LB, UB, new_action_branch

def compare_children(T, s_i):
    """
    Inputs: a tree/subtree T and sample size s_i,
    T - populated instance of BeliefTree
    s_i - integer
    Outputs:
    mean_bounds - nested dict that stores the upper/lower bounds for each action.
                            {'lb': lb, 'ub': ub }
    """
    print("compare_children running")

    # action_mean_bounds = {}
    # for action in T.actions:    
        #mean future bounds over possible observations for this action
    mean_bounds = []
    child_nodes = T.get_all_children()
    print("iterating through children")
    for child in child_nodes:
        print("iterating to child", child)
        # print(adapt_simplification(child, s_i))
        LB, UB, _ = adapt_simplification(child, s_i)
        child.set_bounds(lb=LB, ub=UB)
        print(LB, UB)
        # mean_bounds.append({'lb': LB, 'ub': UB})

    return child_nodes #, mean_bounds

def prune_children(T):#, mean_bounds):
    """
    Input: action_mean_bounds: nested dict (see compare_children for structure/description)

    Output:
        False - if bounds are overlapping.
        optimal action - if bounds are not overlapping.
    """
    print("prune_children running")
    child_nodes = T.get_all_children()
    LB_star = max(child.get_lower_bound() for child in child_nodes)
    print("LB_star", LB_star)
    pruned_children = []
    for child in child_nodes:
        if LB_star > child.get_upper_bound():
            print("Removing child",child)
        else:
            print("Adding to pruned_children", child)
            pruned_children.append(child)
    T.update_children(pruned_children)
    if len(pruned_children) == 1:
        optimal_child = pruned_children[0]
        action_branch = optimal_child.get_optimal_action()
        return optimal_child, action_branch
    else:
        return False, None

def calculate_bounds(x_new, w_new, indices, action, observation, x_s_old, w_s_old):
    print("calculate_bounds running")
    # print("len(x_new)", len(x_new))
    # print("indices", indices)
    pf = ParticleFilter(x_new.copy(), w_new.copy())
    m = 1.0
    eps = 1e-10  # Small constant to prevent log(0)
    lower_bound, upper_bound = 0, 0
    if len(indices) >= len(x_new):
        for i, (x_i, w_i) in enumerate(zip(x_new, w_new)):
            term = w_i * np.log(pf.observation_model(x_i, observation) * sum(pf.transition_probability(x_i, x_j, action) * w_j for x_j, w_j in zip(x_s_old, w_s_old)) + eps)
            lower_bound -= term
            upper_bound -= term
    else:
        for i, (x_i, w_i) in enumerate(zip(x_new, w_new)):
            P_z_x_i = pf.observation_model(x_i, observation) #P(z_k+1 | x_i_k+1)
            if i not in indices:
                # print("w_i * np.log(m * P_z_x_i + eps)",w_i * np.log(m * P_z_x_i + eps))
                lower_bound -= w_i * np.log(m * P_z_x_i + eps) #w_i = w_i_k+1 
            else:
                #transition_probability(x_i, x_j) = P(x_i | x_j, a_k)
                sum_term = sum(pf.transition_probability(x_i, x_j, action) * w_j for x_j, w_j in zip(x_s_old, w_s_old))
                lower_bound -= w_i * np.log(P_z_x_i * sum_term + eps)
                # print("w_i * np.log(P_z_x_i * sum_term + eps)", w_i * np.log(P_z_x_i * sum_term + eps))
        #here we need to sum over the PREVIOUS state k (j in As_k)
        for x_i, w_i in zip(x_new, w_new):
            sum_term = sum(pf.observation_model(x_j, observation) * pf.transition_probability(x_i, x_j, action) * w_j for x_j, w_j in zip(x_s_old, w_s_old))
            upper_bound -= w_i * np.log(sum_term + eps)

    a = np.log(sum(pf.observation_model(x_i, observation) * w_i for x_i, w_i in zip(x_new, w_new)) + eps)
    # print("lower_bound",lower_bound)
    # print("upper_bound",upper_bound)
    # print("a",a)
    if upper_bound < lower_bound:
        lower_bound = upper_bound
    return lower_bound + a, upper_bound + a
