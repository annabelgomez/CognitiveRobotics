# This runs!!!!!!!!! (corrected version 5)

import numpy as np
from scipy.stats import norm
import time 

class ParticleFilter:
    def __init__(self, particles, weights, transition_std=0.1, observation_std=0.1):
        self.particles = np.array(particles)
        self.weights = np.array(weights)
        self.transition_std = transition_std
        self.observation_std = observation_std

    def update(self, action, observation, transition_positions, observation_model):
        self.particles = transition_positions(self.particles, action)
        likelihoods = observation_model(self.particles, observation)
        self.weights *= likelihoods
        if np.sum(self.weights) == 0:
            self.weights += 1e-10
        self.weights /= np.sum(self.weights)

    def simplify(self, num_simplified_particles):
        indices = np.random.choice(len(self.particles), size=num_simplified_particles, replace=False, p=self.weights)
        simplified_particles = self.particles[indices]
        simplified_weights = self.weights[indices]
        simplified_weights /= np.sum(simplified_weights)
        return simplified_particles, simplified_weights, indices

    def calculate_entropy_bounds(self, x_new, w_new, indices, observation, action, x_s_old, w_s_old):
        m = 1.0

        lower_bound, upper_bound = calculate_bounds(
            x_new, w_new, 
            indices, #need to code 
            action, observation, #observation = z_next
            self.transition_probability,
            self.observation_model,
            m,
            x_s_old,
            w_s_old
        )

        return lower_bound, upper_bound

    def transition_positions(self, x, a):
        noise = np.random.normal(0, self.transition_std, x.shape)
        return x + a + noise
    
    def transition_probability(self, x_next, x_prev, a): #NEED A VERSION OF THIS SO THAT IT GIVES US P(x_k+1 | x_k, a_k)
        # Mean of the distribution
        mean = x_prev + a
        # Standard deviation of the distribution
        std = self.transition_std
        # Compute the probability density of x_next given the normal distribution
        probability = norm.pdf(x_next, loc=mean, scale=std)
        return probability

    def observation_model(self, x, z):
        if x.ndim == 1:  # If x is 1D, use element-wise norm
            norm = np.linalg.norm(x - z)
        else:  # Otherwise, use axis 1 norm for multi-dimensional x
            norm = np.linalg.norm(x - z, axis=1)
        return np.exp(-0.5 * norm ** 2 / self.observation_std ** 2)

class POMDP:
    def __init__(self, initial_particles, initial_weights, actions, observations, transition_std=0.1, observation_std=0.1):
        self.particle_filter = ParticleFilter(initial_particles, initial_weights, transition_std, observation_std)
        self.actions = actions
        self.observations = observations

    def find_optimal_policy(T):
        s = 50
        return self.adapt_simplification(T,s)
    
    def prune_branches(T, bounds):
        LB_max = max( all LBs )
        for child in T:
            UB_m = upper bound of child
            if LB_max > UB_m:
                T.delete (child)

    def adapt_simplification(T, s):
        if T is leaf:
            lb, ub = calculate_entropy_bounds(pf_new.particles, pf_new.weights, indices, observation, action, x_s_old, w_s_old)
            return lb, ub

        for all subtrees in T:
            actions,LB_next,UB_next = adapt_simplification(subtree, s)
            LB_sj, UB_sj = ???
        
        prune_branches(T, bounds)
        
        while not pruned branches > 1:
            adapt_simplification(T, s+50)
        
        update LB_sj, UB_sj according to (13)
        return optimal_action, LB_sj, UB_sj


    def adapt_simplification(self, sn, x_s_old, w_s_old):
        actions = []
        bounds = []

        for action in self.actions:
            for observation in self.observations:
                pf_new = ParticleFilter(self.particle_filter.particles.copy(), self.particle_filter.weights.copy(), self.particle_filter.transition_std, self.particle_filter.observation_std)
                pf_new.update(action, observation, pf_new.transition_positions, pf_new.observation_model)
                x_s_new, w_s_new, indices = pf_new.simplify(sn)
                #possible: implement way to incrementally calculate bounds after increasing sample size
                #instead of starting from scratch
                lb, ub = pf_new.calculate_entropy_bounds(pf_new.particles, pf_new.weights, indices, observation, action, x_s_old, w_s_old)
                sub_actions, LB_future, UB_future = self.optimal_policy(horizon, x_s_new, w_s_new, depth + 1)

                # Convert ub and value to floats
                # if isinstance(ub, np.ndarray) and ub.size == 1:
                #     ub = ub.item()
                # elif isinstance(ub, np.ndarray):
                #     ub = float(np.mean(ub))

                LB_new = lb + LB_future#/sn
                UB_new = ub + UB_future#/sn

                # if isinstance(value, np.ndarray) and value.size == 1:
                #     value = value.item()
                # elif isinstance(value, np.ndarray):
                #     value = float(np.mean(value))
    
                # expected_value += ub + value

            # Consider only the best sequence from the current action forward
            # if expected_value > best_value:
            #     best_value = expected_value
            #     best_action_sequence = [action] + sub_actions

            actions.append(action)
            bounds.append((LB_new, UB_new))

        return actions, bounds


    def optimal_policy(self, horizon, x_s_old, w_s_old, depth=0):
        if depth == horizon:
            return [], 0

        best_value = float('-inf')
        best_action_sequence = []

        sn = 100
        actions, bounds = self.adapt_simplification(sn, x_s_old, w_s_old)

        while check_overlaps(bounds) == True:
            sn += 50
            actions, bounds = self.adapt_simplification(sn, x_s_old, w_s_old)
        else:
            best_action = #highest upper bound action
            best_action_sequence = [best_action] + sub_actions

        return best_action_sequence, lb, ub

def calculate_bounds(x_new, w_new, indices, action, observation, transition_probability, observation_model, m, x_s_old, w_s_old):
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

def check_overlaps(intervals): #input as list of tuples
    # Sort intervals based on the start time
    intervals.sort(key=lambda x: x[0])
    # Iterate through the sorted intervals
    for i in range(1, len(intervals)):
        # Check if the current interval starts before the previous one ends
        if intervals[i][0] < intervals[i - 1][1]:
            return True  # Overlap detected
    return False  # No overlaps found

# # Example usage
# intervals = [(1, 3), (5, 8), (2, 6)]
# print(check_overlaps(intervals))  # Output: True

# Example usage
pn = 1000
initial_particles = np.random.randn(pn, 2)
initial_weights = np.ones(pn) / pn
actions = [np.array([1, 0]), np.array([0, 1])]
observations = [np.array([10, 10]), np.array([10, 11])]

pomdp = POMDP(initial_particles, initial_weights, actions, observations)
action, value = pomdp.optimal_policy(3, initial_particles, initial_weights)
print("Optimal Action:", action, "Expected Value:", value)
