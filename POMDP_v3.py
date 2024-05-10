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

    def update(self, action, observation, transition_model, observation_model):
        self.particles = transition_model(self.particles, action)
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

    def calculate_entropy_bounds(self, simplified_particles, simplified_weights, z_next, action):
        m = 1.0
        As_k = set(range(len(self.particles)))
        As_k_plus_1 = set(range(len(simplified_particles)))

        lower_bound, upper_bound = calculate_bounds(
            self.particles, self.weights,
            action, z_next,
            self.transition_model,
            self.observation_model,
            m,
            As_k,
            As_k_plus_1
        )
        return lower_bound, upper_bound

    def transition_model(self, x_next, x_prev, a): #NEED A VERSION OF THIS SO THAT IT GIVES US P(x_k+1 | x_k, a_k)
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

    def optimal_policy(self, horizon, depth=0):
        if depth == horizon:
            return None, 0

        best_action = None
        best_value = float('-inf')

        sn = 100
        for action in self.actions:
            expected_value = 0
            for observation in self.observations:
                pf_copy = ParticleFilter(self.particle_filter.particles.copy(), self.particle_filter.weights.copy(), self.particle_filter.transition_std, self.particle_filter.observation_std)
                pf_copy.update(action, observation, pf_copy.transition_model, pf_copy.observation_model)
                simplified_particles, simplified_weights, indices = pf_copy.simplify(an)
                lb, ub = pf_copy.calculate_entropy_bounds(simplified_particles, simplified_weights, observation, action)
                _, value = self.optimal_policy(horizon, depth + 1)

                # Convert ub and value to floats
                if isinstance(ub, np.ndarray) and ub.size == 1:
                    ub = ub.item()
                elif isinstance(ub, np.ndarray):
                    ub = float(np.mean(ub))

                if isinstance(value, np.ndarray) and value.size == 1:
                    value = value.item()
                elif isinstance(value, np.ndarray):
                    value = float(np.mean(value))

                expected_value += ub + value

            if expected_value > best_value:
                best_value = expected_value
                best_action = action

        return best_action, best_value

def calculate_bounds(particles, weights, action, z_next, transition_model, observation_model, m, As_k, As_k_plus_1):
    eps = 1e-10  # Small constant to prevent log(0)
    lower_bound, upper_bound = 0, 0
    for i, (x_i, w_i) in enumerate(zip(particles, weights)):
        P_z_x_i = observation_model(x_i, z_next) #P(z_k+1 | x_i_k+1)
        if i not in As_k_plus_1:
            lower_bound -= w_i * np.log(m * P_z_x_i + eps) #w_i = w_i_k+1 ?? 
        else:
            #transition_model(x_i, x_j) = P(x_i | x_j, a_k)
            sum_term = sum(transition_model(x_i, x_j) * w_j for w_j in weights) #this is not right 
            #need new transition model that gives probabilities given past states + action
            #we need all the weights from step k (w_j_k) in the input?? 

            lower_bound -= w_i * np.log(P_z_x_i * sum_term + eps)
    #here we need to sum over the PREVIOUS state k (j in As_k)
    for x_i, w_i in zip(particles, weights):
        sum_term = sum(observation_model(x_j, z_next) * transition_model(x_j, x_i) * w_j for j, (x_j, w_j) in enumerate(zip(particles, weights)) if j in As_k)
        upper_bound -= w_i * np.log(sum_term + eps)
    return lower_bound, upper_bound

# Example usage
pn = 1000
initial_particles = np.random.randn(pn, 2)
initial_weights = np.ones(pn) / pn
actions = [np.array([1, 0]), np.array([0, 1])]
observations = [np.array([10, 10]), np.array([10, 11])]

pomdp = POMDP(initial_particles, initial_weights, actions, observations)
action, value = pomdp.optimal_policy(horizon=3)
print("Optimal Action:", action, "Expected Value:", value)
