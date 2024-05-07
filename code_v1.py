import numpy as np

class ParticleFilter:
    def __init__(self, particles, weights):
        self.particles = np.array(particles)
        self.weights = np.array(weights)

    def update(self, action, observation, transition_model, observation_model):
        """ Update particles and weights based on action and observation """
        num_particles = len(self.particles)
        # Transition update
        self.particles = transition_model(self.particles, action)
        # Weight update based on observation likelihood
        likelihoods = observation_model(self.particles, observation)
        self.weights *= likelihoods
        if np.sum(self.weights) == 0:
            self.weights += 1e-10  # prevent division by zero
        self.weights /= np.sum(self.weights)  # Normalize weights

    def simplify(self, num_simplified_particles):
        """ Simplify the particle set to reduce computational complexity """
        if num_simplified_particles >= len(self.particles):
            return self.particles.copy(), self.weights.copy()
        indices = np.random.choice(range(len(self.particles)), size=num_simplified_particles, replace=False, p=self.weights)
        simplified_particles = self.particles[indices]
        simplified_weights = self.weights[indices]
        simplified_weights /= np.sum(simplified_weights)  # Normalize weights
        return simplified_particles, simplified_weights

    def calculate_entropy_bounds(self, simplified_particles, simplified_weights):
        """ Calculate differential entropy bounds for a simplified belief """
        # Placeholder for entropy calculation
        entropy = -np.sum(simplified_weights * np.log(simplified_weights + 1e-10))  # Adding small epsilon to avoid log(0)
        return entropy - 0.1, entropy + 0.1  # Example bounds

def transition_model(particles, action):
    """ Apply the transition model to particles """
    # Simple additive noise model for demonstration
    return particles + action + np.random.normal(0, 0.1, particles.shape)

def observation_model(particles, observation):
    """ Calculate likelihood of observation given particle states """
    # Assuming Gaussian noise in observations
    return np.exp(-0.5 * np.linalg.norm(particles - observation, axis=1)**2 / 0.1**2)

def optimal_policy(particle_filter, actions, observations, horizon, depth=0):
    if depth == horizon:
        return None, 0

    best_action = None
    best_value = float('-inf')

    for action in actions:
        expected_value = 0
        for observation in observations:
            # Copy particle filter for hypothetical updates
            pf_copy = ParticleFilter(particle_filter.particles.copy(), particle_filter.weights.copy())
            pf_copy.update(action, observation, transition_model, observation_model)
            simplified_particles, simplified_weights = pf_copy.simplify(100)  # Simplify to 100 particles
            lb, ub = pf_copy.calculate_entropy_bounds(simplified_particles, simplified_weights)
            _, value = optimal_policy(pf_copy, actions, observations, horizon, depth + 1)
            expected_value += ub + value  # Use upper bound for optimistic planning

        if expected_value > best_value:
            best_value = expected_value
            best_action = action

    return best_action, best_value

# Example usage
initial_particles = np.random.randn(1000, 2)  # 1000 particles in a 2D space
initial_weights = np.ones(1000) / 1000  # Uniform initial weights
particle_filter = ParticleFilter(initial_particles, initial_weights)
actions = [np.array([1, 0]), np.array([0, 1])]  # Example actions
observations = [np.array([10, 10]), np.array([10, 11])]  # Example observations

action, value = optimal_policy(particle_filter, actions, observations, horizon=3)
print("Optimal Action:", action, "Expected Value:", value)
