import numpy as np

class ParticleFilter:
    def __init__(self, particles, weights):
        self.particles = np.array(particles)
        self.weights = np.array(weights)

    def update(self, action, observation, transition_model, observation_model):
        self.particles = transition_model(self.particles, action)
        likelihoods = observation_model(self.particles, observation)
        self.weights *= likelihoods
        if np.sum(self.weights) == 0:
            self.weights += 1e-10  # Prevent division by zero
        self.weights /= np.sum(self.weights)  # Normalize weights

    def simplify(self, num_simplified_particles):
        indices = np.random.choice(range(len(self.particles)), size=num_simplified_particles, replace=False, p=self.weights)
        simplified_particles = self.particles[indices]
        simplified_weights = self.weights[indices]
        simplified_weights /= np.sum(simplified_weights)  # Normalize weights
        return simplified_particles, simplified_weights

    def calculate_entropy_bounds(self, simplified_particles, simplified_weights):
        entropy = -np.sum(simplified_weights * np.log(simplified_weights + 1e-10))
        return entropy - 0.1, entropy + 0.1  # Placeholder bounds

class POMDP:
    def __init__(self, initial_particles, initial_weights, actions, observations, transition_std=0.1, observation_std=0.1):
        self.particle_filter = ParticleFilter(initial_particles, initial_weights)
        self.actions = actions
        self.observations = observations
        self.transition_std = transition_std
        self.observation_std = observation_std

    def transition_model(self, particles, action):
        """ Apply the transition model to particles """
        noise = np.random.normal(0, self.transition_std, particles.shape)
        return particles + action + noise

    def observation_model(self, particles, observation):
        """ Calculate likelihood of observation given particle states """
        distances = np.linalg.norm(particles - observation, axis=1)
        return np.exp(-0.5 * (distances ** 2) / self.observation_std ** 2)

    def optimal_policy(self, horizon, depth=0):
        if depth == horizon:
            return None, 0

        best_action = None
        best_value = float('-inf')

        for action in self.actions:
            expected_value = 0
            for observation in self.observations:
                # Simulate belief update
                pf_copy = ParticleFilter(self.particle_filter.particles.copy(), self.particle_filter.weights.copy())
                pf_copy.update(action, observation, self.transition_model, self.observation_model)
                simplified_particles, simplified_weights = pf_copy.simplify(100)  # Simplify to 100 particles
                lb, ub = pf_copy.calculate_entropy_bounds(simplified_particles, simplified_weights)
                _, value = self.optimal_policy(horizon, depth + 1)
                expected_value += ub + value  # Use upper bound for optimistic planning

            if expected_value > best_value:
                best_value = expected_value
                best_action = action

        return best_action, best_value

# Example usage
initial_particles = np.random.randn(1000, 2)  # 1000 particles in a 2D space
initial_weights = np.ones(1000) / 1000  # Uniform initial weights
actions = [np.array([1, 0]), np.array([0, 1])]  # Example actions
observations = [np.array([10, 10]), np.array([10, 11])]  # Example observations

pomdp = POMDP(initial_particles, initial_weights, actions, observations)
action, value = pomdp.optimal_policy(horizon=3)
print("Optimal Action:", action, "Expected Value:", value)
