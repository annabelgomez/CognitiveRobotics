import numpy as np
from scipy.stats import norm
import random

class Node:
    def __init__(self, particles, weights, parent=None, prev_action=None, prev_observation=None):
        self.particles = particles
        self.weights = weights

        if parent == None:
            self.head_node = True
            self.parent = None
        else:
            self.head_node = False
            self.parent = parent

        self.children = []
        self.actions = [0, 1]
        self.observations = [0, 1]
        
        if prev_action:
            self.prev_action = prev_action
        if prev_observation:
            self.prev_observation = prev_observation

        self.child_dict = {
            0 : {},
            1 : {}
        }

        self.bounds = None
        self.optimal_action = None

        self.is_risky = False

    def update_children(self, pruned_children):
        new_child_dict = {}
        for child in pruned_children:
            prev_action = child.prev_action
            prev_observation = child.prev_observation
            if prev_action in new_child_dict:
                new_child_dict[prev_action][prev_observation] = child
            else:
                new_child_dict[prev_action] = {}
        self.child_dict = new_child_dict
        self.children = pruned_children

    def set_bounds(self, lb, ub):
        self.bounds = {'lb': lb, 'ub': ub}

    def get_bounds_as_tuple(self):
        return (round(self.get_lower_bound(), 2), round(self.get_upper_bound(),2))
    
    def get_upper_bound(self):
        if self.bounds:
            if self.bounds['ub']:
                return self.bounds['ub']
            else:
                return False
        else:
            return False
    
    def get_lower_bound(self):
        if self.bounds:
            if self.bounds['lb']:
                return self.bounds['lb']
            else:
                return False
        else:
            return False

    def is_head_node(self):
        if self.head_node == True:
            return True
        else:
            return False
    
    def get_risky_state(self):
        if self.is_risky == True:
            return True
        else:
            return False

    def set_risky_state(self):
        self.is_risky = True

    def add_child(self, child, action, observation):
        child.prev_action = action
        child.prev_observation = observation
        self.child_dict[action][observation] = child
        self.children.append(child)

    def get_child(self, action, observation):
        return self.child_dict[action][observation]

    def get_children_by_action(self, action):
        children = []
        for observation in self.child_dict[action].keys():
            children.append(self.child_dict[action][observation])
        return children
    
    def get_all_children(self):
        children = self.children
        return children

    def get_all_children_p(self):
        child_particles = []
        for action in self.actions:
            for observation in self.observations:
                child_particles.append(self.child_dict[action][observation].particles)
        return child_particles

    def populate_children(self):
        risky_index = random.randint(0,3)
        iter = 0
        for action in self.actions:
            for observation in self.observations:
                pf = ParticleFilter(self.particles.copy(), self.weights.copy())
                pf.update(action, observation)
                child = Node(pf.particles.copy(), pf.weights.copy(), self)
                if iter == risky_index:
                    child.set_risky_state()
                self.add_child(child, action, observation)
                iter+=1
    
    def is_leaf(self):
        if len(self.children) == 0:
            return True
        else:
            return False
    
    def get_optimal_action(self):
        if self.optimal_action:
            return self.optimal_action
        else:
            if self.is_leaf() == True:
                return []
            else:
                print("No optimal action set")
                return False
        
    def set_optimal_action(self, action_branch):
        """
        action_branch must be an array.
        """
        self.optimal_action = action_branch

class BeliefTree:
    def __init__(self, horizon, particles, weights):
        """
        In our working implementation, we  consider 2 possible actions a1 and a2
        and 2 possible observations z1 and z2. 
        """
        self.horizon = horizon
        self.particles = np.array(particles)
        self.weights = np.array(weights)
        self.head_node = None

    def construct_belief_tree(self):
        """
        Constructs a belief tree with horizon = 2
        """
        self.head_node = Node(self.particles, self.weights, parent=None)
        self.head_node.populate_children()
        for child in self.head_node.get_all_children():
            child.populate_children()

class ParticleFilter:
    def __init__(self, particles, weights):
        self.particles = np.array(particles)
        self.weights = np.array(weights)
        self.transition_std = 0.1
        self.observation_std = 0.1

    def update(self, action, observation):
        self.particles = self.transition_positions(self.particles, action)
        likelihoods = self.observation_model(self.particles, observation)
        self.weights *= likelihoods
        if np.sum(self.weights) == 0:
            self.weights += 1e-10
        self.weights /= np.sum(self.weights)

    def simplify(self, num_simplified_particles):
        num_particles = len(self.particles)
        num_simplified_particles = min(num_simplified_particles, num_particles)  # Ensure we do not exceed the number of particles
        non_zero_weights = self.weights > 0
        num_non_zero_weights = np.sum(non_zero_weights)
        
        if num_non_zero_weights < num_simplified_particles:
            # Option 2: Assign a small non-zero probability to zero-weight particles
            adjusted_weights = np.where(self.weights > 0, self.weights, 1e-10)
            adjusted_weights /= adjusted_weights.sum()
        else:
            adjusted_weights = self.weights
        indices = np.random.choice(num_particles, size=num_simplified_particles, replace=False, p=adjusted_weights)
        simplified_particles = self.particles[indices]
        simplified_weights = self.weights[indices]
        simplified_weights /= np.sum(simplified_weights)
        return simplified_particles, simplified_weights, indices

    def transition_positions(self, x, a):
        noise = np.random.normal(0, self.transition_std, x.shape)
        return x + a + noise
    
    def transition_probability(self, x_next, x_prev, a):
        # Mean of the distribution
        mean = x_prev + a
        # Standard deviation of the distribution
        std = self.transition_std
        # Compute the probability density of x_next given the normal distribution
        probability = norm.pdf(x_next, loc=mean, scale=std)
        return probability[0]

    def observation_model(self, x, z):
        if x.ndim == 1:  # If x is 1D, use element-wise norm
            norm = np.linalg.norm(x - z)
        else:  # Otherwise, use axis 1 norm for multi-dimensional x
            norm = np.linalg.norm(x - z, axis=1)
        return np.exp(-0.5 * norm ** 2 / self.observation_std ** 2)

