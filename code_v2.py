# This is what chat gbt gave annabel for the entire code implementation for the simplification method 

import numpy as np

class SimplificationFramework:
    def __init__(self, bounds_fn, levels):
        self.bounds_fn = bounds_fn  # Function to calculate bounds
        self.levels = levels  # Available levels of simplification

    def adapt_simplification(self, belief_tree, level):
        """
        Adaptively simplifies and prunes the belief tree.
        """
        if belief_tree.is_leaf():
            return self.bounds_fn(belief_tree, level)

        # Initialize the simplification level
        s = level
        bounds = []
        optimal_policy = None

        for subtree in belief_tree.subtrees:
            bounds.append(self.adapt_simplification(subtree, s))
            bounds[-1]["action"] = subtree.action

        bounds = sorted(bounds, key=lambda x: x["lower_bound"])

        # Pruning
        for i in range(len(bounds) - 1):
            if bounds[i]["upper_bound"] < bounds[i+1]["lower_bound"]:
                bounds = bounds[:i+1]
                break

        # If more than one branch remains, increase simplification level
        while len(bounds) > 1 and s < max(self.levels):
            s += 1
            for i in range(len(bounds)):
                bounds[i] = self.adapt_simplification(bounds[i]["tree"], s)
                bounds[i]["action"] = bounds[i]["tree"].action

            bounds = sorted(bounds, key=lambda x: x["lower_bound"])

            for i in range(len(bounds) - 1):
                if bounds[i]["upper_bound"] < bounds[i+1]["lower_bound"]:
                    bounds = bounds[:i+1]
                    break

        return {
            "lower_bound": bounds[0]["lower_bound"],
            "upper_bound": bounds[0]["upper_bound"],
            "action": bounds[0]["action"],
            "tree": bounds[0]["tree"]
        }

    def find_optimal_policy(self, belief_tree):
        """
        Returns the optimal policy using adaptive simplification.
        """
        return self.adapt_simplification(belief_tree, self.levels[0])


class BeliefTree:
    def __init__(self, action, is_leaf=False, subtrees=None):
        """
        Represents a belief tree node.
        """
        self.action = action
        self.is_leaf = is_leaf
        self.subtrees = subtrees or []


def differential_entropy_bounds(belief, level):
    """
    Example bounds function for differential entropy.
    """
    bounds = {
        "lower_bound": np.log(1 + level) / 2,
        "upper_bound": np.log(2 + level) / 2,
        "tree": belief
    }
    return bounds


# Example usage
levels = [0, 1, 2, 3, 4]
framework = SimplificationFramework(differential_entropy_bounds, levels)

# Example belief tree with multiple subtrees
tree = BeliefTree("root", subtrees=[
    BeliefTree("action1", subtrees=[
        BeliefTree("subaction1", is_leaf=True),
        BeliefTree("subaction2", is_leaf=True)
    ]),
    BeliefTree("action2", subtrees=[
        BeliefTree("subaction3", is_leaf=True),
        BeliefTree("subaction4", is_leaf=True)
    ])
])

optimal_policy = framework.find_optimal_policy(tree)
print(f"Optimal policy: {optimal_policy['action']}")
print(f"Lower bound: {optimal_policy['lower_bound']}, Upper bound: {optimal_policy['upper_bound']}")
