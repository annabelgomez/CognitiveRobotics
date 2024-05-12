from graphviz import Digraph

class Node:
    def __init__(self, name, children=None):
        self.name = name
        self.children = children if children else []

    def is_leaf(self):
        return len(self.children) == 0

def add_nodes_edges(graph, node):
    """Recursively add nodes and edges to the Graphviz graph."""
    if node.is_leaf():
        # Leaf node, draw a doublecircle
        graph.node(str(node.get_bounds_as_tuple()), shape='circle')
    elif node.is_head_node():
        graph.node("Head Node", shape='doublecirlce')
    else:
        # Internal node, draw a circle
        graph.node(str(node.get_bounds_as_tuple()), shape='circle')

    for child in node.get_all_children():
        label = "action: "+str(child.prev_action)
        # For each child, draw the edge from the current node to the child
        if node.is_head_node():
            graph.edge("Head Node", str(child.get_bounds_as_tuple()), label)
        else:
            graph.edge(str(node.get_bounds_as_tuple()), str(child.get_bounds_as_tuple()), label)
        # Recursively add nodes and edges for the child
        add_nodes_edges(graph, child)

def visualize_belief_tree(root):
    """Creates and renders a belief tree using Graphviz."""
    graph = Digraph(comment='Belief Tree', format='png')
    add_nodes_edges(graph, root)
    # Render the graph to a file and display it
    graph.render('output/belief_tree', view=True)

# Example Usage
# root = Node('Root', children=[
#     Node('Child1', children=[
#         Node('Grandchild1'),
#         Node('Grandchild2')
#     ]),
#     Node('Child2')
# ])

# visualize_belief_tree(root)