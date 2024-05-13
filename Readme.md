# CognitiveRobotics
Group 5 Final Project


To test:
Run either test_basic.py or test_risky_state.py.
The output visualization of the pruned tree will be in output/belief_tree.png.


Overview of Python files:

belief tree.py - implements the initial belief tree. The
actions that an be taken, and observations are all de-
fined in this file. The risky states are also defined in
this file.

adapt simplification.py - implements the entire high
level adapt simplification algorithm and pruning.

adapt simplification risky states.py - implements the
simplification algorithm and pruning including our
risky state extension.

test basic.py - runs the code for the basic algorithm in
adapt simplification.py.

test risky state.py - runs the code for
the extension risky state algorithm in
adapt simplification risky states.py.

graph tree.py - is used to visualise the final pruned tree
output. Note that only actions and belief states are
shown here in order to make it easier to understand and
view the tree.