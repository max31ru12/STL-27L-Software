from matplotlib import pyplot as plt

from common.models import MoveNode

x, y = [1, 2, 3], [1, 2, 3]

x_2, y_2 = [5, 5, 5], [1, 2, 3]

plt.scatter(x, y)

node_1 = MoveNode(straight_move=4, theta=0)

node_2 = MoveNode(straight_move=4, theta=0, prev=node_1)

node_3 = MoveNode(straight_move=40, theta=0, prev=node_2)

plt.scatter(*node_3.get_current_coordinates(x_2, y_2))

plt.show()
