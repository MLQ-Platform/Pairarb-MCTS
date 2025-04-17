import math
import numpy as np
from pmcts.core.mdp import ACTION_SPACE
from pmcts.core.node import Node


class Node:
    """
    Node of the MCTS tree.
    """

    def __init__(self, parent=None, state=None, action=None):
        """
        Initialize a node.

        Args:
            parent (Node, optional): Parent node.
            state: State associated with the node.
            action: Action leading to this node.
        """
        self.state = state
        self.action = action
        self.parent = parent

        self.children = []
        self.untried_actions = list(ACTION_SPACE)
        self.visits = 0
        self.total_reward = 0

    def select_child(self) -> Node:
        """
        Select a child node using UCT.

        Returns:
            Node: The child node with the highest UCT value.
        """
        C = math.sqrt(2)
        best_child = None
        best_uct = -float("inf")

        for child in self.children:
            avg_reward = child.total_reward / child.visits
            uct_value = avg_reward + C * math.sqrt(math.log(self.visits) / child.visits)

            if uct_value > best_uct:
                best_uct = uct_value
                best_child = child

        return best_child

    def add_child(self, state: np.ndarray, action: float) -> Node:
        """
        Add a child node for the given state and action.

        Args:
            state: State for the new child node.
            action: Action leading to the new child node.

        Returns:
            Node: The newly created child node.
        """
        new_node = Node(parent=self, state=state, action=action)

        self.untried_actions.remove(action)
        self.children.append(new_node)
        return new_node

    def update(self, reward: float):
        """
        Update the node with the simulation reward.

        Args:
            reward (float): Reward from the simulation.
        """
        self.visits += 1
        self.total_reward += reward
