import random
import numpy as np
from pmcts.core.node import Node
from pmcts.core.env import AllocationEnv
from pmcts.core.mdp import ACTION_SPACE


def MCTS(states: np.ndarray, iteration: int, max_steps: int):
    """
    Perform the Vanilla MCTS algorithm.

    Args:
        states (np.ndarray): The states for the environment.
        iteration (int): Number of iterations for the MCTS.
        max_steps (int): Maximum steps allowed in the environment.

    Returns:
        The action with the highest visitation count from the root node's children.
    """
    # Initialize environment
    env = AllocationEnv(states, max_steps)

    # Initialize root node
    root_node = Node(state=env.now_state, action=None)

    for _ in range(iteration):
        env.reset()
        node = root_node
        total_reward = 0

        # 1. UCT based Selection if all actions are expanded
        while node.untried_actions == [] and node.children and not env.is_terminal:
            node = node.select_child()
            total_reward += env.execute(node.action)

        # 2. Expansion based on untried actions
        if node.untried_actions:
            action = random.choice(node.untried_actions)
            total_reward += env.execute(action)
            node = node.add_child(state=env.now_state, action=action)

        # 3. Rollout
        while not env.is_terminal:
            action = random.choice(ACTION_SPACE)
            total_reward += env.execute(action)

        # 4. Backpropagation through visited nodes
        while node is not None:
            node.update(total_reward)
            node = node.parent

    # Optimal Action: Based on visitation count
    best_child = max(root_node.children, key=lambda c: c.visits)
    return best_child.action
