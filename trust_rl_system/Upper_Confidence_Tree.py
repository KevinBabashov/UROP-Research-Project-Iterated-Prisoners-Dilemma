import math
import random

class UCTNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.action = action
        self.rollout_value = None  # Optional: can cache neural rollout value

    def uct_score(self, exploration_constant=1.41):
        if self.visits == 0:
            return float('inf')  # Ensure unexplored nodes get picked
        avg_reward = self.total_reward / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return avg_reward + exploration

    def best_child(self, exploration_constant=1.41):
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.uct_score(exploration_constant))

    def expand(self, action, next_state):
        new_node = UCTNode(state=next_state, parent=self, action=action)
        self.children.append(new_node)
        return new_node

    def is_fully_expanded(self, legal_actions):
        tried_actions = {child.action for child in self.children}
        return all(action in tried_actions for action in legal_actions)

    def get_untried_actions(self, legal_actions):
        tried = {child.action for child in self.children}
        return [a for a in legal_actions if a not in tried]

    def select_or_expand(self, legal_actions, rollout_fn):
        """
        Chooses a new unexplored action to expand or selects best child.
        rollout_fn(state, action) -> (next_state, reward)
        """
        untried = self.get_untried_actions(legal_actions)
        if untried:
            action = random.choice(untried)
            next_state, reward = rollout_fn(self.state, action)
            return self.expand(action, next_state)
        return self.best_child()

    def backpropagate(self, reward):
        self.visits += 1
        self.total_reward += reward
        if self.parent:
            self.parent.backpropagate(reward)


