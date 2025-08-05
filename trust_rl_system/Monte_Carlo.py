import math, random
from typing import List, Tuple
# Reuse action constants and payoff matrix
from GameSetup import COOPERATE, DEFECT, ABSTAIN, PAYOFFS

class UCTNode:
    """Node in the MCTS tree."""
    def __init__(self, state, parent=None, action=None):
        self.state = state                    # (agent1, agent2, history)
        self.parent = parent
        self.children: List['UCTNode'] = []
        self.visits = 0
        self.total_reward = 0.0
        self.action = action                  # action taken from parent->this

    def uct_score(self, exploration_constant=1.41):
        if self.visits == 0:
            return float('inf')  # prioritize unexplored nodes
        avg_reward = self.total_reward / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return avg_reward + exploration

    def best_child(self, exploration_constant=1.41):
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

    def backpropagate(self, reward):
        self.visits += 1
        self.total_reward += reward
        if self.parent:
            self.parent.backpropagate(reward)


class MCTS:
    """Base Monte Carlo Tree Search."""
    def __init__(self, action_space, simulations=100, max_depth=5, exploration_constant=1.41):
        self.action_space = action_space
        self.simulations = simulations
        self.max_depth = max_depth
        self.c = exploration_constant

    def run(self, root_state):
        root = UCTNode(state=root_state)

        for _ in range(self.simulations):
            node = self.tree_policy(root)
            reward = self.rollout(node.state)
            node.backpropagate(reward)

        # Choose the child with the most visits (robust child)
        if not root.children:
            # Fallback: no children (shouldn't happen, but be safe)
            return random.choice(self.action_space)
        best = max(root.children, key=lambda c: c.visits)
        return best  # return node, not just action

    def tree_policy(self, node: UCTNode) -> UCTNode:
        while not self._is_terminal(node.state):
            if not node.is_fully_expanded(self.action_space):
                return self.expand_node(node)
            else:
                node = node.best_child(self.c)
        return node

    def expand_node(self, node: UCTNode) -> UCTNode:
        untried = node.get_untried_actions(self.action_space)
        action = random.choice(untried)  # <- randomness to avoid "always C"
        next_state = self.simulate_transition(node.state, action)
        return node.expand(action, next_state)

    def rollout(self, state) -> float:
        # Default rollout: random playout with payoff heuristic
        agent1, agent2, history = state
        # do a very short random rollout to max_depth
        sim_hist = list(history)
        for _ in range(self.max_depth - len(sim_hist)):
            a1 = random.choice(self.action_space)
            a2 = random.choice(self.action_space)
            sim_hist.append((a1, a2))
        # simple heuristic: sum immediate payoffs for agent1
        total = 0.0
        for (a1, a2) in sim_hist[len(history):]:
            p1, _ = PAYOFFS[(a1, a2)]
            total += p1
        return total

    def simulate_transition(self, state, action) -> Tuple:
        """Simulate taking 'action' by agent1 and respond with agent2's action. Returns new state (agent1, agent2, history)."""
        agent1, agent2, history = state
        new_history = list(history)

        # Agent1 takes the action we decided
        agent1_move = action

        # Agent2 responds (if fixed strategy -> use it, else simple trust-based heuristic)
        if hasattr(agent2, 'strategy') and agent2.strategy is not None:
            self_hist = [b for (_, b) in new_history]
            opp_hist = [a for (a, _) in new_history]
            agent2_move = agent2.strategy(self_hist, opp_hist)
        elif hasattr(agent2, 'trust_model') and agent2.trust_model is not None:
            trust_level = agent2.trust.get(agent1.name, 0.5)
            if trust_level < 0.3:
                agent2_move = ABSTAIN
            elif trust_level < 0.5:
                agent2_move = DEFECT
            else:
                agent2_move = COOPERATE
        else:
            agent2_move = random.choice(self.action_space)

        new_history.append((agent1_move, agent2_move))
        return (agent1, agent2, new_history)

    def _is_terminal(self, state) -> bool:
        # Accept both (agent1, agent2) or (agent1, agent2, history)
        if len(state) == 2:
            # No history -> not terminal but we can't go depth-wise: treat as start
            return False
        _, _, history = state
        return len(history) >= self.max_depth
    
    def run_simulation(self, player, opponent):
        """
        Placeholder simulation logic.
        In a real implementation, this would involve tree search.
        """
        return player.decide_action(opponent, shared_trust={})



class MCTSWithLearningModel(MCTS):
    """MCTS that uses a learned model (Trust GNN) to evaluate rollouts."""
    def __init__(self, action_space, simulations=50, max_depth=5,
                 env_model=None, gnn_model=None, build_graph_fn=None, trust_model=None,
                 exploration_constant=1.41):
        super().__init__(action_space, simulations, max_depth, exploration_constant)
        self.gnn_model = gnn_model      # Pretrained GNN to estimate trust/value
        self.build_graph_fn = build_graph_fn

    def rollout(self, state):
        # Use the learned model to score the leaf state
        agent1, agent2, history = state if len(state) == 3 else (*state, [])
        try:
            features = [
                [agent1.trust.get(agent2.name, 0.5), agent1.wealth, 0.0, 0.0, 0.0],
                [agent2.trust.get(agent1.name, 0.5), agent2.wealth, 0.0, 0.0, 0.0]
            ]
            edges = [(0, 1), (1, 0)]
            graph = self.build_graph_fn(features, edges)
            pred = self.gnn_model(graph).squeeze()
            return float(pred.mean().item())
        except Exception as e:
            # Fallback if the GNN errors out
            return super().rollout(state)

    def select_action(self, agent, opponent):
        # Proper initial state: include empty history
        state = (agent, opponent, [])
        selected_node = self.run(state)
        return selected_node.action
