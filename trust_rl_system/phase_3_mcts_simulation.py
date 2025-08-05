import random
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from Graph_Neural_Network import TrustGNN
from Monte_Carlo import MCTS
from GameSetup import Agent, COOPERATE, DEFECT, ABSTAIN

class Phase3Simulator:
    def __init__(self, agent1, agent2, mcts1, mcts2, num_episodes=5, max_rounds=5):
        self.agent1 = agent1
        self.agent2 = agent2
        self.mcts1 = mcts1
        self.mcts2 = mcts2
        self.num_episodes = num_episodes
        self.max_rounds = max_rounds
    def _decide_with_possible_mcts(self, player, opponent, mcts):
        # Inject some randomness for trust-based agents
        exploration_rate = 0.2
        rl_agent_names = ["HearsayTrust", "TRAVOSTrust", "PersonalTrust", "DefectiveAgent"]

        if any(name in player.name for name in rl_agent_names) and random.random() < exploration_rate:
            return random.choice(["cooperate", "defect", "abstain"])

        if mcts is None:
            return player.decide_action(opponent, shared_trust={})
        
        return mcts.run_simulation(player, opponent)


    def run(self):
        data = []
        for episode in range(self.num_episodes):
            self.agent1.reset()
            self.agent2.reset()
            for round in range(self.max_rounds):
                action1 = self._decide_with_possible_mcts(self.agent1, self.agent2, self.mcts1)
                action2 = self._decide_with_possible_mcts(self.agent2, self.agent1, self.mcts2)

                payoff1, payoff2 = self.get_payoff(action1, action2)
                self.agent1.wealth += payoff1
                self.agent2.wealth += payoff2

                self.agent1.history.append(action1)
                self.agent2.history.append(action2)
                self.agent1.opponent_history.append(action2)
                self.agent2.opponent_history.append(action1)

            result = {
                "agent": self.agent1.name,
                "total_wealth": self.agent1.wealth,
                "num_cooperate": self.agent1.history.count(COOPERATE),
                "num_defect": self.agent1.history.count(DEFECT),
                "num_abstain": self.agent1.history.count(ABSTAIN),
            }
            data.append(result)
        return pd.DataFrame(data)

    def get_payoff(self, action1, action2):
        return {
            (COOPERATE, COOPERATE): (3, 3),
            (DEFECT, DEFECT): (-1, -1),
            (COOPERATE, DEFECT): (-5, 5),
            (DEFECT, COOPERATE): (5, -5),
            (ABSTAIN, ABSTAIN): (0, 0),
            (COOPERATE, ABSTAIN): (0, 0),
            (ABSTAIN, COOPERATE): (0, 0),
            (DEFECT, ABSTAIN): (0, 0),
            (ABSTAIN, DEFECT): (0, 0),
        }[(action1, action2)]

def build_rl_agents(trust_model=1):
    gnn = TrustGNN(input_dim=5, hidden_dim=16, output_dim=2)

    try:
        gnn.load_state_dict(torch.load("saved_models/trust_gnn.pth", map_location="cpu"))
    except Exception as e:
        print(f"❌ Error loading model weights: {e}")

    agent1 = Agent("RLAgent1", trust_model=trust_model)
    agent2 = Agent("RLAgent2", trust_model=trust_model)
    mcts1 = MCTS(agent1, gnn)
    mcts2 = MCTS(agent2, gnn)
    return agent1, mcts1, agent2, mcts2
