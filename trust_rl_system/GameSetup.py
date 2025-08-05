from typing import List, Dict
import random

COOPERATE = "C"
DEFECT = "D"
ABSTAIN = "A"

# Payoff matrix for actions: (agent1_action, agent2_action) -> (agent1_payoff, agent2_payoff)
PAYOFFS = {
    (COOPERATE, COOPERATE): (3, 3),
    (DEFECT, DEFECT): (-1, -1),
    (COOPERATE, DEFECT): (-5, 5),
    (DEFECT, COOPERATE): (5, -5),
    (ABSTAIN, ABSTAIN): (0, 0),
    (COOPERATE, ABSTAIN): (0, 0),
    (ABSTAIN, COOPERATE): (0, 0),
    (DEFECT, ABSTAIN): (0, 0),
    (ABSTAIN, DEFECT): (0, 0)
}

class Agent:
    def __init__(self, name: str, strategy_fn=None, trust_model=None, 
                 trust=None, wealth=None, evidence=None, beliefs=None):
        self.name = name
        self.strategy = strategy_fn  # If provided, this agent uses a fixed strategy function
        self.trust_model = trust_model  # Trust model ID (1-5) if this is a trust-based agent
        self.history = []
        self.opponent_history = []
        self.trust = trust if trust is not None else {}
        self.wealth = wealth if wealth is not None else 10  # initial wealth
        self.evidence = evidence if evidence is not None else {}  # for trust models using evidence (e.g., TRAVOS)
        self.beliefs = beliefs if beliefs is not None else {}    # for trust models using belief distributions
        self.last_action = None

    def play(self):
        """For non-RL agents with a strategy function, decide an action given histories."""
        move = self.strategy(self.history, self.opponent_history)
        self.history.append(move)
        return move

    def reset(self):
        """Reset agent state for a new tournament or simulation."""
        self.history = []
        self.opponent_history = []
        self.last_action = None

    def beta_expected_value(self, success: int, fail: int) -> float:
        """Compute Beta distribution expected value given successes/failures (for trust calculation)."""
        return (success + 1) / (success + fail + 2)

    def update_beliefs(self, opponent_name: str, action: str):
        """Bayesian update of opponent type beliefs (used in trust_model 4)."""
        if opponent_name not in self.beliefs:
            # Belief distribution over opponent being Cooperative (C), Liar (L), or Adversary (A)
            self.beliefs[opponent_name] = {"C": 1/3, "L": 1/3, "A": 1/3}
        prior = self.beliefs[opponent_name]
        # Likelihoods of observing an action given opponent type
        likelihoods = {
            "C": {COOPERATE: 0.8, DEFECT: 0.1, ABSTAIN: 0.1},
            "L": {COOPERATE: 0.2, DEFECT: 0.4, ABSTAIN: 0.4},
            "A": {COOPERATE: 0.1, DEFECT: 0.7, ABSTAIN: 0.2},
        }
        marginal = sum(likelihoods[t][action] * prior[t] for t in prior)
        new_belief = {
            t: (likelihoods[t][action] * prior[t]) / marginal for t in prior
        }
        self.beliefs[opponent_name] = new_belief

    def update_trust(self, opponent_name: str, action: str):
        """Update trust and evidence based on opponent's observed action."""
        if opponent_name not in self.trust:
            self.trust[opponent_name] = 0.5  # start neutral trust
        if opponent_name not in self.evidence:
            self.evidence[opponent_name] = {"success": 0, "fail": 0}
        if action == COOPERATE:
            self.evidence[opponent_name]["success"] += 1
        elif action == DEFECT:
            self.evidence[opponent_name]["fail"] += 1
        # Simple trust adjustment: increase trust on opponent's cooperation, decrease on defection
        if action == DEFECT:
            self.trust[opponent_name] -= 0.1
        elif action == COOPERATE:
            self.trust[opponent_name] += 0.1
        # Keep trust in [0,1]
        self.trust[opponent_name] = max(0, min(1, self.trust[opponent_name]))

    def decide_action(self, opponent: 'Agent', shared_trust: dict) -> str:
        """Decide an action based on trust model or fixed strategy. Returns 'C', 'D', or 'A'."""
        if self.strategy is not None:
            # Use predefined strategy function
            return self.strategy(self.history, opponent.history)

        # Default trust level (if no info, neutral 0.5)
        trust_level = self.trust.get(opponent.name, 0.5)
        # Different trust model behaviors:
        if self.trust_model == 1:
            # PersonalTrust: use personal experience (Beta distribution expected value of cooperative rate)
            ev = self.evidence.get(opponent.name, {"success": 0, "fail": 0})
            trust_level = self.beta_expected_value(ev["success"], ev["fail"])
        elif self.trust_model == 2:
            # TRAVOS: combine personal trust and shared (reputation) trust
            ev = self.evidence.get(opponent.name, {"success": 0, "fail": 0})
            own_trust = self.beta_expected_value(ev["success"], ev["fail"])
            shared_value = shared_trust.get(opponent.name, 0.5)
            recommender_trust = shared_trust.get("RecommenderAgent", 0.5)  # trust in the recommender if applicable
            # Weighted combination of own trust and shared trust
            trust_level = (own_trust + shared_value * recommender_trust) / (1 + recommender_trust)
        elif self.trust_model == 3:
            # Hearsay/Reputation: rely only on shared trust information
            trust_level = shared_trust.get(opponent.name, 0.5)
        elif self.trust_model == 4:
            # DefectiveAgent model: use beliefs about opponent type
            beliefs = self.beliefs.get(opponent.name, {"C": 1/3, "L": 1/3, "A": 1/3})
            expected_coop = 0.8 * beliefs["C"] + 0.2 * beliefs["L"] + 0.1 * beliefs["A"]
            if beliefs["A"] > 0.6:
                return ABSTAIN  # likely adversary, refuse to play
            elif expected_coop < 0.3:
                return DEFECT  # if opponent likely to defect, defect preemptively
            else:
                return COOPERATE
        elif self.trust_model == 5:
            # AdversaryAgent model: exploitative and retaliatory behavior
            if len(opponent.history) == 0:
                return DEFECT  # start by defecting to test opponent
            elif opponent.history[-1] == COOPERATE:
                return DEFECT  # exploit opponent's cooperation
            elif opponent.history[-1] == DEFECT:
                # If opponent defected last time, continue defecting if we ever cooperated (escalate),
                # otherwise cooperate once to confuse
                return DEFECT if self.history.count(COOPERATE) > 0 else COOPERATE

        # If none of the above returned, decide based on trust threshold
        if trust_level < 0.3:
            return ABSTAIN
        elif trust_level < 0.5:
            return DEFECT
        else:
            return COOPERATE

class Environment:
    def __init__(self, agents: List[Agent], rounds=100):
        self.agents = agents
        self.rounds = rounds
        self.wealth_history = {agent.name: [] for agent in agents}
        self.match_scores: Dict[(str, str), float] = {}

    def reset(self):
        """Reset environment before a tournament."""
        for agent in self.agents:
            agent.trust = {}
            agent.wealth = 10
            agent.history = []
            agent.opponent_history = []
            agent.last_action = None
        self.history = []
        self.current_round = 0
        self.match_scores = {}

    def perform_action(self, agent: Agent, action: str):
        """Apply an agent's action in a sequential interaction (not used in simultaneous play)."""
        opponent = next(a for a in self.agents if a != agent)
        # Simple sequential payoff logic: (for demonstration; simultaneous uses play_round)
        if not hasattr(agent, 'last_action'):
            agent.last_action = None
        if not hasattr(opponent, 'last_action'):
            opponent.last_action = None
        if action == COOPERATE and opponent.last_action == COOPERATE:
            agent.wealth += 3
        elif action == COOPERATE and opponent.last_action == DEFECT:
            agent.wealth += 0
        elif action == DEFECT and opponent.last_action == COOPERATE:
            agent.wealth += 5
        elif action == DEFECT and opponent.last_action == DEFECT:
            agent.wealth += 1
        elif action == ABSTAIN:
            agent.wealth += 1  # reward (or cost) for abstaining
        agent.last_action = action

    def get_rewards(self) -> Dict[str, float]:
        """Get current wealth of all agents."""
        return {agent.name: agent.wealth for agent in self.agents}

    def run(self):
        """Run a round-robin tournament for the specified number of rounds."""
        for agent in self.agents:
            agent.wealth = 0
            agent.history = []
        self.wealth_history = {agent.name: [] for agent in self.agents}
        for _ in range(self.rounds):
            for i, agent1 in enumerate(self.agents):
                for j, agent2 in enumerate(self.agents):
                    if i >= j:
                        continue
                    self.play_round(agent1, agent2)
                    # Record wealth after each interaction
                    for agent in self.agents:
                        self.wealth_history[agent.name].append(agent.wealth)

    def play_round(self, agent1: Agent, agent2: Agent):
        """Play a single simultaneous round between two agents."""
        shared_trust = self.calculate_shared_trust()
        action1 = agent1.decide_action(agent2, shared_trust)
        action2 = agent2.decide_action(agent1, shared_trust)
        payoff1, payoff2 = PAYOFFS[(action1, action2)]
        agent1.wealth += payoff1
        agent2.wealth += payoff2
        # Track cumulative match scores
        key1 = (agent1.name, agent2.name)
        key2 = (agent2.name, agent1.name)
        if key1 not in self.match_scores:
            self.match_scores[key1] = 0
        if key2 not in self.match_scores:
            self.match_scores[key2] = 0
        self.match_scores[key1] += payoff1
        self.match_scores[key2] += payoff2
        # Update trust and beliefs based on actions
        if action1 != ABSTAIN:
            agent1.update_trust(agent2.name, action2)
            agent1.update_beliefs(agent2.name, action2)
        if action2 != ABSTAIN:
            agent2.update_trust(agent1.name, action1)
            agent2.update_beliefs(agent1.name, action1)

    def calculate_shared_trust(self) -> Dict[str, float]:
        """Compute shared trust (reputation) values for each known agent from all agents' perspectives."""
        trust_scores: Dict[str, List[float]] = {}
        for agent in self.agents:
            for other_name, ev in agent.evidence.items():
                # Use Beta expected value as reputation score
                val = (ev["success"] + 1) / (ev["success"] + ev["fail"] + 2)
                trust_scores.setdefault(other_name, []).append(val)
        # Shared trust: average of all agents' trust scores for each target
        return {name: sum(vals)/len(vals) for name, vals in trust_scores.items()}

    def results(self) -> str:
        """Return final ranking of agents by wealth."""
        sorted_agents = sorted([(agent.name, agent.wealth) for agent in self.agents], key=lambda x: -x[1])
        return "\n".join([f"{name}: {wealth}" for name, wealth in sorted_agents])
