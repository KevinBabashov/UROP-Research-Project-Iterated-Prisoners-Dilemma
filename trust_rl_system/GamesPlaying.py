from GameSetup import Agent, Environment
from strategies import (
    deterministic_strategies,
    stochastic_strategies,
    deceptive_strategies,
    probing_strategies,
    evolutionary_strategies,
    group_aware_strategies,
)
from strategies.utility import plot_wealth_over_time

import matplotlib.pyplot as plt

if __name__ == "__main__":
    trust_agents = [
        Agent("PersonalTrust", trust_model=1),
        Agent("TRAVOSTrust", trust_model=2),
        Agent("HearsayTrust", trust_model=3),
        Agent("DefectiveAgent", trust_model=4),
        Agent("AdversaryAgent", trust_model=5),
    ]
    strategy_categories = {
        "Deterministic": deterministic_strategies.all_strategies,
        "Stochastic": stochastic_strategies.all_strategies,
        "Deceptive": deceptive_strategies.all_strategies,
        "Probing": probing_strategies.all_strategies,
        "Evolutionary": evolutionary_strategies.all_strategies,
        "GroupAware": group_aware_strategies.all_strategies,
    }
    strategy_all_agents = (
        deterministic_strategies.all_strategies + stochastic_strategies.all_strategies + deceptive_strategies.all_strategies + 
        probing_strategies.all_strategies + evolutionary_strategies.all_strategies + group_aware_strategies.all_strategies
    )
    print("Number of Strategies Implemented: " + str(len(strategy_all_agents)))

    for category_name, strategy_list in strategy_categories.items():
        print(f"\n--- Running {category_name} Strategies vs Trust Agents ---")

        strategy_agents = [Agent(name=strat.__name__, strategy_fn=strat) for strat in strategy_list]
        agents = trust_agents + strategy_agents

        env = Environment(agents, rounds=25)
        env.run()

        plot_wealth_over_time(env.wealth_history)

        print(env.results())
        
# Final run with all strategies vs trust agents
print("\n--- Running Final Test: All Strategies vs Trust Agents ---")
final_strategy_agents = [Agent(name=strat.__name__, strategy_fn=strat) for strat in strategy_all_agents]
final_agents = trust_agents + final_strategy_agents

env = Environment(final_agents, rounds=25)
env.run()
plot_wealth_over_time(env.wealth_history)
print(env.results())

