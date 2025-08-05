from phase_3_mcts_simulation import build_rl_agents, Phase3Simulator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from GameSetup import Agent
from strategies.deceptive_strategies import all_strategies as deceptive_strategies
from strategies.deterministic_strategies import all_strategies as deterministic_strategies
from strategies.evolutionary_strategies import all_strategies as evolutionary_strategies
from strategies.group_aware_strategies import all_strategies as group_aware_strategies
from strategies.probing_strategies import all_strategies as probing_strategies
from strategies.stochastic_strategies import all_strategies as stochastic_strategies

if __name__ == "__main__":
    trust_rl_strategies = [1, 2, 3, 4, 5]  # All trust models
    all_results = []

    for trust_model in trust_rl_strategies:
        a1, mcts1, a2, mcts2 = build_rl_agents(trust_model=trust_model)
        model_name = {1: "PersonalTrust", 2: "TRAVOSTrust", 3: "HearsayTrust",
                      4: "DefectiveAgent", 5: "AdversaryAgent"}[trust_model]
        rl_label = f"RL + {model_name}"

        # Gather ALL strategies from all categories
        all_strategy_sets = [
            deceptive_strategies, deterministic_strategies, evolutionary_strategies,
            group_aware_strategies, probing_strategies, stochastic_strategies
        ]
        opponents = []
        for strategy_set in all_strategy_sets:
            for strategy_fn in strategy_set:
                name = strategy_fn.__name__
                opponents.append((Agent(name, strategy_fn=strategy_fn), name))

        # Shuffle for randomness/variability
        random.shuffle(opponents)

        for opponent, opp_name in opponents:
            # Add variability by changing random seed each match
            random.seed(hash(f"{trust_model}-{opp_name}") % 1_000_000)

            sim = Phase3Simulator(a1, opponent, mcts1, mcts2, num_episodes=5, max_rounds=3)
            df = sim.run()
            df["opponent"] = opp_name
            df["rl_variant"] = rl_label
            all_results.append(df)

    if all_results:
        full_df = pd.concat(all_results, ignore_index=True)
        full_df.to_csv("phase3_vs_all_results.csv", index=False)
        print("Saved tournament results to phase3_vs_all_results.csv")

        avg_wealth = (full_df[full_df.agent == "RLAgent1"]
                      .groupby(["rl_variant", "opponent"])["total_wealth"].mean()
                      .unstack().fillna(0))
        plt.figure(figsize=(16, 7))
        avg_wealth.T.plot(kind="bar")
        plt.title("RL Trust Variants vs Opponent Strategies (Wealth)")
        plt.ylabel("Average Wealth")
        plt.xlabel("Opponent Strategy")
        plt.tight_layout()
        plt.legend(title="RL Trust Variant")
        plt.savefig("rl_all_trust_barplot.png")
        print("Saved bar plot to rl_all_trust_barplot.png")

        action_means = (full_df[full_df.agent == "RLAgent1"]
                        .groupby(["rl_variant", "opponent"])[["num_cooperate", "num_defect", "num_abstain"]]
                        .mean())
        for action in ["num_cooperate", "num_defect", "num_abstain"]:
            pivot = action_means[action].unstack().fillna(0)
            plt.figure(figsize=(14, 6))
            sns.heatmap(pivot, annot=True, cmap="YlGnBu")
            plt.title(f"{action.replace('num_', '').capitalize()} Heatmap: RL Variants vs Opponents")
            plt.tight_layout()
            plt.savefig(f"rl_all_trust_{action}_heatmap.png")
            print(f"Saved heatmap to rl_all_trust_{action}_heatmap.png")
    else:
        print("⚠No data generated from simulations.")
