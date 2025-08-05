import random

def random_agent(history, opponent_history):
    return 'C' if random.random() < 0.5 else 'D'

def generous_tit_for_tat(history, opponent_history):
    if not opponent_history:
        return 'C'
    if opponent_history[-1] == 'C':
        return 'C'
    # Occasionally forgive a defection ~30% of the time
    if random.random() < 0.3:
        return 'C'
    return 'D'

def noisy_tft(history, opponent_history):
    if not opponent_history:
        return 'C'
    # Occasionally do the opposite of tit-for-tat (10% noise)
    if random.random() < 0.1:
        return 'D' if opponent_history[-1] == 'C' else 'C'
    return opponent_history[-1]

def random_tit_for_tat(history, opponent_history):
    if not opponent_history:
        return 'C'
    if random.random() < 0.5:
        return opponent_history[-1]  # mimic half the time
    # Otherwise do opposite of opponent's last move
    return 'C' if opponent_history[-1] == 'D' else 'D'

def stochastic_grudger(history, opponent_history):
    if 'D' in opponent_history:
        # After a defection, defect with high probability, otherwise occasionally forgive
        return 'D' if random.random() > 0.3 else 'C'
    return 'C'

def sometimes_cooperate(history, opponent_history):
    return 'C' if random.random() < 0.7 else 'D'

def sometimes_defect(history, opponent_history):
    return 'D' if random.random() < 0.7 else 'C'

all_strategies = [
    random_agent, generous_tit_for_tat, noisy_tft, random_tit_for_tat,
    stochastic_grudger, sometimes_cooperate, sometimes_defect
]
