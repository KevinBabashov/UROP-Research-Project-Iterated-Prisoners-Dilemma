import random

def prober(history, opponent_history):
    # Probe with a fixed initial sequence: D, C, C
    probe_sequence = ['D', 'C', 'C']
    if len(history) < len(probe_sequence):
        return probe_sequence[len(history)]
    # After probing, if opponent retaliated during probe, switch to tit-for-tat; otherwise always defect
    if 'D' in opponent_history[:3]:
        return opponent_history[-1]  # tit-for-tat
    return 'D'

def deceitful_grim(history, opponent_history):
    # Cooperate until a certain move, defect occasionally thereafter
    if not history:
        return 'C'
    if 'D' in opponent_history:
        return 'D'
    if len(history) % 5 == 0:
        return 'D'  # defect on every 5th move as a deception
    return 'C'

def double_agent(history, opponent_history):
    # Cooperate for first 10 rounds, then defect always (lulls opponent into trust, then betrays)
    if len(history) < 10:
        return 'C'
    return 'D'

def fake_forgiver(history, opponent_history):
    # Occasionally "forgive" but mostly defect
    if len(history) % 7 == 0:
        return 'D'
    return 'C'

def manipulator(history, opponent_history):
    # If opponent defects, immediately cooperate (to entice), otherwise defect
    if opponent_history and opponent_history[-1] == 'D':
        return 'C'
    return 'D'

def sneak_attack(history, opponent_history):
    # Behave cooperatively initially, then randomly defect with low probability
    if len(history) < 5:
        return 'C'
    if random.random() < 0.2:
        return 'D'
    return 'C'

all_strategies = [prober, deceitful_grim, double_agent, fake_forgiver, manipulator, sneak_attack]
