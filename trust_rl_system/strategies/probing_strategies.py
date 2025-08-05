import random

def strategic_probe(history, opponent_history):
    # Fixed sequence of moves to test opponent: C, D, C, D
    sequence = ['C', 'D', 'C', 'D']
    if len(history) < len(sequence):
        return sequence[len(history)]
    # After probing, switch to tit-for-tat
    return opponent_history[-1]

def detective(history, opponent_history):
    # "Detective" strategy: start with a sequence to learn opponent's reaction
    probe_sequence = ['C', 'D', 'C', 'C']
    if len(history) < len(probe_sequence):
        return probe_sequence[len(history)]
    # If opponent defected during probes, defect forever; else mirror
    if 'D' in opponent_history[:4]:
        return 'D'
    return opponent_history[-1]

def invasive_probe(history, opponent_history):
    # Defect first few rounds to test opponent's tolerance, then adapt
    if len(history) < 3:
        return 'D'
    # If opponent has more defects than cooperations, defect; otherwise cooperate
    if opponent_history.count('D') > opponent_history.count('C'):
        return 'D'
    return 'C'

def annoyed_probe(history, opponent_history):
    # Probe by defecting two in a row and see response
    if len(opponent_history) < 2:
        return 'C'
    if opponent_history[-1] == 'D' and opponent_history[-2] == 'D':
        return 'D'
    return 'C'

def suspicious_probe(history, opponent_history):
    # Start suspicious (defect first), then mirror
    if not history:
        return 'D'
    return opponent_history[-1]

all_strategies = [strategic_probe, detective, invasive_probe, annoyed_probe, suspicious_probe]
