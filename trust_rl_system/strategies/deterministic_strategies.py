def always_cooperate(history, opponent_history):
    return 'C'

def always_defect(history, opponent_history):
    return 'D'

def tit_for_tat(history, opponent_history):
    if not opponent_history:
        return 'C'
    return opponent_history[-1]  # mirror opponent's last move

def grudger(history, opponent_history):
    if 'D' in opponent_history:
        return 'D'
    return 'C'

def forgiver(history, opponent_history):
    if not opponent_history or opponent_history[-1] == 'C':
        return 'C'
    # Defect once if opponent just defected, then forgive
    return 'D'

def snob(history, opponent_history):
    # Cooperate first 3 rounds, then defect if opponent defected
    if len(opponent_history) < 3:
        return 'C'
    return 'D' if opponent_history[-1] == 'D' else 'C'

def slow_tit_for_tat(history, opponent_history):
    if len(opponent_history) < 3:
        return 'C'
    return opponent_history[-1]

def win_stay_lose_shift(history, opponent_history):
    if not history:
        return 'C'
    # If last round was mutual cooperation, continue cooperating; otherwise defect
    if history[-1] == 'C' and opponent_history[-1] == 'C':
        return 'C'
    return 'D'

def soft_majority(history, opponent_history):
    # Cooperate if opponent cooperated more than defected so far
    return 'C' if opponent_history.count('C') > opponent_history.count('D') else 'D'

def hard_majority(history, opponent_history):
    # Cooperate if opponent cooperations >= defections; else defect
    return 'C' if opponent_history.count('C') >= opponent_history.count('D') else 'D'

def cooperative_tft(history, opponent_history):
    # Tit-for-tat that starts cooperatively
    if not history:
        return 'C'
    return opponent_history[-1]

def anti_tft(history, opponent_history):
    # Opposite of tit-for-tat (start defecting)
    if not history:
        return 'D'
    return opponent_history[-1]

def grim_trigger(history, opponent_history):
    # Cooperate until opponent defects once, then defect forever
    if 'D' in opponent_history:
        return 'D'
    return 'C'

def retaliator(history, opponent_history):
    # Simple retaliation: mirror last opponent move (like TFT)
    if not opponent_history:
        return 'C'
    return 'D' if opponent_history[-1] == 'D' else 'C'

def suspicious_tft(history, opponent_history):
    # Tit-for-tat starting with suspicion (first move defect)
    if not opponent_history:
        return 'D'
    return opponent_history[-1]

def grudging_tft(history, opponent_history):
    # Like tit-for-tat but holds a grudge: once opponent defects, always defect thereafter
    if 'D' in opponent_history:
        return 'D'
    if not opponent_history:
        return 'C'
    return opponent_history[-1]

all_strategies = [
    always_cooperate, always_defect, tit_for_tat, grudger, forgiver, snob,
    slow_tit_for_tat, win_stay_lose_shift, soft_majority, hard_majority,
    cooperative_tft, anti_tft, grim_trigger, retaliator, suspicious_tft, grudging_tft
]
