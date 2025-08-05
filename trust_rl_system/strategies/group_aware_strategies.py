def forgiving_majority(history, opponent_history):
    # Cooperate if opponent's cooperations >= opponent's defections - 1 (slightly forgiving)
    return 'C' if opponent_history.count('C') >= opponent_history.count('D') - 1 else 'D'

def harsh_majority(history, opponent_history):
    # Defect if opponent's defections >= opponent's cooperations
    return 'D' if opponent_history.count('D') >= opponent_history.count('C') else 'C'

def tft_in_small_groups(history, opponent_history):
    # Identical to majority logic here (assuming group context not explicitly tracked)
    return 'C' if opponent_history.count('C') > opponent_history.count('D') else 'D'

def nice_tft(history, opponent_history):
    # Cooperate if opponent cooperated at least as much as defected
    return 'C' if opponent_history.count('C') >= opponent_history.count('D') else 'D'

def skeptical_majority(history, opponent_history):
    # Require opponent to have significantly more cooperation than defection to cooperate
    if opponent_history.count('D') > opponent_history.count('C') + 1:
        return 'D'
    return 'C'

def cooperative_majority(history, opponent_history):
    # Cooperate if opponent cooperated at least as often as defected
    return 'C' if opponent_history.count('C') >= opponent_history.count('D') else 'D'

all_strategies = [forgiving_majority, harsh_majority, tft_in_small_groups, nice_tft, skeptical_majority, cooperative_majority]
