def adaptive_majority(history, opponent_history):
    # After 5 rounds, cooperate if opponent cooperated in at least 3 of last 5, else defect
    if len(opponent_history) < 5:
        return 'C'
    recent = opponent_history[-5:]
    return 'C' if recent.count('C') >= 3 else 'D'

def trend_follower(history, opponent_history):
    # If last move we made was C and opponent responded with C, continue C, else defect
    if not history:
        return 'C'
    if history[-1] == 'C':
        return 'C' if opponent_history[-1] == 'C' else 'D'
    return 'D'

def evolver(history, opponent_history):
    # Simple evolutionary heuristic: mimic if opponent did same action twice in a row
    if len(opponent_history) < 3:
        return 'C'
    if opponent_history[-1] == opponent_history[-2]:
        return opponent_history[-1]
    return 'C'

def persistent_tft(history, opponent_history):
    # Cooperate unless opponent defected last turn
    if not opponent_history:
        return 'C'
    return 'C' if opponent_history[-1] == 'C' else 'D'

def revenge_seeker(history, opponent_history):
    # If opponent defected in the last 5 rounds, defect; otherwise cooperate
    if 'D' in opponent_history[-5:]:
        return 'D'
    return 'C'

def flexible_grudger(history, opponent_history):
    # Defect if opponent defected more than cooperated overall
    if opponent_history.count('D') > opponent_history.count('C'):
        return 'D'
    return 'C'

def cautious_trend(history, opponent_history):
    # If opponent's last two moves are same, assume trend will continue
    if len(history) < 2:
        return 'C'
    if opponent_history[-1] == opponent_history[-2]:
        return opponent_history[-1]
    return 'C'

all_strategies = [adaptive_majority, trend_follower, evolver, persistent_tft, revenge_seeker, flexible_grudger, cautious_trend]
