# evolution/bots/heuristic/heuristic_template.py

import random


class Bot:
    def __init__(self, name: str, aggression: float, bluff_freq: float, tightness: float):
        self.name = name
        self.aggression = aggression
        self.bluff_freq = bluff_freq
        self.tightness = tightness

    def decide(self, state):
        # state: Fullhouse state dict
        legal = state["legal_actions"]  # e.g. ["fold","call","raise","check"]
        rnd = random.random()

        # Very crude, safe heuristic
        if "raise" in legal and rnd < self.aggression:
            return {"action": "raise", "amount": state.get("min_raise", state.get("min_bet", 0))}
        if "call" in legal and rnd < (1.0 - self.tightness):
            return {"action": "call"}
        if "check" in legal:
            return {"action": "check"}
        return {"action": "fold"}
    
bot = Bot()

def decide(state):
    return bot.decide(state)