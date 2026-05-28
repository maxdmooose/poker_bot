import random

class Bot:
    def __init__(self):
        self.name = "h_000"
        self.aggression = 0.62
        self.bluff_freq = 0.275
        self.tightness = 0.318

    def decide(self, state):
        legal = state["legal_actions"]
        rnd = random.random()
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
