import random

class Bot:
    def __init__(self):
        self.name = "l_000"
        self.looseness = 0.243
        self.aggression = 0.601

    def decide(self, state):
        legal = state["legal_actions"]
        rnd = random.random()
        if "raise" in legal and rnd < self.aggression:
            return {"action": "raise", "amount": state.get("min_raise", state.get("min_bet", 0))}
        if "call" in legal and rnd < self.looseness:
            return {"action": "call"}
        if "check" in legal:
            return {"action": "check"}
        return {"action": "fold"}

bot = Bot()

def decide(state):
    return bot.decide(state)
