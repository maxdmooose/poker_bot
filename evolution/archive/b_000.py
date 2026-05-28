import random

class Bot:
    def __init__(self):
        self.name = "b_000"
        self.value_weight = 0.861
        self.bluff_weight = 0.427

    def decide(self, state):
        legal = state["legal_actions"]
        rnd = random.random()
        threshold = self.value_weight + self.bluff_weight * 0.5
        if "raise" in legal and rnd < threshold:
            return {"action": "raise", "amount": state.get("min_raise", state.get("min_bet", 0))}
        if "call" in legal and rnd < 0.7:
            return {"action": "call"}
        if "check" in legal:
            return {"action": "check"}
        return {"action": "fold"}

bot = Bot()

def decide(state):
    return bot.decide(state)
