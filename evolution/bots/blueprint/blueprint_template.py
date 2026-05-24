# evolution/bots/blueprint/blueprint_template.py

import random


class Bot:
    def __init__(self, name: str, value_weight: float, bluff_weight: float):
        self.name = name
        self.value_weight = value_weight
        self.bluff_weight = bluff_weight

    def decide(self, state):
        legal = state["legal_actions"]
        rnd = random.random()

        # Very crude "blueprint-like" behaviour:
        # value_weight biases toward aggression, bluff_weight adds some bluffs.
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