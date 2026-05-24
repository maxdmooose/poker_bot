# evolution/bots/lookup/lookup_template.py

import random


class Bot:
    def __init__(self, name: str, looseness: float, aggression: float):
        self.name = name
        self.looseness = looseness
        self.aggression = aggression

    def decide(self, state):
        legal = state["legal_actions"]
        rnd = random.random()

        # Very crude "lookup-like" behaviour: looseness controls folding,
        # aggression controls raising when allowed.
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