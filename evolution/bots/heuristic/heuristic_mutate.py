# evolution/bots/heuristic/heuristic_mutate.py

import os
import random
from typing import Dict

from config import BOTS_DIR
from file_ops import write_file, copy_file


HEURISTIC_TEMPLATE_CODE = """\
import random

class Bot:
    def __init__(self):
        self.name = "{name}"
        self.aggression = {aggression}
        self.bluff_freq = {bluff_freq}
        self.tightness = {tightness}

    def decide(self, state):
        legal = state["legal_actions"]
        rnd = random.random()
        if "raise" in legal and rnd < self.aggression:
            return {{"action": "raise", "amount": state.get("min_raise", state.get("min_bet", 0))}}
        if "call" in legal and rnd < (1.0 - self.tightness):
            return {{"action": "call"}}
        if "check" in legal:
            return {{"action": "check"}}
        return {{"action": "fold"}}

bot = Bot()

def decide(state):
    return bot.decide(state)
"""


def _new_id(state: dict, species: str) -> str:
    nid = state["next_id"][species]
    state["next_id"][species] += 1
    return f"h_{nid:03d}"


def _write_heuristic_bot(state: dict, name: str) -> Dict:
    params = {
        "name": name,
        "aggression": round(random.uniform(0.2, 0.8), 3),
        "bluff_freq": round(random.uniform(0.05, 0.3), 3),
        "tightness": round(random.uniform(0.2, 0.8), 3),
    }
    code = HEURISTIC_TEMPLATE_CODE.format(**params)
    path = os.path.join(BOTS_DIR, "heuristic", f"{name}.py")
    write_file(path, code)
    archive_path = f"evolution/archive/{name}.py"
    copy_file(path, archive_path)
    return {"id": name, "path": path, "species": "heuristic", "ev": 0.0}


def create_initial_heuristic_bot(state: dict) -> dict:
    name = _new_id(state, "heuristic")
    bot = _write_heuristic_bot(state, name)
    state["bots"]["heuristic"].append(bot)
    return state


def mutate_heuristic_bot(state: dict, parent_bot: Dict) -> Dict:
    name = _new_id(state, "heuristic")
    bot = _write_heuristic_bot(state, name)
    state["bots"]["heuristic"].append(bot)
    return bot
