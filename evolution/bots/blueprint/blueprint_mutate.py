# evolution/bots/blueprint/blueprint_mutate.py

import os
import random
from typing import Dict

from config import BOTS_DIR
from file_ops import write_file, copy_file


BLUEPRINT_TEMPLATE_CODE = """\
import random

class Bot:
    def __init__(self):
        self.name = "{name}"
        self.value_weight = {value_weight}
        self.bluff_weight = {bluff_weight}

    def decide(self, state):
        legal = state["legal_actions"]
        rnd = random.random()
        threshold = self.value_weight + self.bluff_weight * 0.5
        if "raise" in legal and rnd < threshold:
            return {{"action": "raise", "amount": state.get("min_raise", state.get("min_bet", 0))}}
        if "call" in legal and rnd < 0.7:
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
    return f"b_{nid:03d}"


def _write_blueprint_bot(state: dict, name: str) -> Dict:
    params = {
        "name": name,
        "value_weight": round(random.uniform(0.3, 0.9), 3),
        "bluff_weight": round(random.uniform(0.0, 0.5), 3),
    }
    code = BLUEPRINT_TEMPLATE_CODE.format(**params)
    path = os.path.join(BOTS_DIR, "blueprint", f"{name}.py")
    write_file(path, code)
    archive_path = f"evolution/archive/{name}.py"
    copy_file(path, archive_path)
    return {"id": name, "path": path, "species": "blueprint", "ev": 0.0}


def create_initial_blueprint_bot(state: dict) -> dict:
    name = _new_id(state, "blueprint")
    bot = _write_blueprint_bot(state, name)
    state["bots"]["blueprint"].append(bot)
    return state


def mutate_blueprint_bot(state: dict, parent_bot: Dict) -> Dict:
    name = _new_id(state, "blueprint")
    bot = _write_blueprint_bot(state, name)
    state["bots"]["blueprint"].append(bot)
    return bot
