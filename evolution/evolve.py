# evolution/evolve.py

import json
import os
import datetime
from typing import Dict, Any, List

from config import (
    NUM_GENERATIONS,
    CHILDREN_PER_SPECIES,
    SURVIVORS_PER_SPECIES,
    SPECIES,
    STATE_PATH,
    LOG_DIR,
)
from file_ops import ensure_dirs
from species_registry import SPECIES_REGISTRY
from match_runner import evaluate_bot
from hall_of_fame import maybe_add_to_hof


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str):
    print(f"[{_now()}] {msg}")
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "evolution.log"), "a", encoding="utf-8") as f:
        f.write(f"[{_now()}] {msg}\n")


def _init_state() -> Dict[str, Any]:
    return {
        "generation": 0,
        "bots": {sp: [] for sp in SPECIES},
        "next_id": {sp: 0 for sp in SPECIES},
    }


def _load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        return _init_state()
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(state: Dict[str, Any]):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _ensure_initial_population(state: Dict[str, Any]) -> Dict[str, Any]:
    for sp in SPECIES:
        if len(state["bots"][sp]) == 0:
            init_fn = SPECIES_REGISTRY[sp]["init"]
            for _ in range(SURVIVORS_PER_SPECIES):
                state = init_fn(state)
            _log(f"Initialized species {sp} with {SURVIVORS_PER_SPECIES} bots.")
    return state


def _evaluate_species(state: Dict[str, Any], species: str):
    for bot in state["bots"][species]:
        ev = evaluate_bot(bot["path"])
        bot["ev"] = ev
        _log(f"Evaluated {bot['id']} ({species}) EV={ev:.2f} BB/100")
        maybe_add_to_hof(bot)


def _select_and_mutate(state: Dict[str, Any], species: str) -> Dict[str, Any]:
    bots: List[Dict] = state["bots"][species]
    bots_sorted = sorted(bots, key=lambda b: b.get("ev", -9999.0), reverse=True)
    survivors = bots_sorted[:SURVIVORS_PER_SPECIES]
    _log(f"Survivors for {species}: {[b['id'] for b in survivors]}")

    mutate_fn = SPECIES_REGISTRY[species]["mutate"]
    children: List[Dict] = []
    import random

    for _ in range(CHILDREN_PER_SPECIES):
        parent = random.choice(survivors)
        child = mutate_fn(state, parent)
        children.append(child)

    state["bots"][species] = survivors + children
    return state


def main():
    ensure_dirs()
    state = _load_state()
    state = _ensure_initial_population(state)
    _save_state(state)

    start_gen = state["generation"]
    end_gen = start_gen + NUM_GENERATIONS

    for gen in range(start_gen, end_gen):
        _log(f"=== Generation {gen} ===")

        for sp in SPECIES:
            _evaluate_species(state, sp)

        for sp in SPECIES:
            state = _select_and_mutate(state, sp)

        state["generation"] = gen + 1
        _save_state(state)
        _log(f"=== End of generation {gen} ===")

    _log("Evolution complete. You can now inspect logs, hall_of_fame, and bots.")


if __name__ == "__main__":
    main()
