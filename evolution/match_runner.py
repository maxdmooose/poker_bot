# evolution/match_runner.py

import subprocess
import json
import os
import random

from config import HANDS_PER_ROUND, ROUNDS_PER_EVAL, STATE_PATH


def _load_population_paths() -> list:
    """Loads all bot paths from population_state.json."""
    if not os.path.exists(STATE_PATH):
        return []

    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        bots = []
        for species_bots in data["bots"].values():
            for bot in species_bots:
                bots.append(bot["path"])

        return bots

    except Exception:
        return []


def _choose_population_opponents(bot_path: str, num_opponents: int = 5) -> list:
    """
    Selects opponents ONLY from the current population.
    Excludes the bot being evaluated.
    """
    all_bots = _load_population_paths()

    # Remove the bot being evaluated
    opponents = [b for b in all_bots if b != bot_path]

    # If not enough opponents exist, duplicate randomly
    if len(opponents) < num_opponents:
        while len(opponents) < num_opponents:
            opponents.append(random.choice(all_bots))

    return random.sample(opponents, num_opponents)


def _run_single_round(bot_path: str, opponents: list) -> float:
    """
    Runs ONE Fullhouse match (one 'round') of HANDS_PER_ROUND hands.
    Returns chip delta for the evaluated bot.
    """

    cmd = [
        "python3",
        "sandbox/match.py",
        bot_path,
    ] + opponents + [
        "--hands",
        str(HANDS_PER_ROUND),
        "--json"
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    try:
        data = json.loads(result.stdout)
        # bot under evaluation is always index 0
        chip_delta = data["results"][0]["chips"]
        return float(chip_delta)

    except Exception as e:
        print("Error evaluating bot:", e)
        print("Raw output:", result.stdout)
        return -999999.0  # catastrophic failure → bot dies


def evaluate_bot(bot_path: str) -> float:
    """
    Runs ROUNDS_PER_EVAL rounds of HANDS_PER_ROUND hands each.
    Sums chip deltas across all rounds.
    Converts to BB/100.
    """

    opponents = _choose_population_opponents(bot_path, num_opponents=5)

    total_chips = 0.0

    for _ in range(ROUNDS_PER_EVAL):
        delta = _run_single_round(bot_path, opponents)
        total_chips += delta

    # Convert chips → BB/100
    # Fullhouse uses 1 chip = 1 BB (big blind)
    total_hands = ROUNDS_PER_EVAL * HANDS_PER_ROUND
    bb100 = (total_chips / total_hands) * 100.0

    return bb100
