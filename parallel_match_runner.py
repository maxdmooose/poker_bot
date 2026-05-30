import subprocess
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

MATCH_PATH = "sandbox/match.py"

# Up to 6 bots
BOTS = [
    "bots/maximus_bot/bot.py",
    "bots/maximus_bot/bot2.py",
    "bots/maximus_bot/bot3.py",
    "bots/maximus_bot/CFR.py",
    "bots/maximus_bot/best.py",
    "bots/maximus_bot/gametree.py",
]

HANDS_PER_ROUND = 50
ROUNDS = 30
MAX_WORKERS = 30   # number of parallel matches


# ---------------------------------------------------------
# BOT NAME EXTRACTION
# ---------------------------------------------------------

def bot_name_from_path(path):
    """
    Fullhouse uses the filename (without .py) as the bot name.
    Example: bots/maximus_bot/bot3.py → bot3
    """
    base = os.path.basename(path)
    return base.replace(".py", "")


# ---------------------------------------------------------
# WORKER FUNCTION
# ---------------------------------------------------------

def run_single_round(round_index):
    """
    Runs one match.py instance and returns chip deltas for all bots.
    """

    cmd = [
        "python",
        MATCH_PATH,
        "--json",                     # MUST come before bot list
        "--hands", str(HANDS_PER_ROUND),
        *BOTS
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    stdout = result.stdout.strip()

    # Fullhouse prints JSON on the FIRST LINE, then human output.
    # So we must extract ONLY the first line.
    try:
        first_line = stdout.splitlines()[0].strip()
        data = json.loads(first_line)

        deltas_dict = data["chip_delta"]

        # Preserve bot order
        deltas = [deltas_dict.get(bot_name_from_path(b), 0) for b in BOTS]

        return deltas

    except Exception as e:
        print(f"[Round {round_index}] ERROR parsing output")
        print("Raw output:", stdout)
        print("Exception:", e)
        return [0] * len(BOTS)


# ---------------------------------------------------------
# MAIN PARALLEL EXECUTION
# ---------------------------------------------------------

def main():
    cumulative = [0] * len(BOTS)

    print(f"Running {ROUNDS} rounds in parallel...\n")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_single_round, i): i
            for i in range(ROUNDS)
        }

        for future in as_completed(futures):
            idx = futures[future]
            deltas = future.result()
            cumulative = [c + d for c, d in zip(cumulative, deltas)]
            print(f"Round {idx+1}/{ROUNDS} complete → {deltas}")

    print("\n==============================")
    print(" FINAL CUMULATIVE CHIP DELTAS ")
    print("==============================")
    for bot, delta in zip(BOTS, cumulative):
        print(f"{bot:40s}  {delta:+.1f}")


if __name__ == "__main__":
    main()
