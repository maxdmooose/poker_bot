import subprocess
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import random
import csv
from datetime import datetime

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

MATCH_PATH = "sandbox/match.py"

BOTS = [
    "bots/maximus_bot/CFRoptimized.py",
    "bots/maximus_bot/tierS.py",
#    "bots/final/best.py",
    "bots/maximus_bot/CFRtierS.py",
    "bots/maximus_bot/optimized.py",
    "bots/final/final.py",
]

HANDS_PER_ROUND = 100
ROUNDS = 120
MAX_WORKERS = 60   # be sane: ~#physical cores

LOG_TO_CSV = True
CSV_PATH = f"results_fullhouse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


# ---------------------------------------------------------
# BOT NAME EXTRACTION
# ---------------------------------------------------------

def bot_name_from_path(path):
    base = os.path.basename(path)
    return base.replace(".py", "")


# ---------------------------------------------------------
# WORKER FUNCTION
# ---------------------------------------------------------

def run_single_round(round_index, bots_for_round):
    """
    Runs one match.py instance with a specific seating order
    and returns (bot_paths_in_this_round, chip_deltas_in_same_order).
    """
    cmd = [
        "python",
        MATCH_PATH,
        "--json",
        "--hands", str(HANDS_PER_ROUND),
        *bots_for_round,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    stdout = result.stdout.strip()

    try:
        first_line = stdout.splitlines()[0].strip()
        data = json.loads(first_line)
        deltas_dict = data["chip_delta"]

        deltas = [deltas_dict.get(bot_name_from_path(b), 0.0) for b in bots_for_round]
        return bots_for_round, deltas

    except Exception as e:
        print(f"[Round {round_index}] ERROR parsing output")
        print("Raw output:", stdout)
        print("Exception:", e)
        return bots_for_round, [0.0] * len(bots_for_round)


# ---------------------------------------------------------
# MAIN PARALLEL EXECUTION
# ---------------------------------------------------------

def main():
    # global cumulative results keyed by bot path
    cumulative = {b: 0.0 for b in BOTS}
    total_hands_per_bot = {b: 0 for b in BOTS}

    if LOG_TO_CSV:
        csv_file = open(CSV_PATH, "w", newline="")
        csv_writer = csv.writer(csv_file)
        header = ["round_index"] + [bot_name_from_path(b) for b in BOTS]
        csv_writer.writerow(header)
    else:
        csv_file = csv_writer = None

    print(f"Running {ROUNDS} rounds in parallel...\n")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}

        for i in range(ROUNDS):
            # randomize seating each round to remove positional bias
            bots_for_round = BOTS[:]
            random.shuffle(bots_for_round)
            fut = executor.submit(run_single_round, i, bots_for_round)
            futures[fut] = (i, bots_for_round)

        for future in as_completed(futures):
            round_idx, bots_for_round = futures[future]
            bots_round, deltas = future.result()

            # accumulate into global per-bot totals
            for b, d in zip(bots_round, deltas):
                cumulative[b] += d
                total_hands_per_bot[b] += HANDS_PER_ROUND

            # log per-round row in canonical bot order
            if csv_writer is not None:
                row = [round_idx]
                # map from this round's bot to its delta
                d_map = {b: d for b, d in zip(bots_round, deltas)}
                for b in BOTS:
                    row.append(d_map.get(b, 0.0))
                csv_writer.writerow(row)

            print(f"Round {round_idx+1}/{ROUNDS} complete → {deltas}")

    if csv_file is not None:
        csv_file.close()
        print(f"\nCSV written to: {CSV_PATH}")

    print("\n==============================")
    print(" FINAL CUMULATIVE CHIP DELTAS ")
    print("==============================")
    for b in BOTS:
        name = bot_name_from_path(b)
        delta = cumulative[b]
        hands = total_hands_per_bot[b]
        ev_per_100 = (delta / hands) * 100 if hands > 0 else 0.0
        print(f"{name:20s}  Δchips = {delta:+.1f}   EV/100 = {ev_per_100:+.2f}")


if __name__ == "__main__":
    main()
