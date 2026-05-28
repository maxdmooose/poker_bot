# evolution/config.py

# --- core evolution parameters ---
NUM_GENERATIONS = 5          # change to 50, 100, 150, etc.
CHILDREN_PER_SPECIES = 1      # 4 children per species → 12 total
SURVIVORS_PER_SPECIES = 1

ROUNDS_PER_EVAL = 1
HANDS_PER_ROUND = 100

# --- hall of fame ---
HOF_EV_THRESHOLD_BB100 = 5.0  # archive if EV > +5 BB/100

# --- species ---
SPECIES = ["heuristic", "lookup", "blueprint"]

# --- paths (relative to repo root) ---
EVOLUTION_ROOT = "evolution"
BOTS_DIR = f"{EVOLUTION_ROOT}/bots"
HOF_DIR = f"{EVOLUTION_ROOT}/hall_of_fame"
LOG_DIR = f"{EVOLUTION_ROOT}/logs"
STATE_PATH = f"{EVOLUTION_ROOT}/population_state.json"
