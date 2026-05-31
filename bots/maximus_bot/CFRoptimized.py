# bot.py — Monte Carlo + Lightweight CFR Hybrid Bot (Optimized)

import random
import time
import eval7

RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK = [r + s for r in RANKS for s in SUITS]

# CFR storage: regrets & strategy sums
REGRETS = {}
STRAT_SUM = {}

ACTIONS = [0, 1, 2]  # fold/check, call, raise


# ------------------ SAFE EVAL ------------------

def safe_eval(cards):
    try:
        if len(cards) < 5:
            return 0
        return eval7.evaluate(cards[:7])
    except Exception:
        return 0


# ------------------ OPTIMIZED MONTE CARLO ------------------

def estimate_equity_multi_timed(your_cards, board_cards, n_opponents, time_limit=1.8):
    start = time.perf_counter()

    yc = [eval7.Card(c) for c in your_cards]
    bc = [eval7.Card(c) for c in board_cards]

    dead = set(your_cards + board_cards)
    deck = [c for c in FULL_DECK if c not in dead]
    deck_eval = [eval7.Card(c) for c in deck]
    deck_len = len(deck_eval)

    wins = ties = 0
    iters = 0
    need = 5 - len(board_cards)

    if n_opponents * 2 + max(0, need) > deck_len:
        return 0.5

    opp_cards = [None] * (2 * n_opponents)
    MAX_ITERS = 12000

    while time.perf_counter() - start < time_limit and iters < MAX_ITERS:
        sample = random.sample(deck_eval, 2 * n_opponents + need)

        idx = 0
        for i in range(2 * n_opponents):
            opp_cards[i] = sample[idx]
            idx += 1

        sim_board = bc
        if need > 0:
            sim_board = bc + sample[idx:idx + need]

        our_score = safe_eval(yc + sim_board)

        better = equal = 0
        k = 0
        for _ in range(n_opponents):
            opp_score = safe_eval([opp_cards[k], opp_cards[k+1]] + sim_board)
            k += 2
            if opp_score > our_score:
                better += 1
            elif opp_score == our_score:
                equal += 1

        if better == 0 and equal == 0:
            wins += 1
        elif better == 0:
            ties += 1

        iters += 1

    if iters == 0:
        return 0.5

    return (wins + 0.5 * ties) / iters


# ------------------ BOARD TEXTURE ------------------

def board_texture_features(board_cards):
    if not board_cards:
        return {"connected": False, "flushy": False, "paired": False}

    ranks = [c[0] for c in board_cards]
    suits = [c[1] for c in board_cards]

    idxs = sorted([RANKS.index(r) for r in ranks])
    gaps = [idxs[i+1] - idxs[i] for i in range(len(idxs)-1)]
    min_gap = min(gaps) if gaps else 10

    flush_counts = {s: suits.count(s) for s in SUITS}
    max_flush = max(flush_counts.values())

    paired = len(set(ranks)) < len(ranks)

    return {
        "connected": min_gap <= 2,
        "flushy": max_flush >= 3,
        "paired": paired,
    }


# ------------------ OPPONENT MODEL ------------------

def opponent_aggression_factor(state):
    log = state.get("action_log", [])
    raises = sum(1 for e in log if e.get("action") in ("raise", "bet"))
    calls = sum(1 for e in log if e.get("action") == "call")
    total = raises + calls
    if total == 0:
        return 1.0
    agg = raises / total
    return 0.7 + 0.6 * agg


# ------------------ CFR HELPERS ------------------

def get_infoset(street, equity_bucket, pot_bucket, to_call):
    """Small abstraction for CFR."""
    return (street, equity_bucket, pot_bucket, 1 if to_call > 0 else 0)


def regret_matching(regrets):
    pos = [max(r, 0) for r in regrets]
    s = sum(pos)
    if s > 1e-9:
        return [p / s for p in pos]
    return [1/3, 1/3, 1/3]


# ------------------ MAIN DECIDE ------------------

def decide(state: dict) -> dict:
    try:
        your_cards = state["your_cards"]
        board = state["community_cards"]
        street = state["street"]
        pot = float(state["pot"])
        stack = float(state["your_stack"])
        to_call = float(state["amount_owed"])
        can_check = state["can_check"]
        min_raise_to = int(state["min_raise_to"])

        if stack <= 0:
            return {"action": "call"} if to_call > 0 else {"action": "check"}

        n_opps = sum(1 for p in state["players"] if not p.get("has_folded", False)) - 1
        n_opps = max(1, n_opps)

        # Monte Carlo equity (optimized)
        equity = estimate_equity_multi_timed(
            your_cards,
            board,
            n_opps,
            time_limit=1.8
        )

        # ------------------ CFR STATE ------------------

        # Equity bucket (4 buckets)
        if equity < 0.25:
            eq_bucket = 0
        elif equity < 0.50:
            eq_bucket = 1
        elif equity < 0.75:
            eq_bucket = 2
        else:
            eq_bucket = 3

        # Pot bucket
        pot_bucket = 0 if pot < 20 else 1 if pot < 60 else 2

        infoset = get_infoset(street, eq_bucket, pot_bucket, to_call)

        if infoset not in REGRETS:
            REGRETS[infoset] = [0.0, 0.0, 0.0]
            STRAT_SUM[infoset] = [0.0, 0.0, 0.0]

        regrets = REGRETS[infoset]
        strat = regret_matching(regrets)

        # Sample CFR action
        r = random.random()
        cum = 0
        for i, p in enumerate(strat):
            cum += p
            if r <= cum:
                cfr_action = i
                break
        else:
            cfr_action = 2

        # ------------------ HEURISTIC ACTION ------------------

        texture = board_texture_features(board)
        dry = not texture["connected"] and not texture["flushy"] and not texture["paired"]
        wet = texture["connected"] or texture["flushy"]

        # Pot odds
        pot_odds = to_call / (pot + to_call) if to_call > 0 else 0

        # Heuristic thresholds
        margin = 0.03
        if street == "preflop":
            raise_thresh = 0.62
            call_thresh = max(pot_odds + margin, 0.25)
        elif street == "flop":
            raise_thresh = 0.60
            call_thresh = pot_odds + margin
        elif street == "turn":
            raise_thresh = 0.62
            call_thresh = pot_odds + margin
        else:
            raise_thresh = 0.66
            call_thresh = pot_odds + margin

        agg = opponent_aggression_factor(state)
        call_thresh *= (1 - 0.15 * (agg - 1))
        raise_thresh *= (1 - 0.10 * (agg - 1))

        # ------------------ BLEND CFR + HEURISTIC ------------------

        # CFR action meanings:
        # 0 = fold/check
        # 1 = call
        # 2 = raise

        # Heuristic override if extreme
        if equity < call_thresh * 0.7:
            chosen = 0
        elif equity > raise_thresh * 1.2:
            chosen = 2
        else:
            chosen = cfr_action

        # ------------------ EXECUTE ACTION ------------------

        # Facing a bet
        if to_call > 0:
            if chosen == 0:
                return {"action": "fold"}
            if chosen == 1:
                return {"action": "call"}
            # raise
            mult = 1.8 if dry else 1.4
            amount = max(min_raise_to, int(state["current_bet"] + to_call * mult))
            if amount >= state["current_bet"] + stack * 0.95:
                return {"action": "all_in"}
            return {"action": "raise", "amount": amount}

        # No bet to call
        if can_check:
            if chosen == 0:
                return {"action": "check"}
            if chosen == 1:
                return {"action": "check"}
            # raise
            mult = 0.6 if dry else 0.9
            amount = max(min_raise_to, int(state["current_bet"] + pot * mult))
            if amount >= state["current_bet"] + stack * 0.95:
                return {"action": "all_in"}
            return {"action": "raise", "amount": amount}

        # Weird edge case
        return {"action": "call"}

    except Exception:
        if state.get("can_check", False) and state.get("amount_owed", 0) == 0:
            return {"action": "check"}
        return {"action": "call"}
