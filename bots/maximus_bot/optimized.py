# bot.py — Fullhouse Monte Carlo "maxed" bot (optimized version)

import random
import time
import eval7

RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK = [r + s for r in RANKS for s in SUITS]


# ------------------ SAFE EVAL7 WRAPPER ------------------

def safe_eval(cards):
    """eval7.evaluate but guaranteed not to crash."""
    try:
        if len(cards) < 5:
            return 0
        return eval7.evaluate(cards[:7])
    except Exception:
        return 0


# ------------------ EQUITY (OPTIMIZED) ------------------

def estimate_equity_multi_timed(your_cards, board_cards, n_opponents, time_limit=1.0):
    """
    Ultra-optimized Monte Carlo equity vs n_opponents random hands.
    Time-bounded to ~time_limit seconds.
    """

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
    """
    Simple board texture: connectedness, flushiness, pairing.
    """
    if not board_cards:
        return {
            "connected": False,
            "very_connected": False,
            "flushy": False,
            "paired": False,
        }

    ranks = [c[0] for c in board_cards]
    suits = [c[1] for c in board_cards]

    rank_idxs = sorted([RANKS.index(r) for r in ranks])
    gaps = [rank_idxs[i+1] - rank_idxs[i] for i in range(len(rank_idxs)-1)]
    min_gap = min(gaps) if gaps else 10

    flush_counts = {s: suits.count(s) for s in SUITS}
    max_flush = max(flush_counts.values()) if flush_counts else 0

    paired = len(set(ranks)) < len(ranks)

    return {
        "connected": min_gap <= 2,
        "very_connected": min_gap == 1,
        "flushy": max_flush >= 3,
        "paired": paired,
    }


# ------------------ OPPONENT MODEL ------------------

def opponent_aggression_factor(state):
    """
    Crude opponent model from action_log:
    ratio of raises/bets to (raises+bets+calls).
    """
    log = state.get("action_log", [])
    raises = 0
    calls = 0
    for entry in log:
        act = entry.get("action")
        if act in ("raise", "bet"):
            raises += 1
        elif act == "call":
            calls += 1
    total = raises + calls
    if total == 0:
        return 1.0  # neutral
    agg = raises / total
    return 0.7 + 0.6 * agg  # map [0,1] to [0.7,1.3]


# ------------------ MAIN STRATEGY ------------------

def decide(state: dict) -> dict:
    """
    Fullhouse decide() entrypoint.
    """
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
            if to_call > 0:
                return {"action": "call"}
            return {"action": "check"} if can_check else {"action": "fold"}

        n_opponents = sum(1 for p in state["players"] if not p.get("has_folded", False)) - 1
        n_opponents = max(1, n_opponents)

        # give more time to Monte Carlo but keep safety margin
        equity = estimate_equity_multi_timed(
            your_cards,
            board,
            n_opponents=n_opponents,
            time_limit=1.55,  # previously 1.75
        )

        if to_call > 0:
            pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 1.0
        else:
            pot_odds = 0.0

        margin = 0.03
        if street == "preflop":
            base_raise_thresh = 0.62
            base_call_thresh = max(pot_odds + margin, 0.25)
        elif street == "flop":
            base_raise_thresh = 0.60
            base_call_thresh = pot_odds + margin
        elif street == "turn":
            base_raise_thresh = 0.62
            base_call_thresh = pot_odds + margin
        else:  # river
            base_raise_thresh = 0.66
            base_call_thresh = pot_odds + margin

        agg_factor = opponent_aggression_factor(state)
        call_thresh = max(0.05, min(0.95, base_call_thresh * (1.0 - 0.15 * (agg_factor - 1.0))))
        raise_thresh = max(0.05, min(0.95, base_raise_thresh * (1.0 - 0.10 * (agg_factor - 1.0))))

        texture = board_texture_features(board)
        dry_board = not texture["connected"] and not texture["flushy"] and not texture["paired"]
        wet_board = texture["connected"] or texture["flushy"]

        r = random.random()

        # ---------- Facing a bet ----------
        if to_call > 0:
            if equity < call_thresh:
                if pot_odds < 0.15 and equity > pot_odds and r < 0.25:
                    return {"action": "call"}
                if pot_odds < 0.20 and equity > pot_odds and agg_factor > 1.1 and r < 0.35:
                    return {"action": "call"}
                return {"action": "fold"}

            if equity >= raise_thresh and stack > to_call * 2:
                base_freq = (equity - raise_thresh) / 0.2
                if dry_board:
                    base_freq += 0.10
                if wet_board:
                    base_freq -= 0.08
                if agg_factor < 0.9:
                    base_freq += 0.08
                if agg_factor > 1.1:
                    base_freq -= 0.05

                raise_freq = min(0.9, max(0.15, base_freq))
                if r < raise_freq:
                    mult_min = 1.5 if wet_board else 1.8
                    mult_max = 2.2 if wet_board else 2.7
                    min_total = max(min_raise_to, int(state["current_bet"] + to_call * mult_min))
                    max_total = int(min(state["current_bet"] + to_call * mult_max,
                                        state["current_bet"] + stack))
                    if max_total <= min_total:
                        amount = min_total
                    else:
                        alpha = random.random()
                        amount = int(min_total + alpha * (max_total - min_total))

                    if amount >= state["current_bet"] + stack * 0.95:
                        return {"action": "all_in"}

                    return {"action": "raise", "amount": amount}

            return {"action": "call"}

        # ---------- No bet to call (we can check or bet) ----------
        if can_check:
            if equity >= raise_thresh:
                value_freq = 0.85
                if dry_board:
                    value_freq += 0.05
                if agg_factor < 0.9:
                    value_freq += 0.05
                value_freq = min(0.95, value_freq)

                if r < value_freq:
                    if dry_board:
                        min_mult, max_mult = 0.45, 1.0
                    elif wet_board:
                        min_mult, max_mult = 0.55, 1.2
                    else:
                        min_mult, max_mult = 0.5, 1.1

                    min_total = max(min_raise_to, state["current_bet"] + int(pot * min_mult))
                    max_total = max(min_total, state["current_bet"] + int(pot * max_mult))
                    alpha = random.random()
                    amount = int(min_total + alpha * (max_total - min_total))

                    if amount >= state["current_bet"] + stack * 0.95:
                        return {"action": "all_in"}

                    return {"action": "raise", "amount": amount}

                return {"action": "check"}

            bluff_low, bluff_high = 0.18, 0.42
            if bluff_low <= equity <= bluff_high and street in ("flop", "turn"):
                base_bluff_freq = 0.18
                if dry_board:
                    base_bluff_freq += 0.06
                if wet_board:
                    base_bluff_freq -= 0.04
                if texture["paired"]:
                    base_bluff_freq += 0.04
                if agg_factor < 0.9:
                    base_bluff_freq += 0.05
                if agg_factor > 1.1:
                    base_bluff_freq -= 0.05

                base_bluff_freq = max(0.05, min(0.30, base_bluff_freq))

                if r < base_bluff_freq:
                    if dry_board:
                        min_mult, max_mult = 0.45, 0.75
                    else:
                        min_mult, max_mult = 0.55, 0.90

                    min_total = max(min_raise_to, state["current_bet"] + int(pot * min_mult))
                    max_total = max(min_total, state["current_bet"] + int(pot * max_mult))
                    alpha = random.random()
                    amount = int(min_total + alpha * (max_total - min_total))

                    if amount >= state["current_bet"] + stack * 0.95:
                        return {"action": "all_in"}

                    return {"action": "raise", "amount": amount}

            return {"action": "check"}

        return {"action": "call"}

    except Exception:
        if state.get("can_check", False) and state.get("amount_owed", 0) == 0:
            return {"action": "check"}
        return {"action": "call"}
