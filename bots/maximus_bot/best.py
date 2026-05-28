# bot.py — Max-strength Monte Carlo bot for Fullhouse Hackathon
# Uses: multi-opponent equity, time-adaptive rollouts, board texture,
#       opponent modelling, dynamic thresholds, calibrated mixed strategies.

import random
import time
import eval7

RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK = [r + s for r in RANKS for s in SUITS]


# ------------------------------------------------------------
# Monte Carlo Equity (multi-opponent, time-bounded)
# ------------------------------------------------------------

def estimate_equity_multi_timed(your_cards, board_cards, n_opponents, time_limit):
    start = time.perf_counter()

    your_eval = [eval7.Card(c) for c in your_cards]
    board_eval = [eval7.Card(c) for c in board_cards]

    dead = set(your_cards + board_cards)
    deck = [c for c in FULL_DECK if c not in dead]
    deck_eval = [eval7.Card(c) for c in deck]

    wins = ties = 0
    iters = 0
    board_needed = 5 - len(board_cards)

    while True:
        if time.perf_counter() - start >= time_limit:
            break

        random.shuffle(deck_eval)

        # deal opponents
        opp_hands = []
        idx = 0
        for _ in range(n_opponents):
            opp_hands.append(deck_eval[idx:idx+2])
            idx += 2

        # complete board
        sim_board = board_eval[:]
        if board_needed > 0:
            sim_board = sim_board + deck_eval[idx:idx + board_needed]

        our_score = eval7.evaluate(your_eval + sim_board)

        better = equal = 0
        for opp in opp_hands:
            opp_score = eval7.evaluate(opp + sim_board)
            if opp_score > our_score:
                better += 1
            elif opp_score == our_score:
                equal += 1

        if better == 0 and equal == 0:
            wins += 1
        elif better == 0 and equal > 0:
            ties += 1

        iters += 1

    if iters == 0:
        return 0.5

    return (wins + 0.5 * ties) / iters


# ------------------------------------------------------------
# Board Texture
# ------------------------------------------------------------

def board_texture(board):
    if not board:
        return {"dry": True, "wet": False, "paired": False}

    ranks = [c[0] for c in board]
    suits = [c[1] for c in board]

    idxs = sorted([RANKS.index(r) for r in ranks])
    gaps = [idxs[i+1] - idxs[i] for i in range(len(idxs)-1)]
    min_gap = min(gaps) if gaps else 10

    flush_counts = {s: suits.count(s) for s in SUITS}
    max_flush = max(flush_counts.values())

    paired = len(set(ranks)) < len(ranks)
    connected = min_gap <= 2
    wet = connected or max_flush >= 3

    return {
        "dry": not wet and not paired,
        "wet": wet,
        "paired": paired
    }


# ------------------------------------------------------------
# Opponent Model
# ------------------------------------------------------------

def opponent_model(state):
    """
    Tracks fold/call/raise frequencies of opponents.
    """
    log = state["action_log"]
    folds = calls = raises = 0

    for entry in log:
        act = entry["action"]
        if act == "fold":
            folds += 1
        elif act == "call":
            calls += 1
        elif act in ("raise", "bet"):
            raises += 1

    total = folds + calls + raises
    if total == 0:
        return {"fold": 0.33, "call": 0.33, "raise": 0.33}

    return {
        "fold": folds / total,
        "call": calls / total,
        "raise": raises / total
    }


# ------------------------------------------------------------
# Main Strategy
# ------------------------------------------------------------

def decide(state: dict) -> dict:
    your_cards = state["your_cards"]
    board = state["community_cards"]
    street = state["street"]
    pot = float(state["pot"])
    stack = float(state["your_stack"])
    to_call = float(state["amount_owed"])
    can_check = state["can_check"]
    min_raise_to = int(state["min_raise_to"])

    # trivial case
    if stack <= 0:
        return {"action": "call"} if to_call > 0 else {"action": "check"}

    # number of active opponents
    n_opponents = sum(1 for p in state["players"] if not p["has_folded"]) - 1
    n_opponents = max(1, n_opponents)

    # time budget: leave ~0.25s for logic
    equity = estimate_equity_multi_timed(
        your_cards,
        board,
        n_opponents=n_opponents,
        time_limit=1.95
    )

    # pot odds
    pot_odds = to_call / (pot + to_call) if to_call > 0 else 0.0

    # board texture + opponent tendencies
    tex = board_texture(board)
    opp = opponent_model(state)

    # dynamic thresholds
    margin = 0.03
    base_call = pot_odds + margin
    base_raise = 0.60 if street == "flop" else 0.62
    if street == "river":
        base_raise = 0.66

    # exploit adjustments
    call_thresh = base_call * (1 - 0.20 * (opp["raise"] - 0.33))
    raise_thresh = base_raise * (1 - 0.15 * (opp["call"] - 0.33))

    call_thresh = max(0.05, min(0.95, call_thresh))
    raise_thresh = max(0.05, min(0.95, raise_thresh))

    r = random.random()

    # ------------------------------------------------------------
    # Facing a bet
    # ------------------------------------------------------------
    if to_call > 0:
        # fold region
        if equity < call_thresh:
            # defend vs tiny bets
            if pot_odds < 0.15 and equity > pot_odds and r < 0.30:
                return {"action": "call"}
            return {"action": "fold"}

        # raise region
        if equity >= raise_thresh and stack > to_call * 2:
            # raise frequency
            freq = (equity - raise_thresh) / 0.25
            if tex["dry"]:
                freq += 0.10
            if opp["fold"] > 0.45:
                freq += 0.10
            freq = max(0.15, min(0.90, freq))

            if r < freq:
                # sizing
                mult_min = 1.6 if tex["dry"] else 1.3
                mult_max = 2.4 if tex["dry"] else 1.8

                min_total = max(min_raise_to, int(state["current_bet"] + to_call * mult_min))
                max_total = int(min(state["current_bet"] + to_call * mult_max,
                                    state["current_bet"] + stack))

                if max_total <= min_total:
                    amount = min_total
                else:
                    amount = int(min_total + random.random() * (max_total - min_total))

                if amount >= state["current_bet"] + stack * 0.95:
                    return {"action": "all_in"}

                return {"action": "raise", "amount": amount}

        return {"action": "call"}

    # ------------------------------------------------------------
    # No bet to call (we can check or bet)
    # ------------------------------------------------------------
    if can_check:
        # value betting
        if equity >= raise_thresh:
            freq = 0.85
            if tex["dry"]:
                freq += 0.05
            if opp["call"] < 0.25:
                freq += 0.05
            freq = min(0.95, freq)

            if r < freq:
                min_mult = 0.45 if tex["dry"] else 0.55
                max_mult = 1.0 if tex["dry"] else 1.2

                min_total = max(min_raise_to, state["current_bet"] + int(pot * min_mult))
                max_total = max(min_total, state["current_bet"] + int(pot * max_mult))

                amount = int(min_total + random.random() * (max_total - min_total))

                if amount >= state["current_bet"] + stack * 0.95:
                    return {"action": "all_in"}

                return {"action": "raise", "amount": amount}

            return {"action": "check"}

        # bluffing region
        bluff_low, bluff_high = 0.18, 0.42
        if bluff_low <= equity <= bluff_high and street in ("flop", "turn"):
            freq = 0.18
            if tex["dry"]:
                freq += 0.06
            if opp["fold"] > 0.40:
                freq += 0.08
            if opp["call"] > 0.45:
                freq -= 0.06
            freq = max(0.05, min(0.30, freq))

            if r < freq:
                min_mult = 0.45 if tex["dry"] else 0.55
                max_mult = 0.75 if tex["dry"] else 0.90

                min_total = max(min_raise_to, state["current_bet"] + int(pot * min_mult))
                max_total = max(min_total, state["current_bet"] + int(pot * max_mult))

                amount = int(min_total + random.random() * (max_total - min_total))

                if amount >= state["current_bet"] + stack * 0.95:
                    return {"action": "all_in"}

                return {"action": "raise", "amount": amount}

        return {"action": "check"}

    # fallback
    return {"action": "call"}
