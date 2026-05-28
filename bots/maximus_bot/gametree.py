# bot.py — Game-tree search bot (Fullhouse)
# Depth-limited lookahead with eval7 equity evaluation, time-bounded to ~2s.

import random
import time
import eval7

RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK = [r + s for r in RANKS for s in SUITS]


# ----------------- Equity evaluator -----------------

def estimate_equity_vs_random(your_cards, board_cards, n_opponents, iters, rng):
    your_eval = [eval7.Card(c) for c in your_cards]
    board_eval = [eval7.Card(c) for c in board_cards]

    dead = set(your_cards + board_cards)
    deck = [c for c in FULL_DECK if c not in dead]
    deck_eval = [eval7.Card(c) for c in deck]

    wins = ties = 0
    board_needed = 5 - len(board_cards)

    for _ in range(iters):
        rng.shuffle(deck_eval)

        opp_hands = []
        idx = 0
        for _ in range(n_opponents):
            opp_hands.append(deck_eval[idx:idx+2])
            idx += 2

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

    total = float(iters)
    if total == 0:
        return 0.5
    return (wins + 0.5 * ties) / total


# ----------------- Board texture -----------------

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


# ----------------- Game-tree search -----------------

def evaluate_terminal(pot, stack, to_call, your_cards, board, n_opponents, rng, time_left):
    """
    Evaluate EV of calling/folding at terminal node using equity.
    """
    if time_left() <= 0:
        # fallback: neutral
        return 0.0

    # if we fold, EV = 0 relative to current pot (we give up our invested chips)
    fold_ev = 0.0

    # if we call, we invest to_call and play for pot + to_call
    # approximate equity with small number of iterations
    iters = 200
    eq = estimate_equity_vs_random(your_cards, board, n_opponents, iters, rng)
    win_pot = pot + to_call
    call_ev = eq * win_pot - (1 - eq) * to_call

    return max(fold_ev, call_ev)


def search_action(state, time_left):
    """
    Depth-limited search over immediate actions:
    - If facing a bet: consider fold / call / raise
    - If not facing a bet: consider check / bet
    Use equity-based EV at leaves.
    """
    rng = random.Random()

    your_cards = state["your_cards"]
    board = state["community_cards"]
    street = state["street"]
    pot = float(state["pot"])
    stack = float(state["your_stack"])
    to_call = float(state["amount_owed"])
    can_check = state["can_check"]
    min_raise_to = int(state["min_raise_to"])
    current_bet = float(state["current_bet"])

    n_opponents = sum(1 for p in state["players"] if not p["has_folded"]) - 1
    n_opponents = max(1, n_opponents)

    tex = board_texture(board)

    # facing a bet
    if to_call > 0:
        # 1) Fold EV: 0 (relative)
        fold_ev = 0.0

        # 2) Call EV: evaluate terminal
        call_ev = evaluate_terminal(
            pot=pot,
            stack=stack,
            to_call=to_call,
            your_cards=your_cards,
            board=board,
            n_opponents=n_opponents,
            rng=rng,
            time_left=time_left
        )

        # 3) Raise EV: approximate by assuming villain calls with some frequency
        # and we realize equity on a bigger pot.
        if time_left() <= 0:
            # choose between fold and call
            return "fold" if fold_ev >= call_ev else "call"

        # choose a raise size family
        if tex["dry"]:
            mult_min, mult_max = 2.0, 2.8
        else:
            mult_min, mult_max = 1.6, 2.2

        raise_total = max(min_raise_to, int(current_bet + to_call * mult_min))
        raise_total = min(raise_total, int(current_bet + stack))
        if raise_total <= current_bet + to_call:
            raise_ev = call_ev  # can't really raise
        else:
            # approximate: villain folds some %; if called, we play for bigger pot
            # use small equity sample
            iters = 150
            eq = estimate_equity_vs_random(your_cards, board, n_opponents, iters, rng)
            pot_if_called = pot + to_call + (raise_total - current_bet)
            risk = raise_total - current_bet

            # assume villain folds more on dry boards
            fold_freq = 0.35 if tex["wet"] else 0.50
            call_freq = 1.0 - fold_freq

            win_ev = eq * pot_if_called - (1 - eq) * risk
            raise_ev = fold_freq * pot + call_freq * win_ev

        # choose best
        best_ev = max(fold_ev, call_ev, raise_ev)
        if best_ev == fold_ev:
            return "fold"
        elif best_ev == call_ev:
            return "call"
        else:
            return ("raise", raise_total)

    # no bet to call
    if can_check:
        # consider check vs bet
        # 1) Check EV: approximate as 0 baseline
        check_ev = 0.0

        if time_left() <= 0:
            return "check"

        # 2) Bet EV: bluff/value depending on equity
        # approximate equity quickly
        iters = 200
        eq = estimate_equity_vs_random(your_cards, board, n_opponents, iters, rng)

        if tex["dry"]:
            min_mult, max_mult = 0.45, 1.0
        else:
            min_mult, max_mult = 0.55, 1.2

        bet_total = max(min_raise_to, int(current_bet + pot * min_mult))
        bet_total = min(bet_total, int(current_bet + stack))

        # assume villain folds some %; if called, we realize equity
        fold_freq = 0.40 if tex["dry"] else 0.30
        call_freq = 1.0 - fold_freq

        pot_if_called = pot + (bet_total - current_bet)
        risk = bet_total - current_bet

        win_ev = eq * pot_if_called - (1 - eq) * risk
        bet_ev = fold_freq * pot + call_freq * win_ev

        if bet_ev > check_ev:
            return ("bet", bet_total)
        else:
            return "check"

    # fallback
    return "call"


# ----------------- Main decide -----------------

def decide(state: dict) -> dict:
    start = time.perf_counter()
    budget = 1.90  # leave a bit of safety

    def time_left():
        return budget - (time.perf_counter() - start)

    your_cards = state["your_cards"]
    board = state["community_cards"]
    pot = float(state["pot"])
    stack = float(state["your_stack"])
    to_call = float(state["amount_owed"])
    can_check = state["can_check"]

    if stack <= 0:
        return {"action": "call"} if to_call > 0 else {"action": "check"}

    # run one depth-limited search within time budget
    action = search_action(state, time_left)

    if isinstance(action, tuple):
        kind, amount = action
        if kind == "raise" or kind == "bet":
            # engine uses "raise" even when first in
            if amount >= state["current_bet"] + stack * 0.95:
                return {"action": "all_in"}
            return {"action": "raise", "amount": int(amount)}

    if action == "fold":
        return {"action": "fold"}
    if action == "call":
        if to_call == 0 and can_check:
            return {"action": "check"}
        return {"action": "call"}
    if action == "check":
        if can_check:
            return {"action": "check"}
        return {"action": "call"}

    # safety fallback
    return {"action": "call"}
