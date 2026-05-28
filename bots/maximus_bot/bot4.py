# bot.py — Tiny CFR-style abstraction with hard-coded blueprint

import random
import time
import eval7

RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK = [r + s for r in RANKS for s in SUITS]


def estimate_equity_timed(your_cards, board_cards, time_limit=1.9):
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
        opp = deck_eval[:2]
        sim_board = board_eval[:]
        if board_needed > 0:
            sim_board = sim_board + deck_eval[2:2 + board_needed]

        our_score = eval7.evaluate(your_eval + sim_board)
        opp_score = eval7.evaluate(opp + sim_board)

        if our_score > opp_score:
            wins += 1
        elif our_score == opp_score:
            ties += 1
        iters += 1

    if iters == 0:
        return 0.5
    return (wins + 0.5 * ties) / iters


def preflop_bucket(your_cards):
    """
    Very small preflop abstraction:
    0 = trash, 1 = medium, 2 = strong, 3 = premium
    """
    c1, c2 = your_cards
    r1, s1 = c1[0], c1[1]
    r2, s2 = c2[0], c2[1]
    i1, i2 = RANKS.index(r1), RANKS.index(r2)
    high = max(i1, i2)
    low = min(i1, i2)
    pair = (r1 == r2)
    suited = (s1 == s2)
    gap = abs(i1 - i2) - 1

    if pair and high >= RANKS.index("T"):
        return 3
    if pair and high >= RANKS.index("7"):
        return 2
    if high >= RANKS.index("A") and low >= RANKS.index("T"):
        return 2
    if suited and gap <= 1 and high >= RANKS.index("9"):
        return 2
    if high >= RANKS.index("Q") and low >= RANKS.index("7"):
        return 1
    if suited and gap <= 2 and high >= RANKS.index("7"):
        return 1
    return 0


# simple "blueprint" for preflop: [fold, call, raise] probabilities
PREFLOP_STRAT = {
    0: [0.7, 0.25, 0.05],  # trash
    1: [0.2, 0.55, 0.25],  # medium
    2: [0.05, 0.35, 0.60], # strong
    3: [0.0, 0.15, 0.85],  # premium
}


def postflop_bucket(equity):
    """
    Tiny postflop abstraction:
    0 = weak, 1 = medium, 2 = strong, 3 = nutted
    """
    if equity < 0.30:
        return 0
    if equity < 0.55:
        return 1
    if equity < 0.75:
        return 2
    return 3


# postflop blueprint: [check/fold, call, bet/raise]
POSTFLOP_STRAT = {
    0: [0.8, 0.18, 0.02],
    1: [0.35, 0.55, 0.10],
    2: [0.10, 0.45, 0.45],
    3: [0.02, 0.18, 0.80],
}


def sample_from_probs(probs):
    r = random.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += float(p)
        if r <= cum:
            return i
    return len(probs) - 1


def decide(state: dict) -> dict:
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

    if street == "preflop":
        bucket = preflop_bucket(your_cards)
        probs = PREFLOP_STRAT[bucket]
        act_idx = sample_from_probs(probs)
        # 0 = fold, 1 = call/limp, 2 = raise
        if act_idx == 0:
            if to_call > 0:
                return {"action": "fold"}
            return {"action": "check"} if can_check else {"action": "fold"}
        if act_idx == 1:
            if to_call == 0:
                return {"action": "check"}
            return {"action": "call"}
        if act_idx == 2:
            # simple size: 3x open / 3x raise
            if to_call == 0:
                base = max(min_raise_to, state["current_bet"] + int(pot * 0.75) + 2)
            else:
                base = max(min_raise_to, state["current_bet"] + int(3 * to_call))
            if base >= state["current_bet"] + stack * 0.95:
                return {"action": "all_in"}
            return {"action": "raise", "amount": base}

    # postflop: use equity -> bucket -> blueprint
    equity = estimate_equity_timed(your_cards, board, time_limit=1.20)
    bucket = postflop_bucket(equity)
    probs = POSTFLOP_STRAT[bucket]
    act_idx = sample_from_probs(probs)
    # 0 = check/fold, 1 = call, 2 = bet/raise

    if to_call > 0:
        if act_idx == 0:
            return {"action": "fold"}
        if act_idx == 1:
            return {"action": "call"}
        if act_idx == 2:
            min_total = max(min_raise_to, state["current_bet"] + int(pot * 0.7))
            max_total = max(min_total, state["current_bet"] + int(pot * 1.3))
            alpha = random.random()
            amount = int(min_total + alpha * (max_total - min_total))
            if amount >= state["current_bet"] + stack * 0.95:
                return {"action": "all_in"}
            return {"action": "raise", "amount": amount}
    else:
        if can_check:
            if act_idx == 0:
                return {"action": "check"}
            if act_idx == 1:
                return {"action": "check"}  # no bet to call
            if act_idx == 2:
                min_total = max(min_raise_to, state["current_bet"] + int(pot * 0.5))
                max_total = max(min_total, state["current_bet"] + int(pot * 1.0))
                alpha = random.random()
                amount = int(min_total + alpha * (max_total - min_total))
                if amount >= state["current_bet"] + stack * 0.95:
                    return {"action": "all_in"}
                return {"action": "raise", "amount": amount}
        return {"action": "call"}
