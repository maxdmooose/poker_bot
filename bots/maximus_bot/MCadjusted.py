# bot.py — Correct eval7-based hand evaluator + safe Monte Carlo

import random
import time
import eval7

RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK = [r + s for r in RANKS for s in SUITS]

# Pre-allocate eval7.Card objects for speed
EVAL7_DECK = [eval7.Card(c) for c in FULL_DECK]


# ------------------ CARD ENCODING ------------------

def card_to_int(card):
    return RANKS.index(card[0]) + 13 * SUITS.index(card[1])


# ------------------ PREFLOP APPROX ------------------

PREFLOP_TABLE = {
    (("A", "A"), False): 0.85,
    (("K", "K"), False): 0.82,
    (("Q", "Q"), False): 0.80,
    (("J", "J"), False): 0.78,
    (("T", "T"), False): 0.76,
    (("A", "K"), True): 0.66,
    (("A", "K"), False): 0.63,
    (("A", "Q"), True): 0.64,
    (("A", "Q"), False): 0.60,
}

def preflop_equity_approx(hole, n_opps):
    r1, s1 = hole[0][0], hole[0][1]
    r2, s2 = hole[1][0], hole[1][1]
    suited = (s1 == s2)
    key = tuple(sorted((r1, r2), key=lambda x: RANKS.index(x))), suited
    base = PREFLOP_TABLE.get(key, 0.50)
    adj = base - 0.03 * max(0, n_opps - 1)
    return max(0.05, min(0.95, adj))


# ------------------ EVAL7 HAND EVALUATION ------------------

def eval7_score(cards_int):
    """cards_int: list of 7 ints"""
    cards = [EVAL7_DECK[i] for i in cards_int]
    return eval7.evaluate(cards)   # HIGHER = STRONGER (correct)


# ------------------ SAFE, CORRECT MONTE CARLO ------------------

def estimate_equity_multi_timed(your_cards, board_cards, n_opponents, time_limit):
    start = time.perf_counter()

    hero_int = [card_to_int(c) for c in your_cards]
    board_int = [card_to_int(c) for c in board_cards]

    # Remove dead cards using INTs (correct)
    dead = {card_to_int(c) for c in your_cards + board_cards}
    deck_int = [card_to_int(c) for c in FULL_DECK if card_to_int(c) not in dead]
    deck_len = len(deck_int)

    need = 5 - len(board_cards)
    if n_opponents * 2 + need > deck_len:
        return 0.5

    wins = ties = 0

    while time.perf_counter() - start < time_limit:
        # sample all needed cards at once
        idxs = random.sample(range(deck_len), 2 * n_opponents + need)

        # build simulated board
        sim_board = board_int[:]
        k = 2 * n_opponents
        for i in range(need):
            sim_board.append(deck_int[idxs[k]])
            k += 1

        # hero 7-card hand
        hero_buf = [hero_int[0], hero_int[1], *sim_board]
        hero_score = eval7_score(hero_buf)

        # evaluate opponents
        better = equal = 0
        k = 0
        for _ in range(n_opponents):
            opp1 = deck_int[idxs[k]]
            opp2 = deck_int[idxs[k+1]]
            k += 2

            opp_buf = [opp1, opp2, *sim_board]
            opp_score = eval7_score(opp_buf)

            if opp_score > hero_score:
                better += 1
            elif opp_score == hero_score:
                equal += 1

        if better == 0 and equal == 0:
            wins += 1
        elif better == 0:
            ties += 1

    total = wins + ties
    if total == 0:
        return 0.5

    return (wins + 0.5 * ties) / total


# ------------------ BOARD TEXTURE ------------------

def board_texture_features(board_cards):
    if not board_cards:
        return {"connected": False, "very_connected": False, "flushy": False, "paired": False}

    ranks = [c[0] for c in board_cards]
    suits = [c[1] for c in board_cards]

    idxs = sorted([RANKS.index(r) for r in ranks])
    gaps = [idxs[i+1] - idxs[i] for i in range(len(idxs)-1)]
    min_gap = min(gaps) if gaps else 10

    flush_counts = {s: suits.count(s) for s in SUITS}
    paired = len(set(ranks)) < len(ranks)

    return {
        "connected": min_gap <= 2,
        "very_connected": min_gap == 1,
        "flushy": max(flush_counts.values()) >= 3,
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
    return 0.7 + 0.6 * (raises / total)


# ------------------ MAIN STRATEGY ------------------

def decide(state):
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

        n_opponents = max(1, sum(1 for p in state["players"] if not p.get("has_folded", False)) - 1)

        # EQUITY ESTIMATION
        if street == "preflop":
            equity = preflop_equity_approx(your_cards, n_opponents)
        elif street == "flop":
            equity = estimate_equity_multi_timed(your_cards, board, n_opponents, 0.9)
        elif street == "turn":
            equity = estimate_equity_multi_timed(your_cards, board, n_opponents, 1.2)
        else:
            equity = estimate_equity_multi_timed(your_cards, board, n_opponents, 1.5)

        # POT ODDS
        pot_odds = to_call / (pot + to_call) if to_call > 0 else 0.0

        # THRESHOLDS
        margin = 0.03
        base_call = pot_odds + margin
        base_raise = {"preflop": 0.62, "flop": 0.60, "turn": 0.62, "river": 0.66}[street]

        agg = opponent_aggression_factor(state)
        call_thresh = max(0.05, min(0.95, base_call * (1 - 0.15 * (agg - 1))))
        raise_thresh = max(0.05, min(0.95, base_raise * (1 - 0.10 * (agg - 1))))

        texture = board_texture_features(board)
        dry = not texture["connected"] and not texture["flushy"] and not texture["paired"]
        wet = texture["connected"] or texture["flushy"]

        r = random.random()

        # ------------------ FACING A BET ------------------
        if to_call > 0:
            if equity < call_thresh:
                if pot_odds < 0.15 and equity > pot_odds and r < 0.25:
                    return {"action": "call"}
                if pot_odds < 0.20 and equity > pot_odds and agg > 1.1 and r < 0.35:
                    return {"action": "call"}
                return {"action": "fold"}

            if equity >= raise_thresh and stack > to_call * 2:
                freq = (equity - raise_thresh) / 0.2
                if dry: freq += 0.10
                if wet: freq -= 0.08
                if agg < 0.9: freq += 0.08
                if agg > 1.1: freq -= 0.05
                freq = min(0.9, max(0.15, freq))

                if r < freq:
                    mult_min = 1.5 if wet else 1.8
                    mult_max = 2.2 if wet else 2.7
                    min_total = max(min_raise_to, int(state["current_bet"] + to_call * mult_min))
                    max_total = min(int(state["current_bet"] + to_call * mult_max), int(state["current_bet"] + stack))
                    amount = min_total if max_total <= min_total else int(min_total + random.random() * (max_total - min_total))
                    if amount >= state["current_bet"] + stack * 0.95:
                        return {"action": "all_in"}
                    return {"action": "raise", "amount": amount}

            return {"action": "call"}

        # ------------------ NO BET TO CALL ------------------
        if can_check:
            if equity >= raise_thresh:
                freq = 0.85
                if dry: freq += 0.05
                if agg < 0.9: freq += 0.05
                freq = min(0.95, freq)

                if r < freq:
                    if dry: min_mult, max_mult = 0.45, 1.0
                    elif wet: min_mult, max_mult = 0.55, 1.2
                    else: min_mult, max_mult = 0.5, 1.1

                    min_total = max(min_raise_to, state["current_bet"] + int(pot * min_mult))
                    max_total = max(min_total, state["current_bet"] + int(pot * max_mult))
                    amount = int(min_total + random.random() * (max_total - min_total))

                    if amount >= state["current_bet"] + stack * 0.95:
                        return {"action": "all_in"}
                    return {"action": "raise", "amount": amount}

                return {"action": "check"}

            # bluffing
            if 0.18 <= equity <= 0.42 and street in ("flop", "turn"):
                freq = 0.18
                if dry: freq += 0.06
                if wet: freq -= 0.04
                if texture["paired"]: freq += 0.04
                if agg < 0.9: freq += 0.05
                if agg > 1.1: freq -= 0.05
                freq = max(0.05, min(0.30, freq))

                if r < freq:
                    if dry: min_mult, max_mult = 0.45, 0.75
                    else: min_mult, max_mult = 0.55, 0.90

                    min_total = max(min_raise_to, state["current_bet"] + int(pot * min_mult))
                    max_total = max(min_total, state["current_bet"] + int(pot * max_mult))
                    amount = int(min_total + random.random() * (max_total - min_total))

                    if amount >= state["current_bet"] + stack * 0.95:
                        return {"action": "all_in"}
                    return {"action": "raise", "amount": amount}

            return {"action": "check"}

        return {"action": "call"}

    except Exception:
        return {"action": "check"} if state.get("can_check", False) else {"action": "call"}
