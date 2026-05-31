# bot.py — Pure-Python evaluator + optimized Monte Carlo

import random
import time

RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK = [r + s for r in RANKS for s in SUITS]


# ------------------ CARD ENCODING ------------------

def card_to_int(card):
    r = RANKS.index(card[0])
    s = SUITS.index(card[1])
    return s * 13 + r  # 0..51

def int_to_rank(i):
    return i % 13

def int_to_suit(i):
    return i // 13


# ------------------ PURE-PYTHON 7-CARD EVALUATOR ------------------
# Not “research-grade” like TwoPlusTwo, but fast and consistent.
# Hand strength: larger = stronger.

def evaluate_7cards(cards_int):
    """
    cards_int: list of 7 ints (0..51)
    returns: integer strength (higher is better)
    """
    ranks = [int_to_rank(c) for c in cards_int]
    suits = [int_to_suit(c) for c in cards_int]

    rank_counts = [0] * 13
    suit_counts = [0] * 4
    for r in ranks:
        rank_counts[r] += 1
    for s in suits:
        suit_counts[s] += 1

    # sort ranks high→low for kickers
    unique_ranks = sorted(set(ranks), reverse=True)

    # ----- helpers -----

    def is_flush():
        for s in range(4):
            if suit_counts[s] >= 5:
                # collect flush ranks
                fr = [r for (r, su) in zip(ranks, suits) if su == s]
                fr = sorted(fr, reverse=True)
                return True, fr
        return False, []

    def is_straight(rank_list):
        """Return highest straight rank or -1."""
        rl = sorted(set(rank_list))
        # wheel
        if set([0, 1, 2, 3, 12]).issubset(rl):
            best = 3  # 5-high straight
        else:
            best = -1
        # normal
        for i in range(len(rl) - 4):
            if rl[i+4] - rl[i] == 4 and rl[i+1] - rl[i] == 1 and rl[i+2] - rl[i+1] == 1 and rl[i+3] - rl[i+2] == 1:
                best = max(best, rl[i+4])
        return best

    # ----- classify -----

    # counts
    fours = [r for r in range(13) if rank_counts[r] == 4]
    trips = [r for r in range(13) if rank_counts[r] == 3]
    pairs = [r for r in range(13) if rank_counts[r] == 2]

    is_flush_flag, flush_ranks = is_flush()
    straight_high = is_straight(ranks)
    straight_flush_high = -1
    if is_flush_flag:
        straight_flush_high = is_straight(flush_ranks)

    # category ordering:
    # 8: straight flush
    # 7: four of a kind
    # 6: full house
    # 5: flush
    # 4: straight
    # 3: trips
    # 2: two pair
    # 1: one pair
    # 0: high card

    # Straight flush
    if straight_flush_high >= 0:
        return (8 << 20) + (straight_flush_high << 16)

    # Four of a kind
    if fours:
        four = max(fours)
        kicker = max([r for r in unique_ranks if r != four])
        return (7 << 20) + (four << 16) + (kicker << 12)

    # Full house
    if trips:
        t1 = max(trips)
        remaining_trips_pairs = [r for r in trips + pairs if r != t1]
        if remaining_trips_pairs:
            t2 = max(remaining_trips_pairs)
            return (6 << 20) + (t1 << 16) + (t2 << 12)

    # Flush
    if is_flush_flag:
        top5 = flush_ranks[:5]
        score = (5 << 20)
        shift = 16
        for r in top5:
            score += r << shift
            shift -= 4
        return score

    # Straight
    if straight_high >= 0:
        return (4 << 20) + (straight_high << 16)

    # Trips
    if trips:
        t = max(trips)
        kickers = [r for r in unique_ranks if r != t][:2]
        score = (3 << 20) + (t << 16)
        shift = 12
        for r in kickers:
            score += r << shift
            shift -= 4
        return score

    # Two pair
    if len(pairs) >= 2:
        p1, p2 = sorted(pairs, reverse=True)[:2]
        kicker = max([r for r in unique_ranks if r not in (p1, p2)])
        return (2 << 20) + (p1 << 16) + (p2 << 12) + (kicker << 8)

    # One pair
    if len(pairs) == 1:
        p = pairs[0]
        kickers = [r for r in unique_ranks if r != p][:3]
        score = (1 << 20) + (p << 16)
        shift = 12
        for r in kickers:
            score += r << shift
            shift -= 4
        return score

    # High card
    top5 = unique_ranks[:5]
    score = 0
    shift = 16
    for r in top5:
        score += r << shift
        shift -= 4
    return score


# ------------------ OPTIMIZED MONTE CARLO ------------------

def estimate_equity_multi_timed(your_cards, board_cards, n_opponents, time_limit=1.0):
    """
    Pure-Python, int-based Monte Carlo.
    """
    start = time.perf_counter()

    hero_int = [card_to_int(c) for c in your_cards]
    board_int = [card_to_int(c) for c in board_cards]

    dead = set(your_cards + board_cards)
    deck = [c for c in FULL_DECK if c not in dead]
    deck_int = [card_to_int(c) for c in deck]
    deck_len = len(deck_int)

    wins = ties = 0
    iters = 0
    need = 5 - len(board_cards)

    if n_opponents * 2 + max(0, need) > deck_len:
        return 0.5

    total_needed = 2 * n_opponents + need
    opp_cards = [0] * (2 * n_opponents)
    board_buf = [0] * 7  # hero 2 + board 5

    MAX_ITERS = 20000

    while time.perf_counter() - start < time_limit and iters < MAX_ITERS:
        # reservoir-like sampling of indices
        idxs = random.sample(range(deck_len), total_needed)

        # opponents
        k = 0
        for i in range(2 * n_opponents):
            opp_cards[i] = deck_int[idxs[k]]
            k += 1

        # board
        sim_board = board_int[:]
        for i in range(need):
            sim_board.append(deck_int[idxs[k]])
            k += 1

        # hero 7
        board_buf[0] = hero_int[0]
        board_buf[1] = hero_int[1]
        for i in range(5):
            board_buf[2 + i] = sim_board[i]

        our_score = evaluate_7cards(board_buf)

        better = equal = 0
        k = 0
        for _ in range(n_opponents):
            board_buf[0] = opp_cards[k]
            board_buf[1] = opp_cards[k+1]
            k += 2
            opp_score = evaluate_7cards(board_buf)
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
        return 1.0
    agg = raises / total
    return 0.7 + 0.6 * agg


# ------------------ MAIN STRATEGY (YOUR ORIGINAL LOGIC) ------------------

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
            if to_call > 0:
                return {"action": "call"}
            return {"action": "check"} if can_check else {"action": "fold"}

        n_opponents = sum(1 for p in state["players"] if not p.get("has_folded", False)) - 1
        n_opponents = max(1, n_opponents)

        equity = estimate_equity_multi_timed(
            your_cards,
            board,
            n_opponents=n_opponents,
            time_limit=1.55,
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

        # ---------- No bet to call ----------
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
