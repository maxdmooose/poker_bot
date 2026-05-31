# bot.py — Pure-Python evaluator + optimized Monte Carlo + JSON preflop ranges

import random
import time
import os
import json

RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK = [r + s for r in RANKS for s in SUITS]

# ------------------ DATA DIR + PREFLOP RANGES ------------------

DATA_DIR = os.environ.get("BOT_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
try:
    with open(os.path.join(DATA_DIR, "preflop.json"), "r") as f:
        PREFLOP = json.load(f)
except Exception:
    PREFLOP = {}


# ------------------ CARD ENCODING ------------------

def card_to_int(card):
    r = RANKS.index(card[0])
    s = SUITS.index(card[1])
    return s * 13 + r  # 0..51

def int_to_rank(i):
    return i % 13

def int_to_suit(i):
    return i // 13

# ------------------ PREFLOP APPROX EQUITY TABLE (FALLBACK) ------------------

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

# ------------------ PURE-PYTHON 7-CARD EVALUATOR ------------------

def evaluate_7cards(cards_int):
    ranks = [int_to_rank(c) for c in cards_int]
    suits = [int_to_suit(c) for c in cards_int]

    rank_counts = [0] * 13
    suit_counts = [0] * 4
    for r in ranks:
        rank_counts[r] += 1
    for s in suits:
        suit_counts[s] += 1

    unique_ranks = sorted(set(ranks), reverse=True)

    def is_flush():
        for s in range(4):
            if suit_counts[s] >= 5:
                fr = [r for (r, su) in zip(ranks, suits) if su == s]
                fr = sorted(fr, reverse=True)
                return True, fr
        return False, []

    def is_straight(rank_list):
        rl = sorted(set(rank_list))
        if set([0, 1, 2, 3, 12]).issubset(rl):
            best = 3
        else:
            best = -1
        for i in range(len(rl) - 4):
            if (rl[i+4] - rl[i] == 4 and
                rl[i+1] - rl[i] == 1 and
                rl[i+2] - rl[i+1] == 1 and
                rl[i+3] - rl[i+2] == 1):
                best = max(best, rl[i+4])
        return best

    fours = [r for r in range(13) if rank_counts[r] == 4]
    trips = [r for r in range(13) if rank_counts[r] == 3]
    pairs = [r for r in range(13) if rank_counts[r] == 2]

    is_flush_flag, flush_ranks = is_flush()
    straight_high = is_straight(ranks)
    straight_flush_high = -1
    if is_flush_flag:
        straight_flush_high = is_straight(flush_ranks)

    if straight_flush_high >= 0:
        return (8 << 20) + (straight_flush_high << 16)

    if fours:
        four = max(fours)
        kicker = max([r for r in unique_ranks if r != four])
        return (7 << 20) + (four << 16) + (kicker << 12)

    if trips:
        t1 = max(trips)
        remaining_trips_pairs = [r for r in trips + pairs if r != t1]
        if remaining_trips_pairs:
            t2 = max(remaining_trips_pairs)
            return (6 << 20) + (t1 << 16) + (t2 << 12)

    if is_flush_flag:
        top5 = flush_ranks[:5]
        score = (5 << 20)
        shift = 16
        for r in top5:
            score += r << shift
            shift -= 4
        return score

    if straight_high >= 0:
        return (4 << 20) + (straight_high << 16)

    if trips:
        t = max(trips)
        kickers = [r for r in unique_ranks if r != t][:2]
        score = (3 << 20) + (t << 16)
        shift = 12
        for r in kickers:
            score += r << shift
            shift -= 4
        return score

    if len(pairs) >= 2:
        p1, p2 = sorted(pairs, reverse=True)[:2]
        kicker = max([r for r in unique_ranks if r not in (p1, p2)])
        return (2 << 20) + (p1 << 16) + (p2 << 12) + (kicker << 8)

    if len(pairs) == 1:
        p = pairs[0]
        kickers = [r for r in unique_ranks if r != p][:3]
        score = (1 << 20) + (p << 16)
        shift = 12
        for r in kickers:
            score += r << shift
            shift -= 4
        return score

    top5 = unique_ranks[:5]
    score = 0
    shift = 16
    for r in top5:
        score += r << shift
        shift -= 4
    return score

# ------------------ OPTIMIZED MONTE CARLO ------------------

def estimate_equity_multi_timed(your_cards, board_cards, n_opponents, time_limit=1.0):
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
    board_buf = [0] * 7

    MAX_ITERS = 20000

    while time.perf_counter() - start < time_limit and iters < MAX_ITERS:
        idxs = random.sample(range(deck_len), total_needed)

        k = 0
        for i in range(2 * n_opponents):
            opp_cards[i] = deck_int[idxs[k]]
            k += 1

        sim_board = board_int[:]
        for i in range(need):
            sim_board.append(deck_int[idxs[k]])
            k += 1

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

# ------------------ PREFLOP HELPERS ------------------

def canonical_hand(hole):
    r1, s1 = hole[0][0], hole[0][1]
    r2, s2 = hole[1][0], hole[1][1]
    ranks = "".join(sorted([r1, r2], reverse=True))
    suited = "s" if s1 == s2 else "o"
    return ranks + suited

def get_position(state):
    players = state["players"]
    n = len(players)

    # try explicit hero seat if provided
    hero_seat = state.get("your_seat", None)
    if not isinstance(hero_seat, int) or not (0 <= hero_seat < n):
        # fallback: first player with is_hero flag, else seat 0
        hero_seat = 0
        for i, p in enumerate(players):
            if p.get("is_hero", False):
                hero_seat = i
                break

    # find button seat (fallback to 0 if missing)
    button_seat = 0
    for i, p in enumerate(players):
        if p.get("is_button", False):
            button_seat = i
            break

    # in 6-max: UTG is first to act after BB preflop
    # seat order: BTN, SB, BB, UTG, MP, CO (or rotated)
    # we approximate UTG as (button + 3) % n
    utg_seat = (button_seat + 3) % n
    offset = (hero_seat - utg_seat) % n

    positions = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
    return positions[offset] if offset < len(positions) else "UTG"

def infer_big_blind(state):
    """
    Very simple, robust-ish BB inference:
    - preflop: current_bet is usually the BB when no raise yet
    - otherwise fall back to min_raise_to if needed
    """
    current_bet = int(state.get("current_bet", 0))
    min_raise_to = int(state.get("min_raise_to", 0))

    if current_bet > 0:
        return current_bet
    if min_raise_to > 0:
        return min_raise_to
    return 1

def preflop_decision(state):
    """
    Range-based preflop engine:
    - open-raise using 'open'
    - 3-bet using '3bet'
    - flat vs open using 'call_vs_open'
    - BB defend using 'defend_vs_open'
    No shove-mode here: safer and more stable.
    """
    if not PREFLOP:
        return None

    your_cards = state["your_cards"]
    to_call = int(state["amount_owed"])
    current_bet = int(state["current_bet"])
    min_raise_to = int(state["min_raise_to"])
    stack = int(state["your_stack"])
    can_check = state.get("can_check", False)

    pos = get_position(state)
    ranges = PREFLOP.get(pos, {})
    hand = canonical_hand(your_cards)

    bb = infer_big_blind(state)

    # ---------- open-raise spots ----------
    # We treat as "open" when:
    # - street is preflop
    # - there is only the blind out there (current_bet == bb)
    # - hero cannot check (so not BB facing no raise)
    # - hero's to_call equals the blind (first to act or folded to)
    if state["street"] == "preflop":
        if current_bet == bb and not can_check and to_call == bb:
            # hero is first to act vs blinds (or folded to) → open-raise node
            if hand in ranges.get("open", []):
                # simple positional open sizing
                if pos in ("UTG", "MP"):
                    mult = 2.5
                elif pos == "CO":
                    mult = 2.3
                elif pos == "BTN":
                    mult = 2.2
                else:  # SB open-raise
                    mult = 3.0
                target = int(bb * mult)
                amount = max(min_raise_to, target)
                amount = min(amount, current_bet + stack)
                if amount >= current_bet + int(stack * 0.95):
                    return {"action": "all_in"}
                return {"action": "raise", "amount": amount}
            # SB limp range (optional, only if defined)
            if pos == "SB" and hand in ranges.get("limp", []):
                return {"action": "call"}
            return {"action": "fold"}

        # BB facing no raise: can_check == True, to_call == 0, current_bet == bb
        if pos == "BB" and can_check and to_call == 0 and current_bet == bb:
            # you *could* add a BB iso-raise range here; for now just check
            return {"action": "check"}

    # ---------- facing a raise (single-raise node) ----------
    if to_call > 0:
        # 3-bet range
        if hand in ranges.get("3bet", []):
            # simple 3-bet sizing: 3x IP, 4x OOP
            if pos in ("BTN", "CO"):  # likely IP vs earlier opens
                mult = 3.0
            else:
                mult = 4.0
            target = current_bet + int(to_call * mult)
            amount = max(min_raise_to, target)
            amount = min(amount, current_bet + stack)
            if amount >= current_bet + int(stack * 0.95):
                return {"action": "all_in"}
            return {"action": "raise", "amount": amount}

        # flat vs open
        if hand in ranges.get("call_vs_open", []):
            return {"action": "call"}

        # BB defend vs open
        if pos == "BB" and hand in ranges.get("defend_vs_open", []):
            return {"action": "call"}

        # if not in any range → fold
        return {"action": "fold"}

    # no clear preflop action from ranges
    return None

# ------------------ MAIN STRATEGY ------------------

def decide(state: dict) -> dict:
    if state["street"] == "preflop":
        print("DEBUG_PREFLOP_STATE:", state)

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

        # ---------- PREFLOP: use range-based logic ----------
        if street == "preflop":
            action = preflop_decision(state)
            if action is not None:
                return action
            equity = preflop_equity_approx(your_cards, n_opponents)
        else:
            # ---------- POSTFLOP: Monte Carlo equity ----------
            if street == "flop":
                equity = estimate_equity_multi_timed(
                    your_cards,
                    board,
                    n_opponents=n_opponents,
                    time_limit=0.6,
                )
            elif street == "turn":
                equity = estimate_equity_multi_timed(
                    your_cards,
                    board,
                    n_opponents=n_opponents,
                    time_limit=1.0,
                )
            else:  # river
                equity = estimate_equity_multi_timed(
                    your_cards,
                    board,
                    n_opponents=n_opponents,
                    time_limit=1.2,
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
        else:
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
