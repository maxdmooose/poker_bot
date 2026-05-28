# bot.py — Hybrid CFR + Monte Carlo + range-estimation bot for Fullhouse
# - Preflop: CFR-style bucketed blueprint with regret updates
# - Postflop: bucketed CFR blueprint + Monte Carlo equity + shallow search
# - Villain range estimation via simple bucket frequencies
# - Uses ~2s per decision, with safety margins

import random
import time
import eval7
from collections import defaultdict

RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK = [r + s for r in RANKS for s in SUITS]


# ------------------ SAFE EVAL7 WRAPPER ------------------

def safe_eval(cards):
    try:
        if len(cards) < 5:
            return 0
        return eval7.evaluate(cards[:7])
    except Exception:
        return 0


# ------------------ MONTE CARLO EQUITY ------------------

def estimate_equity_vs_range_timed(your_cards, board_cards, villain_range, n_opponents, time_limit, rng):
    """
    Monte Carlo equity vs n_opponents, where each opponent is sampled from villain_range (bucketed).
    villain_range: list of (hand_str, weight) or None -> random.
    """
    start = time.perf_counter()

    your_eval = [eval7.Card(c) for c in your_cards]
    board_eval = [eval7.Card(c) for c in board_cards]

    dead = set(your_cards + board_cards)
    deck = [c for c in FULL_DECK if c not in dead]
    deck_eval = [eval7.Card(c) for c in deck]

    wins = ties = 0
    iters = 0
    board_needed = 5 - len(board_cards)

    if n_opponents * 2 + max(0, board_needed) > len(deck_eval):
        return 0.5

    # Precompute villain hands if range is given
    if villain_range:
        vr_hands, vr_weights = zip(*villain_range)
        total_w = sum(vr_weights)
        vr_probs = [w / total_w for w in vr_weights]
    else:
        vr_hands = vr_probs = None

    while time.perf_counter() - start < time_limit:
        rng.shuffle(deck_eval)
        idx = 0

        opp_hands = []
        used = set()
        for _ in range(n_opponents):
            if villain_range:
                # sample a hand from villain_range, ensure no card conflict
                for _try in range(20):
                    h = rng.choices(vr_hands, weights=vr_probs, k=1)[0]
                    c1, c2 = h[:2], h[2:]
                    if c1 not in dead and c2 not in dead and c1 not in used and c2 not in used:
                        opp_hands.append([eval7.Card(c1), eval7.Card(c2)])
                        used.add(c1); used.add(c2)
                        break
                else:
                    # fallback: random from deck_eval
                    opp_hands.append(deck_eval[idx:idx+2])
                    idx += 2
            else:
                opp_hands.append(deck_eval[idx:idx+2])
                idx += 2

        sim_board = board_eval[:]
        if board_needed > 0:
            sim_board = sim_board + deck_eval[idx:idx + board_needed]

        our_score = safe_eval(your_eval + sim_board)

        better = equal = 0
        for opp in opp_hands:
            opp_score = safe_eval(opp + sim_board)
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


# ------------------ BUCKETING ------------------

def preflop_bucket(your_cards):
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


def postflop_bucket(equity, tex):
    """
    Postflop bucket by equity + texture.
    0 = weak, 1 = medium, 2 = strong, 3 = nutted
    """
    if equity < 0.25:
        return 0
    if equity < 0.55:
        return 1
    if equity < 0.80:
        return 2
    return 3


# ------------------ CFR BLUEPRINTS + REGRETS ------------------

# Actions: 0 = fold/check, 1 = call, 2 = bet/raise
PREFLOP_STRAT = {
    0: [0.7, 0.25, 0.05],
    1: [0.2, 0.55, 0.25],
    2: [0.05, 0.35, 0.60],
    3: [0.0, 0.15, 0.85],
}

POSTFLOP_STRAT = {
    0: [0.8, 0.18, 0.02],
    1: [0.35, 0.55, 0.10],
    2: [0.10, 0.45, 0.45],
    3: [0.02, 0.18, 0.80],
}

# Regrets and strategy sums (global, updated over match)
PREFLOP_REGRETS = defaultdict(lambda: [0.0, 0.0, 0.0])
PREFLOP_STRAT_SUM = defaultdict(lambda: [0.0, 0.0, 0.0])

POSTFLOP_REGRETS = defaultdict(lambda: [0.0, 0.0, 0.0])
POSTFLOP_STRAT_SUM = defaultdict(lambda: [0.0, 0.0, 0.0])


def regret_matching(regrets, base_strat):
    """Convert regrets + base blueprint into a mixed strategy."""
    pos_regrets = [max(r, 0.0) for r in regrets]
    s = sum(pos_regrets)
    if s > 1e-9:
        return [r / s for r in pos_regrets]
    # fallback to blueprint
    return base_strat[:]


def sample_from_probs(probs, rng):
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += float(p)
        if r <= cum:
            return i
    return len(probs) - 1


# ------------------ SIMPLE VILLAIN RANGE MODEL ------------------

# Track villain preflop bucket frequencies by position (we'll just use a single pool)
VILLAIN_PREFLOP_BUCKET_COUNTS = [1.0, 1.0, 1.0, 1.0]  # start uniform


def update_villain_preflop_bucket(observed_bucket):
    if 0 <= observed_bucket <= 3:
        VILLAIN_PREFLOP_BUCKET_COUNTS[observed_bucket] += 1.0


def build_villain_range_from_buckets():
    """
    Build a crude villain range as a list of (hand_str, weight) using bucket frequencies.
    For speed, we just sample a representative subset of hands per bucket.
    """
    weights = VILLAIN_PREFLOP_BUCKET_COUNTS
    total = sum(weights)
    if total <= 0:
        weights = [1.0, 1.0, 1.0, 1.0]
        total = 4.0
    bucket_probs = [w / total for w in weights]

    # Pre-generate some representative hands per bucket
    bucket_hands = {0: [], 1: [], 2: [], 3: []}
    for c1 in FULL_DECK:
        for c2 in FULL_DECK:
            if c1 >= c2:
                continue
            b = preflop_bucket([c1, c2])
            if len(bucket_hands[b]) < 80:  # cap per bucket
                bucket_hands[b].append(c1 + c2)

    villain_range = []
    for b in range(4):
        if not bucket_hands[b]:
            continue
        w = bucket_probs[b]
        per_hand_w = w / len(bucket_hands[b])
        for h in bucket_hands[b]:
            villain_range.append((h, per_hand_w))
    return villain_range


# ------------------ POSTFLOP CFR DECISION ------------------

def postflop_cfr_action(state, your_cards, board, n_opps, time_left, rng):
    pot = float(state["pot"])
    stack = float(state["your_stack"])
    to_call = float(state["amount_owed"])
    can_check = state["can_check"]
    min_raise_to = int(state["min_raise_to"])
    current_bet = float(state["current_bet"])

    villain_range = build_villain_range_from_buckets()

    # equity vs villain range
    eq_time = min(0.5, max(0.1, time_left() * 0.4))
    equity = estimate_equity_vs_range_timed(
        your_cards,
        board,
        villain_range=villain_range,
        n_opponents=n_opps,
        time_limit=eq_time,
        rng=rng
    )

    tex = board_texture(board)
    bucket = postflop_bucket(equity, tex)
    key = (state["street"], bucket)

    base_strat = POSTFLOP_STRAT[bucket]
    regrets = POSTFLOP_REGRETS[key]
    strat = regret_matching(regrets, base_strat)

    act_idx = sample_from_probs(strat, rng)
    # 0 = fold/check, 1 = call, 2 = bet/raise

    # --- Compute simple action EVs for regret update ---
    # We'll approximate EVs using equity and pot odds, not full tree search (time).

    # fold/check EV
    if to_call > 0:
        ev_fold = 0.0
    else:
        ev_fold = 0.0  # check EV baseline

    # call EV
    if to_call > 0:
        win_pot = pot + to_call
        ev_call = equity * win_pot - (1 - equity) * to_call
    else:
        ev_call = 0.0  # checking behind

    # bet/raise EV (simple model)
    if to_call > 0:
        # raise over bet
        mult = 2.0 if tex["dry"] else 1.6
        total = max(min_raise_to, int(current_bet + to_call * mult))
        total = min(total, int(current_bet + stack))
        if total <= current_bet + to_call:
            ev_raise = ev_call
        else:
            bet_size = total - current_bet
            risk = bet_size
            pot_if_called = pot + bet_size + to_call
            fold_freq = 0.45 if tex["dry"] else 0.30
            call_freq = 1.0 - fold_freq
            win_ev = equity * pot_if_called - (1 - equity) * risk
            ev_raise = fold_freq * (pot + to_call) + call_freq * win_ev
    else:
        # bet from check
        mult = 0.7 if tex["dry"] else 0.9
        total = max(min_raise_to, int(current_bet + pot * mult))
        total = min(total, int(current_bet + stack))
        if total <= current_bet:
            ev_raise = ev_call
        else:
            bet_size = total - current_bet
            risk = bet_size
            pot_if_called = pot + bet_size
            fold_freq = 0.40 if tex["dry"] else 0.30
            call_freq = 1.0 - fold_freq
            win_ev = equity * pot_if_called - (1 - equity) * risk
            ev_raise = fold_freq * pot + call_freq * win_ev

    evs = [ev_fold, ev_call, ev_raise]
    strat_ev = sum(p * e for p, e in zip(strat, evs))

    # CFR regret update
    for a in range(3):
        POSTFLOP_REGRETS[key][a] += evs[a] - strat_ev
        POSTFLOP_STRAT_SUM[key][a] += strat[a]

    # --- Map chosen action to engine action ---
    if to_call > 0:
        if act_idx == 0:
            return {"action": "fold"}
        if act_idx == 1:
            return {"action": "call"}
        if act_idx == 2:
            mult = 2.0 if tex["dry"] else 1.6
            total = max(min_raise_to, int(current_bet + to_call * mult))
            total = min(total, int(current_bet + stack))
            if total >= current_bet + stack * 0.95:
                return {"action": "all_in"}
            return {"action": "raise", "amount": total}
    else:
        if can_check:
            if act_idx == 0:
                return {"action": "check"}
            if act_idx == 1:
                return {"action": "check"}
            if act_idx == 2:
                mult = 0.7 if tex["dry"] else 0.9
                total = max(min_raise_to, int(current_bet + pot * mult))
                total = min(total, int(current_bet + stack))
                if total >= current_bet + stack * 0.95:
                    return {"action": "all_in"}
                return {"action": "raise", "amount": total}
        # weird case: can't check but to_call == 0
        return {"action": "call"}


# ------------------ MAIN DECIDE ------------------

def decide(state: dict) -> dict:
    start = time.perf_counter()
    BUDGET = 1.95

    def time_left():
        return BUDGET - (time.perf_counter() - start)

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

        rng = random.Random()

        # -------- PREFLOP CFR with regret updating --------
        if street == "preflop":
            bucket = preflop_bucket(your_cards)
            key = bucket
            base_strat = PREFLOP_STRAT[bucket]
            regrets = PREFLOP_REGRETS[key]
            strat = regret_matching(regrets, base_strat)

            act_idx = sample_from_probs(strat, rng)
            # approximate EVs for regret update using equity vs random
            eq_time = min(0.4, max(0.1, time_left() * 0.4))
            equity = estimate_equity_vs_range_timed(
                your_cards,
                board,
                villain_range=None,
                n_opponents=n_opps,
                time_limit=eq_time,
                rng=rng
            )

            # fold/check EV
            if to_call > 0:
                ev_fold = 0.0
            else:
                ev_fold = 0.0

            # call EV
            if to_call > 0:
                win_pot = pot + to_call
                ev_call = equity * win_pot - (1 - equity) * to_call
            else:
                ev_call = 0.0

            # raise EV (simple)
            if to_call > 0:
                base = max(min_raise_to, int(state["current_bet"] + 3 * to_call))
            else:
                base = max(min_raise_to, int(pot * 0.75) + 2)
            base = min(base, int(state["current_bet"] + stack))
            if base <= state["current_bet"]:
                ev_raise = ev_call
            else:
                risk = base - state["current_bet"]
                pot_if_called = pot + risk + to_call
                fold_freq = 0.35
                call_freq = 0.65
                win_ev = equity * pot_if_called - (1 - equity) * risk
                ev_raise = fold_freq * (pot + to_call) + call_freq * win_ev

            evs = [ev_fold, ev_call, ev_raise]
            strat_ev = sum(p * e for p, e in zip(strat, evs))

            for a in range(3):
                PREFLOP_REGRETS[key][a] += evs[a] - strat_ev
                PREFLOP_STRAT_SUM[key][a] += strat[a]

            # map action
            if act_idx == 0:
                if to_call > 0:
                    return {"action": "fold"}
                return {"action": "check"} if can_check else {"action": "fold"}
            if act_idx == 1:
                if to_call == 0:
                    return {"action": "check"}
                return {"action": "call"}
            if act_idx == 2:
                amount = base
                if amount >= state["current_bet"] + stack * 0.95:
                    return {"action": "all_in"}
                return {"action": "raise", "amount": amount}

        # -------- POSTFLOP CFR blueprint + range-based equity --------
        action = postflop_cfr_action(state, your_cards, board, n_opps, time_left, rng)
        return action

    except Exception:
        if state.get("can_check", False) and state.get("amount_owed", 0) == 0:
            return {"action": "check"}
        return {"action": "call"}
