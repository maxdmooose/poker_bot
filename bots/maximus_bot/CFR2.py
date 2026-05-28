# bot.py — Practical CFR++ bot for Fullhouse
# - Multi-street information sets (IS)
# - Preflop + postflop CFR with regret matching
# - 5-action postflop abstraction (check/fold, call, small/med/large bet)
# - Villain range narrowing by bucket frequencies
# - Monte Carlo equity vs villain range
# - Fits in ~2s per decision

import random
import time
import eval7
from collections import defaultdict

RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK = [r + s for r in RANKS for s in SUITS]

# ------------------ SAFE EVAL ------------------

def safe_eval(cards):
    try:
        if len(cards) < 5:
            return 0
        return eval7.evaluate(cards[:7])
    except Exception:
        return 0

# ------------------ BUCKETING ------------------

def preflop_bucket(cards):
    c1, c2 = cards
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
    return {"dry": not wet and not paired, "wet": wet, "paired": paired}

def postflop_bucket(equity, tex):
    if equity < 0.25:
        return 0
    if equity < 0.55:
        return 1
    if equity < 0.80:
        return 2
    return 3

def pot_class(pot, stack):
    r = pot / max(1.0, stack)
    if r < 0.5:
        return 0
    if r < 1.5:
        return 1
    return 2

def action_class(state):
    log = state.get("action_log", [])
    raises = sum(1 for e in log if e.get("action") in ("raise", "bet"))
    if raises == 0:
        return 0
    if raises == 1:
        return 1
    return 2

def position_class(state):
    # crude: if we're first to act postflop, treat as OOP; else IP
    # engine doesn't give seat index, so approximate via action_log length
    log = state.get("action_log", [])
    return 0 if len(log) == 0 else 1  # 0 = OOP, 1 = IP

# ------------------ MONTE CARLO VS RANGE ------------------

def estimate_equity_vs_range_timed(your_cards, board_cards, villain_range, n_opps, time_limit, rng):
    start = time.perf_counter()
    yc = [eval7.Card(c) for c in your_cards]
    bc = [eval7.Card(c) for c in board_cards]
    dead = set(your_cards + board_cards)
    deck = [c for c in FULL_DECK if c not in dead]
    deck_eval = [eval7.Card(c) for c in deck]
    wins = ties = 0
    iters = 0
    need = 5 - len(board_cards)
    if n_opps * 2 + max(0, need) > len(deck_eval):
        return 0.5

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
        for _ in range(n_opps):
            if villain_range:
                for _try in range(20):
                    h = rng.choices(vr_hands, weights=vr_probs, k=1)[0]
                    c1, c2 = h[:2], h[2:]
                    if c1 not in dead and c2 not in dead and c1 not in used and c2 not in used:
                        opp_hands.append([eval7.Card(c1), eval7.Card(c2)])
                        used.add(c1); used.add(c2)
                        break
                else:
                    opp_hands.append(deck_eval[idx:idx+2])
                    idx += 2
            else:
                opp_hands.append(deck_eval[idx:idx+2])
                idx += 2

        sim_board = bc[:]
        if need > 0:
            sim_board += deck_eval[idx:idx+need]

        our_score = safe_eval(yc + sim_board)
        better = equal = 0
        for opp in opp_hands:
            os = safe_eval(opp + sim_board)
            if os > our_score:
                better += 1
            elif os == our_score:
                equal += 1
        if better == 0 and equal == 0:
            wins += 1
        elif better == 0:
            ties += 1
        iters += 1

    if iters == 0:
        return 0.5
    return (wins + 0.5 * ties) / iters

# ------------------ CFR STRUCTURES ------------------

# Actions: 0 = fold/check, 1 = call, 2 = small, 3 = medium, 4 = large
PREFLOP_BLUEPRINT = {
    0: [0.75, 0.20, 0.05, 0.0, 0.0],
    1: [0.25, 0.50, 0.20, 0.05, 0.0],
    2: [0.05, 0.35, 0.35, 0.20, 0.05],
    3: [0.0, 0.15, 0.35, 0.30, 0.20],
}

POSTFLOP_BLUEPRINT = {
    0: [0.85, 0.10, 0.03, 0.015, 0.005],
    1: [0.40, 0.45, 0.10, 0.04, 0.01],
    2: [0.10, 0.40, 0.25, 0.20, 0.05],
    3: [0.02, 0.10, 0.25, 0.35, 0.28],
}

REGRETS = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
STRAT_SUM = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])

# villain bucket counts: (street, action_class) -> [b0..b3]
VILLAIN_BUCKET_COUNTS = defaultdict(lambda: [1.0, 1.0, 1.0, 1.0])

def regret_matching(regrets, blueprint):
    pos = [max(r, 0.0) for r in regrets]
    s = sum(pos)
    if s > 1e-9:
        return [r / s for r in pos]
    return blueprint[:]

def sample_from_probs(probs, rng):
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += float(p)
        if r <= cum:
            return i
    return len(probs) - 1

def build_villain_range(street, act_cls):
    counts = VILLAIN_BUCKET_COUNTS[(street, act_cls)]
    total = sum(counts)
    if total <= 0:
        counts = [1.0, 1.0, 1.0, 1.0]
        total = 4.0
    probs = [c / total for c in counts]
    bucket_hands = {0: [], 1: [], 2: [], 3: []}
    for i, c1 in enumerate(FULL_DECK):
        for c2 in FULL_DECK[i+1:]:
            b = preflop_bucket([c1, c2])
            if len(bucket_hands[b]) < 60:
                bucket_hands[b].append(c1 + c2)
    vr = []
    for b in range(4):
        if not bucket_hands[b]:
            continue
        w = probs[b]
        per = w / len(bucket_hands[b])
        for h in bucket_hands[b]:
            vr.append((h, per))
    return vr

# ------------------ CFR DECISION HELPERS ------------------

def preflop_cfr_decide(state, your_cards, n_opps, time_left, rng):
    pot = float(state["pot"])
    stack = float(state["your_stack"])
    to_call = float(state["amount_owed"])
    can_check = state["can_check"]
    current_bet = float(state["current_bet"])
    min_raise_to = int(state["min_raise_to"])

    bucket = preflop_bucket(your_cards)
    pc = pot_class(pot, stack)
    ac = action_class(state)
    pos = position_class(state)
    IS = ("preflop", bucket, pc, ac, pos)

    blueprint = PREFLOP_BLUEPRINT[bucket]
    regrets = REGRETS[IS]
    strat = regret_matching(regrets, blueprint)

    # approximate EVs via equity vs random
    eq_time = min(0.4, max(0.1, time_left() * 0.4))
    equity = estimate_equity_vs_range_timed(
        your_cards, [], None, n_opps, eq_time, rng
    )

    # EVs for 5 actions
    # 0: fold/check
    ev0 = 0.0
    # 1: call
    if to_call > 0:
        win_pot = pot + to_call
        ev1 = equity * win_pot - (1 - equity) * to_call
    else:
        ev1 = 0.0
    # 2/3/4: small/med/large raise
    evs = [ev0, ev1, ev1, ev1, ev1]
    for idx, mult in zip([2, 3, 4], [2.5, 3.5, 4.5]):
        if to_call > 0:
            total = max(min_raise_to, int(current_bet + mult * to_call))
        else:
            total = max(min_raise_to, int(pot * (0.5 + 0.2 * (idx-2))) + 2)
        total = min(total, int(current_bet + stack))
        if total <= current_bet:
            evs[idx] = ev1
        else:
            risk = total - current_bet
            pot_if_called = pot + risk + to_call
            fold_freq = 0.35
            call_freq = 0.65
            win_ev = equity * pot_if_called - (1 - equity) * risk
            evs[idx] = fold_freq * (pot + to_call) + call_freq * win_ev

    strat_ev = sum(p * e for p, e in zip(strat, evs))
    for a in range(5):
        REGRETS[IS][a] += evs[a] - strat_ev
        STRAT_SUM[IS][a] += strat[a]

    act_idx = sample_from_probs(strat, rng)

    # map to engine action
    if act_idx == 0:
        if to_call > 0:
            return {"action": "fold"}
        return {"action": "check"} if can_check else {"action": "fold"}
    if act_idx == 1:
        if to_call == 0:
            return {"action": "check"}
        return {"action": "call"}
    # raises
    if to_call > 0:
        mults = {2: 2.5, 3: 3.5, 4: 4.5}
        mult = mults.get(act_idx, 3.0)
        total = max(min_raise_to, int(current_bet + mult * to_call))
    else:
        mults = {2: 0.6, 3: 0.9, 4: 1.2}
        mult = mults.get(act_idx, 0.8)
        total = max(min_raise_to, int(current_bet + pot * mult))
    total = min(total, int(current_bet + stack))
    if total >= current_bet + stack * 0.95:
        return {"action": "all_in"}
    return {"action": "raise", "amount": total}

def postflop_cfr_decide(state, your_cards, board, n_opps, time_left, rng):
    pot = float(state["pot"])
    stack = float(state["your_stack"])
    to_call = float(state["amount_owed"])
    can_check = state["can_check"]
    current_bet = float(state["current_bet"])
    min_raise_to = int(state["min_raise_to"])
    street = state["street"]

    pc = pot_class(pot, stack)
    ac = action_class(state)
    pos = position_class(state)
    tex = board_texture(board)

    villain_range = build_villain_range(street, ac)
    eq_time = min(0.5, max(0.1, time_left() * 0.4))
    equity = estimate_equity_vs_range_timed(
        your_cards, board, villain_range, n_opps, eq_time, rng
    )
    bucket = postflop_bucket(equity, tex)
    IS = (street, bucket, pc, ac, pos)

    blueprint = POSTFLOP_BLUEPRINT[bucket]
    regrets = REGRETS[IS]
    strat = regret_matching(regrets, blueprint)

    # EVs
    # 0: fold/check
    ev0 = 0.0
    # 1: call/check
    if to_call > 0:
        win_pot = pot + to_call
        ev1 = equity * win_pot - (1 - equity) * to_call
    else:
        ev1 = 0.0

    evs = [ev0, ev1, ev1, ev1, ev1]
    # 2/3/4: small/med/large bet/raise
    if to_call > 0:
        mults = {2: 2.0, 3: 2.7, 4: 3.5}
        for idx, mult in mults.items():
            total = max(min_raise_to, int(current_bet + mult * to_call))
            total = min(total, int(current_bet + stack))
            if total <= current_bet + to_call:
                evs[idx] = ev1
            else:
                risk = total - current_bet
                pot_if_called = pot + risk + to_call
                fold_freq = 0.45 if tex["dry"] else 0.30
                call_freq = 1.0 - fold_freq
                win_ev = equity * pot_if_called - (1 - equity) * risk
                evs[idx] = fold_freq * (pot + to_call) + call_freq * win_ev
    else:
        mults = {2: 0.5, 3: 0.8, 4: 1.1}
        for idx, mult in mults.items():
            total = max(min_raise_to, int(current_bet + pot * mult))
            total = min(total, int(current_bet + stack))
            if total <= current_bet:
                evs[idx] = ev1
            else:
                risk = total - current_bet
                pot_if_called = pot + risk
                fold_freq = 0.40 if tex["dry"] else 0.30
                call_freq = 1.0 - fold_freq
                win_ev = equity * pot_if_called - (1 - equity) * risk
                evs[idx] = fold_freq * pot + call_freq * win_ev

    strat_ev = sum(p * e for p, e in zip(strat, evs))
    for a in range(5):
        REGRETS[IS][a] += evs[a] - strat_ev
        STRAT_SUM[IS][a] += strat[a]

    act_idx = sample_from_probs(strat, rng)

    # map to engine action
    if to_call > 0:
        if act_idx == 0:
            return {"action": "fold"}
        if act_idx == 1:
            return {"action": "call"}
        mults = {2: 2.0, 3: 2.7, 4: 3.5}
        mult = mults.get(act_idx, 2.3)
        total = max(min_raise_to, int(current_bet + mult * to_call))
        total = min(total, int(current_bet + stack))
        if total >= current_bet + stack * 0.95:
            return {"action": "all_in"}
        return {"action": "raise", "amount": total}
    else:
        if can_check:
            if act_idx in (0, 1):
                return {"action": "check"}
            mults = {2: 0.5, 3: 0.8, 4: 1.1}
            mult = mults.get(act_idx, 0.7)
            total = max(min_raise_to, int(current_bet + pot * mult))
            total = min(total, int(current_bet + stack))
            if total >= current_bet + stack * 0.95:
                return {"action": "all_in"}
            return {"action": "raise", "amount": total}
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
        stack = float(state["your_stack"])
        to_call = float(state["amount_owed"])
        can_check = state["can_check"]

        if stack <= 0:
            return {"action": "call"} if to_call > 0 else {"action": "check"}

        n_opps = sum(1 for p in state["players"] if not p.get("has_folded", False)) - 1
        n_opps = max(1, n_opps)
        rng = random.Random()

        if street == "preflop":
            return preflop_cfr_decide(state, your_cards, n_opps, time_left, rng)
        else:
            return postflop_cfr_decide(state, your_cards, board, n_opps, time_left, rng)

    except Exception:
        if state.get("can_check", False) and state.get("amount_owed", 0) == 0:
            return {"action": "check"}
        return {"action": "call"}
