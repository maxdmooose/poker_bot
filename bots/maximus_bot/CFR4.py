# bot.py — DeepStack-lite CFR + value approximation bot
# - Depth-limited lookahead with CFR-style policies at nodes
# - Value function = Monte Carlo equity vs random (or simple range)
# - Multi-action abstraction (fold/check, call, small/med/large bet)
# - Uses ~2s per decision

import random
import time
import eval7
from collections import defaultdict

RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK = [r + s for r in RANKS for s in SUITS]

def safe_eval(cards):
    try:
        if len(cards) < 5:
            return 0
        return eval7.evaluate(cards[:7])
    except Exception:
        return 0

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
    log = state.get("action_log", [])
    return 0 if len(log) == 0 else 1

REGRETS = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
STRAT_SUM = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])

BLUEPRINT = {
    "preflop": {
        0: [0.75, 0.20, 0.05, 0.0, 0.0],
        1: [0.25, 0.50, 0.20, 0.05, 0.0],
        2: [0.05, 0.35, 0.35, 0.20, 0.05],
        3: [0.0, 0.15, 0.35, 0.30, 0.20],
    },
    "postflop": {
        0: [0.85, 0.10, 0.03, 0.015, 0.005],
        1: [0.40, 0.45, 0.10, 0.04, 0.01],
        2: [0.10, 0.40, 0.25, 0.20, 0.05],
        3: [0.02, 0.10, 0.25, 0.35, 0.28],
    }
}

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

def estimate_equity_timed(your_cards, board_cards, n_opps, time_limit, rng):
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
    while time.perf_counter() - start < time_limit:
        rng.shuffle(deck_eval)
        idx = 0
        opps = []
        for _ in range(n_opps):
            opps.append(deck_eval[idx:idx+2])
            idx += 2
        sim_board = bc[:]
        if need > 0:
            sim_board += deck_eval[idx:idx+need]
        our = safe_eval(yc + sim_board)
        better = equal = 0
        for opp in opps:
            os = safe_eval(opp + sim_board)
            if os > our:
                better += 1
            elif os == our:
                equal += 1
        if better == 0 and equal == 0:
            wins += 1
        elif better == 0:
            ties += 1
        iters += 1
    if iters == 0:
        return 0.5
    return (wins + 0.5 * ties) / iters

def value_function(state, your_cards, board, n_opps, time_left, rng):
    if time_left() <= 0:
        return 0.0
    pot = float(state["pot"])
    to_call = float(state["amount_owed"])
    eq_time = min(0.4, max(0.1, time_left() * 0.5))
    equity = estimate_equity_timed(your_cards, board, n_opps, eq_time, rng)
    win_pot = pot + to_call
    call_ev = equity * win_pot - (1 - equity) * to_call
    return max(0.0, call_ev)

def clone_state(state):
    return {
        "your_cards": state["your_cards"],
        "community_cards": state["community_cards"],
        "street": state["street"],
        "pot": float(state["pot"]),
        "your_stack": float(state["your_stack"]),
        "amount_owed": float(state["amount_owed"]),
        "can_check": bool(state["can_check"]),
        "current_bet": float(state["current_bet"]),
        "min_raise_to": int(state["min_raise_to"]),
        "players": state["players"],
        "action_log": state["action_log"],
    }

def deepstack_node(state, your_cards, board, n_opps, depth, max_depth, time_left, rng, reach_prob):
    if depth >= max_depth or time_left() <= 0:
        return value_function(state, your_cards, board, n_opps, time_left, rng)

    pot = float(state["pot"])
    stack = float(state["your_stack"])
    to_call = float(state["amount_owed"])
    can_check = state["can_check"]
    current_bet = float(state["current_bet"])
    min_raise_to = int(state["min_raise_to"])
    street = state["street"]

    if street == "preflop":
        bucket = preflop_bucket(your_cards)
        mode = "preflop"
    else:
        eq_time = min(0.25, max(0.05, time_left() * 0.3))
        equity = estimate_equity_timed(your_cards, board, n_opps, eq_time, rng)
        tex = board_texture(board)
        bucket = postflop_bucket(equity, tex)
        mode = "postflop"

    pc = pot_class(pot, stack)
    ac = action_class(state)
    pos = position_class(state)
    IS = (mode, street, bucket, pc, ac, pos)

    blueprint = BLUEPRINT[mode][bucket]
    regrets = REGRETS[IS]
    strat = regret_matching(regrets, blueprint)

    # actions: 0 fold/check, 1 call, 2 small, 3 med, 4 large
    evs = [0.0] * 5

    # fold/check
    if to_call > 0:
        evs[0] = 0.0
    else:
        evs[0] = 0.0

    # call
    if to_call > 0:
        call_state = clone_state(state)
        call_state["pot"] = pot + to_call
        call_state["your_stack"] = max(0.0, stack - to_call)
        call_state["amount_owed"] = 0.0
        call_state["can_check"] = True
        evs[1] = deepstack_node(call_state, your_cards, board, n_opps,
                                depth + 1, max_depth, time_left, rng, reach_prob * strat[1])
    else:
        evs[1] = 0.0

    # bet/raise sizes
    mults_call = {2: 2.0, 3: 2.7, 4: 3.5}
    mults_bet = {2: 0.5, 3: 0.8, 4: 1.1}

    if to_call > 0:
        for idx, mult in mults_call.items():
            total = max(min_raise_to, int(current_bet + mult * to_call))
            total = min(total, int(current_bet + stack))
            if total <= current_bet + to_call:
                evs[idx] = evs[1]
            else:
                risk = total - current_bet
                raise_state = clone_state(state)
                raise_state["pot"] = pot + risk + to_call
                raise_state["your_stack"] = max(0.0, stack - risk - to_call)
                raise_state["amount_owed"] = 0.0
                raise_state["can_check"] = True
                evs[idx] = deepstack_node(raise_state, your_cards, board, n_opps,
                                          depth + 1, max_depth, time_left, rng, reach_prob * strat[idx])
    else:
        for idx, mult in mults_bet.items():
            total = max(min_raise_to, int(current_bet + pot * mult))
            total = min(total, int(current_bet + stack))
            if total <= current_bet:
                evs[idx] = evs[1]
            else:
                risk = total - current_bet
                bet_state = clone_state(state)
                bet_state["pot"] = pot + risk
                bet_state["your_stack"] = max(0.0, stack - risk)
                bet_state["amount_owed"] = 0.0
                bet_state["can_check"] = True
                evs[idx] = deepstack_node(bet_state, your_cards, board, n_opps,
                                          depth + 1, max_depth, time_left, rng, reach_prob * strat[idx])

    strat_ev = sum(p * e for p, e in zip(strat, evs))
    for i in range(5):
        REGRETS[IS][i] += reach_prob * (evs[i] - strat_ev)
        STRAT_SUM[IS][i] += reach_prob * strat[i]

    # return EV of best action (for value backup)
    return max(evs)

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
        pot = float(state["pot"])
        current_bet = float(state["current_bet"])
        min_raise_to = int(state["min_raise_to"])

        if stack <= 0:
            return {"action": "call"} if to_call > 0 else {"action": "check"}

        n_opps = sum(1 for p in state["players"] if not p.get("has_folded", False)) - 1
        n_opps = max(1, n_opps)
        rng = random.Random()

        max_depth = 3
        deepstack_node(state, your_cards, board, n_opps, 0, max_depth, time_left, rng, reach_prob=1.0)

        # choose action greedily from current strategy
        if street == "preflop":
            bucket = preflop_bucket(your_cards)
            mode = "preflop"
        else:
            eq_time = min(0.25, max(0.05, time_left() * 0.3))
            equity = estimate_equity_timed(your_cards, board, n_opps, eq_time, rng)
            tex = board_texture(board)
            bucket = postflop_bucket(equity, tex)
            mode = "postflop"

        pc = pot_class(pot, stack)
        ac = action_class(state)
        pos = position_class(state)
        IS = (mode, street, bucket, pc, ac, pos)

        blueprint = BLUEPRINT[mode][bucket]
        regrets = REGRETS[IS]
        strat = regret_matching(regrets, blueprint)
        a = sample_from_probs(strat, rng)

        if to_call > 0:
            if a == 0:
                return {"action": "fold"}
            if a == 1:
                return {"action": "call"}
            mults = {2: 2.0, 3: 2.7, 4: 3.5}
            mult = mults.get(a, 2.3)
            total = max(min_raise_to, int(current_bet + mult * to_call))
            total = min(total, int(current_bet + stack))
            if total >= current_bet + stack * 0.95:
                return {"action": "all_in"}
            return {"action": "raise", "amount": total}
        else:
            if can_check:
                if a in (0, 1):
                    return {"action": "check"}
                mults = {2: 0.5, 3: 0.8, 4: 1.1}
                mult = mults.get(a, 0.7)
                total = max(min_raise_to, int(current_bet + pot * mult))
                total = min(total, int(current_bet + stack))
                if total >= current_bet + stack * 0.95:
                    return {"action": "all_in"}
                return {"action": "raise", "amount": total}
            return {"action": "call"}

    except Exception:
        if state.get("can_check", False) and state.get("amount_owed", 0) == 0:
            return {"action": "check"}
        return {"action": "call"}
