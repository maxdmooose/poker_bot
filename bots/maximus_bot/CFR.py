# bot.py — Hybrid CFR-style + Monte Carlo search bot for Fullhouse Hackathon
# - Preflop: CFR-style bucketed blueprint (range-based, mixed strategies)
# - Postflop: Monte Carlo equity + local game-tree search (depth-limited)
# - Uses full ~2s per decision with time budgeting
# - Safe: no eval7 crashes, no illegal actions, no time bombs

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


# ------------------ MONTE CARLO EQUITY ------------------

def estimate_equity_multi_timed(your_cards, board_cards, n_opponents, time_limit, rng):
    """
    Monte Carlo equity vs n_opponents random hands.
    Time-bounded to ~time_limit seconds.
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

    # sanity: if we can't even deal everyone + board, bail to neutral
    if n_opponents * 2 + max(0, board_needed) > len(deck_eval):
        return 0.5

    while time.perf_counter() - start < time_limit:
        rng.shuffle(deck_eval)

        idx = 0
        opp_hands = []
        for _ in range(n_opponents):
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


# ------------------ BOARD TEXTURE ------------------

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


# ------------------ OPPONENT MODEL ------------------

def opponent_model(state):
    """
    Tracks fold/call/raise frequencies of opponents.
    """
    log = state.get("action_log", [])
    folds = calls = raises = 0

    for entry in log:
        act = entry.get("action")
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


# ------------------ PREFLOP CFR-STYLE BLUEPRINT ------------------

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


def sample_from_probs(probs, rng):
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += float(p)
        if r <= cum:
            return i
    return len(probs) - 1


# ------------------ LOCAL GAME-TREE SEARCH (POSTFLOP) ------------------

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


def evaluate_leaf(state, your_cards, board, n_opps, rng, time_left, iters_hint):
    if time_left() <= 0:
        return 0.0
    pot = float(state["pot"])
    to_call = float(state["amount_owed"])
    iters = max(60, int(iters_hint))
    eq = estimate_equity_multi_timed(your_cards, board, n_opps, time_limit=min(0.4, time_left()), rng=rng)
    win_pot = pot + to_call
    call_ev = eq * win_pot - (1 - eq) * to_call
    fold_ev = 0.0
    return max(call_ev, fold_ev)


def villain_response_probs(tex, pot, bet_size, street):
    pot_ratio = bet_size / max(1.0, pot)
    pot_ratio = max(0.1, min(3.0, pot_ratio))

    base_fold = 0.25 + 0.15 * (pot_ratio - 0.5)
    base_call = 0.60 - 0.10 * (pot_ratio - 0.5)
    base_raise = 0.15 - 0.05 * (pot_ratio - 0.5)

    if tex["dry"]:
        base_fold += 0.05
        base_raise -= 0.03
    if tex["wet"]:
        base_fold -= 0.05
        base_call += 0.03

    if street == "river":
        base_raise *= 0.7

    base_fold = max(0.05, min(0.8, base_fold))
    base_raise = max(0.02, min(0.25, base_raise))
    base_call = max(0.05, 1.0 - base_fold - base_raise)

    s = base_fold + base_call + base_raise
    if s <= 0:
        return (0.33, 0.33, 0.34)
    return (base_fold / s, base_call / s, base_raise / s)


def search_node(state, your_cards, board, n_opps, depth, max_depth, rng, time_left, iters_hint):
    if depth >= max_depth or time_left() <= 0:
        return evaluate_leaf(state, your_cards, board, n_opps, rng, time_left, iters_hint)

    pot = float(state["pot"])
    stack = float(state["your_stack"])
    to_call = float(state["amount_owed"])
    can_check = state["can_check"]
    min_raise = int(state["min_raise_to"])
    cur_bet = float(state["current_bet"])
    street = state["street"]
    tex = board_texture(board)

    # Facing a bet
    if to_call > 0:
        fold_ev = 0.0

        call_state = clone_state(state)
        call_state["pot"] = pot + to_call
        call_state["your_stack"] = max(0.0, stack - to_call)
        call_state["amount_owed"] = 0.0
        call_state["can_check"] = True
        call_state["current_bet"] = cur_bet
        call_ev = search_node(call_state, your_cards, board, n_opps,
                              depth + 1, max_depth, rng, time_left, iters_hint * 0.7)

        raise_evs = []

        if tex["dry"]:
            mults = [1.8, 2.3, 2.8]
        else:
            mults = [1.4, 1.8, 2.2]

        for m in mults:
            if time_left() <= 0:
                break
            total = max(min_raise, int(cur_bet + to_call * m))
            total = min(total, int(cur_bet + stack))
            if total <= cur_bet + to_call:
                continue

            bet_size = total - cur_bet
            risk = bet_size
            pot_if_called = pot + bet_size + to_call

            fold_p, call_p, raise_p = villain_response_probs(tex, pot, bet_size, street)

            fold_outcome = pot + to_call

            call_state2 = clone_state(state)
            call_state2["pot"] = pot_if_called
            call_state2["your_stack"] = max(0.0, stack - risk - to_call)
            call_state2["amount_owed"] = 0.0
            call_state2["can_check"] = True
            call_state2["current_bet"] = total

            if depth + 1 >= max_depth or time_left() <= 0:
                eq = estimate_equity_multi_timed(your_cards, board, n_opps,
                                                 time_limit=min(0.3, time_left()), rng=rng)
                call_ev_local = eq * pot_if_called - (1 - eq) * risk
            else:
                call_ev_local = search_node(call_state2, your_cards, board, n_opps,
                                            depth + 1, max_depth, rng, time_left, iters_hint * 0.6)

            reraise_ev = -0.3 * risk

            total_ev = fold_p * fold_outcome + call_p * call_ev_local + raise_p * reraise_ev
            raise_evs.append(total_ev)

        best_raise_ev = max(raise_evs) if raise_evs else float("-inf")

        return max(fold_ev, call_ev, best_raise_ev)

    # No bet to call
    if can_check:
        check_state = clone_state(state)
        check_state["amount_owed"] = 0.0
        check_state["can_check"] = True
        if depth + 1 >= max_depth or time_left() <= 0:
            check_ev = evaluate_leaf(check_state, your_cards, board, n_opps, rng, time_left, iters_hint)
        else:
            check_ev = search_node(check_state, your_cards, board, n_opps,
                                   depth + 1, max_depth, rng, time_left, iters_hint * 0.7)

        bet_evs = []
        if tex["dry"]:
            mults = [0.45, 0.75, 1.0]
        else:
            mults = [0.55, 0.85, 1.2]

        for m in mults:
            if time_left() <= 0:
                break
            total = max(min_raise, int(cur_bet + pot * m))
            total = min(total, int(cur_bet + stack))
            if total <= cur_bet:
                continue

            bet_size = total - cur_bet
            risk = bet_size
            pot_if_called = pot + bet_size

            fold_p, call_p, raise_p = villain_response_probs(tex, pot, bet_size, street)

            fold_outcome = pot

            call_state = clone_state(state)
            call_state["pot"] = pot_if_called
            call_state["your_stack"] = max(0.0, stack - risk)
            call_state["amount_owed"] = 0.0
            call_state["can_check"] = True
            call_state["current_bet"] = total

            if depth + 1 >= max_depth or time_left() <= 0:
                eq = estimate_equity_multi_timed(your_cards, board, n_opps,
                                                 time_limit=min(0.3, time_left()), rng=rng)
                call_ev_local = eq * pot_if_called - (1 - eq) * risk
            else:
                call_ev_local = search_node(call_state, your_cards, board, n_opps,
                                            depth + 1, max_depth, rng, time_left, iters_hint * 0.6)

            reraise_ev = -0.4 * risk

            total_ev = fold_p * fold_outcome + call_p * call_ev_local + raise_p * reraise_ev
            bet_evs.append((total_ev, total))

        best_bet_ev = max(bet_evs, key=lambda x: x[0])[0] if bet_evs else float("-inf")

        return max(check_ev, best_bet_ev)

    return evaluate_leaf(state, your_cards, board, n_opps, rng, time_left, iters_hint)


def choose_postflop_action(state, your_cards, board, n_opps, time_left):
    rng = random.Random()
    pot = float(state["pot"])
    stack = float(state["your_stack"])
    to_call = float(state["amount_owed"])
    can_check = state["can_check"]
    min_raise = int(state["min_raise_to"])
    cur_bet = float(state["current_bet"])
    street = state["street"]
    tex = board_texture(board)

    max_depth = 3
    iters_hint = 200.0

    # Facing bet
    if to_call > 0:
        fold_ev = 0.0

        call_state = clone_state(state)
        call_state["pot"] = pot + to_call
        call_state["your_stack"] = max(0.0, stack - to_call)
        call_state["amount_owed"] = 0.0
        call_state["can_check"] = True
        call_state["current_bet"] = cur_bet
        call_ev = search_node(call_state, your_cards, board, n_opps,
                              1, max_depth, rng, time_left, iters_hint)

        raise_candidates = []
        if tex["dry"]:
            mults = [1.8, 2.3, 2.8]
        else:
            mults = [1.4, 1.8, 2.2]

        for m in mults:
            if time_left() <= 0:
                break
            total = max(min_raise, int(cur_bet + to_call * m))
            total = min(total, int(cur_bet + stack))
            if total <= cur_bet + to_call:
                continue

            bet_size = total - cur_bet
            risk = bet_size
            pot_if_called = pot + bet_size + to_call

            fold_p, call_p, raise_p = villain_response_probs(tex, pot, bet_size, street)

            fold_outcome = pot + to_call

            call_state2 = clone_state(state)
            call_state2["pot"] = pot_if_called
            call_state2["your_stack"] = max(0.0, stack - risk - to_call)
            call_state2["amount_owed"] = 0.0
            call_state2["can_check"] = True
            call_state2["current_bet"] = total

            if time_left() <= 0:
                eq = estimate_equity_multi_timed(your_cards, board, n_opps,
                                                 time_limit=0.2, rng=rng)
                call_ev_local = eq * pot_if_called - (1 - eq) * risk
            else:
                call_ev_local = search_node(call_state2, your_cards, board, n_opps,
                                            1, max_depth, rng, time_left, iters_hint * 0.7)

            reraise_ev = -0.3 * risk

            total_ev = fold_p * fold_outcome + call_p * call_ev_local + raise_p * reraise_ev
            raise_candidates.append((total_ev, total))

        if raise_candidates:
            best_raise_ev, best_raise_amt = max(raise_candidates, key=lambda x: x[0])
        else:
            best_raise_ev, best_raise_amt = (float("-inf"), None)

        best_ev = fold_ev
        best_action = ("fold", None)

        if call_ev > best_ev:
            best_ev = call_ev
            best_action = ("call", None)

        if best_raise_ev > best_ev and best_raise_amt is not None:
            best_ev = best_raise_ev
            best_action = ("raise", best_raise_amt)

        return best_action

    # No bet to call
    if can_check:
        check_state = clone_state(state)
        check_state["amount_owed"] = 0.0
        check_state["can_check"] = True
        check_ev = search_node(check_state, your_cards, board, n_opps,
                               1, max_depth, rng, time_left, iters_hint)

        bet_candidates = []
        if tex["dry"]:
            mults = [0.45, 0.75, 1.0]
        else:
            mults = [0.55, 0.85, 1.2]

        for m in mults:
            if time_left() <= 0:
                break
            total = max(min_raise, int(cur_bet + pot * m))
            total = min(total, int(cur_bet + stack))
            if total <= cur_bet:
                continue

            bet_size = total - cur_bet
            risk = bet_size
            pot_if_called = pot + bet_size

            fold_p, call_p, raise_p = villain_response_probs(tex, pot, bet_size, street)

            fold_outcome = pot

            call_state = clone_state(state)
            call_state["pot"] = pot_if_called
            call_state["your_stack"] = max(0.0, stack - risk)
            call_state["amount_owed"] = 0.0
            call_state["can_check"] = True
            call_state["current_bet"] = total

            if time_left() <= 0:
                eq = estimate_equity_multi_timed(your_cards, board, n_opps,
                                                 time_limit=0.2, rng=rng)
                call_ev_local = eq * pot_if_called - (1 - eq) * risk
            else:
                call_ev_local = search_node(call_state, your_cards, board, n_opps,
                                            1, max_depth, rng, time_left, iters_hint * 0.7)

            reraise_ev = -0.4 * risk

            total_ev = fold_p * fold_outcome + call_p * call_ev_local + raise_p * reraise_ev
            bet_candidates.append((total_ev, total))

        if bet_candidates:
            best_bet_ev, best_bet_amt = max(bet_candidates, key=lambda x: x[0])
        else:
            best_bet_ev, best_bet_amt = (float("-inf"), None)

        if best_bet_ev > check_ev and best_bet_amt is not None:
            return ("bet", best_bet_amt)
        else:
            return ("check", None)

    return ("call", None)


# ------------------ MAIN DECIDE ------------------

def decide(state: dict) -> dict:
    start = time.perf_counter()
    BUDGET = 1.95  # leave a bit of safety

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

        # -------- PREFLOP: CFR-style blueprint + small Monte Carlo tweak --------
        if street == "preflop":
            bucket = preflop_bucket(your_cards)
            probs = PREFLOP_STRAT[bucket][:]

            # If facing a raise, nudge probabilities by equity vs random
            if to_call > 0 and time_left() > 0.4:
                eq = estimate_equity_multi_timed(
                    your_cards,
                    board,
                    n_opponents=n_opps,
                    time_limit=min(0.4, time_left()),
                    rng=rng
                )
                # shift mass from fold->call->raise as equity increases
                f, c, r_ = probs
                shift = (eq - 0.5) * 0.8  # in [-0.4, 0.4]
                if shift > 0:
                    take = min(f, shift * 0.6)
                    f -= take
                    c += take * 0.5
                    r_ += take * 0.5
                    take2 = min(c, shift * 0.4)
                    c -= take2
                    r_ += take2
                else:
                    shift = -shift
                    take = min(r_, shift * 0.6)
                    r_ -= take
                    c += take * 0.5
                    f += take * 0.5
                s = f + c + r_
                if s <= 0:
                    probs = [0.7, 0.25, 0.05]
                else:
                    probs = [f / s, c / s, r_ / s]

            act_idx = sample_from_probs(probs, rng)
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
                if to_call == 0:
                    base = max(min_raise_to, int(pot * 0.75) + 2)
                else:
                    base = max(min_raise_to, int(state["current_bet"] + 3 * to_call))
                if base >= state["current_bet"] + stack * 0.95:
                    return {"action": "all_in"}
                return {"action": "raise", "amount": base}

        # -------- POSTFLOP: hybrid Monte Carlo + local game-tree search --------
        action, amount = choose_postflop_action(state, your_cards, board, n_opps, time_left)

        if action == "fold":
            if to_call > 0:
                return {"action": "fold"}
            return {"action": "check"} if can_check else {"action": "fold"}

        if action == "call":
            if to_call == 0 and can_check:
                return {"action": "check"}
            return {"action": "call"}

        if action in ("raise", "bet"):
            if amount is None:
                if to_call == 0 and can_check:
                    return {"action": "check"}
                return {"action": "call"}
            amount = int(amount)
            current_bet = float(state["current_bet"])
            if amount >= current_bet + stack * 0.95:
                return {"action": "all_in"}
            return {"action": "raise", "amount": amount}

        if action == "check":
            if can_check:
                return {"action": "check"}
            return {"action": "call"}

        return {"action": "call"}

    except Exception:
        # absolute safety net
        to_call = float(state.get("amount_owed", 0))
        can_check = bool(state.get("can_check", False))
        if to_call == 0 and can_check:
            return {"action": "check"}
        return {"action": "call"}
