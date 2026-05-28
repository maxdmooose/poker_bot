# bot.py — Depth-3 game-tree search bot for Fullhouse
# - Uses eval7 for equity
# - Depth-3 local game tree with villain response modelling
# - Iterative deepening within ~1.9s per decision
# - Returns only valid actions per hackathon spec

import random
import time
import eval7

RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK = [r + s for r in RANKS for s in SUITS]


# ----------------- Equity -----------------

def equity_sample(your_cards, board_cards, n_opps, iters, rng):
    yc = [eval7.Card(c) for c in your_cards]
    bc = [eval7.Card(c) for c in board_cards]

    dead = set(your_cards + board_cards)
    deck = [c for c in FULL_DECK if c not in dead]
    deck_eval = [eval7.Card(c) for c in deck]

    wins = ties = 0
    need = 5 - len(board_cards)

    for _ in range(iters):
        rng.shuffle(deck_eval)

        opps = []
        idx = 0
        for _ in range(n_opps):
            opps.append(deck_eval[idx:idx+2])
            idx += 2

        sim_board = bc[:]
        if need > 0:
            sim_board += deck_eval[idx:idx+need]

        our = eval7.evaluate(yc + sim_board)

        better = equal = 0
        for opp in opps:
            os = eval7.evaluate(opp + sim_board)
            if os > our:
                better += 1
            elif os == our:
                equal += 1

        if better == 0 and equal == 0:
            wins += 1
        elif better == 0:
            ties += 1

    if iters == 0:
        return 0.5
    total = float(iters)
    return (wins + 0.5 * ties) / total


# ----------------- Texture -----------------

def texture(board):
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


# ----------------- Villain model -----------------

def villain_response_probs(board_tex, pot, bet_size, street):
    # Simple heuristic villain model: fold/call/raise probabilities
    # Larger bets and drier boards -> more folds, fewer raises.
    pot_ratio = bet_size / max(1.0, pot)
    pot_ratio = max(0.1, min(3.0, pot_ratio))

    base_fold = 0.25 + 0.15 * (pot_ratio - 0.5)
    base_call = 0.60 - 0.10 * (pot_ratio - 0.5)
    base_raise = 0.15 - 0.05 * (pot_ratio - 0.5)

    if board_tex["dry"]:
        base_fold += 0.05
        base_raise -= 0.03
    if board_tex["wet"]:
        base_fold -= 0.05
        base_call += 0.03

    if street == "river":
        base_raise *= 0.7

    base_fold = max(0.05, min(0.8, base_fold))
    base_raise = max(0.02, min(0.25, base_raise))
    base_call = max(0.05, 1.0 - base_fold - base_raise)

    s = base_fold + base_call + base_raise
    return (base_fold / s, base_call / s, base_raise / s)


# ----------------- Game tree search -----------------

def evaluate_leaf(state, your_cards, board, n_opps, rng, time_left, iters_hint):
    if time_left() <= 0:
        return 0.0
    pot = float(state["pot"])
    to_call = float(state["amount_owed"])
    iters = max(60, int(iters_hint))
    eq = equity_sample(your_cards, board, n_opps, iters, rng)
    win_pot = pot + to_call
    call_ev = eq * win_pot - (1 - eq) * to_call
    fold_ev = 0.0
    return max(call_ev, fold_ev)


def clone_state(state):
    # Shallow clone with numeric fields adjusted manually when needed
    return {
        "your_cards": state["your_cards"],
        "community_cards": state["community_cards"],
        "street": state["street"],
        "pot": state["pot"],
        "your_stack": state["your_stack"],
        "amount_owed": state["amount_owed"],
        "can_check": state["can_check"],
        "current_bet": state["current_bet"],
        "min_raise_to": state["min_raise_to"],
        "players": state["players"],
        "action_log": state["action_log"],
    }


def advance_street_if_needed(state):
    # We don't simulate future cards explicitly here; tree is within current street.
    return state


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
    tex = texture(board)

    # Facing a bet: actions = fold, call, raise (3 sizes)
    if to_call > 0:
        # Fold EV
        fold_ev = 0.0

        # Call EV: create new state where we call and villain checks down
        call_state = clone_state(state)
        call_state["pot"] = pot + to_call
        call_state["your_stack"] = max(0.0, stack - to_call)
        call_state["amount_owed"] = 0.0
        call_state["can_check"] = True
        call_state["current_bet"] = cur_bet
        call_state = advance_street_if_needed(call_state)
        call_ev = search_node(call_state, your_cards, board, n_opps,
                              depth + 1, max_depth, rng, time_left, iters_hint * 0.7)

        # Raise EVs: three sizes
        raise_evs = []
        raise_sizes = []

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
            raise_sizes.append(total)

            bet_size = total - cur_bet
            risk = bet_size
            pot_if_called = pot + bet_size + to_call

            fold_p, call_p, raise_p = villain_response_probs(tex, pot, bet_size, street)

            # Villain folds: we win pot + to_call
            fold_outcome = pot + to_call

            # Villain calls: we go to leaf or deeper node
            call_state2 = clone_state(state)
            call_state2["pot"] = pot_if_called
            call_state2["your_stack"] = max(0.0, stack - risk - to_call)
            call_state2["amount_owed"] = 0.0
            call_state2["can_check"] = True
            call_state2["current_bet"] = total
            call_state2 = advance_street_if_needed(call_state2)

            # Approximate EV of called raise via equity
            if depth + 1 >= max_depth or time_left() <= 0:
                eq = equity_sample(your_cards, board, n_opps, max(40, int(iters_hint * 0.5)), rng)
                call_ev_local = eq * pot_if_called - (1 - eq) * risk
            else:
                call_ev_local = search_node(call_state2, your_cards, board, n_opps,
                                            depth + 1, max_depth, rng, time_left, iters_hint * 0.6)

            # Villain re-raises: we approximate as bad for us, small negative EV
            # (we could expand further, but depth is limited)
            reraise_ev = -0.3 * risk

            total_ev = fold_p * fold_outcome + call_p * call_ev_local + raise_p * reraise_ev
            raise_evs.append(total_ev)

        best_raise_ev = max(raise_evs) if raise_evs else float("-inf")

        return max(fold_ev, call_ev, best_raise_ev)

    # No bet to call: actions = check, bet (3 sizes)
    if can_check:
        # Check EV: assume pot control, go to leaf or shallow node
        check_state = clone_state(state)
        check_state["amount_owed"] = 0.0
        check_state["can_check"] = True
        check_state = advance_street_if_needed(check_state)
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
            total = max(min_raise, int(state["current_bet"] + pot * m))
            total = min(total, int(state["current_bet"] + stack))
            if total <= state["current_bet"]:
                continue

            bet_size = total - state["current_bet"]
            risk = bet_size
            pot_if_called = pot + bet_size

            fold_p, call_p, raise_p = villain_response_probs(tex, pot, bet_size, street)

            # Villain folds: we win pot
            fold_outcome = pot

            # Villain calls: go deeper or leaf
            call_state = clone_state(state)
            call_state["pot"] = pot_if_called
            call_state["your_stack"] = max(0.0, stack - risk)
            call_state["amount_owed"] = 0.0
            call_state["can_check"] = True
            call_state["current_bet"] = total
            call_state = advance_street_if_needed(call_state)

            if depth + 1 >= max_depth or time_left() <= 0:
                eq = equity_sample(your_cards, board, n_opps, max(40, int(iters_hint * 0.5)), rng)
                call_ev_local = eq * pot_if_called - (1 - eq) * risk
            else:
                call_ev_local = search_node(call_state, your_cards, board, n_opps,
                                            depth + 1, max_depth, rng, time_left, iters_hint * 0.6)

            # Villain raises over our bet: approximate as bad
            reraise_ev = -0.4 * risk

            total_ev = fold_p * fold_outcome + call_p * call_ev_local + raise_p * reraise_ev
            bet_evs.append((total_ev, total))

        best_bet_ev = max(bet_evs, key=lambda x: x[0])[0] if bet_evs else float("-inf")

        return max(check_ev, best_bet_ev)

    # Fallback if can_check is False but to_call == 0 (shouldn't happen often)
    return evaluate_leaf(state, your_cards, board, n_opps, rng, time_left, iters_hint)


def choose_root_action(state, your_cards, board, n_opps, max_depth, time_left):
    rng = random.Random()
    pot = float(state["pot"])
    stack = float(state["your_stack"])
    to_call = float(state["amount_owed"])
    can_check = state["can_check"]
    min_raise = int(state["min_raise_to"])
    cur_bet = float(state["current_bet"])
    street = state["street"]
    tex = texture(board)

    iters_hint = 200.0

    # Facing bet: evaluate fold, call, 3 raise sizes
    if to_call > 0:
        # Fold
        fold_ev = 0.0

        # Call
        call_state = clone_state(state)
        call_state["pot"] = pot + to_call
        call_state["your_stack"] = max(0.0, stack - to_call)
        call_state["amount_owed"] = 0.0
        call_state["can_check"] = True
        call_state["current_bet"] = cur_bet
        call_state = advance_street_if_needed(call_state)
        call_ev = search_node(call_state, your_cards, board, n_opps,
                              1, max_depth, rng, time_left, iters_hint)

        # Raises
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
            call_state2 = advance_street_if_needed(call_state2)

            if time_left() <= 0:
                eq = equity_sample(your_cards, board, n_opps, 80, rng)
                call_ev_local = eq * pot_if_called - (1 - eq) * risk
            else:
                call_ev_local = search_node(call_state2, your_cards, board, n_opps,
                                            1, max_depth, rng, time_left, iters_hint * 0.7)

            reraise_ev = -0.3 * risk

            total_ev = fold_p * fold_outcome + call_p * call_ev_local + raise_p * reraise_ev
            raise_candidates.append((total_ev, total))

        best_raise_ev, best_raise_amt = (float("-inf"), None)
        if raise_candidates:
            best_raise_ev, best_raise_amt = max(raise_candidates, key=lambda x: x[0])

        # Choose best root action
        best_ev = fold_ev
        best_action = ("fold", None)

        if call_ev > best_ev:
            best_ev = call_ev
            best_action = ("call", None)

        if best_raise_ev > best_ev:
            best_ev = best_raise_ev
            best_action = ("raise", best_raise_amt)

        return best_action

    # No bet to call: check vs 3 bet sizes
    if can_check:
        check_state = clone_state(state)
        check_state["amount_owed"] = 0.0
        check_state["can_check"] = True
        check_state = advance_street_if_needed(check_state)
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
            call_state = advance_street_if_needed(call_state)

            if time_left() <= 0:
                eq = equity_sample(your_cards, board, n_opps, 80, rng)
                call_ev_local = eq * pot_if_called - (1 - eq) * risk
            else:
                call_ev_local = search_node(call_state, your_cards, board, n_opps,
                                            1, max_depth, rng, time_left, iters_hint * 0.7)

            reraise_ev = -0.4 * risk

            total_ev = fold_p * fold_outcome + call_p * call_ev_local + raise_p * reraise_ev
            bet_candidates.append((total_ev, total))

        best_bet_ev, best_bet_amt = (float("-inf"), None)
        if bet_candidates:
            best_bet_ev, best_bet_amt = max(bet_candidates, key=lambda x: x[0])

        if best_bet_ev > check_ev:
            return ("bet", best_bet_amt)
        else:
            return ("check", None)

    # Fallback
    return ("call", None)


# ----------------- Main decide -----------------

def decide(state: dict) -> dict:
    start = time.perf_counter()
    BUDGET = 1.90  # leave a bit of safety under 2s

    def time_left():
        return BUDGET - (time.perf_counter() - start)

    your_cards = state["your_cards"]
    board = state["community_cards"]
    stack = float(state["your_stack"])
    to_call = float(state["amount_owed"])
    can_check = state["can_check"]

    if stack <= 0:
        return {"action": "call"} if to_call > 0 else {"action": "check"}

    n_opps = sum(1 for p in state["players"] if not p["has_folded"]) - 1
    n_opps = max(1, n_opps)

    # Depth-3 search
    max_depth = 3
    action, amount = choose_root_action(state, your_cards, board, n_opps, max_depth, time_left)

    # Map to engine actions
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
            # fallback
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

    # Safety fallback
    return {"action": "call"}
