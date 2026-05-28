# bot.py
import random
import time
import eval7

RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK = [r + s for r in RANKS for s in SUITS]


def estimate_equity_timed(your_cards, board_cards, time_limit=1.90):
    """
    Monte Carlo equity vs 1 random opponent.
    Runs until time_limit seconds have elapsed.
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

    # Run until time budget is exhausted
    while True:
        now = time.perf_counter()
        if now - start >= time_limit:
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
        return 0.5  # fallback

    return (wins + 0.5 * ties) / iters


def decide(state: dict) -> dict:
    your_cards = state["your_cards"]
    board = state["community_cards"]
    street = state["street"]
    pot = float(state["pot"])
    stack = float(state["your_stack"])
    to_call = float(state["amount_owed"])
    can_check = state["can_check"]
    min_raise_to = int(state["min_raise_to"])

    # If stack is gone, trivial action
    if stack <= 0:
        if to_call > 0:
            return {"action": "call"}
        return {"action": "check"} if can_check else {"action": "fold"}

    # --------- Equity estimation (time-bounded) ---------

    # Always use eval7 Monte Carlo, even preflop
    equity = estimate_equity_timed(
        your_cards,
        board,
        time_limit= 1.9  # leave 0.1s safety margin
    )

    # --------- Pot odds ---------

    if to_call > 0:
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 1.0
    else:
        pot_odds = 0.0

    # --------- Thresholds ---------

    margin = 0.03

    if street == "preflop":
        raise_thresh = 0.62
        call_thresh = max(pot_odds + margin, 0.25)
    elif street == "flop":
        raise_thresh = 0.60
        call_thresh = pot_odds + margin
    elif street == "turn":
        raise_thresh = 0.62
        call_thresh = pot_odds + margin
    else:  # river
        raise_thresh = 0.66
        call_thresh = pot_odds + margin

    r = random.random()

    # --------- Facing a bet ---------

    if to_call > 0:
        if equity < call_thresh:
            if pot_odds < 0.15 and equity > pot_odds and r < 0.25:
                return {"action": "call"}
            return {"action": "fold"}

        if equity >= raise_thresh and stack > to_call * 2:
            raise_freq = min(0.9, max(0.2, (equity - raise_thresh) / 0.2))
            if r < raise_freq:
                min_total = max(min_raise_to, int(state["current_bet"] + to_call * 1.5))
                max_total = int(min(state["current_bet"] + pot * 1.5, state["current_bet"] + stack))

                if max_total <= min_total:
                    amount = min_total
                else:
                    alpha = random.random()
                    amount = int(min_total + alpha * (max_total - min_total))

                if amount >= state["current_bet"] + stack * 0.95:
                    return {"action": "all_in"}

                return {"action": "raise", "amount": amount}

        return {"action": "call"}

    # --------- No bet to call ---------

    if can_check:
        if equity >= raise_thresh:
            if r < 0.85:
                half_pot = int(pot * 0.5)
                full_pot = int(pot * 1.1)
                min_total = max(min_raise_to, state["current_bet"] + half_pot)
                max_total = max(min_total, state["current_bet"] + full_pot)
                alpha = random.random()
                amount = int(min_total + alpha * (max_total - min_total))

                if amount >= state["current_bet"] + stack * 0.95:
                    return {"action": "all_in"}

                return {"action": "raise", "amount": amount}
            else:
                return {"action": "check"}

        bluff_low, bluff_high = 0.18, 0.42
        if bluff_low <= equity <= bluff_high and street in ("flop", "turn"):
            base_bluff_freq = {"preflop": 0.10, "flop": 0.20, "turn": 0.14, "river": 0.06}[street]
            if r < base_bluff_freq:
                half_pot = int(pot * 0.45)
                three_quarter = int(pot * 0.75)
                min_total = max(min_raise_to, state["current_bet"] + half_pot)
                max_total = max(min_total, state["current_bet"] + three_quarter)
                alpha = random.random()
                amount = int(min_total + alpha * (max_total - min_total))

                if amount >= state["current_bet"] + stack * 0.95:
                    return {"action": "all_in"}

                return {"action": "raise", "amount": amount}

        return {"action": "check"}

    return {"action": "call"}
