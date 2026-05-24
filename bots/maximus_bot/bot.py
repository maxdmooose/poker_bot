# bot.py
import random
import eval7

# --------- card / equity helpers ---------

RANKS = "23456789TJQKA"
SUITS = "cdhs"
FULL_DECK = [r + s for r in RANKS for s in SUITS]


def estimate_equity(your_cards, board_cards, iters=800):
    """
    Monte Carlo equity vs 1 random opponent.
    Returns equity in [0,1].
    """
    # convert to eval7.Card once for speed
    your_eval = [eval7.Card(c) for c in your_cards]
    board_eval = [eval7.Card(c) for c in board_cards]

    dead = set(your_cards + board_cards)
    deck = [c for c in FULL_DECK if c not in dead]
    deck_eval = [eval7.Card(c) for c in deck]

    wins = ties = 0

    board_needed = 5 - len(board_cards)

    for _ in range(iters):
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

    total = float(iters)
    return (wins + 0.5 * ties) / total if total > 0 else 0.0


def preflop_strength(your_cards):
    """
    Cheap heuristic preflop strength in [0,1].
    Not GTO, but decent ordering for opening ranges.
    """
    c1, c2 = your_cards
    r1, s1 = c1[0], c1[1]
    r2, s2 = c2[0], c2[1]

    i1, i2 = RANKS.index(r1), RANKS.index(r2)
    high = max(i1, i2)
    low = min(i1, i2)

    pair = (r1 == r2)
    suited = (s1 == s2)
    gap = abs(i1 - i2) - 1  # 0 = connectors

    # base on high card
    strength = high / (len(RANKS) - 1)

    if pair:
        strength += 0.25 + 0.03 * high
    if suited:
        strength += 0.05
    if gap == 0:
        strength += 0.05
    elif gap == 1:
        strength += 0.03
    elif gap >= 3:
        strength -= 0.05 * (gap - 2)

    # small bonus for both reasonably high
    if low >= RANKS.index("T"):
        strength += 0.05

    return max(0.0, min(1.0, strength))


# --------- main strategy ---------

def decide(state: dict) -> dict:
    """
    Fullhouse decide() entrypoint.

    state keys:
      - your_cards: list[str]
      - community_cards: list[str]
      - street: 'preflop'/'flop'/'turn'/'river'
      - pot: int
      - your_stack: int
      - amount_owed: int
      - can_check: bool
      - current_bet: int
      - min_raise_to: int
      - players: list[dict]
      - action_log: list[dict]
    """
    your_cards = state["your_cards"]
    board = state["community_cards"]
    street = state["street"]
    pot = float(state["pot"])
    stack = float(state["your_stack"])
    to_call = float(state["amount_owed"])
    can_check = state["can_check"]
    min_raise_to = int(state["min_raise_to"])

    # safety: if we're effectively all in or no stack, just call/check
    if stack <= 0:
        if to_call > 0:
            return {"action": "call"}
        return {"action": "check"} if can_check else {"action": "fold"}

    # --------- estimate equity / strength ---------

    if street == "preflop":
        equity = preflop_strength(your_cards)
    else:
        # postflop: Monte Carlo equity vs 1 random opponent
        equity = estimate_equity(your_cards, board, iters=700)

    # pot odds if facing a bet
    if to_call > 0:
        pot_odds = to_call / (pot + to_call) if (pot + to_call) > 0 else 1.0
    else:
        pot_odds = 0.0

    # --------- thresholds (tunable) ---------

    # margin above breakeven to actually continue
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

    # --------- facing a bet (to_call > 0) ---------

    if to_call > 0:
        # fold region
        if equity < call_thresh:
            # occasionally defend a bit wider vs tiny bets
            if pot_odds < 0.15 and equity > pot_odds and r < 0.25:
                return {"action": "call"}
            return {"action": "fold"}

        # call vs raise
        if equity >= raise_thresh and stack > to_call * 2:
            # raise frequency grows with equity
            raise_freq = min(0.9, max(0.2, (equity - raise_thresh) / 0.2))
            if r < raise_freq:
                # choose a raise size between ~2.5x and pot-sized
                # note: amount is total bet, not raise-by
                min_total = max(min_raise_to, int(state["current_bet"] + to_call * 1.5))
                # cap at all-in
                max_total = int(min(state["current_bet"] + pot * 1.5, state["current_bet"] + stack))

                if max_total <= min_total:
                    amount = min_total
                else:
                    alpha = random.random()
                    amount = int(min_total + alpha * (max_total - min_total))

                # if we're effectively shoving, just use all_in
                if amount >= state["current_bet"] + stack * 0.95:
                    return {"action": "all_in"}

                return {"action": "raise", "amount": amount}

        # profitable call but not strong enough to raise
        return {"action": "call"}

    # --------- no bet to call (we can check or bet) ---------

    if can_check:
        # treat "bet" as a raise to some total; engine uses "raise" even when first in
        # value bet region
        if equity >= raise_thresh:
            # bet for value with high frequency
            if r < 0.85:
                # size between 1/2 pot and full pot
                half_pot = int(pot * 0.5)
                full_pot = int(pot * 1.1)
                min_total = max(min_raise_to, state["current_bet"] + half_pot)
                max_total = max(min_total, state["current_bet"] + full_pot)
                alpha = random.random()
                amount = int(min_total + alpha * (max_total - min_total))

                # cap at all-in
                if amount >= state["current_bet"] + stack * 0.95:
                    return {"action": "all_in"}

                return {"action": "raise", "amount": amount}
            else:
                return {"action": "check"}

        # bluff region: middling equity, earlier streets
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

        # default: take the free card / pot control
        return {"action": "check"}

    # can't check but to_call == 0 is weird; fall back to call
    return {"action": "call"}
