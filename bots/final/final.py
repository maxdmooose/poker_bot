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
    """
    Map hero seat to positional label using button-relative order.

    Seat order (relative to button) assumed (6-max):
        0: BTN
        1: SB
        2: BB
        3: UTG
        4: MP
        5: CO
    """
    players = state["players"]
    n = len(players)

    # hero seat: prefer explicit your_seat, else is_hero, else 0
    hero_seat = state.get("your_seat", None)
    if not isinstance(hero_seat, int) or not (0 <= hero_seat < n):
        hero_seat = 0
        for i, p in enumerate(players):
            if p.get("is_hero", False):
                hero_seat = i
                break

    # button seat: first with is_button, else 0
    button_seat = 0
    for i, p in enumerate(players):
        if p.get("is_button", False):
            button_seat = i
            break

    rel = (hero_seat - button_seat) % n
    order = ["BTN", "SB", "BB", "UTG", "MP", "CO"]
    return order[rel] if rel < len(order) else "UTG"


def infer_big_blind(state):
    """
    Infer big blind size.

    Priority:
      1. state["big_blind"] if present
      2. largest posted blind in preflop action_log
      3. min_raise_to, then current_bet
      4. fallback 1
    """
    bb = state.get("big_blind", None)
    if isinstance(bb, (int, float)) and bb > 0:
        return int(bb)

    log = state.get("action_log", [])
    posted = []
    for e in log:
        if e.get("street") == "preflop" and e.get("action") in ("post", "blind"):
            amt = e.get("amount", 0)
            if isinstance(amt, (int, float)) and amt > 0:
                posted.append(amt)
    if posted:
        return int(max(posted))

    current_bet = int(state.get("current_bet", 0))
    min_raise_to = int(state.get("min_raise_to", 0))
    if min_raise_to > 0:
        return min_raise_to
    if current_bet > 0:
        return current_bet
    return 1


def _classify_preflop_node(state, bb):
    """
    Classify hero's preflop decision node.

    Returns:
      'open_spot'     – hero can open (no prior raise, facing blinds/limps)
      'bb_free_check' – BB facing no raise
      'vs_open'       – facing a single raise
      'vs_3bet'       – facing a 3-bet or higher
    """
    if state["street"] != "preflop":
        return None

    to_call = int(state["amount_owed"])
    can_check = state.get("can_check", False)
    current_bet = int(state.get("current_bet", 0))
    pos = get_position(state)

    log = [e for e in state.get("action_log", []) if e.get("street") == "preflop"]
    raises = sum(1 for e in log if e.get("action") in ("raise", "bet"))

    # BB free check
    if pos == "BB" and can_check and to_call == 0 and current_bet >= bb:
        return "bb_free_check"

    if raises == 0:
        # no raises yet → either open spot or limped pot
        if to_call == bb and not can_check:
            # facing blind amount → first to act / folded to
            return "open_spot"
        if to_call > 0:
            # overlimp / weird structure → treat as vs_open-ish
            return "vs_open"
        # no one has put in more than blinds, hero can open
        return "open_spot"

    if raises == 1:
        return "vs_open"

    return "vs_3bet"


def preflop_decision(state):
    """
    Range-based preflop engine using full JSON ranges:

      - open
      - limp (SB only)
      - call_vs_open
      - 3bet
      - call_vs_3bet
      - 4bet
      - squeeze
      - defend_vs_open (BB only)
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
    node = _classify_preflop_node(state, bb)

    # ---------- OPEN SPOT ----------
    if node == "open_spot":
        if hand in ranges.get("open", []):
            # positional open sizing
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

        # SB limp range
        if pos == "SB" and hand in ranges.get("limp", []):
            return {"action": "call"}

        return {"action": "fold"}

    # ---------- BB FREE CHECK ----------
    if node == "bb_free_check":
        # optional: iso-raise vs limpers using squeeze range
        # simple version: just check
        return {"action": "check"}

    # ---------- VS OPEN (single raise) ----------
    if node == "vs_open" and to_call > 0:
        log = [e for e in state.get("action_log", []) if e.get("street") == "preflop"]
        calls_before_hero = sum(1 for e in log if e.get("action") == "call")
        is_squeeze_spot = calls_before_hero > 0

        # 3-bet / squeeze
        if is_squeeze_spot and hand in ranges.get("squeeze", []):
            mult = 4.0  # bigger sizing for squeeze
        elif hand in ranges.get("3bet", []):
            # 3-bet sizing: 3x IP, 4x OOP
            if pos in ("BTN", "CO"):
                mult = 3.0
            else:
                mult = 4.0
        else:
            mult = None

        if mult is not None:
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

        return {"action": "fold"}

    # ---------- VS 3-BET OR HIGHER ----------
    if node == "vs_3bet" and to_call > 0:
        # 4-bet
        if hand in ranges.get("4bet", []):
            target = current_bet + int(to_call * 2.3)  # ~2.3x 3-bet
            amount = max(min_raise_to, target)
            amount = min(amount, current_bet + stack)
            if amount >= current_bet + int(stack * 0.90):
                return {"action": "all_in"}
            return {"action": "raise", "amount": amount}

        # call vs 3-bet
        if hand in ranges.get("call_vs_3bet", []):
            return {"action": "call"}

        return {"action": "fold"}

    # no clear preflop action from ranges
    return None

# ------------------ MAIN STRATEGY ------------------

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

        # ---------- PREFLOP: use range-based logic ----------
        if street == "preflop":
            action = preflop_decision(state)
            if action is not None:
                return action
            # fallback to old equity-based preflop if JSON missing
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
