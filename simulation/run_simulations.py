#!/usr/bin/env python3
"""
==========================================================================
Cinquillo 2.0  —  Simulation Runner  (run_simulations.py)
==========================================================================
Runs ALL experiments and saves raw results + aggregate statistics to disk.
A separate script (visualise_results.py) reads these files to produce
every figure, table, and LaTeX fragment.

Place at:  simulation/run_simulations.py
Run from PROJECT ROOT:
    python simulation/run_simulations.py
    python simulation/run_simulations.py --quick          # ~500 games / exp
    python simulation/run_simulations.py --medium         # ~2 000 games / exp
    python simulation/run_simulations.py --skip-mcts
    python simulation/run_simulations.py --skip-rl
    python simulation/run_simulations.py --exp 1 8 9      # specific experiments

Outputs (all under output/data/):
    experiments/exp_<N>_<name>.json   aggregated statistics
    raw/exp_<N>_<name>_<variant>.csv  per-game result rows
==========================================================================
"""

# ---------------------------------------------------------------------------
# Path fix — must come before project imports
# ---------------------------------------------------------------------------
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import csv
import time
import warnings
from collections import defaultdict
from itertools import permutations, combinations as _combs
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from game.entities import (
    VariantConfig, GoodDiceEffect, BadDiceEffect,
    ScoringMode, MatchEndMode,
)
from game.rules import Rules
from agents.base_agents import (
    Agent, RandomAgent, HeuristicAgent,
    create_aggressive_heuristic, create_defensive_heuristic,
    create_balanced_heuristic, create_risky_heuristic,
)

try:
    from agents.mcts_agent import (
        MCTSAgent, MCTSAgentSuperFast, MCTSAgentFast,
        MCTSAgentStandard, MCTSAgentDeep,
    )
    HAS_MCTS = True
except ImportError:
    HAS_MCTS = False

try:
    from agents.rl_agent import RLAgent
    HAS_RL = True
except ImportError:
    HAS_RL = False

try:
    from simulation.tournament import GameSimulator, GameResult
    HAS_SIM = True
except ImportError:
    HAS_SIM = False
    print("ERROR: simulation.tournament not found. Ensure you run from PRJ root.")
    sys.exit(1)

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from scipy import stats as scipy_stats
    from scipy.stats import kruskal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ===========================================================================
# TERMINAL HELPERS
# ===========================================================================

def _header(title: str, n: int = 0, total: int = 0):
    bar = '━' * 72
    tag = f'EXP {n}/{total}  ' if n else ''
    print(f'\n{bar}\n  {tag}{title}\n{bar}')

def _sub(msg): print(f'  ▸ {msg}')
def _ok(path): print(f'    ✓ saved → {os.path.basename(path)}')
def _info(msg): print(f'    {msg}')
def _warn(msg): print(f'  ⚠  {msg}')


# ===========================================================================
# VARIANT FACTORY
# ===========================================================================

W   = GoodDiceEffect.WILD
DP  = GoodDiceEffect.DOUBLE_PLAY
GV  = GoodDiceEffect.GIVE_CARD          # Transfer / Give card
INF = GoodDiceEffect.INFO_REVEAL        # Information

TC  = BadDiceEffect.TAKE_CARDS
NP  = BadDiceEffect.NEGATIVE_POINTS
FP  = BadDiceEffect.FORCED_PASS         # Forced Pass / Skip
RH  = BadDiceEffect.REVEAL_HAND         # Exposed

WTA = ScoringMode.WINNER_TAKES_ALL
DBL = ScoringMode.DOUBLE_PENALTY
FR  = MatchEndMode.FIXED_ROUNDS
TS  = MatchEndMode.TARGET_SCORE


def _v(dice_prob: float = 0.5,
       good: GoodDiceEffect = W,
       bad: BadDiceEffect = TC,
       bad_cards: int = 2,
       bad_pts: int = 2,
       scoring: ScoringMode = WTA,
       pts_card: int = 1,
       pass_pen: int = 1,
       rounds: int = 5,
       pts_target: Optional[int] = None) -> VariantConfig:
    """
    Convenience factory.  If pts_target is set the match ends when any player
    reaches that many total points (TARGET_SCORE mode); otherwise FIXED_ROUNDS.
    target_score_multiplier is stored as pts_target // 4 so that 4-player
    games hit the intended total.
    """
    if pts_target is not None:
        multiplier = max(1, pts_target // 4)
        return VariantConfig(
            dice_good_probability=dice_prob,
            dice_good_effect=good,
            dice_bad_effect=bad,
            dice_bad_cards_count=bad_cards,
            dice_bad_penalty_points=bad_pts,
            scoring_mode=scoring,
            points_per_card=pts_card,
            voluntary_pass_penalty=pass_pen,
            match_end_mode=TS,
            target_score_multiplier=multiplier,
            fixed_rounds_count=5,
        )
    return VariantConfig(
        dice_good_probability=dice_prob,
        dice_good_effect=good,
        dice_bad_effect=bad,
        dice_bad_cards_count=bad_cards,
        dice_bad_penalty_points=bad_pts,
        scoring_mode=scoring,
        points_per_card=pts_card,
        voluntary_pass_penalty=pass_pen,
        match_end_mode=FR,
        fixed_rounds_count=rounds,
        target_score_multiplier=10,
    )


# ===========================================================================
# ALL 33 NAMED VARIANTS
# ===========================================================================

def make_baseline():        return _v()
def make_blitz():           return _v(pass_pen=2, rounds=1)
def make_card_exchange():   return _v(good=GV, bad=NP, bad_pts=2, scoring=DBL, pass_pen=1, pts_target=10)
def make_card_flood():      return _v(bad_cards=4)
def make_chaos_mode():      return _v(dice_prob=0.9, bad=NP, bad_pts=5, scoring=DBL, pass_pen=3, pts_target=20)
def make_combo_rush():      return _v(dice_prob=0.6, good=DP, bad=FP, scoring=DBL, pass_pen=2)
def make_double_edge():     return _v(good=DP, pass_pen=0)
def make_double_spy():      return _v(dice_prob=0.6, good=DP, bad=FP, pass_pen=1)
def make_endurance():       return _v(rounds=15)
def make_fortunes_wheel():  return _v(dice_prob=1.0, bad=FP)
def make_gamblers_run():    return _v(dice_prob=0.45, good=DP, bad=NP, bad_pts=2, scoring=DBL, pts_card=2, pass_pen=2, pts_target=20)
def make_ghost_hand():      return _v(dice_prob=0.4, good=GV, bad=RH, scoring=DBL, pass_pen=2, pts_target=15)
def make_glass_cannon():    return _v(dice_prob=0.3, good=DP, bad=NP, bad_pts=5, scoring=DBL, pass_pen=2)
def make_hand_swap():       return _v(dice_prob=0.55, good=GV, bad_cards=2, pass_pen=1)
def make_heavy_toll():      return _v(dice_prob=0.4, bad=NP, bad_pts=3, scoring=DBL, pts_card=2, pass_pen=3, pts_target=20)
def make_high_roller():     return _v(dice_prob=0.6, bad=NP, bad_pts=2, scoring=DBL, pts_card=3, pass_pen=2, pts_target=15)
def make_intel_war():       return _v(good=INF, bad=RH, scoring=DBL, pass_pen=1, pts_target=15)
def make_lucky_draw():      return _v(dice_prob=0.8, bad=NP, bad_pts=1)
def make_marathon():        return _v(rounds=10)
def make_mirror_match():    return _v(good=GV, bad=FP, pass_pen=1)
def make_open_book():       return _v(good=INF, bad=RH, pass_pen=0, rounds=8)
def make_pass_and_peek():   return _v(dice_prob=0.45, good=INF, bad_cards=2, pass_pen=2)
def make_point_race():      return _v(bad=NP, bad_pts=1, pass_pen=1, pts_target=20)
def make_power_play():      return _v(dice_prob=0.7, good=DP, bad=NP, bad_pts=3, scoring=DBL, pts_card=2, pass_pen=2)
def make_pure_strategy():   return _v(dice_prob=0.0, bad=FP)
def make_reveal_rush():     return _v(dice_prob=0.6, good=DP, bad=RH, pass_pen=1)
def make_risk_reward():     return _v(bad=NP, bad_pts=4, scoring=DBL, pass_pen=3, pts_target=25)
def make_safe_harbour():    return _v(bad=FP, pass_pen=0)
def make_score_doubler():   return _v(scoring=DBL, pts_target=15)
def make_scouts_edge():     return _v(dice_prob=0.7, good=INF, bad=FP, pass_pen=0)
def make_slow_burn():       return _v(pass_pen=0, rounds=20)
def make_sprint():          return _v(rounds=3)
def make_spy_game():        return _v(dice_prob=0.65, good=INF, bad=FP, pass_pen=1, pts_target=10)


ALL_VARIANTS: Dict[str, callable] = {
    'Baseline':         make_baseline,
    'Blitz':            make_blitz,
    'Card Exchange':    make_card_exchange,
    'Card Flood':       make_card_flood,
    'Chaos Mode':       make_chaos_mode,
    'Combo Rush':       make_combo_rush,
    'Double Edge':      make_double_edge,
    'Double Spy':       make_double_spy,
    'Endurance':        make_endurance,
    "Fortune's Wheel":  make_fortunes_wheel,
    "Gambler's Run":    make_gamblers_run,
    'Ghost Hand':       make_ghost_hand,
    'Glass Cannon':     make_glass_cannon,
    'Hand Swap':        make_hand_swap,
    'Heavy Toll':       make_heavy_toll,
    'High Roller':      make_high_roller,
    'Intel War':        make_intel_war,
    'Lucky Draw':       make_lucky_draw,
    'Marathon':         make_marathon,
    'Mirror Match':     make_mirror_match,
    'Open Book':        make_open_book,
    'Pass & Peek':      make_pass_and_peek,
    'Point Race':       make_point_race,
    'Power Play':       make_power_play,
    'Pure Strategy':    make_pure_strategy,
    'Reveal Rush':      make_reveal_rush,
    'Risk & Reward':    make_risk_reward,
    'Safe Harbour':     make_safe_harbour,
    'Score Doubler':    make_score_doubler,
    "Scout's Edge":     make_scouts_edge,
    'Slow Burn':        make_slow_burn,
    'Sprint':           make_sprint,
    'Spy Game':         make_spy_game,
}

# Thematic groupings for per-experiment analysis
INFO_VARIANTS   = ['Intel War', 'Open Book', 'Pass & Peek', "Scout's Edge", 'Spy Game']
DOUBLE_PLAY_VARIANTS = ['Combo Rush', 'Double Edge', 'Double Spy', "Gambler's Run",
                         'Glass Cannon', 'Power Play', 'Reveal Rush']
FORCED_PASS_VARIANTS = ['Combo Rush', 'Double Spy', "Fortune's Wheel", 'Mirror Match',
                         'Pure Strategy', 'Safe Harbour', "Scout's Edge", 'Spy Game']
REVEAL_VARIANTS = ['Ghost Hand', 'Intel War', 'Open Book', 'Reveal Rush']
TRANSFER_VARIANTS = ['Card Exchange', 'Ghost Hand', 'Hand Swap', 'Mirror Match']
LUCK_RANGE      = ['Pure Strategy', 'Safe Harbour', 'Baseline', 'Lucky Draw',
                   "Fortune's Wheel", 'Power Play', 'Double Edge']
SCORING_KEY     = ['Baseline', 'Score Doubler', 'Combo Rush', 'Heavy Toll',
                   'Risk & Reward', 'High Roller', "Gambler's Run"]
ROUND_VARIANTS  = ['Blitz', 'Sprint', 'Baseline', 'Marathon', 'Endurance', 'Slow Burn']


# ===========================================================================
# 10 PARAMETRIC SWEEP GROUPS
# ===========================================================================

def _pgood_ladder():
    """p(good) ladder: 0.0 → 1.0 in steps of 0.1, Wild/Take-2, WTA, 5R, pen=1."""
    return {f'p={p:.2f}': _v(dice_prob=p) for p in np.arange(0.0, 1.01, 0.1)}

def _pass_penalty_ladder():
    """Pass-penalty ladder: 0 → 5, Wild/Forced-Pass, WTA, p=0.50."""
    return {f'pen={pen}': _v(bad=FP, pass_pen=pen) for pen in range(6)}

def _take_n_ladder():
    """Take-N ladder: bad roll takes 1 → 5 cards, Wild/Take-N, WTA, p=0.50."""
    return {f'take={n}': _v(bad_cards=n) for n in range(1, 6)}

def _point_penalty_ladder():
    """Point-penalty ladder: bad roll costs 1 → 5 pts, Wild/NegPts, DP, p=0.50."""
    return {f'-{n}pts': _v(bad=NP, bad_pts=n, scoring=DBL, pts_card=1) for n in range(1, 6)}

def _round_count_ladder():
    """Round-count ladder: 1,2,3,5,8,10,15,20 rounds, Wild/Take-2, WTA, p=0.50."""
    return {f'{r}R': _v(rounds=r) for r in [1, 2, 3, 5, 8, 10, 15, 20]}

def _points_target_ladder():
    """Points-target ladder: 5,10,15,20,25,30 point targets, Wild/Take-2, WTA, p=0.50."""
    return {f'{t}pts': _v(pts_target=t) for t in [5, 10, 15, 20, 25, 30]}

def _scoring_mode_sweep():
    """Scoring-mode sweep: WTA vs DP at p=0.30,0.50,0.70."""
    out = {}
    for sm, sn in [(WTA, 'WTA'), (DBL, 'DP')]:
        for p in [0.30, 0.50, 0.70]:
            out[f'{sn} p={p:.2f}'] = _v(scoring=sm, dice_prob=p)
    return out

def _good_effect_sweep():
    """Good-effect sweep: Wild / DP / Transfer / Info with Take-2, p=0.50, WTA."""
    return {
        'Wild':       _v(good=W),
        'Double Play':_v(good=DP),
        'Transfer':   _v(good=GV),
        'Information':_v(good=INF),
    }

def _bad_effect_sweep():
    """Bad-effect sweep: Take-2/4, ForcedPass, Exposed, -2pts, Wild good, p=0.50, WTA."""
    return {
        'Take 2':     _v(bad=TC, bad_cards=2),
        'Take 4':     _v(bad=TC, bad_cards=4),
        'Forced Pass':_v(bad=FP),
        'Exposed':    _v(bad=RH),
        '-2 pts':     _v(bad=NP, bad_pts=2),
    }

def _pgood_coarse_dp():
    """p(good) coarse DP: DP / -2pts at p=0.0,0.25,0.50,0.75,1.0, DP scoring, pen=2."""
    return {f'p={p:.2f}': _v(dice_prob=p, good=DP, bad=NP, bad_pts=2,
                              scoring=DBL, pass_pen=2)
            for p in [0.0, 0.25, 0.50, 0.75, 1.0]}

VARIANT_GROUPS: Dict[str, Dict[str, VariantConfig]] = {
    'Dice Probability':   _pgood_ladder(),
    'Pass Penalty':       _pass_penalty_ladder(),
    'Cards on Bad Roll':  _take_n_ladder(),
    'Point Penalty':      _point_penalty_ladder(),
    'Round Count':        _round_count_ladder(),
    'Points Target':      _points_target_ladder(),
    'Scoring Mode':       _scoring_mode_sweep(),
    'Good Effect':        _good_effect_sweep(),
    'Bad Effect':         _bad_effect_sweep(),
    'DP Coarse p-sweep':  _pgood_coarse_dp(),
}


# ===========================================================================
# GAME RUNNER
# ===========================================================================

def run_batch(agents: List[Agent],
              variant: VariantConfig,
              num_games: int,
              players_per_game: int = 4,
              rotate: bool = True,
              desc: str = '') -> List[GameResult]:
    """Run num_games games, rotating seat assignments."""
    results: List[GameResult] = []
    n = len(agents)
    ppg = min(players_per_game, n)

    if n <= ppg:
        perms = list(permutations(range(n))) if rotate else [tuple(range(n))]
        seq = [(perms[i % len(perms)],) for i in range(num_games)]
    else:
        combos = list(_combs(range(n), ppg))
        seq = []
        for i in range(num_games):
            combo = list(combos[i % len(combos)])
            if rotate:
                rot = (i // len(combos)) % len(combo)
                combo = combo[rot:] + combo[:rot]
            seq.append((tuple(combo),))

    itr = range(num_games)
    if HAS_TQDM:
        label = f'  {desc[:22]:22s}'
        itr = _tqdm(itr, total=num_games, desc=label, unit='g', ncols=82,
                    leave=False,
                    bar_format='{desc} {bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    for i in itr:
        perm = seq[i][0]
        game_agents = [agents[j] for j in perm]
        gr = GameSimulator.simulate_game(game_agents, variant, verbose=False)
        gr.starting_positions = list(perm)
        results.append(gr)
    return results


# ===========================================================================
# TIMING HELPERS  (used by EXP 16)
# ===========================================================================

def _instrument_timing(agent) -> None:
    """Monkey-patch agent.choose_move to record wall-clock time per call."""
    agent._orig_choose_move = agent.choose_move
    agent._move_times = []  # type: ignore[attr-defined]
    def _timed(*args, **kwargs):
        t0 = time.perf_counter()
        m = agent._orig_choose_move(*args, **kwargs)
        agent._move_times.append(time.perf_counter() - t0)
        return m
    agent.choose_move = _timed


def _restore_timing(agent) -> None:
    """Remove timing patch."""
    if hasattr(agent, '_orig_choose_move'):
        agent.choose_move = agent._orig_choose_move
        del agent._orig_choose_move


def _extract_timing(agent) -> dict:
    """Return timing statistics (ms) from a patched agent."""
    ts = getattr(agent, '_move_times', [])
    if not ts:
        return {'mean_ms': 0.0, 'median_ms': 0.0, 'std_ms': 0.0,
                'p95_ms': 0.0, 'min_ms': 0.0, 'max_ms': 0.0,
                'total_moves': 0}
    ts_ms = [t * 1000.0 for t in ts]
    return {
        'mean_ms':    float(np.mean(ts_ms)),
        'median_ms':  float(np.median(ts_ms)),
        'std_ms':     float(np.std(ts_ms)),
        'p95_ms':     float(np.percentile(ts_ms, 95)),
        'min_ms':     float(np.min(ts_ms)),
        'max_ms':     float(np.max(ts_ms)),
        'total_moves': len(ts),
    }


def _reset_move_times(agent) -> None:
    agent._move_times = []


# ===========================================================================
# STATISTICS
# ===========================================================================

def binomial_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)

def chi_square_uniformity(observed):
    n, k = sum(observed), len(observed)
    if n == 0 or k < 2:
        return {'chi2': 0.0, 'df': k - 1, 'p_value': float('nan')}
    exp = n / k
    chi2 = sum((o - exp)**2 / exp for o in observed)
    df = k - 1
    p = float(1 - scipy_stats.chi2.cdf(chi2, df)) if HAS_SCIPY else float('nan')
    return {'chi2': float(chi2), 'df': df, 'p_value': p}

def gini_coefficient(values):
    arr = np.array(sorted(values), dtype=float)
    if arr.sum() == 0:
        return 0.0
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * idx - n - 1).dot(arr) / (n * arr.sum()))

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return float('nan')
    m1, m2 = np.mean(g1), np.mean(g2)
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    return float((m1 - m2) / pooled) if pooled else 0.0

def kruskal_wallis(groups: Dict[str, list]) -> dict:
    if not HAS_SCIPY or len(groups) < 2:
        return {'H': float('nan'), 'p_value': float('nan')}
    arrs = [np.array(v) for v in groups.values() if len(v) > 0]
    if len(arrs) < 2:
        return {'H': float('nan'), 'p_value': float('nan')}
    H, p = kruskal(*arrs)
    return {'H': float(H), 'p_value': float(p)}


# ===========================================================================
# ANALYSIS FUNCTIONS
# ===========================================================================

def analyse_agents(results: List[GameResult]) -> Dict[str, dict]:
    wins, total, scores = defaultdict(int), defaultdict(int), defaultdict(list)
    for gr in results:
        for i, name in enumerate(gr.player_names):
            total[name] += 1
            if gr.winner == i:
                wins[name] += 1
            scores[name].append(gr.final_scores[i])
    out = {}
    for name in sorted(total):
        w, n = wins[name], total[name]
        lo, hi = binomial_ci(w, n)
        sc = scores[name]
        out[name] = {
            'wins': int(w), 'total': int(n),
            'win_rate': float(w/n) if n else 0.0,
            'ci_low': float(lo), 'ci_high': float(hi),
            'mean_score': float(np.mean(sc)),
            'std_score': float(np.std(sc)),
            'median_score': float(np.median(sc)),
        }
    return out

def analyse_positions(results: List[GameResult], num_players: int) -> dict:
    wins   = [0] * num_players
    totals = [0] * num_players
    scores = [[] for _ in range(num_players)]
    for gr in results:
        for seat in range(min(num_players, len(gr.final_scores))):
            totals[seat] += 1
            if gr.winner == seat:
                wins[seat] += 1
            scores[seat].append(gr.final_scores[seat])
    seats = {}
    for s in range(num_players):
        w, n = wins[s], totals[s]
        lo, hi = binomial_ci(w, n)
        # Use STRING keys so the dict survives JSON serialisation round-trips
        # (json.load converts all object keys to str, so int keys become "0","1"…)
        seats[str(s)] = {
            'wins': int(w), 'total': int(n),
            'win_rate': float(w/n) if n else 0.0,
            'ci_low': float(lo), 'ci_high': float(hi),
            'mean_score': float(np.mean(scores[s])) if scores[s] else 0.0,
            'std_score': float(np.std(scores[s])) if scores[s] else 0.0,
        }
    fair = 1.0 / num_players
    wr_list = [seats[str(s)]['win_rate'] for s in range(num_players)]
    return {
        'seats': seats,
        'chi2': chi_square_uniformity(wins),
        'max_deviation': float(max(abs(r - fair) for r in wr_list)),
        'gini': gini_coefficient(wins),
        'fair_rate': float(fair),
    }

def analyse_lengths(results: List[GameResult]) -> dict:
    lengths = [gr.num_turns for gr in results]
    return {
        'mean': float(np.mean(lengths)), 'median': float(np.median(lengths)),
        'std': float(np.std(lengths)),   'min': int(np.min(lengths)),
        'max': int(np.max(lengths)),
        'q25': float(np.percentile(lengths, 25)),
        'q75': float(np.percentile(lengths, 75)),
        'lengths': [int(x) for x in lengths],
    }

def analyse_score_margins(results: List[GameResult]) -> dict:
    margins, all_final = [], []
    for gr in results:
        sc = sorted(gr.final_scores, reverse=True)
        if len(sc) >= 2:
            margins.append(sc[0] - sc[1])
        all_final.extend(gr.final_scores)
    if not margins:
        return {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'margins': [],
                'score_variance': 0.0, 'tight_pct': 0.0}
    return {
        'mean': float(np.mean(margins)),
        'std': float(np.std(margins)),
        'median': float(np.median(margins)),
        'q25': float(np.percentile(margins, 25)),
        'q75': float(np.percentile(margins, 75)),
        'margins': [float(m) for m in margins],
        'score_variance': float(np.var(all_final)),
        'tight_pct': float(np.mean([m <= 1 for m in margins]) * 100),
    }

def variance_decomposition(results: List[GameResult],
                            agent_names: List[str],
                            num_players: int) -> dict:
    name2idx = {nm: i for i, nm in enumerate(agent_names)}
    agent_ids, seat_ids, outcomes = [], [], []
    for gr in results:
        for seat in range(num_players):
            nm = gr.player_names[seat] if seat < len(gr.player_names) else ''
            agent_ids.append(name2idx.get(nm, 0))
            seat_ids.append(seat)
            outcomes.append(1.0 if gr.winner == seat else 0.0)
    outcomes   = np.array(outcomes)
    grand_mean = np.mean(outcomes)
    ss_total   = np.sum((outcomes - grand_mean)**2)
    if ss_total == 0:
        return {'agent_pct': 0.0, 'seat_pct': 0.0, 'residual_pct': 100.0}

    def _ss(ids, n_levels):
        means = {}
        for lvl in range(n_levels):
            mask = [j for j, id_ in enumerate(ids) if id_ == lvl]
            if mask:
                means[lvl] = float(np.mean(outcomes[mask]))
        pred = np.array([means.get(id_, grand_mean) for id_ in ids])
        return float(np.sum((pred - grand_mean)**2))

    a_pct = _ss(agent_ids, len(agent_names)) / ss_total * 100
    s_pct = _ss(seat_ids, num_players) / ss_total * 100
    return {
        'agent_pct': float(a_pct),
        'seat_pct': float(s_pct),
        'residual_pct': float(max(0.0, 100.0 - a_pct - s_pct)),
    }

def head_to_head(results: List[GameResult], names: List[str]) -> dict:
    n = len(names)
    n2i = {nm: i for i, nm in enumerate(names)}
    pair_wins  = np.zeros((n, n))
    pair_total = np.zeros((n, n))
    for gr in results:
        wi = n2i.get(gr.player_names[gr.winner])
        if wi is None:
            continue
        for nm in gr.player_names:
            li = n2i.get(nm)
            if li is None or li == wi:
                continue
            pair_wins[wi][li] += 1
            pair_total[wi][li] += 1
            pair_total[li][wi] += 1
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if pair_total[i][j] > 0:
                mat[i][j] = pair_wins[i][j] / pair_total[i][j]
    return {'names': names, 'matrix': mat.tolist()}


# ===========================================================================
# SERIALISATION
# ===========================================================================

def results_to_rows(results: List[GameResult]) -> List[dict]:
    rows = []
    for gr in results:
        rows.append({
            'winner': int(gr.winner),
            'num_turns': int(gr.num_turns),
            'player_names': list(gr.player_names),
            'final_scores': [float(s) for s in gr.final_scores],
            'starting_positions': list(getattr(gr, 'starting_positions', list(range(len(gr.player_names))))),
        })
    return rows

def save_csv(results: List[GameResult], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = results_to_rows(results)
    if not rows:
        return
    n_players = len(rows[0]['player_names'])
    fieldnames = ['winner', 'num_turns'] + \
                 [f'p{i}_name' for i in range(n_players)] + \
                 [f'p{i}_score' for i in range(n_players)] + \
                 [f'p{i}_seat' for i in range(n_players)]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            flat = {'winner': row['winner'], 'num_turns': row['num_turns']}
            for i in range(n_players):
                flat[f'p{i}_name']  = row['player_names'][i]
                flat[f'p{i}_score'] = row['final_scores'][i]
                flat[f'p{i}_seat']  = row['starting_positions'][i] if i < len(row['starting_positions']) else i
            w.writerow(flat)

def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    _ok(path)


# ===========================================================================
# EXPERIMENT RUNNER HELPERS
# ===========================================================================

def _run_named_variants(agents, variant_names, N, NP, ddir, exp_id, tag,
                        save_raw=True) -> dict:
    """Run a list of named variants, return aggregated analysis dict."""
    out = {}
    agent_names = [a.name for a in agents]
    for vn in variant_names:
        if vn not in ALL_VARIANTS:
            _warn(f'Variant {vn!r} not found — skipping')
            continue
        vcfg = ALL_VARIANTS[vn]()
        res = run_batch(agents, vcfg, N, players_per_game=NP, desc=vn)
        pos = analyse_positions(res, NP)
        gl  = analyse_lengths(res)
        ag  = analyse_agents(res)
        sm  = analyse_score_margins(res)
        vd  = variance_decomposition(res, agent_names, NP)
        out[vn] = {'pos': pos, 'gl': gl, 'ag': ag, 'sm': sm, 'vd': vd}
        if save_raw:
            csv_path = os.path.join(ddir, f'exp{exp_id}_{tag}_{vn.replace(" ","_").replace("/","_").replace("&","and")}.csv')
            save_csv(res, csv_path)
    return out


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    t_start = time.time()

    # ── CLI ──────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description='Cinquillo 2.0 — Simulation Runner')
    parser.add_argument('--quick',     action='store_true', help='500 games/batch')
    parser.add_argument('--medium',    action='store_true', help='2000 games/batch')
    parser.add_argument('--skip-mcts', action='store_true', help='Skip MCTS experiments')
    parser.add_argument('--skip-rl',   action='store_true', help='Skip RL experiments')
    parser.add_argument('--exp',       nargs='+', type=int, help='Only run these experiment IDs')
    args = parser.parse_args()

    # All experiments use the same game count — 5 000 by default.
    # --medium: 10 000 for extra-tight CIs.
    # --quick : 300  for smoke-testing only (~5 min).
    N      = 300  if args.quick else (10000 if args.medium else 5000)
    NS     = 300  if args.quick else (10000 if args.medium else 5000)
    N_mcts = 300  if args.quick else (10000 if args.medium else 5000)
    NP = 4

    all_exps = list(range(1, 17))
    run_exp  = set(args.exp) if args.exp else set(all_exps)

    # ── Directories ──────────────────────────────────────────────────────────
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ddir_raw  = os.path.join(root, 'output', 'data', 'raw')
    ddir_exp  = os.path.join(root, 'output', 'data', 'experiments')
    for d in (ddir_raw, ddir_exp):
        os.makedirs(d, exist_ok=True)

    # ── Agent pool ───────────────────────────────────────────────────────────
    agents_core = [
        RandomAgent('Random'),
        create_aggressive_heuristic(),
        create_defensive_heuristic(),
        create_balanced_heuristic(),
    ]

    # Optional MCTS — Standard (not Fast) as requested
    mcts_agent = None
    if not args.skip_mcts and HAS_MCTS:
        try:
            mcts_agent = MCTSAgentStandard('MCTS')
        except Exception:
            mcts_agent = MCTSAgentFast('MCTS')   # fallback if Standard unavailable

    # Optional RL — RLAgent has no model_path constructor arg;
    # instantiate first then call load_weights().
    rl_agent = None
    if not args.skip_rl and HAS_RL:
        model_path = os.path.join(root, 'rl_agent', 'models', 'rl_agent.pkl')
        if os.path.exists(model_path):
            try:
                rl_agent = RLAgent('RL', epsilon=0.0)   # pure exploitation
                rl_agent.load_weights(model_path)
                _info(f'RL agent loaded from {model_path}')
            except Exception as e:
                _warn(f'Could not load RL agent: {e}')
                rl_agent = None
        else:
            _warn(f'RL model not found at {model_path}')

    agents_full = agents_core.copy()
    if mcts_agent:
        agents_full.append(mcts_agent)
    if rl_agent:
        agents_full.append(rl_agent)

    _header('Cinquillo 2.0 — Simulation Runner', total=16)
    _info(f'N={N:,} games/batch  |  NS={NS:,} sweep/batch  |  NP={NP}')
    _info(f'Agents: {[a.name for a in agents_full]}')
    _info(f'Experiments: {sorted(run_exp)}')
    _info(f'All 3 variants loaded  |  10 sweep groups loaded')

    # ======================================================================
    # EXP 1 — FIRST-PLAYER ADVANTAGE (all 33 variants)
    # ======================================================================
    if 1 in run_exp:
        _header('First-Player Advantage', 1, 16)
        data = {}
        for vn, vfunc in ALL_VARIANTS.items():
            res = run_batch(agents_core, vfunc(), N, players_per_game=NP, desc=vn)
            data[vn] = analyse_positions(res, NP)
        save_json({'N': N, 'NP': NP, 'variants': data},
                  os.path.join(ddir_exp, 'exp1_first_player.json'))

    # ======================================================================
    # EXP 2 — DICE USAGE vs WIN RATE (Dice Probability sweep)
    # ======================================================================
    if 2 in run_exp:
        _header('Dice Usage vs Win Rate', 2, 16)
        data = {}
        for vn, vcfg in _pgood_ladder().items():
            res = run_batch(agents_core, vcfg, NS, players_per_game=NP, desc=vn)
            data[vn] = {
                'ag':  analyse_agents(res),
                'pos': analyse_positions(res, NP),
                'gl':  analyse_lengths(res),
            }
        save_json({'NS': NS, 'NP': NP, 'sweep': data},
                  os.path.join(ddir_exp, 'exp2_dice_usage.json'))

    # ======================================================================
    # EXP 3 — LUCK vs SKILL (variance decomposition across luck-range variants)
    # ======================================================================
    if 3 in run_exp:
        _header('Luck vs Skill', 3, 16)
        agent_names = [a.name for a in agents_core]
        data = _run_named_variants(agents_core, LUCK_RANGE, N, NP, ddir_raw, 3, 'luck')
        # Add all 33 variant decompositions
        all_vd = {}
        for vn, vfunc in ALL_VARIANTS.items():
            res = run_batch(agents_core, vfunc(), NS, players_per_game=NP, desc=vn)
            all_vd[vn] = variance_decomposition(res, agent_names, NP)
        save_json({'N': N, 'NS': NS, 'NP': NP, 'luck_range': data,
                   'all_vd': all_vd, 'agent_names': agent_names},
                  os.path.join(ddir_exp, 'exp3_luck_skill.json'))

    # ======================================================================
    # EXP 4 — COMEBACK / SNOWBALL
    # ======================================================================
    if 4 in run_exp:
        _header('Comeback / Snowball Study', 4, 16)
        test_vns = ['Baseline', 'Heavy Toll', 'Lucky Draw', 'Endurance',
                    'Pure Strategy', 'Chaos Mode', 'Blitz']
        data = _run_named_variants(agents_core, test_vns, N, NP, ddir_raw, 4, 'snowball')
        save_json({'N': N, 'NP': NP, 'variants': data},
                  os.path.join(ddir_exp, 'exp4_comeback.json'))

    # ======================================================================
    # EXP 5 — AGENT WIN RATES (all agents, key variants)
    # ======================================================================
    if 5 in run_exp:
        _header('Agent Win Rates', 5, 16)
        key_vns = ['Baseline', 'Combo Rush', "Fortune's Wheel", 'Lucky Draw',
                   'Open Book', 'Pure Strategy', 'Score Doubler', 'Intel War']
        data = {}
        for vn in key_vns:
            if vn not in ALL_VARIANTS:
                continue
            # Run heuristics at full N first (fast), then add MCTS/RL at N_mcts
            # to keep total runtime manageable.
            res_heur = run_batch(agents_core, ALL_VARIANTS[vn](), N,
                                 players_per_game=NP, desc=f'{vn} (heuristics)')
            res_all = res_heur
            if mcts_agent or rl_agent:
                extra_agents = agents_core + ([mcts_agent] if mcts_agent else []) \
                                           + ([rl_agent]   if rl_agent   else [])
                res_mcts = run_batch(extra_agents, ALL_VARIANTS[vn](), N_mcts,
                                     players_per_game=NP, desc=f'{vn} (MCTS/RL)')
                res_all = res_heur + res_mcts
            data[vn] = {
                'ag':  analyse_agents(res_all),
                'h2h': head_to_head(res_all, [a.name for a in
                       (agents_core + ([mcts_agent] if mcts_agent else [])
                                    + ([rl_agent]   if rl_agent   else []))]),
                'pos': analyse_positions(res_all, NP),
            }
        all_exp5_agents = agents_core + ([mcts_agent] if mcts_agent else []) \
                                      + ([rl_agent]   if rl_agent   else [])
        save_json({'N': N, 'N_mcts': N_mcts, 'NP': NP,
                   'agent_names': [a.name for a in all_exp5_agents],
                   'variants': data},
                  os.path.join(ddir_exp, 'exp5_agent_winrates.json'))

    # ======================================================================
    # EXP 6 — NUMBER OF PLAYERS STUDY
    # ======================================================================
    if 6 in run_exp:
        _header('Number of Players Study', 6, 16)
        test_vns = ['Baseline', 'Pure Strategy', 'Lucky Draw', 'Combo Rush']
        data = {}
        for npl in [2, 3, 4]:
            data[str(npl)] = {}
            for vn in test_vns:
                res = run_batch(agents_core[:npl], ALL_VARIANTS[vn](), NS,
                               players_per_game=npl, desc=f'{npl}P-{vn[:10]}')
                data[str(npl)][vn] = {
                    'pos': analyse_positions(res, npl),
                    'gl':  analyse_lengths(res),
                    'ag':  analyse_agents(res),
                }
        save_json({'NS': NS, 'variants': test_vns, 'nplayer_data': data},
                  os.path.join(ddir_exp, 'exp6_nplayers.json'))

    # ======================================================================
    # EXP 7 — VARIANT FAIRNESS & BALANCE (all 33 variants)
    # ======================================================================
    if 7 in run_exp:
        _header('Variant Fairness & Balance', 7, 16)
        fairness = {}
        for vn, vfunc in ALL_VARIANTS.items():
            res = run_batch(agents_core, vfunc(), NS, players_per_game=NP, desc=vn)
            fairness[vn] = {
                'pos': analyse_positions(res, NP),
                'gl':  analyse_lengths(res),
                'sm':  analyse_score_margins(res),
            }
        # Group-level ginis
        group_fairness = {}
        for gname, gvariants in VARIANT_GROUPS.items():
            group_fairness[gname] = {}
            for vname, vcfg in gvariants.items():
                res = run_batch(agents_core, vcfg, NS//2,
                               players_per_game=NP, desc=f'{gname[:10]}:{vname[:8]}')
                group_fairness[gname][vname] = {
                    'pos': analyse_positions(res, NP),
                    'gl':  analyse_lengths(res),
                }
        save_json({'NS': NS, 'NP': NP, 'fairness': fairness,
                   'group_fairness': group_fairness},
                  os.path.join(ddir_exp, 'exp7_fairness.json'))

    # ======================================================================
    # EXP 8 — INFORMATION REVEAL ANALYSIS (new)
    # ======================================================================
    if 8 in run_exp:
        _header('Information Reveal Analysis', 8, 16)
        # Compare INFO_REVEAL vs REVEAL_HAND dynamics
        data = _run_named_variants(agents_core, INFO_VARIANTS + REVEAL_VARIANTS,
                                   N, NP, ddir_raw, 8, 'info')

        # Key question: does having INFO_REVEAL help or hurt different agent styles?
        # Compare Defensive (blocks well) vs Aggressive (just plays fast)
        # across variants that use INFO_REVEAL vs those that don't
        info_sweep = {}
        for p_info in [0.0, 0.25, 0.50, 0.75, 1.0]:
            vcfg = _v(dice_prob=p_info, good=INF, bad=RH)
            key = f'p_info={p_info:.2f}'
            res = run_batch(agents_core, vcfg, NS, players_per_game=NP, desc=key)
            info_sweep[key] = {
                'ag':  analyse_agents(res),
                'pos': analyse_positions(res, NP),
                'gl':  analyse_lengths(res),
            }

        # Heuristic benefit: Defensive heuristic uses revealed info best
        # → compare Defensive win rate with/without INFO_REVEAL
        info_benefit = {}
        for vn in ['Baseline', 'Open Book', 'Intel War', "Scout's Edge",
                   'Pass & Peek', 'Spy Game']:
            if vn not in ALL_VARIANTS:
                continue
            res = run_batch(agents_core, ALL_VARIANTS[vn](), N,
                           players_per_game=NP, desc=vn)
            info_benefit[vn] = analyse_agents(res)

        save_json({
            'N': N, 'NS': NS, 'NP': NP,
            'info_variants': data,
            'info_sweep': info_sweep,       # p(info) vs agent win rates
            'info_benefit': info_benefit,   # per-agent breakdown by variant
        }, os.path.join(ddir_exp, 'exp8_information.json'))

    # ======================================================================
    # EXP 9 — DOUBLE PLAY DEEP DIVE (new)
    # ======================================================================
    if 9 in run_exp:
        _header('Double Play Deep Dive', 9, 16)
        data = _run_named_variants(agents_core, DOUBLE_PLAY_VARIANTS,
                                   N, NP, ddir_raw, 9, 'dp')

        # p(good) sweep with DOUBLE_PLAY good effect
        dp_sweep = {}
        for p_good in np.arange(0.0, 1.01, 0.1):
            vcfg = _v(dice_prob=float(p_good), good=DP, bad=FP)
            key  = f'p={p_good:.2f}'
            res  = run_batch(agents_core, vcfg, NS, players_per_game=NP, desc=key)
            dp_sweep[key] = {
                'ag':  analyse_agents(res),
                'gl':  analyse_lengths(res),
                'sm':  analyse_score_margins(res),
            }

        # Aggressive vs Defensive: who benefits most from DOUBLE_PLAY?
        dp_penalty_sweep = {}
        for pen in range(6):
            vcfg = _v(good=DP, bad=FP, pass_pen=pen)
            key  = f'pen={pen}'
            res  = run_batch(agents_core, vcfg, NS, players_per_game=NP, desc=key)
            dp_penalty_sweep[key] = analyse_agents(res)

        save_json({
            'N': N, 'NS': NS, 'NP': NP,
            'dp_variants': data,
            'dp_prob_sweep': dp_sweep,
            'dp_penalty_sweep': dp_penalty_sweep,
        }, os.path.join(ddir_exp, 'exp9_double_play.json'))

    # ======================================================================
    # EXP 10 — FORCED PASS DYNAMICS (new)
    # ======================================================================
    if 10 in run_exp:
        _header('Forced Pass Dynamics', 10, 16)
        data = _run_named_variants(agents_core, FORCED_PASS_VARIANTS,
                                   N, NP, ddir_raw, 10, 'fp')

        # Sweep: probability of forced-pass outcome
        fp_sweep = {}
        for p_fp in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            vcfg = _v(dice_prob=1.0 - p_fp, bad=FP)  # p(bad) = p_fp
            key  = f'p_forced={p_fp:.2f}'
            res  = run_batch(agents_core, vcfg, NS, players_per_game=NP, desc=key)
            fp_sweep[key] = {
                'ag':  analyse_agents(res),
                'gl':  analyse_lengths(res),
                'pos': analyse_positions(res, NP),
            }

        # Pure Strategy (dice_prob=0 → always FP) vs Baseline
        comparison = {}
        for vn in ['Pure Strategy', 'Baseline', 'Safe Harbour', 'Combo Rush']:
            res = run_batch(agents_core, ALL_VARIANTS[vn](), N,
                           players_per_game=NP, desc=vn)
            comparison[vn] = {
                'ag': analyse_agents(res),
                'gl': analyse_lengths(res),
                'vd': variance_decomposition(res, [a.name for a in agents_core], NP),
            }

        save_json({
            'N': N, 'NS': NS, 'NP': NP,
            'fp_variants': data,
            'fp_prob_sweep': fp_sweep,
            'comparison': comparison,
        }, os.path.join(ddir_exp, 'exp10_forced_pass.json'))

    # ======================================================================
    # EXP 11 — KNOWLEDGE ADVANTAGE (new — information asymmetry experiment)
    # ======================================================================
    if 11 in run_exp:
        _header('Knowledge Advantage Experiment', 11, 16)
        # Run Balanced heuristic (which uses info when available) vs others
        # on info-heavy variants
        info_heavy = ['Open Book', 'Intel War', "Scout's Edge", 'Spy Game',
                      'Ghost Hand', 'Reveal Rush']
        info_light = ['Pure Strategy', 'Baseline', 'Lucky Draw']

        def _run_and_tag(vnames, tag):
            out = {}
            for vn in vnames:
                if vn not in ALL_VARIANTS:
                    continue
                res = run_batch(agents_core, ALL_VARIANTS[vn](), N,
                               players_per_game=NP, desc=vn)
                out[vn] = {
                    'ag':  analyse_agents(res),
                    'vd':  variance_decomposition(res, [a.name for a in agents_core], NP),
                    'pos': analyse_positions(res, NP),
                }
            return out

        # Cohen's d: Defensive agent score in info variants vs non-info variants
        def _scores_for(results, name):
            return [gr.final_scores[i]
                    for gr in results
                    for i, nm in enumerate(gr.player_names) if nm == name]

        cohens_info = {}
        for agent_n in [a.name for a in agents_core]:
            scores_info  = []
            scores_light = []
            for vn in info_heavy:
                if vn not in ALL_VARIANTS:
                    continue
                res = run_batch(agents_core, ALL_VARIANTS[vn](), NS//2,
                               players_per_game=NP, desc=f'{agent_n[:6]}-{vn[:8]}')
                scores_info.extend(_scores_for(res, agent_n))
            for vn in info_light:
                res = run_batch(agents_core, ALL_VARIANTS[vn](), NS//2,
                               players_per_game=NP, desc=f'{agent_n[:6]}-{vn[:8]}')
                scores_light.extend(_scores_for(res, agent_n))
            cohens_info[agent_n] = {
                'd': cohens_d(scores_info, scores_light),
                'mean_info': float(np.mean(scores_info)) if scores_info else 0.0,
                'mean_light': float(np.mean(scores_light)) if scores_light else 0.0,
            }

        save_json({
            'N': N, 'NS': NS, 'NP': NP,
            'info_heavy': _run_and_tag(info_heavy, 'heavy'),
            'info_light': _run_and_tag(info_light, 'light'),
            'cohens_d_info_vs_light': cohens_info,
        }, os.path.join(ddir_exp, 'exp11_knowledge_advantage.json'))

    # ======================================================================
    # EXP 12 — ALL-VARIANT TOURNAMENT (agent rankings across all 33 variants)
    # ======================================================================
    if 12 in run_exp:
        _header('All-Variant Tournament', 12, 16)
        ranking = {}
        for vn, vfunc in ALL_VARIANTS.items():
            res = run_batch(agents_core, vfunc(), NS,
                           players_per_game=NP, desc=vn)
            ag = analyse_agents(res)
            ranking[vn] = {nm: d['win_rate'] for nm, d in ag.items()}
        save_json({'NS': NS, 'NP': NP,
                   'agent_names': [a.name for a in agents_core],
                   'rankings': ranking},
                  os.path.join(ddir_exp, 'exp12_all_variant_tournament.json'))

    # ======================================================================
    # EXP 13 — PARAMETRIC SWEEP ANALYSIS (all 10 groups)
    # ======================================================================
    if 13 in run_exp:
        _header('Parametric Sweep Analysis', 13, 16)
        all_groups = {}
        for gname, gvariants in VARIANT_GROUPS.items():
            _sub(f'Group: {gname}  ({len(gvariants)} configs)')
            all_groups[gname] = {}
            for vname, vcfg in gvariants.items():
                res = run_batch(agents_core, vcfg, NS,
                               players_per_game=NP, desc=f'{gname[:10]}:{vname[:8]}')
                all_groups[gname][vname] = {
                    'ag':  analyse_agents(res),
                    'pos': analyse_positions(res, NP),
                    'gl':  analyse_lengths(res),
                    'sm':  analyse_score_margins(res),
                }
        kw_results = {}
        for gname, gdata in all_groups.items():
            win_groups = {vn: [gdata[vn]['ag'].get('Balanced', {}).get('win_rate', 0)]
                          for vn in gdata}
            kw_results[gname] = kruskal_wallis(win_groups)
        save_json({'NS': NS, 'NP': NP, 'groups': all_groups,
                   'kruskal_wallis': kw_results},
                  os.path.join(ddir_exp, 'exp13_parametric_sweep.json'))

    # ======================================================================
    # EXP 14 — TARGET SCORE vs FIXED ROUNDS COMPARISON (new)
    # ======================================================================
    if 14 in run_exp:
        _header('Target Score vs Fixed Rounds Comparison', 14, 16)
        # Compare game dynamics between the two end-condition modes
        pts_based_vns  = ['Card Exchange', 'Chaos Mode', "Gambler's Run", 'Ghost Hand',
                          'Heavy Toll', 'High Roller', 'Intel War', 'Open Book',
                          'Point Race', 'Risk & Reward', 'Score Doubler', 'Spy Game']
        round_based_vns = ['Baseline', 'Blitz', 'Combo Rush', 'Double Edge',
                           'Endurance', 'Marathon', 'Pure Strategy', 'Sprint',
                           'Safe Harbour', 'Slow Burn']

        pts_data   = _run_named_variants(agents_core, pts_based_vns,  NS, NP, ddir_raw, 14, 'pts')
        round_data = _run_named_variants(agents_core, round_based_vns, NS, NP, ddir_raw, 14, 'rounds')

        # Aggregate comparisons
        def _agg_gl(d):
            means = [v['gl']['mean'] for v in d.values()]
            return {'mean_of_means': float(np.mean(means)),
                    'std_of_means': float(np.std(means))}

        def _agg_gini(d):
            ginis = [v['pos']['gini'] for v in d.values()]
            return {'mean_gini': float(np.mean(ginis)),
                    'std_gini': float(np.std(ginis))}

        save_json({
            'NS': NS, 'NP': NP,
            'pts_based':   pts_data,
            'round_based': round_data,
            'pts_agg_gl':    _agg_gl(pts_data),
            'round_agg_gl':  _agg_gl(round_data),
            'pts_agg_gini':  _agg_gini(pts_data),
            'round_agg_gini':_agg_gini(round_data),
        }, os.path.join(ddir_exp, 'exp14_end_condition.json'))

    # ======================================================================
    # EXP 15 — TRANSFER EFFECT & HAND SIZE DYNAMICS (new)
    # ======================================================================
    if 15 in run_exp:
        _header('Transfer Effect & Hand Size Dynamics', 15, 16)
        data = _run_named_variants(agents_core, TRANSFER_VARIANTS,
                                   N, NP, ddir_raw, 15, 'transfer')
        # Sweep: p(good) when good effect = GIVE_CARD
        transfer_sweep = {}
        for p_gv in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            for bad in [(TC, 'take'), (FP, 'fp'), (RH, 'rh')]:
                vcfg = _v(dice_prob=p_gv, good=GV, bad=bad[0])
                key  = f'p={p_gv:.2f}_{bad[1]}'
                res  = run_batch(agents_core, vcfg, NS, players_per_game=NP, desc=key)
                transfer_sweep[key] = {
                    'ag':  analyse_agents(res),
                    'gl':  analyse_lengths(res),
                }
        save_json({
            'N': N, 'NS': NS, 'NP': NP,
            'transfer_variants': data,
            'transfer_sweep': transfer_sweep,
        }, os.path.join(ddir_exp, 'exp15_transfer.json'))

    # ======================================================================
    # EXP 16 — MCTS BENCHMARK  (win rate + time per move)
    # ======================================================================
    if 16 in run_exp and not args.skip_mcts:
        _header('MCTS Benchmark — Win Rate & Time per Move', 16, 16)

        if not HAS_MCTS:
            _warn('MCTS not available — skipping EXP 16')
        else:
            # ── Agent definitions ──────────────────────────────────────────
            mcts_configs = [
                ('MCTS-SuperFast', MCTSAgentSuperFast),
                ('MCTS-Fast',      MCTSAgentFast),
                ('MCTS-Standard',  MCTSAgentStandard),
                ('MCTS-Deep',      MCTSAgentDeep),
            ]
            # Use the three core heuristics (no Random — too weak as baseline)
            heuristics = [
                create_aggressive_heuristic(),
                create_defensive_heuristic(),
                create_balanced_heuristic(),
            ]
            # Representative variants: no-luck, medium-luck, high-luck, forced-pass
            bench_variants = ['Pure Strategy', 'Baseline', 'Chaos Mode', 'Combo Rush']
            # Tiny game count — MCTS is ~50–200x slower than heuristics;
            # Deep agent can take ~2 s/move, so even 50 games ≈ 25 min per variant.
            N_b = N_mcts

            _sub(f'Games per batch: {N_b}  |  Benchmark variants: {bench_variants}')
            _sub(f'MCTS configs: {[c[0] for c in mcts_configs]}')

            # ------------------------------------------------------------------
            # Helper: run one timed batch and return (results, timing_dict).
            # Agents that are NOT patched (heuristics) are ignored for timing.
            # ------------------------------------------------------------------
            def _timed_batch(agents_list, vcfg, n_games, desc=''):
                timed_agents = [a for a in agents_list
                                if any(a.name.startswith(tag)
                                       for tag in ('MCTS',))]
                for a in timed_agents:
                    _instrument_timing(a)
                    _reset_move_times(a)

                res = run_batch(agents_list, vcfg, n_games,
                                players_per_game=NP, desc=desc)

                timing = {}
                for a in timed_agents:
                    timing[a.name] = _extract_timing(a)
                    _restore_timing(a)
                return res, timing

            # ------------------------------------------------------------------
            # A. HEAD-TO-HEAD: all four MCTS in one 4-player game
            # ------------------------------------------------------------------
            _sub('Sub-exp A: MCTS head-to-head tournament (all 4 variants)')
            mcts_agents_h2h = [cls(name) for name, cls in mcts_configs]

            h2h_data = {}
            for vn in bench_variants:
                if vn not in ALL_VARIANTS:
                    _warn(f'{vn!r} not in ALL_VARIANTS — skipping'); continue
                vcfg = ALL_VARIANTS[vn]()
                res, timing = _timed_batch(
                    mcts_agents_h2h, vcfg, N_b, desc=f'H2H-{vn[:10]}')
                h2h_data[vn] = {
                    'ag':     analyse_agents(res),
                    'h2h':    head_to_head(res, [a.name for a in mcts_agents_h2h]),
                    'timing': timing,
                }
                csv_path = os.path.join(
                    ddir_raw,
                    f'exp16_h2h_{vn.replace(" ","_").replace("/","_").replace("&","and")}.csv')
                save_csv(res, csv_path)

            # ------------------------------------------------------------------
            # B. MCTS vs HEURISTICS: each MCTS paired with all three heuristics
            # ------------------------------------------------------------------
            _sub('Sub-exp B: each MCTS variant vs Aggressive/Defensive/Balanced')
            vs_heuristics_data = {}
            timing_summary_accum: Dict[str, List[float]] = {
                name: [] for name, _ in mcts_configs}

            for mcts_name, mcts_cls in mcts_configs:
                vs_heuristics_data[mcts_name] = {}
                mcts_ag = mcts_cls(mcts_name)
                team = [mcts_ag] + heuristics   # 4-player game

                for vn in bench_variants:
                    if vn not in ALL_VARIANTS:
                        continue
                    vcfg = ALL_VARIANTS[vn]()
                    res, timing = _timed_batch(
                        team, vcfg, N_b,
                        desc=f'{mcts_name[:10]}-{vn[:8]}')
                    vs_heuristics_data[mcts_name][vn] = {
                        'ag':     analyse_agents(res),
                        'timing': timing.get(mcts_name, {}),
                    }
                    # Accumulate raw move times for cross-variant summary
                    raw_ms = getattr(mcts_ag, '_move_times', [])
                    timing_summary_accum[mcts_name].extend(
                        [t * 1000.0 for t in raw_ms])

                    csv_path = os.path.join(
                        ddir_raw,
                        f'exp16_vsh_{mcts_name.replace("-","_")}'
                        f'_{vn.replace(" ","_").replace("/","_").replace("&","and")}.csv')
                    save_csv(res, csv_path)

            # ------------------------------------------------------------------
            # C. Cross-variant timing summary (pooled over all benchmark variants)
            # ------------------------------------------------------------------
            timing_summary = {}
            for mcts_name, ts_ms in timing_summary_accum.items():
                if ts_ms:
                    timing_summary[mcts_name] = {
                        'mean_ms':    float(np.mean(ts_ms)),
                        'median_ms':  float(np.median(ts_ms)),
                        'std_ms':     float(np.std(ts_ms)),
                        'p95_ms':     float(np.percentile(ts_ms, 95)),
                        'min_ms':     float(np.min(ts_ms)),
                        'max_ms':     float(np.max(ts_ms)),
                        'total_moves': len(ts_ms),
                    }
                else:
                    timing_summary[mcts_name] = {
                        'mean_ms': 0.0, 'median_ms': 0.0, 'std_ms': 0.0,
                        'p95_ms': 0.0, 'min_ms': 0.0, 'max_ms': 0.0,
                        'total_moves': 0,
                    }

            # Mean win rate vs heuristics, averaged across benchmark variants
            mean_wr_vs_heuristics = {}
            for mcts_name in [n for n, _ in mcts_configs]:
                wrs = []
                for vn in bench_variants:
                    ag_data = vs_heuristics_data.get(mcts_name, {}).get(vn, {}).get('ag', {})
                    wr = ag_data.get(mcts_name, {}).get('win_rate', None)
                    if wr is not None:
                        wrs.append(wr)
                mean_wr_vs_heuristics[mcts_name] = float(np.mean(wrs)) if wrs else 0.0

            save_json({
                'N_b': N_b, 'NP': NP,
                'mcts_variants':       [n for n, _ in mcts_configs],
                'bench_variants':      bench_variants,
                'head_to_head':        h2h_data,
                'vs_heuristics':       vs_heuristics_data,
                'timing_summary':      timing_summary,
                'mean_wr_vs_heuristics': mean_wr_vs_heuristics,
            }, os.path.join(ddir_exp, 'exp16_mcts_benchmark.json'))

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f'\n{"━"*72}')
    print(f'  SIMULATIONS COMPLETE  —  elapsed: {elapsed/60:.1f} min')
    print(f'{"━"*72}')
    for label, d in [('Experiments', ddir_exp), ('Raw CSV', ddir_raw)]:
        if os.path.isdir(d):
            files = sorted(os.listdir(d))
            kb = sum(os.path.getsize(os.path.join(d, f)) for f in files if os.path.isfile(os.path.join(d, f))) // 1024
            print(f'  {label:<14} {len(files):>3} files  ({kb} KB)  → {d}/')
    print(f'\n  Next step:  python simulation/visualise_results.py\n')


if __name__ == '__main__':
    main()