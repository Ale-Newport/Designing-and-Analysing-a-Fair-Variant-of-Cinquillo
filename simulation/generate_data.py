#!/usr/bin/env python3
"""
==========================================================================
Dissertation Data Generator for Cinquillo 2.0  —  v2.1
==========================================================================
Generates ALL tables, figures, and statistical analyses for the report.

Place this file at:  simulation/generate_data.py
Run from the PROJECT ROOT:
    cd PRJ
    python simulation/generate_data.py
    python simulation/generate_data.py --quick          # 1 000 games (fast)
    python simulation/generate_data.py --medium         # 5 000 games
    python simulation/generate_data.py --skip-mcts      # skip slow MCTS
    python simulation/generate_data.py --skip-rl        # skip RL agent
    python simulation/generate_data.py --exp 1 4 5      # only exps 1, 4, 5

Outputs:
    output/
    ├── figures/           # PDF figures for LaTeX (includegraphics)
    ├── tables/            # LaTeX table fragments (\\input)
    └── data/              # Raw CSV data

EXPERIMENTS:
    1.  First-Player Advantage
    2.  Dice Usage vs Win Rate
    3.  Luck vs Skill
    4.  Comeback / Snowball Study
    5.  Heuristic Agent Analysis
    6.  Number of Players Study
    7.  Variant Fairness & Balance
    8.  Additional Diagnostics (RL curve, architecture)
==========================================================================
"""

# ---------------------------------------------------------------------------
# Path fix – must come before project imports
# ---------------------------------------------------------------------------
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import csv
import warnings
from collections import defaultdict
from itertools import permutations, combinations as _combs
from typing import List, Dict, Tuple, Optional

import numpy as np
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from game.entities import (
    Card, Suit, Deck, Player, Board, GameState,
    VariantConfig, GoodDiceEffect, BadDiceEffect,
    ScoringMode, MatchEndMode,
)
from game.rules import Rules, PlayCard, RollDice, Pass
from agents.base_agents import (
    Agent, RandomAgent, HeuristicAgent,
    create_aggressive_heuristic, create_defensive_heuristic,
    create_balanced_heuristic, create_risky_heuristic,
)
from agents.mcts_agent import (
    MCTSAgent, MCTSAgentFast, MCTSAgentStandard,
    MCTSAgentDeep, MCTSAgentSuperFast,
)
from simulation.tournament import GameSimulator, Tournament, GameResult, TournamentResult

try:
    from agents.rl_agent import RLAgent, RLAgentExplore, RLAgentExploit
    HAS_RL = True
except ImportError:
    HAS_RL = False

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    import matplotlib.cm as cm

    # Version-aware helpers
    _MPL_VER = tuple(int(x) for x in matplotlib.__version__.split('.')[:2])

    def _get_cmap(name, n=None):
        try:
            cmap = matplotlib.colormaps[name]
        except (AttributeError, KeyError):
            cmap = cm.get_cmap(name)           # old matplotlib fallback
        return cmap.resampled(n) if n is not None else cmap

    # boxplot 'labels' was renamed 'tick_labels' in 3.9
    _BP_LABEL = 'tick_labels' if _MPL_VER >= (3, 9) else 'labels'

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not found – figures will not be generated.")

try:
    from scipy import stats as scipy_stats
    from scipy.stats import kruskal, mannwhitneyu
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ===========================================================================
# FIGURE SETTINGS
# ===========================================================================
if HAS_MATPLOTLIB:
    plt.rcParams.update({
        'font.size': 11, 'font.family': 'serif',
        'axes.labelsize': 12, 'axes.titlesize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.figsize': (7, 5), 'figure.dpi': 300,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.12,
    })

AGENT_COLOURS = {
    'Random': '#E69F00', 'Aggressive': '#56B4E9', 'Defensive': '#009E73',
    'Balanced': '#0072B2', 'Risky': '#CC79A7', 'MCTS': '#D55E00',
    'MCTS-Fast': '#D55E00', 'MCTS-Standard': '#E8A070',
    'MCTS-Deep': '#A0522D', 'RL': '#F0E442', 'RL-v4': '#F0E442',
}
SEAT_COLOURS = ['#56B4E9', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#999999']


# ===========================================================================
# TERMINAL OUTPUT HELPERS
# ===========================================================================

def _header(title: str, exp_num: int = 0, total: int = 8):
    bar = '━' * 72
    tag = f'EXP {exp_num}/{total}  ' if exp_num else ''
    print(f'\n{bar}')
    print(f'  {tag}{title}')
    print(bar)

def _sub(msg: str):
    print(f'  ▸ {msg}')

def _ok(path: str):
    fname = os.path.basename(path)
    print(f'    ✓ {fname}')

def _info(msg: str):
    print(f'    {msg}')

def _warn(msg: str):
    print(f'  ⚠  {msg}')

def _run_note(n: int, desc: str = ''):
    tag = f' ({desc})' if desc else ''
    print(f'  Running {n:,} games{tag}...')


# ===========================================================================
# VARIANT DEFINITIONS
# ===========================================================================

def _v(dice_prob=0.5,
       good_effect=GoodDiceEffect.WILD,
       bad_effect=BadDiceEffect.TAKE_CARDS,
       bad_cards=2, bad_neg_pts=2,
       scoring=ScoringMode.WINNER_TAKES_ALL,
       pts_per_card=1, pass_penalty=1, rounds=5) -> VariantConfig:
    """Convenience VariantConfig factory."""
    return VariantConfig(
        dice_good_probability=dice_prob,
        dice_good_effect=good_effect,
        dice_bad_effect=bad_effect,
        dice_bad_cards_count=bad_cards,
        dice_bad_penalty_points=bad_neg_pts,
        scoring_mode=scoring,
        points_per_card=pts_per_card,
        voluntary_pass_penalty=pass_penalty,
        match_end_mode=MatchEndMode.FIXED_ROUNDS,
        fixed_rounds_count=rounds,
    )

# ── Named canonical variants ─────────────────────────────────────────────────

def make_baseline():
    """Baseline – 50/50 dice, WILD / TAKE_CARDS, Winner-Takes-All, 5 rounds."""
    return _v()

def make_score_doubler():
    """Score Doubler – same dice as Baseline but losers also lose points (Double Penalty)."""
    return _v(scoring=ScoringMode.DOUBLE_PENALTY)

def make_combo_rush():
    """Combo Rush – 60% DOUBLE_PLAY / FORCED_PASS; higher pass penalty amplifies aggression."""
    return _v(dice_prob=0.6, good_effect=GoodDiceEffect.DOUBLE_PLAY,
              bad_effect=BadDiceEffect.FORCED_PASS,
              scoring=ScoringMode.DOUBLE_PENALTY, pass_penalty=2)

def make_heavy_toll():
    """Heavy Toll – 40% WILD; bad roll loses 3 pts; Double Penalty; 2pts/card; 3pt pass fee."""
    return _v(dice_prob=0.4, bad_effect=BadDiceEffect.NEGATIVE_POINTS, bad_neg_pts=3,
              scoring=ScoringMode.DOUBLE_PENALTY, pts_per_card=2, pass_penalty=3)

def make_pure_strategy():
    """Pure Strategy – 0% good dice (always bad: Forced Pass); skill only, no luck."""
    return _v(dice_prob=0.0, bad_effect=BadDiceEffect.FORCED_PASS)

def make_lucky_draw():
    """Lucky Draw – 80% good dice; mild negative-points penalty on bad roll."""
    return _v(dice_prob=0.8, bad_effect=BadDiceEffect.NEGATIVE_POINTS, bad_neg_pts=1)

def make_fortunes_wheel():
    """Fortune's Wheel – 100% good dice (always WILD); pure luck ceiling."""
    return _v(dice_prob=1.0)

def make_power_play():
    """Power Play – 70% DOUBLE_PLAY / NEGATIVE_POINTS (−3); Double Penalty; 2pts/card."""
    return _v(dice_prob=0.7, good_effect=GoodDiceEffect.DOUBLE_PLAY,
              bad_effect=BadDiceEffect.NEGATIVE_POINTS, bad_neg_pts=3,
              scoring=ScoringMode.DOUBLE_PENALTY, pts_per_card=2, pass_penalty=2)

def make_safe_harbour():
    """Safe Harbour – Forced Pass on bad roll; no pass penalty; low-variance play."""
    return _v(bad_effect=BadDiceEffect.FORCED_PASS, pass_penalty=0)

def make_marathon():
    """Marathon – 10 rounds; otherwise Baseline rules. Tests long-game snowball."""
    return _v(rounds=10)

def make_sprint():
    """Sprint – 3 rounds; otherwise Baseline rules. Tests short-game first-move edge."""
    return _v(rounds=3)

def make_double_edge():
    """Double Edge – 50% DOUBLE_PLAY / TAKE_CARDS; no pass penalty. Combo-heavy."""
    return _v(good_effect=GoodDiceEffect.DOUBLE_PLAY, pass_penalty=0)

def make_risk_and_reward():
    """Risk & Reward – NEGATIVE_POINTS −4 on bad roll; heavy penalty for dice usage."""
    return _v(bad_effect=BadDiceEffect.NEGATIVE_POINTS, bad_neg_pts=4,
              scoring=ScoringMode.DOUBLE_PENALTY, pass_penalty=3)

def make_card_flood():
    """Card Flood – TAKE_CARDS(4) on bad roll; extreme hand-size swings."""
    return _v(bad_effect=BadDiceEffect.TAKE_CARDS, bad_cards=4)

def make_tactical_blend():
    """Tactical Blend – 55% DOUBLE_PLAY / TAKE_CARDS; balanced risk–reward."""
    return _v(dice_prob=0.55, good_effect=GoodDiceEffect.DOUBLE_PLAY, pass_penalty=1)

def make_high_roller():
    """High Roller – 60% WILD; Double Penalty; 3pts/card; high-stakes score swings."""
    return _v(dice_prob=0.6, bad_effect=BadDiceEffect.NEGATIVE_POINTS, bad_neg_pts=2,
              scoring=ScoringMode.DOUBLE_PENALTY, pts_per_card=3, pass_penalty=2)

def make_gamblers_run():
    """Gambler's Run – 45% DOUBLE_PLAY / NEGATIVE_POINTS (−2); Double Penalty; 2pts."""
    return _v(dice_prob=0.45, good_effect=GoodDiceEffect.DOUBLE_PLAY,
              bad_effect=BadDiceEffect.NEGATIVE_POINTS, bad_neg_pts=2,
              scoring=ScoringMode.DOUBLE_PENALTY, pts_per_card=2, pass_penalty=1)

def make_sudden_death():
    """Sudden Death – 2 rounds only; maximum first-mover urgency."""
    return _v(rounds=2)

def make_endurance():
    """Endurance – 15 rounds; WTA scoring; tests whether skill compounds over time."""
    return _v(rounds=15)


# ── Full catalogue ────────────────────────────────────────────────────────────

ALL_VARIANTS: Dict[str, callable] = {
    'Baseline':         make_baseline,
    'Score Doubler':    make_score_doubler,
    'Combo Rush':       make_combo_rush,
    'Heavy Toll':       make_heavy_toll,
    'Pure Strategy':    make_pure_strategy,
    'Lucky Draw':       make_lucky_draw,
    "Fortune's Wheel":  make_fortunes_wheel,
    'Power Play':       make_power_play,
    'Safe Harbour':     make_safe_harbour,
    'Marathon':         make_marathon,
    'Sprint':           make_sprint,
    'Double Edge':      make_double_edge,
    'Risk & Reward':    make_risk_and_reward,
    'Card Flood':       make_card_flood,
    'Tactical Blend':   make_tactical_blend,
    'High Roller':      make_high_roller,
    "Gambler's Run":    make_gamblers_run,
    'Sudden Death':     make_sudden_death,
    'Endurance':        make_endurance,
}

# Subsets used by specific experiments
SCORING_VARIANTS  = ['Baseline', 'Score Doubler', 'Combo Rush', 'Heavy Toll',
                     'Risk & Reward', 'High Roller', "Gambler's Run"]
LUCK_RANGE        = ['Pure Strategy', 'Safe Harbour', 'Baseline', 'Lucky Draw',
                     "Fortune's Wheel", 'Power Play', 'Double Edge']
ROUND_VARIANTS    = ['Sudden Death', 'Sprint', 'Baseline', 'Marathon', 'Endurance']
FAIRNESS_KEY      = list(ALL_VARIANTS.keys())   # all variants for fairness study


# ── Variant groups (parametric sweeps) ───────────────────────────────────────

def _dice_prob_series():
    return {f'p={p:.2f}': _v(dice_prob=p)
            for p in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}

def _good_effect_series():
    out = {}
    for en, ef in [('WILD', GoodDiceEffect.WILD), ('DOUBLE_PLAY', GoodDiceEffect.DOUBLE_PLAY)]:
        for p in [0.3, 0.5, 0.7]:
            out[f'{en} p={p:.1f}'] = _v(dice_prob=p, good_effect=ef)
    return out

def _bad_effect_series():
    out = {}
    for bn, be in [('TakeCards', BadDiceEffect.TAKE_CARDS),
                   ('ForcedPass', BadDiceEffect.FORCED_PASS),
                   ('NegPoints',  BadDiceEffect.NEGATIVE_POINTS)]:
        for p in [0.3, 0.5, 0.7]:
            out[f'{bn} p={p:.1f}'] = _v(dice_prob=p, bad_effect=be, bad_neg_pts=2)
    return out

def _pass_penalty_series():
    return {f'Penalty={pen}': _v(pass_penalty=pen) for pen in [0, 1, 2, 3, 4]}

def _round_count_series():
    return {f'{r} rounds': _v(rounds=r) for r in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]}

def _bad_cards_series():
    return {f'Draw {n}': _v(bad_effect=BadDiceEffect.TAKE_CARDS, bad_cards=n)
            for n in [1, 2, 3, 4, 5]}

def _scoring_mode_series():
    out = {}
    for sn, sm in [('WTA', ScoringMode.WINNER_TAKES_ALL),
                   ('DblPen', ScoringMode.DOUBLE_PENALTY)]:
        for p in [0.3, 0.5, 0.7]:
            out[f'{sn} p={p:.1f}'] = _v(dice_prob=p, scoring=sm)
    return out

def _pts_per_card_series():
    return {f'{n} pt/card': _v(pts_per_card=n, scoring=ScoringMode.DOUBLE_PENALTY)
            for n in [1, 2, 3, 4]}

VARIANT_GROUPS: Dict[str, Dict[str, VariantConfig]] = {
    'Dice Probability':  _dice_prob_series(),
    'Good Dice Effect':  _good_effect_series(),
    'Bad Dice Effect':   _bad_effect_series(),
    'Scoring Mode':      _scoring_mode_series(),
    'Pass Penalty':      _pass_penalty_series(),
    'Round Count':       _round_count_series(),
    'Cards on Bad Roll': _bad_cards_series(),
    'Points per Card':   _pts_per_card_series(),
}


# ===========================================================================
# GAME RUNNER
# ===========================================================================

def run_batch(agents: List[Agent],
              variant: VariantConfig,
              num_games: int,
              rotate: bool = True,
              players_per_game: int = 4,
              desc: str = '') -> List[GameResult]:
    """Run num_games games; shows tqdm bar if available."""
    results: List[GameResult] = []
    n = len(agents)
    ppg = min(players_per_game, n)

    # Build iteration sequence
    if n <= ppg:
        all_perms = list(permutations(range(n))) if rotate else [tuple(range(n))]
        seq = [(all_perms[i % len(all_perms)],) for i in range(num_games)]
    else:
        combos = list(_combs(range(n), ppg))
        seq = []
        for i in range(num_games):
            combo = list(combos[i % len(combos)])
            if rotate:
                rot = (i // len(combos)) % len(combo)
                combo = combo[rot:] + combo[:rot]
            seq.append((tuple(combo),))

    iterator = range(num_games)
    if HAS_TQDM:
        label = f'  {desc[:20]:20s}' if desc else '  Running'
        iterator = _tqdm(iterator, total=num_games, desc=label,
                        unit='g', ncols=80, leave=False,
                        bar_format='{desc} {bar}| {n_fmt}/{total_fmt} '
                                   '[{elapsed}<{remaining}, {rate_fmt}]')

    for i in iterator:
        perm = seq[i][0]
        game_agents = [agents[j] for j in perm]
        gr = GameSimulator.simulate_game(game_agents, variant, verbose=False)
        gr.starting_positions = list(perm)
        results.append(gr)

    return results


# ===========================================================================
# SHARED PRE-COMPUTATION
# ===========================================================================

def compute_named_variants(agents_core: List[Agent],
                           N: int,
                           NP: int,
                           variants: Dict[str, callable] = None) -> dict:
    """
    Run every named variant once and cache all derived analyses.
    Returns a dict keyed by variant name with sub-keys:
      results, pos, gl, ag, sm, vd
    """
    if variants is None:
        variants = ALL_VARIANTS
    core_names = [a.name for a in agents_core]
    store = {}
    for vname, vfunc in variants.items():
        res = run_batch(agents_core, vfunc(), N, players_per_game=NP, desc=vname)
        store[vname] = {
            'results': res,
            'pos':     analyse_positions(res, NP),
            'gl':      analyse_lengths(res),
            'ag':      analyse_agents(res),
            'sm':      analyse_score_margins(res),
            'vd':      variance_decomposition(res, core_names, NP),
        }
    return store


def compute_group_variants(agents_core: List[Agent],
                           N_sweep: int,
                           NP: int) -> dict:
    """
    Run every variant-group sweep and cache positional + GL analyses.
    Returns {group_name: {variant_name: {'pos': ..., 'gl': ..., 'vd': ...}}}
    """
    core_names = [a.name for a in agents_core]
    group_store = {}
    for gname, gvariants in VARIANT_GROUPS.items():
        group_store[gname] = {}
        for vname, vcfg in gvariants.items():
            res = run_batch(agents_core, vcfg, N_sweep,
                           players_per_game=NP, desc=f'{gname[:12]}:{vname[:10]}')
            group_store[gname][vname] = {
                'pos': analyse_positions(res, NP),
                'gl':  analyse_lengths(res),
                'vd':  variance_decomposition(res, core_names, NP),
            }
    return group_store


# ===========================================================================
# STATISTICAL HELPERS
# ===========================================================================

def binomial_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    z = 1.96 if confidence == 0.95 else 2.576
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))

def chi_square_uniformity(observed: List[int]) -> dict:
    n, k = sum(observed), len(observed)
    if n == 0 or k < 2:
        return {'chi2': 0.0, 'df': k - 1, 'p_value': float('nan')}
    expected = n / k
    chi2 = sum((o - expected)**2 / expected for o in observed)
    df = k - 1
    p = 1 - scipy_stats.chi2.cdf(chi2, df) if HAS_SCIPY else float('nan')
    return {'chi2': chi2, 'df': df, 'p_value': p}

def cohens_d(g1, g2) -> float:
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return float('nan')
    m1, m2 = np.mean(g1), np.mean(g2)
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return (m1 - m2) / pooled if pooled else 0.0

def gini_coefficient(values: List[float]) -> float:
    arr = np.array(sorted(values), dtype=float)
    if arr.sum() == 0:
        return 0.0
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * idx - n - 1).dot(arr) / (n * arr.sum()))

def kruskal_wallis_test(groups: Dict[str, List[float]]) -> dict:
    if not HAS_SCIPY or len(groups) < 2:
        return {'H': float('nan'), 'p_value': float('nan')}
    arrs = [np.array(v) for v in groups.values() if len(v) > 0]
    if len(arrs) < 2:
        return {'H': float('nan'), 'p_value': float('nan')}
    H, p = kruskal(*arrs)
    return {'H': H, 'p_value': p}


# ===========================================================================
# ANALYSIS FUNCTIONS
# ===========================================================================

def analyse_agents(results: List[GameResult]) -> dict:
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
            'wins': w, 'total': n,
            'win_rate': w / n if n else 0.0,
            'ci_low': lo, 'ci_high': hi,
            'mean_score':   float(np.mean(sc)),
            'std_score':    float(np.std(sc)),
            'median_score': float(np.median(sc)),
            'scores': sc,
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
        seats[s] = {
            'wins': w, 'total': n,
            'win_rate': w / n if n else 0.0,
            'ci_low': lo, 'ci_high': hi,
            'mean_score': float(np.mean(scores[s])) if scores[s] else 0.0,
            'std_score':  float(np.std(scores[s]))  if scores[s] else 0.0,
        }
    fair    = 1.0 / num_players
    wr_list = [seats[s]['win_rate'] for s in range(num_players)]
    return {
        'seats': seats,
        'chi2':          chi_square_uniformity(wins),
        'max_deviation': max(abs(r - fair) for r in wr_list),
        'gini':          gini_coefficient(wins),
    }

def analyse_lengths(results: List[GameResult]) -> dict:
    lengths = [gr.num_turns for gr in results]
    return {
        'mean': float(np.mean(lengths)), 'median': float(np.median(lengths)),
        'std':  float(np.std(lengths)),  'min': int(np.min(lengths)),
        'max':  int(np.max(lengths)),
        'q25':  float(np.percentile(lengths, 25)),
        'q75':  float(np.percentile(lengths, 75)),
        'lengths': lengths,
    }

def analyse_score_margins(results: List[GameResult]) -> dict:
    margins, all_final = [], []
    for gr in results:
        sc = sorted(gr.final_scores, reverse=True)
        if len(sc) >= 2:
            margins.append(sc[0] - sc[1])
        all_final.extend(gr.final_scores)
    if not margins:
        return {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'q25': 0.0,
                'q75': 0.0, 'margins': [], 'score_variance': 0.0,
                'tight_games_pct': 0.0}
    return {
        'mean':            float(np.mean(margins)),
        'std':             float(np.std(margins)),
        'median':          float(np.median(margins)),
        'q25':             float(np.percentile(margins, 25)),
        'q75':             float(np.percentile(margins, 75)),
        'margins':         margins,
        'score_variance':  float(np.var(all_final)),
        'tight_games_pct': float(np.mean([m <= 1 for m in margins]) * 100),
    }

def headtohead(results: List[GameResult], agent_names: List[str]) -> dict:
    n = len(agent_names)
    name2idx = {nm: i for i, nm in enumerate(agent_names)}
    pair_wins  = np.zeros((n, n))
    pair_total = np.zeros((n, n))
    for gr in results:
        wi = name2idx.get(gr.player_names[gr.winner])
        if wi is None:
            continue
        for nm in gr.player_names:
            li = name2idx.get(nm)
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
    return {'names': agent_names, 'matrix': mat}

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

    def _ss_factor(ids, n_levels):
        means = {}
        for lvl in range(n_levels):
            mask = [j for j, id_ in enumerate(ids) if id_ == lvl]
            if mask:
                means[lvl] = float(np.mean(outcomes[mask]))
        pred = np.array([means.get(id_, grand_mean) for id_ in ids])
        return float(np.sum((pred - grand_mean)**2))

    ss_agent = _ss_factor(agent_ids, len(agent_names))
    ss_seat  = _ss_factor(seat_ids,  num_players)
    a_pct    = ss_agent / ss_total * 100
    s_pct    = ss_seat  / ss_total * 100
    return {
        'agent_pct':    a_pct,
        'seat_pct':     s_pct,
        'residual_pct': max(0.0, 100.0 - a_pct - s_pct),
    }


# ===========================================================================
# LATEX TABLE GENERATORS
# ===========================================================================

def _tex_wrap(body_lines, caption, label):
    return '\n'.join([
        r'\begin{table}[H]', r'\centering',
        f'\\caption{{{caption}}}', f'\\label{{{label}}}',
        r'\small',
    ] + body_lines + [r'\end{table}'])

def tex_agent_winrates(analysis, caption, label):
    body = [r'\begin{tabular}{lcccc}', r'\toprule',
            r'\textbf{Agent} & \textbf{Win Rate (\%)} & \textbf{95\% CI} '
            r'& \textbf{Mean Score} & \textbf{Std Score} \\', r'\midrule']
    for nm in sorted(analysis):
        d  = analysis[nm]
        ci = f"[{d['ci_low']*100:.1f}, {d['ci_high']*100:.1f}]"
        body.append(f"  {nm} & {d['win_rate']*100:.1f} & {ci} & "
                    f"{d['mean_score']:.2f} & {d['std_score']:.2f} \\\\")
    body += [r'\bottomrule', r'\end{tabular}']
    return _tex_wrap(body, caption, label)

def tex_positional_table(all_pos, num_players, caption, label):
    hdr  = ' & '.join(f'\\textbf{{Seat {i+1}}}' for i in range(num_players))
    body = [f'\\begin{{tabular}}{{l{"c"*num_players}}}', r'\toprule',
            f'\\textbf{{Variant}} & {hdr} \\\\', r'\midrule']
    for vn, pd in all_pos.items():
        vals = ' & '.join(f"{pd['seats'][s]['win_rate']*100:.1f}"
                          for s in range(num_players))
        body.append(f'  {vn} & {vals} \\\\')
    body += [r'\bottomrule', r'\end{tabular}']
    return _tex_wrap(body, caption, label)

def tex_chi_square(all_pos, caption, label):
    body = [r'\begin{tabular}{lccl}', r'\toprule',
            r"\textbf{Variant} & $\chi^2$ & \textbf{df} & $p$-value \\", r'\midrule']
    for vn, pd in all_pos.items():
        c  = pd['chi2']
        ps = '< 0.001' if c['p_value'] < 0.001 else f"{c['p_value']:.4f}"
        sig = r'$^*$' if c['p_value'] < 0.05 else ''
        body.append(f"  {vn} & {c['chi2']:.2f} & {c['df']} & {ps}{sig} \\\\")
    body += [r'\bottomrule', r'\end{tabular}']
    return _tex_wrap(body, caption, label)

def tex_headtohead(h2h, caption, label):
    names = h2h['names']
    mat   = h2h['matrix']
    sn    = [nm[:11] for nm in names]
    hdr   = ' & '.join(f'\\textbf{{{s}}}' for s in sn)
    body  = [f'\\begin{{tabular}}{{l{"c"*len(names)}}}', r'\toprule',
             f' & {hdr} \\\\', r'\midrule']
    for i, nm in enumerate(sn):
        row = ['--' if i == j else f'{mat[i][j]*100:.1f}' for j in range(len(names))]
        body.append(f'  {nm} & ' + ' & '.join(row) + r' \\')
    body += [r'\bottomrule', r'\end{tabular}']
    return _tex_wrap(body, caption, label)

def tex_game_lengths(all_gl, caption, label):
    body = [r'\begin{tabular}{lccccc}', r'\toprule',
            r'\textbf{Variant} & \textbf{Mean} & \textbf{Median} '
            r'& \textbf{Std} & \textbf{Min} & \textbf{Max} \\', r'\midrule']
    for vn, gl in all_gl.items():
        body.append(f"  {vn} & {gl['mean']:.1f} & {gl['median']:.0f} & "
                    f"{gl['std']:.1f} & {gl['min']} & {gl['max']} \\\\")
    body += [r'\bottomrule', r'\end{tabular}']
    return _tex_wrap(body, caption, label)

def tex_anova(vd, caption, label):
    body = [r'\begin{tabular}{lccc}', r'\toprule',
            r'\textbf{Variant} & \textbf{Agent (\%)} & \textbf{Seat (\%)} '
            r'& \textbf{Residual (\%)} \\', r'\midrule']
    for vn, d in vd.items():
        body.append(f"  {vn} & {d['agent_pct']:.1f} & "
                    f"{d['seat_pct']:.1f} & {d['residual_pct']:.1f} \\\\")
    body += [r'\bottomrule', r'\end{tabular}']
    return _tex_wrap(body, caption, label)

def tex_ttest_table(rows, caption, label):
    body = [r'\begin{tabular}{lccc}', r'\toprule',
            r"\textbf{Comparison} & $t$-stat & $p$-value & Cohen's $d$ \\", r'\midrule']
    body.extend(rows)
    body += [r'\bottomrule', r'\end{tabular}']
    return _tex_wrap(body, caption, label)

def tex_fairness_ranking(all_pos, caption, label):
    rows_data = sorted(
        [(vn, pd['gini'], pd['max_deviation']*100,
          pd['chi2']['chi2'], pd['chi2']['p_value'])
         for vn, pd in all_pos.items()],
        key=lambda x: x[1]
    )
    body = [r'\begin{tabular}{lcccc}', r'\toprule',
            r'\textbf{Variant} & \textbf{Gini} & \textbf{Max Dev.\ (\%)} '
            r'& $\chi^2$ & $p$-value \\', r'\midrule']
    for vn, g, md, c2, pv in rows_data:
        ps  = '< 0.001' if pv < 0.001 else f'{pv:.4f}'
        sig = r'$^*$' if pv < 0.05 else ''
        body.append(f'  {vn} & {g:.4f} & {md:.1f} & {c2:.2f} & {ps}{sig} \\\\')
    body += [r'\bottomrule', r'\end{tabular}']
    return _tex_wrap(body, caption, label)

def tex_score_margins(sm_data, caption, label):
    body = [r'\begin{tabular}{lcccc}', r'\toprule',
            r'\textbf{Variant} & \textbf{Mean Margin} & \textbf{Median} '
            r'& \textbf{Tight Games (\%)} & \textbf{Score Var.} \\', r'\midrule']
    for vn, d in sm_data.items():
        body.append(f"  {vn} & {d['mean']:.2f} & {d['median']:.1f} & "
                    f"{d['tight_games_pct']:.1f} & {d['score_variance']:.2f} \\\\")
    body += [r'\bottomrule', r'\end{tabular}']
    return _tex_wrap(body, caption, label)

def tex_player_count(pc_data, caption, label):
    body = [r'\begin{tabular}{lcccccc}', r'\toprule',
            r'\textbf{N} & \textbf{S1 (\%)} & \textbf{S2 (\%)} & \textbf{S3 (\%)} '
            r'& \textbf{S4 (\%)} & \textbf{Gini} & $p$-val \\', r'\midrule']
    for npl, data in sorted(pc_data.items()):
        seats = data['pos']['seats']
        wrs   = [f"{seats[s]['win_rate']*100:.1f}" if s in seats else '--'
                 for s in range(4)]
        pv = data['pos']['chi2']['p_value']
        ps = '< 0.001' if pv < 0.001 else f'{pv:.4f}'
        body.append(f"  {npl}P & {' & '.join(wrs)} & "
                    f"{data['pos']['gini']:.4f} & {ps} \\\\")
    body += [r'\bottomrule', r'\end{tabular}']
    return _tex_wrap(body, caption, label)


# ===========================================================================
# FIGURE HELPERS
# ===========================================================================

def _save(fig, path):
    if not HAS_MATPLOTLIB:
        return
    fig.savefig(path)
    plt.close(fig)
    _ok(path)

def _agent_colour(name: str) -> str:
    for k, c in AGENT_COLOURS.items():
        if name.startswith(k):
            return c
    return '#888888'


# ===========================================================================
# FIGURE GENERATORS – EXP 1 (First-Player Advantage)
# ===========================================================================

def fig_first_player_advantage(all_pos: dict, path: str, num_players: int = 4):
    if not HAS_MATPLOTLIB: return
    fair = 1.0 / num_players
    vns  = list(all_pos)
    advs = [(all_pos[v]['seats'][0]['win_rate'] - fair) * 100 for v in vns]
    # sort by advantage descending
    paired = sorted(zip(vns, advs), key=lambda x: -x[1])
    vns, advs = zip(*paired) if paired else ([], [])
    cols = ['#D55E00' if a > 0 else '#56B4E9' for a in advs]
    fig, ax = plt.subplots(figsize=(10, max(4, len(vns) * 0.38)))
    ax.barh(range(len(vns)), advs, color=cols, edgecolor='black', lw=0.4)
    ax.axvline(0, color='black', lw=0.8, ls='--')
    ax.set_yticks(range(len(vns)))
    ax.set_yticklabels(vns, fontsize=8)
    ax.set_xlabel('Seat-1 Win-Rate Deviation from Fair (%)')
    ax.set_title('First-Player Advantage by Variant')
    handles = [mpatches.Patch(color='#D55E00', label='Above fair (advantage)'),
               mpatches.Patch(color='#56B4E9', label='Below fair (disadvantage)')]
    ax.legend(handles=handles, loc='lower right')
    plt.tight_layout()
    _save(fig, path)

def fig_positional_heatmap(all_pos: dict, path: str, num_players: int = 4):
    if not HAS_MATPLOTLIB: return
    vnames = list(all_pos)
    mat    = np.array([[all_pos[v]['seats'][s]['win_rate'] * 100
                        for s in range(num_players)] for v in vnames])
    fair   = 100 / num_players
    fig, ax = plt.subplots(figsize=(6, max(5, len(vnames) * 0.42)))
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn',
                   vmin=fair - 10, vmax=fair + 10)
    ax.set_xticks(range(num_players))
    ax.set_xticklabels([f'Seat {i+1}' for i in range(num_players)])
    ax.set_yticks(range(len(vnames)))
    ax.set_yticklabels(vnames, fontsize=8)
    for i in range(len(vnames)):
        for j in range(num_players):
            ax.text(j, i, f'{mat[i,j]:.1f}', ha='center', va='center',
                    fontsize=6.5, color='black')
    fig.colorbar(im, ax=ax, shrink=0.6, label='Win Rate (%)')
    ax.set_title('Positional Win Rates (%)\n(Green = above fair, Red = below fair)')
    plt.tight_layout()
    _save(fig, path)

def fig_positional_grouped(all_pos: dict, path: str, num_players: int = 4):
    if not HAS_MATPLOTLIB: return
    vnames = list(all_pos)
    nv     = len(vnames)
    x      = np.arange(num_players)
    w      = min(0.8 / nv, 0.15)
    cmap   = _get_cmap('tab20', nv)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, vn in enumerate(vnames):
        rates = [all_pos[vn]['seats'][s]['win_rate'] * 100 for s in range(num_players)]
        ax.bar(x + (i - nv/2 + 0.5) * w, rates, w, label=vn,
               color=cmap(i / nv), edgecolor='black', lw=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Seat {i+1}' for i in range(num_players)])
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Positional Win Rates Across Variants')
    ax.axhline(100/num_players, color='gray', ls='--', lw=0.8, label='Fair')
    ax.legend(fontsize=6, ncol=3, loc='upper right')
    ax.set_ylim(0, 50)
    plt.tight_layout()
    _save(fig, path)

def fig_chi2_pvalue_bar(all_pos: dict, path: str):
    if not HAS_MATPLOTLIB: return
    vnames = list(all_pos)
    pvals  = [all_pos[v]['chi2']['p_value'] for v in vnames]
    # sort by p-value ascending (most biased first)
    paired = sorted(zip(vnames, pvals), key=lambda x: x[1])
    vnames, pvals = zip(*paired) if paired else ([], [])
    logp   = [-np.log10(max(p, 1e-7)) for p in pvals]
    cols   = ['#D55E00' if p < 0.05 else '#56B4E9' for p in pvals]
    fig, ax = plt.subplots(figsize=(10, max(4, len(vnames) * 0.38)))
    ax.barh(range(len(vnames)), logp, color=cols, edgecolor='black', lw=0.4)
    ax.axvline(-np.log10(0.05), color='black', ls='--', lw=1, label='p = 0.05')
    ax.set_yticks(range(len(vnames)))
    ax.set_yticklabels(vnames, fontsize=8)
    ax.set_xlabel(r'$-\log_{10}(p)$')
    ax.set_title('Positional Bias Significance')
    ax.legend()
    plt.tight_layout()
    _save(fig, path)


# ===========================================================================
# FIGURE GENERATORS – EXP 2 (Dice Usage vs Win Rate)
# ===========================================================================

def fig_dice_sweep_positional(sweep: dict, path: str):
    if not HAS_MATPLOTLIB: return
    probs  = sorted(sweep)
    nseats = len(sweep[probs[0]]['seats'])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for seat in range(nseats):
        rates = [sweep[p]['seats'][seat]['win_rate'] * 100 for p in probs]
        ax.plot(probs, rates, 'o-', color=SEAT_COLOURS[seat],
                label=f'Seat {seat+1}', lw=1.5, ms=4)
    ax.axhline(100/nseats, color='gray', ls='--', lw=0.8, label='Fair')
    ax.set_xlabel('P(Good Dice Outcome)')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Positional Win Rate vs Dice Probability')
    ax.legend()
    plt.tight_layout()
    _save(fig, path)

def fig_dice_sweep_by_agent(sweep_ag: dict, path: str):
    if not HAS_MATPLOTLIB: return
    probs  = sorted(sweep_ag)
    agents = list(next(iter(sweep_ag.values())).keys())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ag in agents:
        rates = [sweep_ag[p][ag] * 100 for p in probs]
        ax.plot(probs, rates, 'o-', color=_agent_colour(ag),
                label=ag, lw=1.5, ms=4)
    ax.axhline(25, color='gray', ls='--', lw=0.8, label='Random (25%)')
    ax.set_xlabel('P(Good Dice Outcome)')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Agent Win Rate vs Dice Probability')
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, path)

def fig_dice_effect_comparison(sweep_effects: dict, path: str):
    if not HAS_MATPLOTLIB: return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ls_cycle = ['-', '--', '-.']
    for idx, (eff, sw) in enumerate(sweep_effects.items()):
        probs = sorted(sw)
        rates = [sw[p]['seats'][0]['win_rate'] * 100 for p in probs]
        ax.plot(probs, rates, ls_cycle[idx % 3], lw=1.8,
                label=f'{eff} – Seat 1')
    ax.axhline(25, color='gray', ls=':', lw=0.8, label='Fair (25%)')
    ax.set_xlabel('P(Good Dice Outcome)')
    ax.set_ylabel('Seat-1 Win Rate (%)')
    ax.set_title('Good-Dice Effect on First-Player Advantage')
    ax.legend()
    plt.tight_layout()
    _save(fig, path)

def fig_bad_effect_comparison(bad_eff_data: dict, path: str, num_players: int = 4):
    if not HAS_MATPLOTLIB: return
    effects = list(bad_eff_data)
    fair    = 100 / num_players
    fig, axes = plt.subplots(1, len(effects), figsize=(4*len(effects), 4), sharey=True)
    if len(effects) == 1:
        axes = [axes]
    for ax, eff in zip(axes, effects):
        wrs  = [bad_eff_data[eff]['seats'][s]['win_rate'] * 100 for s in range(num_players)]
        ax.bar(range(num_players), wrs, color=SEAT_COLOURS[:num_players],
               edgecolor='black', lw=0.4)
        ax.axhline(fair, color='gray', ls='--', lw=0.8)
        ax.set_xticks(range(num_players))
        ax.set_xticklabels([f'S{i+1}' for i in range(num_players)])
        ax.set_title(eff, fontsize=9)
        ax.set_ylim(0, 50)
    axes[0].set_ylabel('Win Rate (%)')
    fig.suptitle('Positional Win Rates by Bad-Dice Effect (p=0.5)', fontsize=12)
    plt.tight_layout()
    _save(fig, path)

def fig_group_heatmap(group_data: dict, key: str, path: str, num_players: int = 4):
    """Positional heatmap for a single variant group."""
    if not HAS_MATPLOTLIB: return
    vnames = list(group_data)
    mat    = np.array([[group_data[v]['pos']['seats'][s]['win_rate'] * 100
                        for s in range(num_players)] for v in vnames])
    fair   = 100 / num_players
    fig, ax = plt.subplots(figsize=(5, max(4, len(vnames) * 0.4)))
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn',
                   vmin=fair - 10, vmax=fair + 10)
    ax.set_xticks(range(num_players))
    ax.set_xticklabels([f'S{i+1}' for i in range(num_players)])
    ax.set_yticks(range(len(vnames)))
    ax.set_yticklabels(vnames, fontsize=7)
    for i in range(len(vnames)):
        for j in range(num_players):
            ax.text(j, i, f'{mat[i,j]:.1f}', ha='center', va='center', fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.7, label='Win Rate (%)')
    ax.set_title(f'Positional Win Rates – {key}')
    plt.tight_layout()
    _save(fig, path)


# ===========================================================================
# FIGURE GENERATORS – EXP 3 (Luck vs Skill)
# ===========================================================================

def fig_luck_skill_stacked(vd: dict, path: str):
    if not HAS_MATPLOTLIB: return
    labels = list(vd)
    a = [vd[v]['agent_pct']    for v in labels]
    s = [vd[v]['seat_pct']     for v in labels]
    r = [vd[v]['residual_pct'] for v in labels]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 0.55), 5))
    ax.bar(x, a, label='Agent (Skill)',    color='#56B4E9', edgecolor='black', lw=0.3)
    ax.bar(x, s, bottom=a,               label='Seat (Position)', color='#E69F00', edgecolor='black', lw=0.3)
    ax.bar(x, r, bottom=[ai+si for ai, si in zip(a, s)],
           label='Residual (Luck)', color='#AAAAAA', edgecolor='black', lw=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('Luck vs Skill: Variance Decomposition')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)
    plt.tight_layout()
    _save(fig, path)

def fig_luck_skill_scatter(vd: dict, path: str):
    if not HAS_MATPLOTLIB: return
    labels = list(vd)
    xs = [vd[v]['agent_pct']    for v in labels]
    ys = [vd[v]['residual_pct'] for v in labels]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, c=range(len(labels)), cmap='tab20',
               s=70, edgecolors='black', lw=0.5, zorder=3)
    for i, (lbl, x, y) in enumerate(zip(labels, xs, ys)):
        ax.annotate(lbl, (x, y), textcoords='offset points',
                    xytext=(4, 3), fontsize=6.5, color='#333')
    ax.set_xlabel('Variance Explained by Agent Skill (%)')
    ax.set_ylabel('Residual (Luck + Noise) (%)')
    ax.set_title('Luck vs Skill Landscape')
    ax.axhline(50, color='gray', ls=':', lw=0.8)
    ax.axvline(50, color='gray', ls=':', lw=0.8)
    plt.tight_layout()
    _save(fig, path)

def fig_luck_by_group(group_vd: dict, path: str):
    if not HAS_MATPLOTLIB: return
    ngroups = len(group_vd)
    ncols   = min(3, ngroups)
    nrows   = (ngroups + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows))
    axes = np.array(axes).flatten()
    for idx, (gname, vd) in enumerate(group_vd.items()):
        ax  = axes[idx]
        vls = list(vd)
        a   = [vd[v]['agent_pct']    for v in vls]
        s   = [vd[v]['seat_pct']     for v in vls]
        r   = [100 - ai - si for ai, si in zip(a, s)]
        x   = range(len(vls))
        ax.bar(x, a, color='#56B4E9', edgecolor='black', lw=0.3, label='Skill')
        ax.bar(x, s, bottom=a, color='#E69F00', edgecolor='black', lw=0.3, label='Seat')
        ax.bar(x, r, bottom=[ai+si for ai, si in zip(a, s)],
               color='#AAAAAA', edgecolor='black', lw=0.3, label='Luck')
        ax.set_xticks(x)
        ax.set_xticklabels(vls, rotation=30, ha='right', fontsize=6.5)
        ax.set_ylim(0, 110)
        ax.set_title(gname, fontsize=9, fontweight='bold')
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')
    for ax in axes[ngroups:]:
        ax.set_visible(False)
    fig.suptitle('Luck vs Skill by Variant Group', fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save(fig, path)


# ===========================================================================
# FIGURE GENERATORS – EXP 4 (Comeback / Snowball)
# ===========================================================================

def fig_score_margins(sm_data: dict, path: str):
    if not HAS_MATPLOTLIB: return
    vnames = list(sm_data)
    data   = [sm_data[v]['margins'] for v in vnames]
    fig, ax = plt.subplots(figsize=(max(10, len(vnames)*0.7), 5))
    bp = ax.boxplot(data, patch_artist=True, showfliers=True,
                    flierprops={'marker': 'o', 'markersize': 2, 'alpha': 0.3},
                    **{_BP_LABEL: vnames})
    cmap = _get_cmap('tab20', len(vnames))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(cmap(i / len(vnames)))
        patch.set_alpha(0.8)
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_ylabel('Winner − Runner-up Score Margin')
    ax.set_title('Score Margin Distributions')
    plt.xticks(rotation=35, ha='right', fontsize=8)
    plt.tight_layout()
    _save(fig, path)

def fig_tight_games(sm_data: dict, path: str):
    if not HAS_MATPLOTLIB: return
    vnames = list(sm_data)
    tights = [sm_data[v]['tight_games_pct'] for v in vnames]
    cmap   = _get_cmap('RdYlGn')
    cols   = [cmap(t / 100) for t in tights]
    fig, ax = plt.subplots(figsize=(max(10, len(vnames)*0.6), 4))
    ax.bar(range(len(vnames)), tights, color=cols, edgecolor='black', lw=0.4)
    ax.set_xticks(range(len(vnames)))
    ax.set_xticklabels(vnames, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Tight Games (Margin ≤ 1) %')
    ax.set_title('Comeback Potential: Close Finishes')
    plt.tight_layout()
    _save(fig, path)

def fig_score_variance(sm_data: dict, path: str):
    if not HAS_MATPLOTLIB: return
    vnames = list(sm_data)
    svars  = [sm_data[v]['score_variance'] for v in vnames]
    fig, ax = plt.subplots(figsize=(max(10, len(vnames)*0.6), 4))
    ax.bar(range(len(vnames)), svars, color='#56B4E9', edgecolor='black', lw=0.4)
    ax.set_xticks(range(len(vnames)))
    ax.set_xticklabels(vnames, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Score Variance')
    ax.set_title('Score Variance by Variant (Snowball Indicator)')
    plt.tight_layout()
    _save(fig, path)

def fig_short_vs_long(short_ag: dict, long_ag: dict, path: str):
    if not HAS_MATPLOTLIB: return
    agents = sorted(set(short_ag) & set(long_ag))
    x      = [short_ag[ag]['win_rate'] * 100 for ag in agents]
    y      = [long_ag[ag]['win_rate']  * 100 for ag in agents]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for ag, xi, yi in zip(agents, x, y):
        ax.scatter(xi, yi, color=_agent_colour(ag), s=80,
                   edgecolors='black', lw=0.5, zorder=3)
        ax.annotate(ag, (xi, yi), textcoords='offset points',
                    xytext=(5, 3), fontsize=8)
    lo = min(min(x), min(y)) - 2
    hi = max(max(x), max(y)) + 2
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, label='y = x')
    ax.set_xlabel('Win Rate – Sprint (3 rounds, %)')
    ax.set_ylabel('Win Rate – Marathon (10 rounds, %)')
    ax.set_title('Sprint vs Marathon: Snowball Detection')
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, path)

def fig_round_stability(round_ag_data: dict, path: str):
    if not HAS_MATPLOTLIB: return
    rounds = sorted(round_ag_data)
    agents = list(next(iter(round_ag_data.values())).keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    for ag in agents:
        rates = [round_ag_data[r].get(ag, 0) * 100 for r in rounds]
        ax.plot(rounds, rates, 'o-', color=_agent_colour(ag),
                label=ag, lw=1.5, ms=5)
    ax.axhline(25, color='gray', ls='--', lw=0.8, label='Random (25%)')
    ax.set_xlabel('Number of Rounds')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Agent Win-Rate Stability vs Round Count')
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, path)

def fig_scoring_mode_margins(sm_data: dict, scoring_variants: list, path: str):
    """Score margin distributions for scoring-sensitive variants."""
    if not HAS_MATPLOTLIB: return
    subset = {v: sm_data[v] for v in scoring_variants if v in sm_data}
    if not subset:
        return
    fig_score_margins(subset, path)


# ===========================================================================
# FIGURE GENERATORS – EXP 5 (Heuristic Analysis)
# ===========================================================================

def fig_agent_winrates(analysis: dict, path: str, num_players: int = 4):
    if not HAS_MATPLOTLIB: return
    names  = sorted(analysis)
    rates  = [analysis[n]['win_rate'] * 100 for n in names]
    lo_err = [analysis[n]['win_rate']*100 - analysis[n]['ci_low']*100  for n in names]
    hi_err = [analysis[n]['ci_high']*100   - analysis[n]['win_rate']*100 for n in names]
    cols   = [_agent_colour(n) for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(names)), rates, yerr=[lo_err, hi_err], capsize=4,
           color=cols, edgecolor='black', lw=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Agent Win Rates — Baseline Variant')
    ax.axhline(100/num_players, color='gray', ls='--', lw=0.8,
               label=f'Expected ({100/num_players:.0f}%)')
    ax.legend()
    ax.set_ylim(0, max(rates)*1.35)
    plt.tight_layout()
    _save(fig, path)

def fig_heatmap(h2h: dict, path: str):
    if not HAS_MATPLOTLIB: return
    names = h2h['names']
    mat   = h2h['matrix'] * 100
    sn    = [nm[:12] for nm in names]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap='RdYlGn', vmin=0, vmax=100)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(sn, rotation=45, ha='right')
    ax.set_yticklabels(sn)
    for i in range(len(names)):
        for j in range(len(names)):
            t = '--' if i == j else f'{mat[i,j]:.0f}'
            ax.text(j, i, t, ha='center', va='center', fontsize=9,
                    color='black' if 30 < mat[i,j] < 70 else 'white')
    ax.set_title('Head-to-Head Win Rates (%)\n(Row beats Column)')
    fig.colorbar(im, ax=ax, shrink=0.8, label='Win %')
    plt.tight_layout()
    _save(fig, path)

def fig_score_dists(analysis: dict, path: str):
    if not HAS_MATPLOTLIB: return
    names = sorted(analysis)
    data  = [analysis[n]['scores'] for n in names]
    cols  = [_agent_colour(n) for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for i, (pc, c) in enumerate(zip(parts['bodies'], cols)):
        pc.set_facecolor(c); pc.set_alpha(0.7)
    ax.set_xticks(range(1, len(names)+1))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('Match Score')
    ax.set_title('Score Distribution by Agent')
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    plt.tight_layout()
    _save(fig, path)

def fig_agent_radar(analysis: dict, path: str):
    if not HAS_MATPLOTLIB: return
    names   = sorted(analysis)
    metrics = ['Win Rate', 'Mean Score', 'Score\nStability', 'Median Score']
    n_m     = len(metrics)
    angles  = np.linspace(0, 2*np.pi, n_m, endpoint=False).tolist() + [0]

    def _norm(vals):
        mn, mx = min(vals), max(vals)
        return [0.5]*len(vals) if mx == mn else [(v-mn)/(mx-mn) for v in vals]

    wr_n = _norm([analysis[n]['win_rate']                for n in names])
    ms_n = _norm([analysis[n]['mean_score']              for n in names])
    st_n = _norm([1/(analysis[n]['std_score']+1e-6)      for n in names])
    md_n = _norm([analysis[n]['median_score']            for n in names])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i, nm in enumerate(names):
        vals = [wr_n[i], ms_n[i], st_n[i], md_n[i]] + [wr_n[i]]
        ax.plot(angles, vals, lw=1.5, label=nm, color=_agent_colour(nm))
        ax.fill(angles, vals, alpha=0.1, color=_agent_colour(nm))
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title('Agent Profile Radar (Normalised)', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.12), fontsize=8)
    plt.tight_layout()
    _save(fig, path)

def fig_agent_across_variants(all_agent_ag: dict, path: str):
    if not HAS_MATPLOTLIB: return
    vnames  = list(all_agent_ag)
    agents  = sorted(next(iter(all_agent_ag.values())))
    n_v, n_a = len(vnames), len(agents)
    x       = np.arange(n_v)
    w       = 0.8 / n_a
    fig, ax = plt.subplots(figsize=(max(10, n_v*0.75), 5))
    for i, ag in enumerate(agents):
        rates = [all_agent_ag[v].get(ag, {}).get('win_rate', 0) * 100 for v in vnames]
        ax.bar(x + (i - n_a/2 + 0.5)*w, rates, w,
               label=ag, color=_agent_colour(ag), edgecolor='black', lw=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(vnames, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Agent Win Rates Across All Variants')
    ax.axhline(25, color='gray', ls='--', lw=0.8, label='Expected 25%')
    ax.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    _save(fig, path)


# ===========================================================================
# FIGURE GENERATORS – EXP 6 (Player Count)
# ===========================================================================

def fig_player_count_positional(pc_data: dict, path: str):
    if not HAS_MATPLOTLIB: return
    n_counts = sorted(pc_data)
    fig, axes = plt.subplots(1, len(n_counts), figsize=(4*len(n_counts), 4))
    if len(n_counts) == 1: axes = [axes]
    for ax, npl in zip(axes, n_counts):
        pos  = pc_data[npl]['pos']
        fair = 100 / npl
        wrs  = [pos['seats'][s]['win_rate'] * 100 for s in range(npl)]
        lo   = [pos['seats'][s]['ci_low']  * 100 for s in range(npl)]
        hi   = [pos['seats'][s]['ci_high'] * 100 for s in range(npl)]
        errs = [[w-l for w,l in zip(wrs, lo)], [h-w for h,w in zip(hi, wrs)]]
        ax.bar(range(npl), wrs, yerr=errs, capsize=4,
               color=SEAT_COLOURS[:npl], edgecolor='black', lw=0.4)
        ax.axhline(fair, color='gray', ls='--', lw=0.8)
        ax.set_xticks(range(npl))
        ax.set_xticklabels([f'S{i+1}' for i in range(npl)])
        ax.set_title(f'{npl} Players', fontsize=10)
        ax.set_ylim(0, max(fair*2.2, max(wrs)*1.25))
        if npl == n_counts[0]: ax.set_ylabel('Win Rate (%)')
    fig.suptitle('Positional Win Rates by Player Count', fontsize=12)
    plt.tight_layout()
    _save(fig, path)

def fig_player_count_lengths(pc_data: dict, path: str):
    if not HAS_MATPLOTLIB: return
    n_counts = sorted(pc_data)
    data = [pc_data[n]['gl']['lengths'] for n in n_counts]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    **{_BP_LABEL: [f'{n}P' for n in n_counts]})
    for patch, c in zip(bp['boxes'], SEAT_COLOURS):
        patch.set_facecolor(c); patch.set_alpha(0.8)
    ax.set_ylabel('Game Length (turns)')
    ax.set_title('Game Length by Player Count')
    plt.tight_layout()
    _save(fig, path)

def fig_player_count_fairness(pc_data: dict, path: str):
    if not HAS_MATPLOTLIB: return
    n_counts = sorted(pc_data)
    ginis    = [pc_data[n]['pos']['gini']          for n in n_counts]
    max_devs = [pc_data[n]['pos']['max_deviation'] * 100 for n in n_counts]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()
    ax1.plot(n_counts, ginis,    'o-', color='#0072B2', lw=1.8, label='Gini')
    ax2.plot(n_counts, max_devs, 's--', color='#D55E00', lw=1.8, label='Max dev. (%)')
    ax1.set_xlabel('Number of Players')
    ax1.set_ylabel('Gini Coefficient', color='#0072B2')
    ax2.set_ylabel('Max Seat Deviation (%)', color='#D55E00')
    ax1.set_title('Fairness Metrics vs Player Count')
    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, lbl1+lbl2, loc='upper left', fontsize=8)
    plt.tight_layout()
    _save(fig, path)


# ===========================================================================
# FIGURE GENERATORS – EXP 7 (Fairness)
# ===========================================================================

def fig_fairness_scatter(all_pos: dict, path: str):
    if not HAS_MATPLOTLIB: return
    vnames = list(all_pos)
    ginis  = [all_pos[v]['gini']          for v in vnames]
    maxd   = [all_pos[v]['max_deviation'] * 100 for v in vnames]
    pvals  = [all_pos[v]['chi2']['p_value'] for v in vnames]
    cols   = ['#D55E00' if p < 0.05 else '#56B4E9' for p in pvals]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(ginis, maxd, c=cols, s=60, edgecolors='black', lw=0.5, zorder=3)
    for vn, g, m in zip(vnames, ginis, maxd):
        ax.annotate(vn, (g, m), textcoords='offset points', xytext=(4,3), fontsize=6.5)
    sig   = mpatches.Patch(color='#D55E00', label='Significant bias (p<0.05)')
    nosig = mpatches.Patch(color='#56B4E9', label='No significant bias')
    ax.legend(handles=[sig, nosig])
    ax.set_xlabel('Gini Coefficient')
    ax.set_ylabel('Max Seat Deviation (%)')
    ax.set_title('Variant Fairness Landscape')
    plt.tight_layout()
    _save(fig, path)

def fig_fairness_ranking_bar(all_pos: dict, path: str):
    if not HAS_MATPLOTLIB: return
    pairs = sorted([(v, all_pos[v]['gini']) for v in all_pos], key=lambda x: x[1])
    names, vals = zip(*pairs) if pairs else ([], [])
    cols = ['#009E73' if g < 0.03 else ('#E69F00' if g < 0.07 else '#D55E00')
            for g in vals]
    fig, ax = plt.subplots(figsize=(8, max(4, len(names)*0.38)))
    ax.barh(range(len(names)), vals, color=cols, edgecolor='black', lw=0.4)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Gini Coefficient (lower = fairer)')
    ax.set_title('Variant Fairness Ranking')
    handles = [mpatches.Patch(color='#009E73', label='Fair (< 0.03)'),
               mpatches.Patch(color='#E69F00', label='Moderate (0.03–0.07)'),
               mpatches.Patch(color='#D55E00', label='Unfair (> 0.07)')]
    ax.legend(handles=handles)
    plt.tight_layout()
    _save(fig, path)

def fig_group_fairness_summary(group_pos: dict, path: str):
    if not HAS_MATPLOTLIB: return
    gnames = list(group_pos)
    m_gini = [np.mean([group_pos[g][v]['gini'] for v in group_pos[g]]) for g in gnames]
    s_gini = [np.std( [group_pos[g][v]['gini'] for v in group_pos[g]]) for g in gnames]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(len(gnames)), m_gini, yerr=s_gini, capsize=4,
           color='#56B4E9', edgecolor='black', lw=0.4)
    ax.set_xticks(range(len(gnames)))
    ax.set_xticklabels(gnames, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Mean Gini Coefficient')
    ax.set_title('Positional Fairness by Variant Group')
    ax.axhline(0, color='black', lw=0.5)
    plt.tight_layout()
    _save(fig, path)

def fig_game_lengths(all_gl: dict, path: str):
    if not HAS_MATPLOTLIB: return
    names = list(all_gl)
    data  = [all_gl[n]['lengths'] for n in names]
    fig, ax = plt.subplots(figsize=(max(10, len(names)*0.65), 5))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    **{_BP_LABEL: names})
    cmap = _get_cmap('Set2', len(names))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(cmap(i / len(names)))
    ax.set_ylabel('Game Length (turns)')
    ax.set_title('Game Length Distribution by Variant')
    plt.xticks(rotation=35, ha='right', fontsize=8)
    plt.tight_layout()
    _save(fig, path)

def fig_group_game_lengths(group_gl: dict, path: str):
    if not HAS_MATPLOTLIB: return
    ngroups = len(group_gl)
    ncols   = min(3, ngroups)
    nrows   = (ngroups + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows))
    axes = np.array(axes).flatten()
    for idx, (gname, gl_dict) in enumerate(group_gl.items()):
        ax   = axes[idx]
        vls  = list(gl_dict)
        means = [gl_dict[v]['mean'] for v in vls]
        stds  = [gl_dict[v]['std']  for v in vls]
        ax.bar(range(len(vls)), means, yerr=stds, capsize=3,
               color='#0072B2', edgecolor='black', lw=0.3, alpha=0.8)
        ax.set_xticks(range(len(vls)))
        ax.set_xticklabels(vls, rotation=30, ha='right', fontsize=7)
        ax.set_title(gname, fontsize=9)
        ax.set_ylabel('Mean Turns')
    for ax in axes[ngroups:]:
        ax.set_visible(False)
    fig.suptitle('Game Length by Variant Group', fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save(fig, path)


# ===========================================================================
# FIGURE GENERATORS – EXP 8 (Diagnostics)
# ===========================================================================

def fig_architecture(path: str):
    if not HAS_MATPLOTLIB: return
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis('off')
    boxes = [
        (1.0, 5.5, 3.0, 1.0, 'Game Engine\n(Python)', '#56B4E9'),
        (6.0, 5.5, 3.0, 1.0, 'AI Agents\n(Rand/Heur/MCTS/RL)', '#E69F00'),
        (1.0, 3.5, 3.0, 1.0, 'Tournament\n& Analysis', '#009E73'),
        (6.0, 3.5, 3.0, 1.0, 'Web API\n(Flask)', '#D55E00'),
        (6.0, 1.5, 3.0, 1.0, 'Web Frontend\n(HTML/JS)', '#CC79A7'),
        (3.5, 1.5, 2.0, 1.0, 'Data Gen.\n(This script)', '#0072B2'),
    ]
    for bx, by, bw, bh, txt, c in boxes:
        ax.add_patch(mpatches.FancyBboxPatch(
            (bx, by), bw, bh, boxstyle='round,pad=0.1',
            fc=c, ec='black', lw=1.2, alpha=0.85))
        ax.text(bx+bw/2, by+bh/2, txt, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')
    ak = dict(arrowstyle='->', lw=1.5, color='#333')
    ax.annotate('', xy=(6.0, 6.0), xytext=(4.0, 6.0),  arrowprops=ak)
    ax.annotate('', xy=(2.5, 4.5), xytext=(2.5, 5.5),  arrowprops=ak)
    ax.annotate('', xy=(6.0, 4.0), xytext=(4.0, 6.0),  arrowprops=ak)
    ax.annotate('', xy=(7.5, 3.5), xytext=(7.5, 2.5),  arrowprops=ak)
    ax.annotate('', xy=(4.5, 3.5), xytext=(4.5, 2.5),  arrowprops=ak)
    ax.set_title('Cinquillo 2.0 System Architecture',
                 fontsize=13, fontweight='bold', pad=15)
    _save(fig, path)


# ===========================================================================
# CSV OUTPUT
# ===========================================================================

def save_csv(results: List[GameResult], filepath: str):
    if not results: return
    n      = len(results[0].player_names)
    fields = (['winner', 'num_turns']
              + [f'score_{i}' for i in range(n)]
              + [f'agent_{i}' for i in range(n)])
    with open(filepath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for gr in results:
            row = {'winner': gr.winner, 'num_turns': gr.num_turns}
            for i in range(n):
                row[f'score_{i}'] = gr.final_scores[i]
                row[f'agent_{i}'] = gr.player_names[i]
            w.writerow(row)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Cinquillo 2.0 – dissertation data generator v2.1')
    parser.add_argument('--quick',      action='store_true', help='1 000 games')
    parser.add_argument('--medium',     action='store_true', help='5 000 games')
    parser.add_argument('--skip-mcts',  action='store_true')
    parser.add_argument('--skip-rl',    action='store_true')
    parser.add_argument('--output-dir', default='output')
    parser.add_argument('--rl-weights', default='models/rl_agent_v4.pkl')
    parser.add_argument('--exp', nargs='+', type=int, default=None,
                        help='Run only these experiment numbers (1-8)')
    args = parser.parse_args()

    N  = 1000 if args.quick else (5000 if args.medium else 10_000)
    NP = 4
    run_exp = set(args.exp) if args.exp else set(range(1, 9))

    base = args.output_dir
    fdir = os.path.join(base, 'figures')
    tdir = os.path.join(base, 'tables')
    ddir = os.path.join(base, 'data')
    for d in (fdir, tdir, ddir):
        os.makedirs(d, exist_ok=True)

    # ── Banner ────────────────────────────────────────────────────────────
    print('━' * 72)
    print('  CINQUILLO 2.0 — DISSERTATION DATA GENERATOR  v2.1')
    print('━' * 72)
    group_total = sum(len(v) for v in VARIANT_GROUPS.values())
    print(f'  Games/experiment : {N:,}')
    print(f'  Named variants   : {len(ALL_VARIANTS)}')
    print(f'  Variant groups   : {len(VARIANT_GROUPS)} ({group_total} configs)')
    print(f'  Output           : {base}/')
    print(f'  MCTS             : {"SKIP" if args.skip_mcts else "ON"}')
    print(f'  RL               : {"SKIP" if args.skip_rl or not HAS_RL else "ON"}')
    print(f'  tqdm             : {"YES" if HAS_TQDM else "NO (pip install tqdm)"}')
    print(f'  Experiments      : {sorted(run_exp)}')
    print('━' * 72)

    # ── Agents ────────────────────────────────────────────────────────────
    print('\n  Building agents...')
    agents_core: List[Agent] = [
        RandomAgent('Random'),
        create_aggressive_heuristic(),
        create_defensive_heuristic(),
        create_balanced_heuristic(),
    ]
    agents_all = list(agents_core)

    if not args.skip_mcts:
        agents_all.append(MCTSAgentStandard())

    if not args.skip_rl and HAS_RL:
        try:
            rl = RLAgent(name='RL-v4', epsilon=0.0)
            rl.load_weights(args.rl_weights)
            GameSimulator.simulate_game(
                [rl, create_balanced_heuristic(),
                 create_balanced_heuristic(), create_balanced_heuristic()],
                make_baseline(), verbose=False)
            agents_all.append(rl)
            _info(f'RL agent loaded from {args.rl_weights}')
        except ValueError as e:
            _warn(f'RL dim mismatch – skipped. ({e})')
        except Exception as e:
            _warn(f'RL skipped – {e}')

    core_names = [a.name for a in agents_core]
    all_names  = [a.name for a in agents_all]
    _info(f'Core  : {core_names}')
    _info(f'All   : {all_names}')

    # ── Pre-compute all named variants (shared across experiments) ─────────
    print('\n  Pre-computing named variants...')
    t_start = time.time()
    named = compute_named_variants(agents_core, N, NP)
    t_pre  = time.time() - t_start
    _info(f'{len(named)} variants computed in {t_pre:.0f}s '
          f'({N * len(named):,} games total)')

    # Save CSVs for named variants
    for vname, d in named.items():
        safe = vname.replace(' ', '_').replace("'", '').replace('/', '_')
        save_csv(d['results'], os.path.join(ddir, f'{safe}.csv'))

    # Extract convenience dicts
    named_pos = {v: named[v]['pos'] for v in named}
    named_gl  = {v: named[v]['gl']  for v in named}
    named_ag  = {v: named[v]['ag']  for v in named}
    named_sm  = {v: named[v]['sm']  for v in named}
    named_vd  = {v: named[v]['vd']  for v in named}

    # ══════════════════════════════════════════════════════════════════════
    # EXP 1 – FIRST-PLAYER ADVANTAGE
    # ══════════════════════════════════════════════════════════════════════
    if 1 in run_exp:
        _header('First-Player Advantage', 1)

        # Tables
        with open(os.path.join(tdir, 'positional_fairness.tex'), 'w') as f:
            f.write(tex_positional_table(named_pos, NP,
                f'Win rate (\\%) by seat across variants ({N:,} games).',
                'tab:positional_wr'))
        with open(os.path.join(tdir, 'chi_square.tex'), 'w') as f:
            f.write(tex_chi_square(named_pos,
                'Chi-square tests for positional uniformity.', 'tab:chi_square'))
        with open(os.path.join(tdir, 'fairness_ranking.tex'), 'w') as f:
            f.write(tex_fairness_ranking(named_pos,
                'Variant fairness ranking (Gini coefficient).', 'tab:fairness_rank'))
        _info('Tables: positional_fairness, chi_square, fairness_ranking')

        # Figures
        fig_first_player_advantage(named_pos,
            os.path.join(fdir, 'first_player_advantage.pdf'), NP)
        fig_positional_heatmap(named_pos,
            os.path.join(fdir, 'positional_heatmap.pdf'), NP)
        fig_positional_grouped(named_pos,
            os.path.join(fdir, 'positional_grouped.pdf'), NP)
        fig_chi2_pvalue_bar(named_pos,
            os.path.join(fdir, 'chi2_pvalues.pdf'))

        # Console summary
        fair = 1.0 / NP
        rows = [(v, (named_pos[v]['seats'][0]['win_rate'] - fair) * 100,
                 named_pos[v]['chi2']['p_value'],
                 named_pos[v]['gini']) for v in named_pos]
        rows.sort(key=lambda x: -x[1])
        print()
        print(f'  {"Variant":<22} {"S1 adv":>7} {"p-val":>8} {"Gini":>7}')
        print(f'  {"-"*22} {"-"*7} {"-"*8} {"-"*7}')
        for vn, adv, pv, g in rows:
            sig = '*' if pv < 0.05 else ' '
            ps  = '<0.001' if pv < 0.001 else f'{pv:.4f}'
            print(f'  {vn:<22} {adv:>+6.1f}%  {ps:>8}{sig}  {g:>6.4f}')

    # ══════════════════════════════════════════════════════════════════════
    # EXP 2 – DICE USAGE VS WIN RATE
    # ══════════════════════════════════════════════════════════════════════
    if 2 in run_exp:
        _header('Dice Usage vs Win Rate', 2)
        sweep_n = max(500, N // 2)

        # 2a: positional + agent sweep across dice probability
        _sub('Dice probability sweep (positional + per-agent)')
        dice_pos_sw: Dict[float, dict] = {}
        dice_ag_sw:  Dict[float, dict] = {}
        for prob in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            res = run_batch(agents_core, _v(dice_prob=prob), sweep_n,
                           players_per_game=NP, desc=f'p={prob:.1f}')
            dice_pos_sw[prob] = analyse_positions(res, NP)
            dice_ag_sw[prob]  = {nm: analyse_agents(res).get(nm, {}).get('win_rate', 0)
                                 for nm in core_names}

        fig_dice_sweep_positional(dice_pos_sw,
            os.path.join(fdir, 'dice_sweep_positional.pdf'))
        fig_dice_sweep_by_agent(dice_ag_sw,
            os.path.join(fdir, 'dice_sweep_by_agent.pdf'))

        # 2b: good dice effect (WILD vs DOUBLE_PLAY) – seat-1 effect
        _sub('Good dice effect comparison (WILD vs DOUBLE_PLAY)')
        eff_sw = {}
        for ef_name, ef in [('WILD', GoodDiceEffect.WILD),
                             ('DOUBLE_PLAY', GoodDiceEffect.DOUBLE_PLAY)]:
            eff_sw[ef_name] = {}
            for prob in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
                res = run_batch(agents_core, _v(dice_prob=prob, good_effect=ef),
                               sweep_n, players_per_game=NP, desc=f'{ef_name[:6]} {prob:.1f}')
                eff_sw[ef_name][prob] = analyse_positions(res, NP)

        fig_dice_effect_comparison(eff_sw,
            os.path.join(fdir, 'dice_effect_comparison.pdf'))

        # 2c: bad dice effect at fixed probabilities
        _sub('Bad dice effect comparison (p=0.5)')
        bad_eff = {}
        for be_name, be in [('Take Cards', BadDiceEffect.TAKE_CARDS),
                             ('Forced Pass', BadDiceEffect.FORCED_PASS),
                             ('Neg. Points', BadDiceEffect.NEGATIVE_POINTS)]:
            res = run_batch(agents_core, _v(bad_effect=be, bad_neg_pts=2),
                           sweep_n, players_per_game=NP, desc=be_name[:10])
            bad_eff[be_name] = analyse_positions(res, NP)
        fig_bad_effect_comparison(bad_eff,
            os.path.join(fdir, 'bad_effect_comparison.pdf'))

        # 2d: positional heatmap for the dice-probability group
        _sub('Variant group: Dice Probability heatmap')
        dp_grp = {}
        for vname, vcfg in VARIANT_GROUPS['Dice Probability'].items():
            res = run_batch(agents_core, vcfg, sweep_n,
                           players_per_game=NP, desc=vname[:12])
            dp_grp[vname] = {'pos': analyse_positions(res, NP), 'gl': analyse_lengths(res)}
        fig_group_heatmap(dp_grp, 'Dice Probability', 
            os.path.join(fdir, 'dice_prob_series_heatmap.pdf'))

    # ══════════════════════════════════════════════════════════════════════
    # EXP 3 – LUCK VS SKILL
    # ══════════════════════════════════════════════════════════════════════
    if 3 in run_exp:
        _header('Luck vs Skill (Variance Decomposition)', 3)

        with open(os.path.join(tdir, 'anova.tex'), 'w') as f:
            f.write(tex_anova(named_vd,
                'Variance decomposition: agent skill vs positional effect vs residual luck.',
                'tab:anova'))

        fig_luck_skill_stacked(named_vd,
            os.path.join(fdir, 'luck_vs_skill_named.pdf'))
        fig_luck_skill_scatter(named_vd,
            os.path.join(fdir, 'luck_skill_scatter.pdf'))

        # Group-level decomposition
        _sub('Group-level variance decomposition')
        sweep_n3 = max(300, N // 4)
        # Use dice-related groups and round-count group most relevant here
        relevant_groups = ['Dice Probability', 'Good Dice Effect',
                           'Bad Dice Effect', 'Round Count', 'Scoring Mode']
        group_vd = {}
        for gname in relevant_groups:
            group_vd[gname] = {}
            for vname, vcfg in VARIANT_GROUPS[gname].items():
                res = run_batch(agents_core, vcfg, sweep_n3,
                               players_per_game=NP, desc=f'{gname[:8]}:{vname[:8]}')
                group_vd[gname][vname] = variance_decomposition(res, core_names, NP)
            _info(f'{gname}: done ({len(VARIANT_GROUPS[gname])} configs)')

        fig_luck_by_group(group_vd,
            os.path.join(fdir, 'luck_skill_by_group.pdf'))

        # Console table
        print()
        print(f'  {"Variant":<22} {"Skill%":>7} {"Seat%":>7} {"Luck%":>7}')
        print(f'  {"-"*22} {"-"*7} {"-"*7} {"-"*7}')
        for vn, d in named_vd.items():
            print(f'  {vn:<22} {d["agent_pct"]:>7.1f} '
                  f'{d["seat_pct"]:>7.1f} {d["residual_pct"]:>7.1f}')

    # ══════════════════════════════════════════════════════════════════════
    # EXP 4 – COMEBACK / SNOWBALL STUDY
    # ══════════════════════════════════════════════════════════════════════
    if 4 in run_exp:
        _header('Comeback / Snowball Study', 4)

        # Use scoring-sensitive and round-count variants (most relevant)
        scoring_sm = {v: named_sm[v] for v in SCORING_VARIANTS if v in named_sm}
        round_sm   = {v: named_sm[v] for v in ROUND_VARIANTS   if v in named_sm}
        all_sm     = named_sm

        with open(os.path.join(tdir, 'score_margins.tex'), 'w') as f:
            f.write(tex_score_margins(all_sm,
                'Score margin statistics by variant.', 'tab:score_margins'))

        fig_score_margins(all_sm,
            os.path.join(fdir, 'score_margins_all.pdf'))
        fig_score_margins(scoring_sm,
            os.path.join(fdir, 'score_margins_scoring.pdf'))
        fig_tight_games(all_sm,
            os.path.join(fdir, 'tight_games.pdf'))
        fig_score_variance(all_sm,
            os.path.join(fdir, 'score_variance.pdf'))

        # Sprint vs Marathon agent win rates
        _sub('Sprint vs Marathon snowball detection')
        short_ag = named_ag.get('Sprint', {})
        long_ag  = named_ag.get('Marathon', {})
        if short_ag and long_ag:
            fig_short_vs_long(short_ag, long_ag,
                os.path.join(fdir, 'short_vs_long_winrate.pdf'))

        # Round count stability (use the Round Count variant group)
        _sub('Round count stability sweep')
        sweep_n4 = max(300, N // 4)
        round_ag_data: Dict[int, Dict[str, float]] = {}
        for vname, vcfg in VARIANT_GROUPS['Round Count'].items():
            r_num = vcfg.fixed_rounds_count
            res   = run_batch(agents_core, vcfg, sweep_n4,
                             players_per_game=NP, desc=vname[:14])
            ag    = analyse_agents(res)
            round_ag_data[r_num] = {nm: ag[nm]['win_rate']
                                    for nm in core_names if nm in ag}
        fig_round_stability(round_ag_data,
            os.path.join(fdir, 'round_count_stability.pdf'))

        # Console summary
        print()
        print(f'  {"Variant":<22} {"Mean margin":>12} {"Tight%":>7} {"Score var":>10}')
        print(f'  {"-"*22} {"-"*12} {"-"*7} {"-"*10}')
        for vn, d in all_sm.items():
            print(f'  {vn:<22} {d["mean"]:>12.2f} '
                  f'{d["tight_games_pct"]:>7.1f} {d["score_variance"]:>10.2f}')

    # ══════════════════════════════════════════════════════════════════════
    # EXP 5 – HEURISTIC AGENT ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    if 5 in run_exp:
        _header('Heuristic Agent Analysis', 5)

        # Baseline tournament with ALL agents (includes MCTS and RL)
        _sub(f'Baseline tournament – all agents ({N:,} games)')
        bl_res = run_batch(agents_all, make_baseline(), N,
                          players_per_game=NP, desc='Baseline (all)')
        save_csv(bl_res, os.path.join(ddir, 'baseline_all_agents.csv'))
        ag_bl = analyse_agents(bl_res)

        with open(os.path.join(tdir, 'agent_winrates.tex'), 'w') as f:
            f.write(tex_agent_winrates(ag_bl,
                f'Agent win rates (\\%) in {NP}-player Baseline ({N:,} games).',
                'tab:agent_winrates'))

        fig_agent_winrates(ag_bl,
            os.path.join(fdir, 'agent_winrates.pdf'), NP)
        fig_score_dists(ag_bl,
            os.path.join(fdir, 'score_distributions.pdf'))
        fig_agent_radar(ag_bl,
            os.path.join(fdir, 'agent_radar.pdf'))

        h2h = headtohead(bl_res, all_names)
        with open(os.path.join(tdir, 'headtohead.tex'), 'w') as f:
            f.write(tex_headtohead(h2h,
                'Head-to-head win rates (\\%) — Baseline variant.',
                'tab:headtohead'))
        fig_heatmap(h2h,
            os.path.join(fdir, 'heatmap_headtohead.pdf'))

        # Agent win rates across all named variants (core agents only, reuse pre-computed)
        fig_agent_across_variants(named_ag,
            os.path.join(fdir, 'agent_winrates_across_variants.pdf'))

        # t-tests: Baseline vs each variant
        _sub('t-tests Baseline vs variants')
        base_winner_scores = [gr.final_scores[gr.winner] for gr in bl_res]
        ttest_rows = []
        for vname, d in named_ag.items():
            if vname == 'Baseline':
                continue
            var_scores = []
            for nm, ag_d in d.items():
                var_scores.extend(ag_d['scores'])
            if HAS_SCIPY and var_scores:
                t, p = scipy_stats.ttest_ind(base_winner_scores, var_scores)
                dv   = cohens_d(base_winner_scores, var_scores)
                ps   = '< 0.001' if p < 0.001 else f'{p:.4f}'
                ttest_rows.append(
                    f"  Baseline vs.\\ {vname} & {t:.2f} & {ps} & {dv:.3f} \\\\")

        with open(os.path.join(tdir, 'ttest_variants.tex'), 'w') as f:
            f.write(tex_ttest_table(ttest_rows,
                'Two-sample $t$-tests comparing winner scores across variants.',
                'tab:ttest'))

        # Console win-rate table
        print()
        print(f'  {"Agent":<20} {"Win Rate":>9} {"95% CI":>20} {"Mean Score":>11}')
        print(f'  {"-"*20} {"-"*9} {"-"*20} {"-"*11}')
        for nm in sorted(ag_bl):
            d  = ag_bl[nm]
            ci = f"[{d['ci_low']*100:.1f}, {d['ci_high']*100:.1f}]"
            print(f'  {nm:<20} {d["win_rate"]*100:>8.1f}%  {ci:>20} '
                  f'{d["mean_score"]:>10.2f}')

    # ══════════════════════════════════════════════════════════════════════
    # EXP 6 – NUMBER OF PLAYERS
    # ══════════════════════════════════════════════════════════════════════
    if 6 in run_exp:
        _header('Number of Players Study', 6)

        # Use Baseline, Pure Strategy, Lucky Draw for comparison
        test_variants = {
            'Baseline':      make_baseline(),
            'Pure Strategy': make_pure_strategy(),
            'Lucky Draw':    make_lucky_draw(),
        }
        pc_data: Dict[int, dict] = {}

        for npl in [2, 3, 4]:
            _sub(f'{npl}-player games across {len(test_variants)} variants')
            pc_data[npl] = {}
            for vn, vcfg in test_variants.items():
                res = run_batch(agents_core[:npl], vcfg, N,
                               players_per_game=npl, desc=f'{npl}P {vn[:10]}')
                pc_data[npl][vn] = {
                    'pos': analyse_positions(res, npl),
                    'gl':  analyse_lengths(res),
                    'ag':  analyse_agents(res),
                }
                save_csv(res, os.path.join(ddir,
                    f'nplayers_{npl}_{vn.replace(" ","_")}.csv'))

        # Flatten to single-variant dicts for tables/figures (use Baseline)
        pc_baseline = {npl: pc_data[npl]['Baseline'] for npl in pc_data}

        with open(os.path.join(tdir, 'player_count.tex'), 'w') as f:
            f.write(tex_player_count(pc_baseline,
                'Positional win rates by player count (Baseline variant).',
                'tab:player_count'))

        fig_player_count_positional(pc_baseline,
            os.path.join(fdir, 'player_count_positional.pdf'))
        fig_player_count_lengths(pc_baseline,
            os.path.join(fdir, 'player_count_game_lengths.pdf'))
        fig_player_count_fairness(pc_baseline,
            os.path.join(fdir, 'player_count_fairness.pdf'))

        # Variant comparison across player counts (subplots per variant)
        for vn in test_variants:
            fig_player_count_positional(
                {npl: pc_data[npl][vn] for npl in pc_data},
                os.path.join(fdir, f'player_count_{vn.replace(" ","_")}.pdf'))

        # Console summary
        print()
        for npl in sorted(pc_data):
            fair = 100 / npl
            for vn in test_variants:
                pos  = pc_data[npl][vn]['pos']
                seats_str = '  '.join(
                    f'S{s+1}={pos["seats"][s]["win_rate"]*100:.1f}%'
                    for s in range(npl))
                pv  = pos['chi2']['p_value']
                sig = '*' if pv < 0.05 else ' '
                print(f'  {npl}P {vn:<16} {seats_str}  '
                      f'Gini={pos["gini"]:.4f}  p={pv:.4f}{sig}')

    # ══════════════════════════════════════════════════════════════════════
    # EXP 7 – VARIANT FAIRNESS & BALANCE
    # ══════════════════════════════════════════════════════════════════════
    if 7 in run_exp:
        _header('Variant Fairness & Balance', 7)

        # Named variant fairness figures (uses pre-computed named_pos + named_gl)
        with open(os.path.join(tdir, 'game_lengths.tex'), 'w') as f:
            f.write(tex_game_lengths(named_gl,
                'Game length statistics by variant.', 'tab:game_length'))

        fig_fairness_scatter(named_pos,
            os.path.join(fdir, 'fairness_scatter.pdf'))
        fig_fairness_ranking_bar(named_pos,
            os.path.join(fdir, 'fairness_ranking.pdf'))
        fig_game_lengths(named_gl,                     # ← fixed: pass named_gl, not named_pos
            os.path.join(fdir, 'game_length_distributions.pdf'))

        # Group-level fairness analysis
        _sub('Variant group fairness sweep')
        sweep_n7 = max(300, N // 4)
        group_pos: Dict[str, Dict[str, dict]] = {}
        group_gl:  Dict[str, Dict[str, dict]] = {}
        for gname, gvariants in VARIANT_GROUPS.items():
            group_pos[gname] = {}
            group_gl[gname]  = {}
            for vname, vcfg in gvariants.items():
                res = run_batch(agents_core, vcfg, sweep_n7,
                               players_per_game=NP, desc=f'{gname[:8]}:{vname[:8]}')
                group_pos[gname][vname] = analyse_positions(res, NP)
                group_gl[gname][vname]  = analyse_lengths(res)
            _info(f'{gname}: {len(gvariants)} configs done')

        fig_group_fairness_summary(group_pos,
            os.path.join(fdir, 'group_fairness_summary.pdf'))
        fig_group_game_lengths(group_gl,
            os.path.join(fdir, 'group_game_lengths.pdf'))

        # Per-group heatmaps (most diagnostic groups)
        for gname in ['Dice Probability', 'Round Count', 'Scoring Mode']:
            if gname in group_pos:
                safe_g = gname.replace(' ', '_')
                fig_group_heatmap(
                    {v: {'pos': group_pos[gname][v], 'gl': group_gl[gname][v]}
                     for v in group_pos[gname]},
                    gname,
                    os.path.join(fdir, f'fairness_heatmap_{safe_g}.pdf'))

        # Kruskal–Wallis across groups
        group_ginis = {g: [group_pos[g][v]['gini'] for v in group_pos[g]]
                       for g in group_pos}
        kw = kruskal_wallis_test(group_ginis)
        _info(f'Kruskal–Wallis (group ginis): H={kw["H"]:.2f}  p={kw["p_value"]:.4f}')

        # Console fairness ranking
        print()
        print(f'  {"Variant":<22} {"Gini":>7} {"MaxDev%":>8} {"p-val":>8}')
        print(f'  {"-"*22} {"-"*7} {"-"*8} {"-"*8}')
        ranked = sorted(named_pos.items(), key=lambda x: x[1]['gini'])
        for vn, pd in ranked:
            pv  = pd['chi2']['p_value']
            ps  = '<0.001' if pv < 0.001 else f'{pv:.4f}'
            sig = '*' if pv < 0.05 else ' '
            print(f'  {vn:<22} {pd["gini"]:>7.4f} '
                  f'{pd["max_deviation"]*100:>7.1f}%  {ps:>8}{sig}')

    # ══════════════════════════════════════════════════════════════════════
    # EXP 8 – ADDITIONAL DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════════
    if 8 in run_exp:
        _header('Additional Diagnostics', 8)

        # 8a: RL learning curve
        rl_found = False
        for cand in ['models/training_log.json', 'simulation/training_log.json']:
            if os.path.exists(cand):
                try:
                    with open(cand) as f:
                        rl_log = json.load(f)
                    eps = rl_log.get('episodes', rl_log.get('episode', []))
                    wrs = rl_log.get('win_rates', rl_log.get('win_rate', []))
                    if eps and wrs and HAS_MATPLOTLIB:
                        fig, ax = plt.subplots(figsize=(7, 4.5))
                        ax.plot(eps, [w*100 for w in wrs], '-', color='#D55E00', lw=1.5)
                        ax.axhline(25, color='gray', ls='--', lw=0.8, label='Random (25%)')
                        ax.set_xlabel('Training Episodes')
                        ax.set_ylabel('Win Rate vs Heuristics (%)')
                        ax.set_title('RL Agent Learning Curve')
                        ax.legend()
                        _save(fig, os.path.join(fdir, 'rl_learning_curve.pdf'))
                        rl_found = True
                except Exception as e:
                    _warn(f'Could not parse {cand}: {e}')
                break
        if not rl_found:
            _info('No RL training log found – skipping curve.')

        # 8b: Architecture diagram
        fig_architecture(os.path.join(fdir, 'architecture_diagram.pdf'))

        # 8c: Score variance across all named variants
        fig_score_variance(named_sm,
            os.path.join(fdir, 'score_variance_all.pdf'))

    # ── SUMMARY ───────────────────────────────────────────────────────────
    t_total = time.time() - t_start
    print(f'\n{"━"*72}')
    print(f'  COMPLETE  –  total elapsed: {t_total/60:.1f} min')
    print(f'{"━"*72}')
    for subdir_label, subdir in [('Figures', fdir), ('Tables', tdir), ('Data', ddir)]:
        if os.path.isdir(subdir):
            files = sorted(os.listdir(subdir))
            total_kb = sum(os.path.getsize(os.path.join(subdir, f))
                           for f in files) // 1024
            print(f'  {subdir_label:<8} {len(files):>3} files  ({total_kb} KB)  → {subdir}/')
    print()
    print('  LaTeX usage:')
    print(r'    \graphicspath{{output/figures/}}')
    print(r'    \includegraphics[width=0.9\textwidth]{first_player_advantage}')
    print(r'    \input{output/tables/agent_winrates}')
    print()


if __name__ == '__main__':
    main()