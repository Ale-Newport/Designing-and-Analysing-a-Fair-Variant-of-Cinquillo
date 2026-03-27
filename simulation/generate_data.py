#!/usr/bin/env python3
"""
==========================================================================
Dissertation Data Generator for Cinquillo 2.0
==========================================================================
Generates ALL tables, figures, and statistical analyses needed for the
final report.

Place this file at:  simulation/generate_data.py
Run from the PROJECT ROOT:
    cd PRJ
    python simulation/generate_data.py
    python simulation/generate_data.py --quick          # 1000 games (fast)
    python simulation/generate_data.py --skip-mcts      # skip slow MCTS
    python simulation/generate_data.py --skip-rl        # skip RL agent

Outputs:
    dissertation_output/
    ├── figures/           # PDF figures for LaTeX (includegraphics)
    ├── tables/            # LaTeX table fragments  (input)
    └── data/              # Raw CSV data
==========================================================================
"""

# ---------------------------------------------------------------------------
# Path fix – MUST come before any project imports.
# Works whether you run from project root or from simulation/.
# ---------------------------------------------------------------------------
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import json
import csv
import warnings
from collections import defaultdict
from itertools import permutations
from typing import List, Dict, Tuple, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Project imports (now guaranteed to resolve)
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

# Optional RL import (needs a trained model file)
try:
    from agents.rl_agent import RLAgent, RLAgentExplore, RLAgentExploit
    HAS_RL = True
except ImportError:
    HAS_RL = False
    print("WARNING: RL agent module not available. RL experiments will be skipped.")

# Optional matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not found – figures will not be generated.")

# Optional scipy
try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found – some statistical tests will be limited.")


# ===========================================================================
# FIGURE SETTINGS
# ===========================================================================
if HAS_MATPLOTLIB:
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (6.5, 4.5),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })

# Colourblind-friendly palette
COLOURS = {
    'Random':       '#E69F00',
    'Aggressive':   '#56B4E9',
    'Defensive':    '#009E73',
    'Balanced':     '#0072B2',
    'Risky':        '#CC79A7',
    'MCTS':         '#D55E00',
    'MCTS-Fast':    '#D55E00',
    'MCTS-SuperFast': '#E8A070',
    'MCTS-Deep':    '#A0522D',
    'RL':           '#F0E442',
    'RL-v4':        '#F0E442',
}
SEAT_COLOURS = ['#56B4E9', '#E69F00', '#009E73', '#D55E00']


# ===========================================================================
# VARIANT DEFINITIONS  (using the real VariantConfig fields from entities.py)
# ===========================================================================

def make_baseline():
    """Baseline – 50/50 WILD/TAKE_CARDS, WINNER_TAKES_ALL, 5 rounds."""
    return VariantConfig(
        dice_good_probability=0.5,
        dice_good_effect=GoodDiceEffect.WILD,
        dice_bad_effect=BadDiceEffect.TAKE_CARDS,
        dice_bad_cards_count=2,
        scoring_mode=ScoringMode.WINNER_TAKES_ALL,
        points_per_card=1,
        voluntary_pass_penalty=1,
        match_end_mode=MatchEndMode.FIXED_ROUNDS,
        fixed_rounds_count=5,
    )

def make_variant_A():
    """Variant A – same dice, DOUBLE_PENALTY scoring."""
    return VariantConfig(
        dice_good_probability=0.5,
        dice_good_effect=GoodDiceEffect.WILD,
        dice_bad_effect=BadDiceEffect.TAKE_CARDS,
        dice_bad_cards_count=2,
        scoring_mode=ScoringMode.DOUBLE_PENALTY,
        points_per_card=1,
        voluntary_pass_penalty=1,
        match_end_mode=MatchEndMode.FIXED_ROUNDS,
        fixed_rounds_count=5,
    )

def make_variant_B():
    """Variant B – DOUBLE_PLAY / FORCED_PASS dice, higher pass penalty."""
    return VariantConfig(
        dice_good_probability=0.6,
        dice_good_effect=GoodDiceEffect.DOUBLE_PLAY,
        dice_bad_effect=BadDiceEffect.FORCED_PASS,
        scoring_mode=ScoringMode.DOUBLE_PENALTY,
        points_per_card=1,
        voluntary_pass_penalty=2,
        match_end_mode=MatchEndMode.FIXED_ROUNDS,
        fixed_rounds_count=5,
    )

def make_variant_C():
    """Variant C – harsh: NEGATIVE_POINTS bad effect, high per-card points."""
    return VariantConfig(
        dice_good_probability=0.4,
        dice_good_effect=GoodDiceEffect.WILD,
        dice_bad_effect=BadDiceEffect.NEGATIVE_POINTS,
        dice_bad_penalty_points=3,
        scoring_mode=ScoringMode.DOUBLE_PENALTY,
        points_per_card=2,
        voluntary_pass_penalty=3,
        match_end_mode=MatchEndMode.FIXED_ROUNDS,
        fixed_rounds_count=5,
    )

def make_no_dice():
    """No dice (0% good outcome) – pure strategy."""
    return VariantConfig(
        dice_good_probability=0.0,
        dice_good_effect=GoodDiceEffect.WILD,
        dice_bad_effect=BadDiceEffect.FORCED_PASS,
        scoring_mode=ScoringMode.WINNER_TAKES_ALL,
        points_per_card=1,
        voluntary_pass_penalty=1,
        match_end_mode=MatchEndMode.FIXED_ROUNDS,
        fixed_rounds_count=5,
    )

def make_high_luck():
    """High luck – 80% good dice."""
    return VariantConfig(
        dice_good_probability=0.8,
        dice_good_effect=GoodDiceEffect.WILD,
        dice_bad_effect=BadDiceEffect.NEGATIVE_POINTS,
        dice_bad_penalty_points=1,
        scoring_mode=ScoringMode.WINNER_TAKES_ALL,
        points_per_card=1,
        voluntary_pass_penalty=1,
        match_end_mode=MatchEndMode.FIXED_ROUNDS,
        fixed_rounds_count=5,
    )


ALL_VARIANTS = {
    'Baseline':   make_baseline,
    'Variant A':  make_variant_A,
    'Variant B':  make_variant_B,
    'Variant C':  make_variant_C,
    'No Dice':    make_no_dice,
    'High Luck':  make_high_luck,
}


# ===========================================================================
# GAME RUNNER – uses GameSimulator.simulate_game() from tournament.py
# ===========================================================================

def run_batch(agents: List[Agent],
              variant: VariantConfig,
              num_games: int,
              rotate: bool = True,
              verbose: bool = True,
              players_per_game: int = 4) -> List[GameResult]:
    """
    Run *num_games* games using GameSimulator.simulate_game().

    If len(agents) <= players_per_game, rotates seat order among all agents.
    If len(agents) > players_per_game, cycles through combinations of
    *players_per_game* agents and rotates seats within each combo.
    """
    from itertools import combinations as combs

    results: List[GameResult] = []
    n = len(agents)

    if n <= players_per_game:
        # Simple case: all agents fit in one game
        all_perms = list(permutations(range(n))) if rotate else [tuple(range(n))]

        t0 = time.time()
        for i in range(num_games):
            perm = all_perms[i % len(all_perms)]
            game_agents = [agents[j] for j in perm]
            gr = GameSimulator.simulate_game(game_agents, variant, verbose=False)
            gr.starting_positions = list(perm)
            results.append(gr)

            if verbose and (i + 1) % max(1, num_games // 10) == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{i+1}/{num_games}]  {rate:.0f} games/sec")
    else:
        # More agents than seats: cycle through combos of players_per_game
        combos = list(combs(range(n), players_per_game))
        t0 = time.time()
        for i in range(num_games):
            combo = combos[i % len(combos)]
            # Rotate seats within this combo
            seats = list(combo)
            if rotate:
                rot = i // len(combos) % len(seats)
                seats = seats[rot:] + seats[:rot]
            game_agents = [agents[j] for j in seats]
            gr = GameSimulator.simulate_game(game_agents, variant, verbose=False)
            gr.starting_positions = list(seats)
            results.append(gr)

            if verbose and (i + 1) % max(1, num_games // 10) == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{i+1}/{num_games}]  {rate:.0f} games/sec")

    return results


# ===========================================================================
# STATISTICAL HELPERS
# ===========================================================================

def binomial_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    z = 1.96 if confidence == 0.95 else 2.576
    denom = 1 + z ** 2 / n
    centre = (p_hat + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n) / denom
    return (max(0, centre - margin), min(1, centre + margin))


def chi_square_uniformity(observed: List[int]) -> dict:
    n = sum(observed)
    k = len(observed)
    expected = n / k
    chi2 = sum((o - expected) ** 2 / expected for o in observed)
    df = k - 1
    p = 1 - scipy_stats.chi2.cdf(chi2, df) if HAS_SCIPY else float('nan')
    return {'chi2': chi2, 'df': df, 'p_value': p}


def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return float('nan')
    m1, m2 = np.mean(g1), np.mean(g2)
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    return (m1 - m2) / pooled if pooled else 0.0


# ===========================================================================
# ANALYSIS  (works on List[GameResult])
# ===========================================================================

def analyse_agents(results: List[GameResult]) -> dict:
    """Win rates, CIs, scores per agent name."""
    wins = defaultdict(int)
    total = defaultdict(int)
    scores = defaultdict(list)

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
        out[name] = {
            'wins': w, 'total': n,
            'win_rate': w / n if n else 0,
            'ci_low': lo, 'ci_high': hi,
            'mean_score': float(np.mean(scores[name])),
            'std_score': float(np.std(scores[name])),
            'scores': scores[name],
        }
    return out


def analyse_positions(results: List[GameResult], num_players: int) -> dict:
    """Win rates by seat."""
    wins = [0] * num_players
    totals = [0] * num_players
    scores = [[] for _ in range(num_players)]

    for gr in results:
        for seat in range(num_players):
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
            'win_rate': w / n if n else 0,
            'ci_low': lo, 'ci_high': hi,
            'mean_score': float(np.mean(scores[s])),
            'std_score': float(np.std(scores[s])),
        }
    chi2 = chi_square_uniformity(wins)
    return {'seats': seats, 'chi2': chi2}


def analyse_lengths(results: List[GameResult]) -> dict:
    lengths = [gr.num_turns for gr in results]
    return {
        'mean': float(np.mean(lengths)),
        'median': float(np.median(lengths)),
        'std': float(np.std(lengths)),
        'min': int(np.min(lengths)),
        'max': int(np.max(lengths)),
        'q25': float(np.percentile(lengths, 25)),
        'q75': float(np.percentile(lengths, 75)),
        'lengths': lengths,
    }


def headtohead(results: List[GameResult], agent_names: List[str]) -> dict:
    """Pairwise win fractions from multiplayer games."""
    n = len(agent_names)
    name2idx = {nm: i for i, nm in enumerate(agent_names)}
    pair_wins = np.zeros((n, n))
    pair_total = np.zeros((n, n))

    for gr in results:
        winner_name = gr.player_names[gr.winner]
        wi = name2idx.get(winner_name)
        if wi is None:
            continue
        for i, nm in enumerate(gr.player_names):
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
    """% of outcome variance explained by agent vs seat vs residual."""
    name2idx = {nm: i for i, nm in enumerate(agent_names)}
    agent_ids, seat_ids, outcomes = [], [], []

    for gr in results:
        for seat in range(num_players):
            nm = gr.player_names[seat]
            ai = name2idx.get(nm, 0)
            agent_ids.append(ai)
            seat_ids.append(seat)
            outcomes.append(1.0 if gr.winner == seat else 0.0)

    outcomes = np.array(outcomes)
    grand_mean = np.mean(outcomes)
    ss_total = np.sum((outcomes - grand_mean) ** 2)
    if ss_total == 0:
        return {'agent_pct': 0, 'seat_pct': 0, 'residual_pct': 100}

    # Agent factor
    agent_means = {}
    for i in range(len(agent_names)):
        mask = [j for j, ai in enumerate(agent_ids) if ai == i]
        if mask:
            agent_means[i] = np.mean(outcomes[mask])
    agent_pred = np.array([agent_means.get(ai, grand_mean) for ai in agent_ids])
    ss_agent = np.sum((agent_pred - grand_mean) ** 2)

    # Seat factor
    seat_means = {}
    for s in range(num_players):
        mask = [j for j, si in enumerate(seat_ids) if si == s]
        if mask:
            seat_means[s] = np.mean(outcomes[mask])
    seat_pred = np.array([seat_means.get(si, grand_mean) for si in seat_ids])
    ss_seat = np.sum((seat_pred - grand_mean) ** 2)

    a_pct = ss_agent / ss_total * 100
    s_pct = ss_seat / ss_total * 100
    return {'agent_pct': a_pct, 'seat_pct': s_pct,
            'residual_pct': max(0, 100 - a_pct - s_pct)}


# ===========================================================================
# LATEX TABLE GENERATORS
# ===========================================================================

def tex_agent_winrates(analysis, caption, label):
    lines = [
        r'\begin{table}[H]', r'\centering',
        f'\\caption{{{caption}}}', f'\\label{{{label}}}',
        r'\small', r'\begin{tabular}{lcccc}', r'\toprule',
        r'\textbf{Agent} & \textbf{Win Rate (\%)} & \textbf{95\% CI} '
        r'& \textbf{Mean Score} & \textbf{Std Score} \\',
        r'\midrule',
    ]
    for nm in sorted(analysis):
        d = analysis[nm]
        wr = d['win_rate'] * 100
        ci = f"[{d['ci_low']*100:.1f}, {d['ci_high']*100:.1f}]"
        lines.append(f"  {nm} & {wr:.1f} & {ci} & {d['mean_score']:.2f} & {d['std_score']:.2f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def tex_positional_table(all_pos, num_players, caption, label):
    hdr = " & ".join([f"\\textbf{{Seat {i+1}}}" for i in range(num_players)])
    lines = [
        r'\begin{table}[H]', r'\centering',
        f'\\caption{{{caption}}}', f'\\label{{{label}}}',
        r'\small',
        f'\\begin{{tabular}}{{l{"c" * num_players}}}',
        r'\toprule', f'\\textbf{{Variant}} & {hdr} \\\\', r'\midrule',
    ]
    for vn, pd in all_pos.items():
        vals = " & ".join(f"{pd['seats'][s]['win_rate']*100:.1f}" for s in range(num_players))
        lines.append(f"  {vn} & {vals} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def tex_chi_square(all_pos, caption, label):
    lines = [
        r'\begin{table}[H]', r'\centering',
        f'\\caption{{{caption}}}', f'\\label{{{label}}}',
        r'\small', r'\begin{tabular}{lccl}', r'\toprule',
        r"\textbf{Variant} & $\chi^2$ & \textbf{df} & $p$\textbf{-value} \\",
        r'\midrule',
    ]
    for vn, pd in all_pos.items():
        c = pd['chi2']
        ps = "< 0.001" if c['p_value'] < 0.001 else f"{c['p_value']:.4f}"
        lines.append(f"  {vn} & {c['chi2']:.2f} & {c['df']} & {ps} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def tex_headtohead(h2h, caption, label):
    names = h2h['names']
    mat = h2h['matrix']
    n = len(names)
    sn = [nm[:12] for nm in names]
    hdr = ' & '.join(f'\\textbf{{{s}}}' for s in sn)
    lines = [
        r'\begin{table}[H]', r'\centering',
        f'\\caption{{{caption}}}', f'\\label{{{label}}}',
        r'\small', f'\\begin{{tabular}}{{l{"c" * n}}}', r'\toprule',
        f' & {hdr} \\\\', r'\midrule',
    ]
    for i in range(n):
        row = []
        for j in range(n):
            row.append('--' if i == j else f'{mat[i][j]*100:.1f}')
        lines.append(f'  {sn[i]} & ' + ' & '.join(row) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def tex_game_lengths(all_gl, caption, label):
    lines = [
        r'\begin{table}[H]', r'\centering',
        f'\\caption{{{caption}}}', f'\\label{{{label}}}',
        r'\small', r'\begin{tabular}{lccccc}', r'\toprule',
        r'\textbf{Variant} & \textbf{Mean} & \textbf{Median} & \textbf{Std} '
        r'& \textbf{Min} & \textbf{Max} \\',
        r'\midrule',
    ]
    for vn, gl in all_gl.items():
        lines.append(f"  {vn} & {gl['mean']:.1f} & {gl['median']:.0f} & "
                     f"{gl['std']:.1f} & {gl['min']} & {gl['max']} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def tex_anova(variance_data, caption, label):
    lines = [
        r'\begin{table}[H]', r'\centering',
        f'\\caption{{{caption}}}', f'\\label{{{label}}}',
        r'\small', r'\begin{tabular}{lccc}', r'\toprule',
        r'\textbf{Variant} & \textbf{Agent (\%)} & \textbf{Seat (\%)} '
        r'& \textbf{Residual (\%)} \\',
        r'\midrule',
    ]
    for vn, vd in variance_data.items():
        lines.append(f"  {vn} & {vd['agent_pct']:.1f} & {vd['seat_pct']:.1f} "
                     f"& {vd['residual_pct']:.1f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def tex_ttest_table(rows, caption, label):
    lines = [
        r'\begin{table}[H]', r'\centering',
        f'\\caption{{{caption}}}', f'\\label{{{label}}}',
        r'\small', r'\begin{tabular}{lccc}', r'\toprule',
        r"\textbf{Comparison} & $t$\textbf{-stat} & $p$\textbf{-value} "
        r"& \textbf{Cohen's} $d$ \\",
        r'\midrule',
    ]
    lines.extend(rows)
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


# ===========================================================================
# FIGURE GENERATORS
# ===========================================================================

def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"    → {path}")


def fig_agent_winrates(analysis, path):
    if not HAS_MATPLOTLIB:
        return
    names = sorted(analysis)
    rates = [analysis[n]['win_rate'] * 100 for n in names]
    lo = [analysis[n]['ci_low'] * 100 for n in names]
    hi = [analysis[n]['ci_high'] * 100 for n in names]
    errs = [[r - l for r, l in zip(rates, lo)],
            [h - r for h, r in zip(hi, rates)]]
    cols = [COLOURS.get(n, '#888') for n in names]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(len(names)), rates, yerr=errs, capsize=4,
           color=cols, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Win Rates by Agent (Baseline)')
    ax.axhline(25, color='gray', ls='--', lw=0.8, label='Expected 25%')
    ax.legend()
    ax.set_ylim(0, max(rates) * 1.3)
    _save(fig, path)


def fig_positional(all_pos, path, num_players=4):
    if not HAS_MATPLOTLIB:
        return
    vnames = list(all_pos)
    nv = len(vnames)
    x = np.arange(num_players)
    w = 0.8 / nv

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, vn in enumerate(vnames):
        rates = [all_pos[vn]['seats'][s]['win_rate'] * 100 for s in range(num_players)]
        ax.bar(x + (i - nv / 2 + 0.5) * w, rates, w, label=vn,
               edgecolor='black', linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Seat {i+1}' for i in range(num_players)])
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Positional Win Rates Across Variants')
    ax.axhline(25, color='gray', ls='--', lw=0.8)
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, 45)
    _save(fig, path)


def fig_game_lengths(all_gl, path):
    if not HAS_MATPLOTLIB:
        return
    names = list(all_gl)
    data = [all_gl[n]['lengths'] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(data, labels=names, patch_artist=True, showfliers=False)
    for patch, c in zip(bp['boxes'], plt.cm.Set2.colors):
        patch.set_facecolor(c)
    ax.set_ylabel('Game Length (turns)')
    ax.set_title('Game Length Distribution')
    plt.xticks(rotation=30, ha='right')
    _save(fig, path)


def fig_score_dists(analysis, path):
    if not HAS_MATPLOTLIB:
        return
    names = sorted(analysis)
    data = [analysis[n]['scores'] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.violinplot(data, showmeans=True, showmedians=True)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('Match Score')
    ax.set_title('Score Distribution by Agent')
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    _save(fig, path)


def fig_heatmap(h2h, path):
    if not HAS_MATPLOTLIB:
        return
    names = h2h['names']
    mat = h2h['matrix'] * 100
    n = len(names)
    sn = [nm[:12] for nm in names]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap='RdYlGn', vmin=0, vmax=100)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(sn, rotation=45, ha='right')
    ax.set_yticklabels(sn)
    for i in range(n):
        for j in range(n):
            t = '--' if i == j else f'{mat[i][j]:.0f}'
            ax.text(j, i, t, ha='center', va='center', fontsize=9,
                    color='black' if 30 < mat[i][j] < 70 else 'white')
    ax.set_title('Head-to-Head Win Rates (%)\n(Row beats Column)')
    fig.colorbar(im, ax=ax, shrink=0.8, label='Win %')
    _save(fig, path)


def fig_dice_sweep(sweep, path):
    if not HAS_MATPLOTLIB:
        return
    probs = sorted(sweep)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for seat in range(4):
        rates = [sweep[p]['seats'][seat]['win_rate'] * 100 for p in probs]
        ax.plot(probs, rates, 'o-', color=SEAT_COLOURS[seat],
                label=f'Seat {seat+1}', lw=1.5, ms=4)
    ax.axhline(25, color='gray', ls='--', lw=0.8, label='Fair (25%)')
    ax.set_xlabel('P(Good Dice Outcome)')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Positional Win Rate vs Dice Probability')
    ax.legend()
    ax.set_ylim(10, 40)
    _save(fig, path)


def fig_luck_skill(vd, path):
    if not HAS_MATPLOTLIB:
        return
    labels = list(vd)
    a = [vd[v]['agent_pct'] for v in labels]
    s = [vd[v]['seat_pct'] for v in labels]
    r = [100 - ai - si for ai, si in zip(a, s)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(labels))
    ax.bar(x, a, label='Agent (Skill)', color='#56B4E9', edgecolor='black', lw=0.3)
    ax.bar(x, s, bottom=a, label='Seat (Position)', color='#E69F00', edgecolor='black', lw=0.3)
    ax.bar(x, r, bottom=[ai + si for ai, si in zip(a, s)],
           label='Residual (Luck)', color='#999', edgecolor='black', lw=0.3)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('Variance Explained (%)')
    ax.set_title('Luck vs Skill: Variance Decomposition')
    ax.legend(); ax.set_ylim(0, 105)
    _save(fig, path)


def fig_architecture(path):
    if not HAS_MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.set_aspect('equal'); ax.axis('off')
    boxes = [
        (1.5, 5, 3, 1.2, 'Game Engine\n(Python)', '#56B4E9'),
        (5.5, 5, 3, 1.2, 'AI Agents\n(Rand, Heur,\nMCTS, RL)', '#E69F00'),
        (1.5, 2.8, 3, 1.2, 'Tournament\n& Analysis', '#009E73'),
        (5.5, 2.8, 3, 1.2, 'Web API\n(Flask)', '#D55E00'),
        (5.5, 0.8, 3, 1.2, 'Web Frontend\n(HTML/JS)', '#CC79A7'),
    ]
    for bx, by, bw, bh, txt, c in boxes:
        rect = mpatches.FancyBboxPatch((bx, by), bw, bh, boxstyle='round,pad=0.1',
                                        fc=c, ec='black', lw=1.2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(bx + bw/2, by + bh/2, txt, ha='center', va='center', fontsize=8,
                fontweight='bold', color='white')
    arrow_kw = dict(arrowstyle='->', lw=1.5, color='#333')
    ax.annotate('', xy=(5.5, 5.6), xytext=(4.5, 5.6), arrowprops=arrow_kw)
    ax.annotate('', xy=(3, 4.0), xytext=(3, 5.0), arrowprops=arrow_kw)
    ax.annotate('', xy=(5.5, 3.4), xytext=(4.5, 5.0), arrowprops=arrow_kw)
    ax.annotate('', xy=(7, 2.8), xytext=(7, 2.0), arrowprops=arrow_kw)
    ax.set_title('Cinquillo 2.0 System Architecture', fontsize=13, fontweight='bold', pad=15)
    _save(fig, path)


# ===========================================================================
# CSV OUTPUT
# ===========================================================================

def save_csv(results: List[GameResult], filepath: str):
    if not results:
        return
    n = len(results[0].player_names)
    fields = ['winner', 'num_turns'] + \
             [f'score_{i}' for i in range(n)] + \
             [f'agent_{i}' for i in range(n)]
    with open(filepath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for gr in results:
            row = {'winner': gr.winner, 'num_turns': gr.num_turns}
            for i in range(n):
                row[f'score_{i}'] = gr.final_scores[i]
                row[f'agent_{i}'] = gr.player_names[i]
            w.writerow(row)
    print(f"    → {filepath}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate dissertation data')
    parser.add_argument('--quick', action='store_true', help='1 000 games (fast)')
    parser.add_argument('--skip-mcts', action='store_true', help='Skip MCTS (slow)')
    parser.add_argument('--skip-rl', action='store_true', help='Skip RL agent')
    parser.add_argument('--output-dir', default='dissertation_output')
    parser.add_argument('--rl-weights', default='models/rl_agent_v4.pkl',
                        help='Path to trained RL weights')
    args = parser.parse_args()

    N = 1000 if args.quick else 10_000
    NP = 4

    # Directories
    base = args.output_dir
    fdir = os.path.join(base, 'figures')
    tdir = os.path.join(base, 'tables')
    ddir = os.path.join(base, 'data')
    for d in (fdir, tdir, ddir):
        os.makedirs(d, exist_ok=True)

    print("=" * 70)
    print("CINQUILLO 2.0 — DISSERTATION DATA GENERATOR")
    print("=" * 70)
    print(f"  Games per experiment : {N}")
    print(f"  Output               : {base}/")
    print(f"  MCTS                 : {'SKIP' if args.skip_mcts else 'ON'}")
    print(f"  RL                   : {'SKIP' if args.skip_rl or not HAS_RL else 'ON'}")
    print(f"  matplotlib           : {'YES' if HAS_MATPLOTLIB else 'NO'}")
    print(f"  scipy                : {'YES' if HAS_SCIPY else 'NO'}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. CREATE AGENTS
    # ------------------------------------------------------------------
    print("\n[1/8] Creating agents …")
    agents_core = [
        RandomAgent("Random"),
        create_aggressive_heuristic(),   # name = "Aggressive"
        create_defensive_heuristic(),    # name = "Defensive"
        create_balanced_heuristic(),     # name = "Balanced"
    ]

    agents_all = list(agents_core)

    if not args.skip_mcts:
        mcts = MCTSAgentSuperFast()  # 100 iterations – fast enough for 10k games
        agents_all.append(mcts)

    if not args.skip_rl and HAS_RL:
        try:
            rl = RLAgent(name="RL-v4", epsilon=0.0)
            rl.load_weights(args.rl_weights)

            # Smoke-test: play one quick game to verify state/model dims match
            print("  Smoke-testing RL agent …", end=" ")
            _test_var = make_baseline()
            _test_agents = [rl, create_balanced_heuristic(),
                            create_balanced_heuristic(), create_balanced_heuristic()]
            GameSimulator.simulate_game(_test_agents, _test_var, verbose=False)
            print("OK")
            agents_all.append(rl)
        except ValueError as e:
            print(f"\n  WARNING: RL model dimension mismatch – {e}")
            print(f"  The trained model expects a different state encoding size.")
            print(f"  Retrain with:  python simulation/train_rl_v3.py --episodes 5000")
            print(f"  RL agent will be SKIPPED for this run.\n")
        except Exception as e:
            print(f"  WARNING: RL agent skipped – {e}")

    agent_names = [a.name for a in agents_all]
    print(f"  Agents: {agent_names}")

    # ------------------------------------------------------------------
    # 2. BASELINE – Agent Performance (RQ2)
    # ------------------------------------------------------------------
    print(f"\n[2/8] Baseline tournament ({N} games) …")
    baseline_var = make_baseline()
    baseline_res = run_batch(agents_all, baseline_var, N)
    save_csv(baseline_res, os.path.join(ddir, 'baseline.csv'))

    ag = analyse_agents(baseline_res)
    print("\n  Agent Win Rates:")
    for nm in sorted(ag):
        d = ag[nm]
        print(f"    {nm:16s}  {d['win_rate']*100:5.1f}%  "
              f"CI=[{d['ci_low']*100:.1f},{d['ci_high']*100:.1f}]  "
              f"AvgScore={d['mean_score']:.2f}")

    with open(os.path.join(tdir, 'agent_winrates.tex'), 'w') as f:
        f.write(tex_agent_winrates(ag,
            f'Win rates (\\%) in {NP}-player baseline Cinquillo ({N} games).',
            'tab:agent_winrates'))
    fig_agent_winrates(ag, os.path.join(fdir, 'agent_winrates.pdf'))
    fig_score_dists(ag, os.path.join(fdir, 'score_distributions.pdf'))

    h2h = headtohead(baseline_res, agent_names)
    with open(os.path.join(tdir, 'headtohead.tex'), 'w') as f:
        f.write(tex_headtohead(h2h, 'Head-to-head win rates (\\%).', 'tab:headtohead'))
    fig_heatmap(h2h, os.path.join(fdir, 'heatmap_headtohead.pdf'))

    # ------------------------------------------------------------------
    # 3. POSITIONAL FAIRNESS ACROSS VARIANTS (RQ1)
    # ------------------------------------------------------------------
    print(f"\n[3/8] Positional fairness across variants …")
    all_pos = {}
    all_gl = {}
    all_res = {}

    for vname, vfunc in ALL_VARIANTS.items():
        print(f"\n  → {vname}")
        variant = vfunc()
        res = run_batch(agents_core, variant, N)
        all_res[vname] = res
        save_csv(res, os.path.join(ddir, f"{vname.replace(' ','_')}.csv"))

        pos = analyse_positions(res, NP)
        all_pos[vname] = pos

        gl = analyse_lengths(res)
        all_gl[vname] = gl

        c = pos['chi2']
        print(f"    χ²={c['chi2']:.2f}  p={c['p_value']:.4f}")
        for s in range(NP):
            sd = pos['seats'][s]
            print(f"    Seat {s+1}: {sd['win_rate']*100:.1f}%  "
                  f"CI=[{sd['ci_low']*100:.1f},{sd['ci_high']*100:.1f}]")

    with open(os.path.join(tdir, 'positional_fairness.tex'), 'w') as f:
        f.write(tex_positional_table(all_pos, NP,
            f'Win rate (\\%) by seat across variants ({N} games, rotated seats).',
            'tab:positional_wr'))
    with open(os.path.join(tdir, 'chi_square.tex'), 'w') as f:
        f.write(tex_chi_square(all_pos,
            'Chi-square tests for positional uniformity.', 'tab:chi_square'))
    with open(os.path.join(tdir, 'game_lengths.tex'), 'w') as f:
        f.write(tex_game_lengths(all_gl,
            'Game length statistics by variant.', 'tab:game_length'))

    fig_positional(all_pos, os.path.join(fdir, 'positional_fairness.pdf'))
    fig_game_lengths(all_gl, os.path.join(fdir, 'game_length_distributions.pdf'))

    # ------------------------------------------------------------------
    # 4. T-TESTS: BASELINE VS VARIANTS (RQ1, RQ3)
    # ------------------------------------------------------------------
    print(f"\n[4/8] t-tests (baseline vs variants) …")
    ttest_rows = []
    base_scores = [gr.final_scores[gr.winner] for gr in all_res.get('Baseline', baseline_res)]

    for vname in ALL_VARIANTS:
        if vname == 'Baseline':
            continue
        var_scores = [gr.final_scores[gr.winner] for gr in all_res[vname]]
        if HAS_SCIPY:
            t, p = scipy_stats.ttest_ind(base_scores, var_scores)
        else:
            t, p = float('nan'), float('nan')
        d = cohens_d(base_scores, var_scores)
        ps = "< 0.001" if p < 0.001 else f"{p:.4f}"
        ttest_rows.append(f"  Baseline vs.\\ {vname} & {t:.2f} & {ps} & {d:.3f} \\\\")
        print(f"  Baseline vs {vname}: t={t:.2f}  p={ps}  d={d:.3f}")

    with open(os.path.join(tdir, 'ttest_variants.tex'), 'w') as f:
        f.write(tex_ttest_table(ttest_rows,
            'Two-sample $t$-tests comparing winner scores.', 'tab:ttest'))

    # ------------------------------------------------------------------
    # 5. LUCK VS SKILL (RQ2)
    # ------------------------------------------------------------------
    print(f"\n[5/8] Variance decomposition …")
    vd = {}
    core_names = [a.name for a in agents_core]
    for vname in ALL_VARIANTS:
        v = variance_decomposition(all_res[vname], core_names, NP)
        vd[vname] = v
        print(f"  {vname:15s}  Agent={v['agent_pct']:.1f}%  "
              f"Seat={v['seat_pct']:.1f}%  Residual={v['residual_pct']:.1f}%")

    with open(os.path.join(tdir, 'anova.tex'), 'w') as f:
        f.write(tex_anova(vd, 'Variance decomposition by factor.', 'tab:anova'))
    fig_luck_skill(vd, os.path.join(fdir, 'luck_vs_skill.pdf'))

    # ------------------------------------------------------------------
    # 6. DICE PROBABILITY SWEEP (RQ3)
    # ------------------------------------------------------------------
    print(f"\n[6/8] Dice probability sweep …")
    sweep_n = max(500, N // 2)
    sweep = {}
    for prob in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        print(f"  p(good)={prob:.1f} …")
        v = VariantConfig(
            dice_good_probability=prob,
            dice_good_effect=GoodDiceEffect.WILD,
            dice_bad_effect=BadDiceEffect.TAKE_CARDS,
            dice_bad_cards_count=2,
            scoring_mode=ScoringMode.WINNER_TAKES_ALL,
            points_per_card=1,
            voluntary_pass_penalty=1,
            match_end_mode=MatchEndMode.FIXED_ROUNDS,
            fixed_rounds_count=5,
        )
        res = run_batch(agents_core, v, sweep_n, verbose=False)
        sweep[prob] = analyse_positions(res, NP)

    fig_dice_sweep(sweep, os.path.join(fdir, 'dice_probability_sweep.pdf'))

    # ------------------------------------------------------------------
    # 7. RL LEARNING CURVE (if log exists)
    # ------------------------------------------------------------------
    print(f"\n[7/8] RL learning curve …")
    rl_log_found = False
    for candidate in ['models/training_log.json', 'simulation/training_log.json']:
        if os.path.exists(candidate):
            try:
                with open(candidate) as f:
                    rl_log = json.load(f)
                eps = rl_log.get('episodes', rl_log.get('episode', []))
                wrs = rl_log.get('win_rates', rl_log.get('win_rate', []))
                if eps and wrs and HAS_MATPLOTLIB:
                    fig, ax = plt.subplots(figsize=(7, 4.5))
                    ax.plot(eps, [w * 100 for w in wrs], '-', color='#D55E00', lw=1.5)
                    ax.axhline(25, color='gray', ls='--', lw=0.8, label='Random (25%)')
                    ax.set_xlabel('Training Episodes')
                    ax.set_ylabel('Win Rate vs Heuristics (%)')
                    ax.set_title('RL Agent Learning Curve')
                    ax.legend()
                    _save(fig, os.path.join(fdir, 'rl_learning_curve.pdf'))
                    rl_log_found = True
            except Exception as e:
                print(f"  Could not parse {candidate}: {e}")
            break
    if not rl_log_found:
        print("  No training log found — skipping.")

    # ------------------------------------------------------------------
    # 8. ARCHITECTURE DIAGRAM
    # ------------------------------------------------------------------
    print(f"\n[8/8] Architecture diagram …")
    fig_architecture(os.path.join(fdir, 'architecture_diagram.pdf'))

    # ------------------------------------------------------------------
    # DONE
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    for subdir in (fdir, tdir, ddir):
        print(f"\n{subdir}/")
        if os.path.isdir(subdir):
            for fn in sorted(os.listdir(subdir)):
                print(f"  {fn}")
    sep = '─' * 70
    print(f"\n{sep}\nTO USE IN LATEX:\n{sep}")
    print(r"  \graphicspath{{dissertation_output/figures/}}")
    print()
    print(r"  \begin{figure}[H]")
    print(r"    \centering")
    print(r"    \includegraphics[width=0.9\textwidth]{agent_winrates}")
    print(r"    \caption{Win rates by agent type.}")
    print(r"  \end{figure}")
    print()
    print(r"  \input{dissertation_output/tables/agent_winrates}")
    print(sep)
    print()


if __name__ == '__main__':
    main()