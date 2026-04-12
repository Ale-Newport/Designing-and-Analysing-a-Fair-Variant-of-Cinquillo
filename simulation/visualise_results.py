#!/usr/bin/env python3
"""
==========================================================================
Cinquillo 2.0  —  Visualisation & Report Generator  (visualise_results.py)
==========================================================================
Reads JSON/CSV output from run_simulations.py and produces ALL original
figures and tables from generate_data.py PLUS new ones for the expanded
experiment set (Exp 8–16).

Place at:  simulation/visualise_results.py
Run from PROJECT ROOT after run_simulations.py:
    python simulation/visualise_results.py
    python simulation/visualise_results.py --exp 1 8 9
    python simulation/visualise_results.py --format png

Structure
---------
Each experiment is handled by its own run_expN_*() function, which:
  • loads the relevant JSON from output/data/experiments/
  • generates all figures (saved to output/figures/)
  • generates all LaTeX tables (saved to output/tables/)
  • prints a brief description of every artefact it creates

The main() function wires the CLI, builds the path helpers, and dispatches
to whichever experiment runners are selected via --exp.
==========================================================================
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import warnings
from typing import Dict, List, Optional

import numpy as np
warnings.filterwarnings('ignore')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.cm as cm

    _MPL_VER  = tuple(int(x) for x in matplotlib.__version__.split('.')[:2])
    _BP_LABEL = 'tick_labels' if _MPL_VER >= (3, 9) else 'labels'

    def _get_cmap(name, n=None):
        try:
            cmap = matplotlib.colormaps[name]
        except (AttributeError, KeyError):
            cmap = cm.get_cmap(name)
        return cmap.resampled(n) if n is not None else cmap

    HAS_MPL = True
except ImportError:
    print("ERROR: matplotlib is required."); sys.exit(1)

try:
    from scipy import stats as scipy_stats
    from scipy.stats import kruskal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ===========================================================================
# STYLE  (identical to original generate_data.py)
# ===========================================================================

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
    'Balanced': '#0072B2', 'Risky': '#CC79A7',
    'MCTS-SuperFast': '#FF7F0E', 'MCTS-Fast': '#D55E00',
    'MCTS-Standard': '#E8A070', 'MCTS': '#D55E00',
    'MCTS-Deep': '#8B1A00', 'RL': '#F0E442',
}
SEAT_COLOURS = ['#56B4E9', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#999999']


def _agent_colour(name: str) -> str:
    for k, c in AGENT_COLOURS.items():
        if name.startswith(k):
            return c
    return '#888888'


# ===========================================================================
# HELPERS
# ===========================================================================

def _header(title, n=0, total=0):
    bar = '━' * 72
    tag = f'VIS {n}/{total}  ' if n else ''
    print(f'\n{bar}\n  {tag}{title}\n{bar}')

def _sub(msg):  print(f'  ▸ {msg}')
def _ok(path):  print(f'    ✓ {os.path.basename(path)}')
def _warn(msg): print(f'  ⚠  {msg}')
def _info(msg): print(f'    {msg}')


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    _ok(path)


def _load(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        _warn(f'Missing: {path}')
        return None
    with open(path) as f:
        return json.load(f)


def _save_tex(content: str, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    _ok(path)


def _wilson_ci(w, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p     = w / n
    denom = 1 + z**2 / n
    ctr   = (p + z**2 / (2*n)) / denom
    marg  = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denom
    return max(0.0, ctr - marg), min(1.0, ctr + marg)


def _seat(seats: dict, s) -> dict:
    """JSON deserialises integer keys as strings. Try int key first, then str."""
    if s in seats:
        return seats[s]
    if str(s) in seats:
        return seats[str(s)]
    return {'win_rate': 0.0, 'wins': 0, 'total': 1,
            'ci_low': 0.0, 'ci_high': 0.0,
            'mean_score': 0.0, 'std_score': 0.0}


# ===========================================================================
# LATEX TABLE GENERATORS  — ALL ORIGINALS RESTORED
# ===========================================================================

def _tex_wrap(body_lines, caption, label):
    return '\n'.join([
        r'\begin{table}[H]', r'\centering',
        f'\\caption{{{caption}}}', f'\\label{{{label}}}',
        r'\small',
    ] + body_lines + [r'\end{table}'])


def tex_agent_winrates(ag: dict, caption: str, label: str) -> str:
    body = [r'\begin{tabular}{lcccc}', r'\toprule',
            r'\textbf{Agent} & \textbf{Win Rate (\%)} & \textbf{95\% CI} '
            r'& \textbf{Mean Score} & \textbf{Std Score} \\', r'\midrule']
    for nm in sorted(ag):
        d  = ag[nm]
        lo, hi = _wilson_ci(d['wins'], d['total'])
        ci = f"[{lo*100:.1f}, {hi*100:.1f}]"
        body.append(f"  {nm} & {d['win_rate']*100:.1f} & {ci} & "
                    f"{d['mean_score']:.2f} & {d['std_score']:.2f} \\\\")
    body += [r'\bottomrule', r'\end{tabular}']
    return _tex_wrap(body, caption, label)


def tex_positional_table(all_pos: dict, num_players: int,
                          caption: str, label: str) -> str:
    hdr  = ' & '.join(f'\\textbf{{Seat {i+1}}}' for i in range(num_players))
    body = [f'\\begin{{tabular}}{{l{"c"*num_players}}}', r'\toprule',
            f'\\textbf{{Variant}} & {hdr} \\\\', r'\midrule']
    for vn, pd in all_pos.items():
        vals = ' & '.join(f"{_seat(pd['seats'], s)['win_rate']*100:.1f}"
                          for s in range(num_players))
        body.append(f'  {vn} & {vals} \\\\')
    body += [r'\bottomrule', r'\end{tabular}']
    return _tex_wrap(body, caption, label)


def tex_chi_square(all_pos: dict, caption: str, label: str) -> str:
    body = [r'\begin{tabular}{lccl}', r'\toprule',
            r'\textbf{Variant} & $\chi^2$ & \textbf{df} & $p$-value \\',
            r'\midrule']
    for vn, pd in all_pos.items():
        c  = pd['chi2']
        ps = '< 0.001' if c['p_value'] < 0.001 else f"{c['p_value']:.4f}"
        sig = r'$^*$' if c['p_value'] < 0.05 else ''
        body.append(f"  {vn} & {c['chi2']:.2f} & {c['df']} & {ps}{sig} \\\\")
    body += [r'\bottomrule', r'\end{tabular}']
    return _tex_wrap(body, caption, label)


def tex_headtohead(h2h: dict, caption: str, label: str) -> str:
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


def tex_game_lengths(all_gl: dict, caption: str, label: str) -> str:
    body = [r'\begin{tabular}{lccccc}', r'\toprule',
            r'\textbf{Variant} & \textbf{Mean} & \textbf{Median} '
            r'& \textbf{Std} & \textbf{Min} & \textbf{Max} \\', r'\midrule']
    for vn, gl in all_gl.items():
        body.append(f"  {vn} & {gl['mean']:.1f} & {gl['median']:.0f} & "
                    f"{gl['std']:.1f} & {gl['min']} & {gl['max']} \\\\")
    body += [r'\bottomrule', r'\end{tabular}']
    return _tex_wrap(body, caption, label)


def tex_anova(vd: dict, caption: str, label: str) -> str:
    body = [r'\begin{tabular}{lccc}', r'\toprule',
            r'\textbf{Variant} & \textbf{Agent (\%)} & \textbf{Seat (\%)} '
            r'& \textbf{Residual (\%)} \\', r'\midrule']
    for vn, d in vd.items():
        body.append(f"  {vn} & {d['agent_pct']:.1f} & "
                    f"{d['seat_pct']:.1f} & {d['residual_pct']:.1f} \\\\")
    body += [r'\bottomrule', r'\end{tabular}']
    return _tex_wrap(body, caption, label)


def tex_ttest_table(rows: list, caption: str, label: str) -> str:
    body = [r'\begin{tabular}{lccc}', r'\toprule',
            r"\textbf{Comparison} & $t$-stat & $p$-value & Cohen's $d$ \\",
            r'\midrule']
    body.extend(rows)
    body += [r'\bottomrule', r'\end{tabular}']
    return _tex_wrap(body, caption, label)


def tex_fairness_ranking(all_pos: dict, caption: str, label: str) -> str:
    rows_data = sorted(
        [(vn, pd['pos']['gini'], pd['pos']['max_deviation']*100,
          pd['pos']['chi2']['chi2'], pd['pos']['chi2']['p_value'])
         for vn, pd in all_pos.items()],
        key=lambda x: x[1])
    body = [r'\begin{tabular}{lcccc}', r'\toprule',
            r'\textbf{Variant} & \textbf{Gini} & \textbf{Max Dev.\ (\%)} '
            r'& $\chi^2$ & $p$-value \\', r'\midrule']
    for vn, g, md, c2, pv in rows_data:
        ps  = '< 0.001' if pv < 0.001 else f'{pv:.4f}'
        sig = r'$^*$' if pv < 0.05 else ''
        body.append(f'  {vn} & {g:.4f} & {md:.1f} & {c2:.2f} & {ps}{sig} \\\\')
    body += [r'\bottomrule', r'\end{tabular}']
    return _tex_wrap(body, caption, label)


def tex_score_margins(sm_data: dict, caption: str, label: str) -> str:
    body = [r'\begin{tabular}{lcccc}', r'\toprule',
            r'\textbf{Variant} & \textbf{Mean Margin} & \textbf{Median} '
            r'& \textbf{Tight Games (\%)} & \textbf{Score Var.} \\', r'\midrule']
    for vn, d in sm_data.items():
        body.append(f"  {vn} & {d['mean']:.2f} & {d['median']:.1f} & "
                    f"{d.get('tight_pct', d.get('tight_games_pct', 0)):.1f} & "
                    f"{d['score_variance']:.2f} \\\\")
    body += [r'\bottomrule', r'\end{tabular}']
    return _tex_wrap(body, caption, label)


def tex_player_count(pc_data: dict, caption: str, label: str) -> str:
    body = [r'\begin{tabular}{lcccccc}', r'\toprule',
            r'\textbf{N} & \textbf{S1 (\%)} & \textbf{S2 (\%)} & \textbf{S3 (\%)} '
            r'& \textbf{S4 (\%)} & \textbf{Gini} & $p$-val \\', r'\midrule']
    for npl, data in sorted(pc_data.items(), key=lambda x: int(x[0])):
        seats = data['pos']['seats']
        def _wr(s):
            return (seats[s]['win_rate'] if s in seats
                    else seats.get(str(s), {}).get('win_rate', 0)) * 100
        wrs = [f'{_wr(s):.1f}' if s < int(npl) else '--' for s in range(4)]
        pv  = data['pos']['chi2']['p_value']
        ps  = '< 0.001' if pv < 0.001 else f'{pv:.4f}'
        body.append(f"  {npl}P & {' & '.join(wrs)} & "
                    f"{data['pos']['gini']:.4f} & {ps} \\\\")
    body += [r'\bottomrule', r'\end{tabular}']
    return _tex_wrap(body, caption, label)


def tex_information_analysis(info_data: dict, caption: str, label: str) -> str:
    agents = ['Random', 'Aggressive', 'Defensive', 'Balanced']
    hdrs   = ' & '.join(f'\\textbf{{{a}}} WR (\\%)' for a in agents)
    body   = [r'\begin{tabular}{lcccc}', r'\toprule',
              f'\\textbf{{Variant}} & {hdrs} \\\\', r'\midrule']
    for vn, vd in sorted(info_data.items()):
        ag = vd.get('ag', {})
        cells = ' & '.join(f"{ag.get(nm,{}).get('win_rate',0)*100:.1f}"
                           for nm in agents)
        body.append(f'  {vn} & {cells} \\\\')
    body += [r'\bottomrule', r'\end{tabular}']
    return _tex_wrap(body, caption, label)


# ===========================================================================
# EXP 1  —  First-Player Advantage
# ===========================================================================

def fig_first_player_advantage(all_pos: dict, path: str, num_players: int = 4):
    fair   = 1.0 / num_players
    vns    = list(all_pos)
    advs   = [(_seat(all_pos[v]['seats'], 0)['win_rate'] - fair) * 100 for v in vns]
    paired = sorted(zip(vns, advs), key=lambda x: -x[1])
    if not paired:
        return
    vns, advs = zip(*paired)
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
    vnames = list(all_pos)
    mat    = np.array([[_seat(all_pos[v]['seats'], s)['win_rate'] * 100
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
    ax.set_title('Positional Win Rates (%) — All Variants\n'
                 '(Green = above fair, Red = below fair)')
    plt.tight_layout()
    _save(fig, path)


def fig_positional_grouped(all_pos: dict, path: str, num_players: int = 4):
    vnames = list(all_pos)
    nv     = len(vnames)
    x      = np.arange(num_players)
    w      = min(0.8 / max(nv, 1), 0.15)
    cmap   = _get_cmap('tab20', nv)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, vn in enumerate(vnames):
        rates = [_seat(all_pos[vn]['seats'], s)['win_rate'] * 100 for s in range(num_players)]
        ax.bar(x + (i - nv/2 + 0.5) * w, rates, w, label=vn,
               color=cmap(i / max(nv, 1)), edgecolor='black', lw=0.3, alpha=0.85)
    ax.axhline(100/num_players, color='black', ls='--', lw=0.8,
               label=f'Fair ({100/num_players:.0f}%)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Seat {i+1}' for i in range(num_players)])
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Positional Win Rates by Variant (Grouped)')
    if nv <= 8:
        ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    _save(fig, path)


# ===========================================================================
# EXP 2  —  Dice Probability Sweep
# ===========================================================================

def fig_positional_vs_dice_prob(sweep: dict, path: str, nseats: int = 4):
    probs = sorted(sweep.keys())
    pvals = [float(k.split('=')[1]) for k in probs]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for seat in range(nseats):
        rates = [_seat(sweep[k]['pos']['seats'], seat)['win_rate'] * 100 for k in probs]
        ax.plot(pvals, rates, 'o-', color=SEAT_COLOURS[seat],
                label=f'Seat {seat+1}', lw=1.5, ms=4)
    ax.axhline(100/nseats, color='gray', ls='--', lw=0.8, label='Fair')
    ax.set_xlabel('P(Good Dice Outcome)')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Positional Win Rate vs Dice Probability')
    ax.legend()
    plt.tight_layout()
    _save(fig, path)


def fig_dice_sweep_by_agent(sweep: dict, path: str):
    probs  = sorted(sweep.keys())
    pvals  = [float(k.split('=')[1]) for k in probs]
    agents = sorted(sweep[probs[0]]['ag'].keys()) if probs else []
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ag in agents:
        rates = [sweep[k]['ag'].get(ag, {}).get('win_rate', 0) * 100 for k in probs]
        ax.plot(pvals, rates, 'o-', color=_agent_colour(ag),
                label=ag, lw=1.5, ms=4)
    ax.axhline(25, color='gray', ls='--', lw=0.8, label='Random (25%)')
    ax.set_xlabel('P(Good Dice Outcome)')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Agent Win Rate vs Dice Probability')
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, path)


def fig_dice_gini_length(sweep: dict, path: str):
    probs = sorted(sweep.keys())
    pvals = [float(k.split('=')[1]) for k in probs]
    ginis = [sweep[k]['pos']['gini'] for k in probs]
    gl_ms = [sweep[k]['gl']['mean']  for k in probs]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()
    ax1.plot(pvals, ginis, 'o-', color='#D55E00', lw=1.8, label='Gini')
    ax2.plot(pvals, gl_ms, 's--', color='#0072B2', lw=1.5, label='Mean turns')
    ax1.set_xlabel('P(Good Dice Outcome)')
    ax1.set_ylabel('Gini Coefficient', color='#D55E00')
    ax2.set_ylabel('Mean Game Length (turns)', color='#0072B2')
    ax1.set_title('Fairness & Game Length vs Dice Probability')
    l1, b1 = ax1.get_legend_handles_labels()
    l2, b2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, b1+b2, loc='upper left', fontsize=8)
    plt.tight_layout()
    _save(fig, path)


def fig_bad_effect_comparison(bad_eff_data: dict, path: str, num_players: int = 4):
    effects = list(bad_eff_data)
    fair    = 100 / num_players
    fig, axes = plt.subplots(1, len(effects), figsize=(4*len(effects), 4), sharey=True)
    if len(effects) == 1:
        axes = [axes]
    for ax, eff in zip(axes, effects):
        wrs = [_seat(bad_eff_data[eff]['pos']['seats'], s)['win_rate'] * 100
               for s in range(num_players)]
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
    vnames = list(group_data)
    mat    = np.array([[_seat(group_data[v]['pos']['seats'], s)['win_rate'] * 100
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
    ax.set_title(f'Positional Win Rates — {key}')
    plt.tight_layout()
    _save(fig, path)


# ===========================================================================
# EXP 3  —  Luck vs Skill
# ===========================================================================

def fig_luck_skill_stacked(vd: dict, path: str):
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



# ===========================================================================
# EXP 4  —  Comeback / Snowball
# ===========================================================================

def fig_score_margins(sm_data: dict, path: str):
    vnames = list(sm_data)
    data   = [sm_data[v]['margins'] for v in vnames]
    fig, ax = plt.subplots(figsize=(max(10, len(vnames)*0.7), 5))
    bp = ax.boxplot(data, patch_artist=True, showfliers=True,
                    flierprops={'marker': 'o', 'markersize': 2, 'alpha': 0.3},
                    **{_BP_LABEL: vnames})
    cmap = _get_cmap('tab20', len(vnames))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(cmap(i / max(len(vnames), 1)))
        patch.set_alpha(0.8)
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_ylabel('Winner − Runner-up Score Margin')
    ax.set_title('Score Margin Distributions')
    plt.xticks(rotation=35, ha='right', fontsize=8)
    plt.tight_layout()
    _save(fig, path)


def fig_tight_games(sm_data: dict, path: str):
    vnames = list(sm_data)
    tights = [sm_data[v].get('tight_pct', sm_data[v].get('tight_games_pct', 0))
              for v in vnames]
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
    agents = sorted(set(short_ag) & set(long_ag))
    x = [short_ag[ag]['win_rate'] * 100 for ag in agents]
    y = [long_ag[ag]['win_rate']  * 100 for ag in agents]
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
    rounds = sorted(round_ag_data)
    agents = list(next(iter(round_ag_data.values())).keys()) if round_ag_data else []
    fig, ax = plt.subplots(figsize=(8, 5))
    for ag in agents:
        rates = [round_ag_data[r].get(ag, {}).get('win_rate', 0) * 100 for r in rounds]
        ax.plot(rounds, rates, 'o-', color=_agent_colour(ag),
                label=ag, lw=1.5, ms=5)
    ax.axhline(25, color='gray', ls='--', lw=0.8, label='Random (25%)')
    ax.set_xlabel('Number of Rounds')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Agent Win-Rate Stability vs Round Count')
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, path)


# ===========================================================================
# EXP 5  —  Heuristic Agent Analysis
# ===========================================================================

def fig_agent_winrates(analysis: dict, path: str, num_players: int = 4, title: str = 'Agent Win Rates — Baseline Variant'):
    names = sorted(analysis)
    rates = [analysis[n]['win_rate'] * 100 for n in names]
    lo_err = [analysis[n]['win_rate']*100 - _wilson_ci(analysis[n]['wins'], analysis[n]['total'])[0]*100 for n in names]
    hi_err = [_wilson_ci(analysis[n]['wins'], analysis[n]['total'])[1]*100 - analysis[n]['win_rate']*100 for n in names]
    cols = [_agent_colour(n) for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(names)), rates, yerr=[lo_err, hi_err], capsize=4, color=cols, edgecolor='black', lw=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title(title)
    ax.axhline(100/num_players, color='gray', ls='--', lw=0.8, label=f'Expected ({100/num_players:.0f}%)')
    ax.legend()
    ax.set_ylim(0, (max(rates)+ max(hi_err)*1.10) if rates else 60)
    plt.tight_layout()
    _save(fig, path)


def fig_heatmap(h2h: dict, path: str):
    names = h2h['names']
    mat = np.array(h2h['matrix']) * 100
    sn = [nm[:12] for nm in names]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap='RdYlGn', vmin=0, vmax=100)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(sn, rotation=45, ha='right')
    ax.set_yticklabels(sn)
    for i in range(len(names)):
        for j in range(len(names)):
            t = '--' if i == j else f'{mat[i,j]:.0f}'
            ax.text(j, i, t, ha='center', va='center', fontsize=9, color='black' if 30 < mat[i,j] < 70 else 'white')
    ax.set_title('Head-to-Head Win Rates (%)\n(Row beats Column)')
    fig.colorbar(im, ax=ax, shrink=0.8, label='Win %')
    plt.tight_layout()
    _save(fig, path)


def fig_agent_radar(analysis: dict, path: str):
    names = sorted(analysis)
    metrics = ['Win Rate', 'Mean Score', 'Score\nStability', 'Median Score']
    n_m = len(metrics)
    angles = np.linspace(0, 2*np.pi, n_m, endpoint=False).tolist() + [0]

    def _norm(vals):
        mn, mx = min(vals), max(vals)
        return [0.5]*len(vals) if mx == mn else [(v-mn)/(mx-mn) for v in vals]

    wr_n = _norm([analysis[n]['win_rate']           for n in names])
    ms_n = _norm([analysis[n]['mean_score']         for n in names])
    st_n = _norm([1/(analysis[n]['std_score']+1e-6) for n in names])
    md_n = _norm([analysis[n]['median_score']       for n in names])

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
    vnames = list(all_agent_ag)
    agents = sorted(next(iter(all_agent_ag.values())).keys()) if all_agent_ag else []
    nv, na = len(vnames), len(agents)
    x = np.arange(nv)
    w = 0.8 / max(na, 1)
    fig, ax = plt.subplots(figsize=(max(10, nv * 0.7), 5))
    for i, ag in enumerate(agents):
        rates = [all_agent_ag[v].get(ag, {}).get('win_rate', 0) * 100 for v in vnames]
        ax.bar(x + (i - na/2 + 0.5)*w, rates, w, label=ag, color=_agent_colour(ag), edgecolor='black', lw=0.3, alpha=0.85)
    ax.axhline(25, color='black', ls='--', lw=0.8, label='Fair (25%)')
    ax.set_xticks(x)
    ax.set_xticklabels(vnames, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Agent Win Rates Across Variants')
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    _save(fig, path)


# ===========================================================================
# EXP 6  —  Number of Players
# ===========================================================================

def _seat_wr(seats, s):
    return _seat(seats, s)['win_rate'] * 100

def _seat_wi(seats, s):
    d = _seat(seats, s)
    return d.get('wins', 0), d.get('total', 1)


def fig_player_count_positional(pc_data: dict, path: str):
    n_counts = sorted(pc_data.keys(), key=int)
    ncols = len(n_counts)
    fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]
    for ax, npl in zip(axes, n_counts):
        npl_i = int(npl)
        seats  = pc_data[npl]['pos']['seats']
        wrs    = [_seat_wr(seats, s) for s in range(npl_i)]
        cis    = [_wilson_ci(*_seat_wi(seats, s)) for s in range(npl_i)]
        lo_err = [wrs[s] - cis[s][0]*100 for s in range(npl_i)]
        hi_err = [cis[s][1]*100 - wrs[s] for s in range(npl_i)]
        ax.bar(range(npl_i), wrs, yerr=[lo_err, hi_err], capsize=4,
               color=SEAT_COLOURS[:npl_i], edgecolor='black', lw=0.4)
        ax.axhline(100/npl_i, color='gray', ls='--', lw=0.8)
        ax.set_xticks(range(npl_i))
        ax.set_xticklabels([f'S{s+1}' for s in range(npl_i)])
        ax.set_title(f'{npl} Players')
        ax.set_ylim(0, 60)
    axes[0].set_ylabel('Win Rate (%)')
    fig.suptitle('Positional Win Rates by Player Count (Baseline)', fontsize=12)
    plt.tight_layout()
    _save(fig, path)


def fig_player_count_lengths(pc_data: dict, path: str):
    n_counts = sorted(pc_data.keys(), key=int)
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
    n_counts = sorted(pc_data.keys(), key=int)
    ginis    = [pc_data[n]['pos']['gini']             for n in n_counts]
    max_devs = [pc_data[n]['pos']['max_deviation']*100 for n in n_counts]
    n_ints   = [int(n) for n in n_counts]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()
    ax1.plot(n_ints, ginis,    'o-', color='#0072B2', lw=1.8, label='Gini')
    ax2.plot(n_ints, max_devs, 's--', color='#D55E00', lw=1.8, label='Max dev. (%)')
    ax1.set_xlabel('Number of Players')
    ax1.set_ylabel('Gini Coefficient', color='#0072B2')
    ax2.set_ylabel('Max Seat Deviation (%)', color='#D55E00')
    ax1.set_title('Fairness Metrics vs Player Count')
    l1, b1 = ax1.get_legend_handles_labels()
    l2, b2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, b1+b2, loc='upper left', fontsize=8)
    plt.tight_layout()
    _save(fig, path)


# ===========================================================================
# EXP 7  —  Fairness
# ===========================================================================

def _pos(d):
    """Extract the 'pos' sub-dict whether d has a 'pos' key or IS the pos dict."""
    return d['pos'] if 'pos' in d else d

def fig_fairness_scatter(all_pos: dict, path: str):
    vnames = list(all_pos)
    ginis  = [_pos(all_pos[v])['gini']             for v in vnames]
    maxd   = [_pos(all_pos[v])['max_deviation']*100 for v in vnames]
    pvals  = [_pos(all_pos[v])['chi2']['p_value']   for v in vnames]
    cols   = ['#D55E00' if p < 0.05 else '#56B4E9' for p in pvals]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(ginis, maxd, c=cols, s=60, edgecolors='black', lw=0.5, zorder=3)
    for vn, g, m in zip(vnames, ginis, maxd):
        ax.annotate(vn, (g, m), textcoords='offset points', xytext=(4,3), fontsize=6.5)
    sig   = mpatches.Patch(color='#D55E00', label='Significant bias ($p$<0.05)')
    nosig = mpatches.Patch(color='#56B4E9', label='No significant bias')
    ax.legend(handles=[sig, nosig])
    ax.set_xlabel('Gini Coefficient')
    ax.set_ylabel('Max Seat Deviation (%)')
    ax.set_title('Variant Fairness Landscape')
    plt.tight_layout()
    _save(fig, path)


def fig_fairness_ranking_bar(all_pos: dict, path: str):
    pairs = sorted([(v, _pos(all_pos[v])['gini']) for v in all_pos], key=lambda x: x[1])
    if not pairs:
        return
    names, vals = zip(*pairs)
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


def fig_game_lengths(all_gl: dict, path: str):
    names = list(all_gl)
    data  = [all_gl[n]['lengths'] for n in names]
    fig, ax = plt.subplots(figsize=(max(10, len(names)*0.65), 5))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    **{_BP_LABEL: names})
    cmap = _get_cmap('Set2', len(names))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(cmap(i / max(len(names), 1)))
    ax.set_ylabel('Game Length (turns)')
    ax.set_title('Game Length Distribution by Variant')
    plt.xticks(rotation=35, ha='right', fontsize=8)
    plt.tight_layout()
    _save(fig, path)


def fig_group_fairness_summary(group_pos: dict, path: str):
    gnames = list(group_pos)
    m_gini = [np.mean([group_pos[g][v]['pos']['gini'] for v in group_pos[g]])
              for g in gnames]
    s_gini = [np.std( [group_pos[g][v]['pos']['gini'] for v in group_pos[g]])
              for g in gnames]
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


def fig_group_game_lengths(group_gl: dict, path: str):
    ngroups = len(group_gl)
    ncols   = min(3, ngroups)
    nrows   = (ngroups + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows))
    axes = np.array(axes).flatten()
    for idx, (gname, gl_dict) in enumerate(group_gl.items()):
        ax    = axes[idx]
        vls   = list(gl_dict)
        means = [gl_dict[v]['gl']['mean'] for v in vls]
        stds  = [gl_dict[v]['gl']['std']  for v in vls]
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
# EXP 8 DIAGNOSTICS  —  RL curve + architecture
# ===========================================================================

def fig_rl_learning_curve(log_path: str, out_path: str):
    if not os.path.exists(log_path):
        _warn(f'RL log not found: {log_path}'); return
    with open(log_path) as f:
        rl_log = json.load(f)
    eps = rl_log.get('episodes', rl_log.get('episode', []))
    wrs = rl_log.get('win_rates', rl_log.get('win_rate', []))
    if not (eps and wrs):
        _warn('RL log has no episode/win_rate fields'); return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(eps, [w*100 for w in wrs], '-', color='#D55E00', lw=1.5)
    ax.axhline(25, color='gray', ls='--', lw=0.8, label='Random (25%)')
    ax.set_xlabel('Training Episodes')
    ax.set_ylabel('Win Rate vs Heuristics (%)')
    ax.set_title('RL Agent Learning Curve')
    ax.legend()
    _save(fig, out_path)


# ===========================================================================
# EXP 8–15
# ===========================================================================

def fig_info_sweep(info_sweep: dict, path: str):
    if not info_sweep: return
    probs  = sorted(info_sweep.keys())
    pvals  = [float(k.split('=')[1]) for k in probs]
    agents = sorted(info_sweep[probs[0]]['ag'].keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for nm in agents:
        wrs = [info_sweep[k]['ag'].get(nm, {}).get('win_rate', 0)*100 for k in probs]
        axes[0].plot(pvals, wrs, 'o-', label=nm, color=_agent_colour(nm), lw=1.8)
    axes[0].axhline(25, color='gray', ls='--', lw=0.7)
    axes[0].set_xlabel('Probability of Info Reveal')
    axes[0].set_ylabel('Win Rate (%)')
    axes[0].set_title('Win Rate vs Information Reveal Probability')
    axes[0].legend()
    gl_m = [info_sweep[k]['gl']['mean'] for k in probs]
    axes[1].plot(pvals, gl_m, 's-', color='#D55E00', lw=1.8)
    axes[1].set_xlabel('Probability of Info Reveal')
    axes[1].set_ylabel('Mean Turns')
    axes[1].set_title('Game Length vs Info Reveal Probability')
    fig.tight_layout(); _save(fig, path)


def fig_tournament_heatmap(rankings: dict, agent_names: list, path: str):
    vns = sorted(rankings.keys())
    mat = np.array([[rankings[vn].get(nm, 0)*100 for nm in agent_names] for vn in vns])
    fig, ax = plt.subplots(figsize=(len(agent_names)*2.2, max(6, len(vns)*0.32)))
    im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=0, vmax=50)
    ax.set_xticks(range(len(agent_names)))
    ax.set_xticklabels(agent_names, rotation=30, ha='right')
    ax.set_yticks(range(len(vns)))
    ax.set_yticklabels(vns, fontsize=8)
    for i in range(len(vns)):
        for j in range(len(agent_names)):
            ax.text(j, i, f'{mat[i,j]:.0f}', ha='center', va='center',
                    fontsize=6.5, color='black')
    fig.colorbar(im, ax=ax, label='Win Rate (%)')
    ax.set_title('All-Variant Tournament — Agent Win Rates (%)')
    fig.tight_layout(); _save(fig, path)


def fig_parametric_group_overview(groups: dict, path: str):
    gnames = list(groups.keys())
    ncols = 2; nrows = (len(gnames) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    axes = np.array(axes).flatten()
    for idx, gname in enumerate(gnames):
        ax = axes[idx]; gdata = groups[gname]; cfgs = list(gdata.keys())
        for nm, mk in [('Balanced', 'o-'), ('Aggressive', 's-'), ('Defensive', 'D-')]:
            wrs = [gdata[c]['ag'].get(nm, {}).get('win_rate', 0)*100 for c in cfgs]
            ax.plot(range(len(cfgs)), wrs, mk, color=_agent_colour(nm), lw=1.5, ms=4, label=nm)
        ax.axhline(25, color='gray', ls='--', lw=0.7)
        ax.set_xticks(range(len(cfgs)))
        ax.set_xticklabels(cfgs, rotation=35, ha='right', fontsize=7)
        ax.set_title(gname, fontsize=9); ax.set_ylabel('Win Rate (%)'); ax.set_ylim(0, 60)
        if idx == 0: ax.legend(fontsize=7)
    for idx in range(len(gnames), len(axes)): axes[idx].set_visible(False)
    fig.suptitle('Parametric Sweep — Agent Win Rates by Group', fontsize=13)
    fig.tight_layout(); _save(fig, path)


def fig_knowledge_advantage(cohens_d: dict, path: str):
    agents = sorted(cohens_d.keys())
    ds     = [cohens_d[a].get('d', 0) for a in agents]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(agents, ds, color=[_agent_colour(a) for a in agents], alpha=0.85)
    ax.axhline(0,   color='black', lw=0.8)
    ax.axhline(0.1, color='gray', ls=':', lw=0.8, label="Small ($d$=0.1)")
    ax.axhline(0.3, color='gray', ls='--', lw=0.8, label="Medium ($d$=0.3)")
    ax.set_ylabel("Cohen's $d$")
    ax.set_title("Knowledge Advantage: Score Gain in Info-Reveal vs Non-Info Variants")
    ax.legend(); fig.tight_layout(); _save(fig, path)


# ===========================================================================
# EXPERIMENT RUNNERS
# ===========================================================================

def run_exp1_first_player_advantage(F, T, LD):
    """
    Experiment 1 — First-Player Advantage
    ======================================
    Source: exp1_first_player.json

    Figures
    -------
    first_player_advantage
        Horizontal bar chart showing how much each variant's Seat-1 win rate
        deviates from the fair baseline (1/N).  Bars to the right (orange) signal
        a first-player advantage; bars to the left (blue) signal a disadvantage.

    positional_heatmap_all
        Green/red colour-coded grid (variants × seats).  Each cell shows the raw
        win-rate percentage; green cells are above fair, red cells below.  Gives a
        single at-a-glance view of positional bias across the full variant set.

    positional_grouped
        Grouped bar chart where each cluster of bars represents one seat position
        and each bar within the cluster is a variant.  Useful for comparing whether
        a particular seat is systematically advantaged across many variants.

    Tables
    ------
    positional_all_variants  (tab:positional_all)
        LaTeX table of raw seat win-rates (%) for every variant.

    chi2_positional  (tab:chi2_positional)
        Chi-square goodness-of-fit tests for positional uniformity.  A significant
        p-value (starred) indicates the variant has a statistically detectable
        positional bias.
    """
    _header('First-Player Advantage', 1, 16)
    d = LD('exp1_first_player.json')
    if d:
        NP = d.get('NP', 4); data = d['variants']
        fig_first_player_advantage(data, F('first_player_advantage'), NP)
        fig_positional_heatmap(data, F('positional_heatmap_all'), NP)
        fig_positional_grouped(data, F('positional_grouped'), NP)
        _save_tex(tex_positional_table(data, NP,
            'Seat win rates (\\%) across all 33 variants.', 'tab:positional_all'),
            T('positional_all_variants'))
        _save_tex(tex_chi_square(data,
            'Chi-square positional uniformity tests for all 33 variants.',
            'tab:chi2_positional'), T('chi2_positional'))


def run_exp2_dice_probability(F, T, LD):
    """
    Experiment 2 — Dice Probability Sweep
    ======================================
    Source: exp2_dice_usage.json  (+ exp7_fairness.json for bad-effect panel)

    Figures
    -------
    dice_prob_agent_winrate
        Line chart: each agent's win rate plotted against the probability of a
        favourable dice outcome (p from 0 to 1).  Reveals whether higher luck
        amplifies or suppresses agent-skill differences.

    positional_vs_dice_prob
        Line chart: each seat's win rate vs dice probability.  A crossing of the
        fair line (dashed) indicates that dice luck can reverse positional bias.

    dice_prob_fairness_length
        Dual-axis line chart.  Left axis: Gini coefficient (fairness); right axis:
        mean game length.  Shows the trade-off between luck intensity, fairness,
        and game duration simultaneously.

    bad_effect_comparison  (requires exp7_fairness.json)
        Side-by-side bar panels — one per bad-effect type — each showing the
        four seat win rates at p = 0.5.  Illustrates how the type of punishment
        (TAKE_CARDS, FORCED_PASS, REVEAL_HAND) changes the positional landscape.
    """
    _header('Dice Usage vs Win Rate', 2, 16)
    d = LD('exp2_dice_usage.json')
    if d:
        sweep = d['sweep']; NP = d.get('NP', 4)
        fig_dice_sweep_by_agent(sweep, F('dice_prob_agent_winrate'))
        fig_positional_vs_dice_prob(sweep, F('positional_vs_dice_prob'), NP)
        fig_dice_gini_length(sweep, F('dice_prob_fairness_length'))
        d7 = LD('exp7_fairness.json')
        if d7 and 'group_fairness' in d7:
            bad_grp = d7['group_fairness'].get('Bad Effect', {})
            if bad_grp:
                fig_bad_effect_comparison(bad_grp, F('bad_effect_comparison'), NP)


def run_exp3_luck_vs_skill(F, T, LD):
    """
    Experiment 3 — Luck vs Skill
    =============================
    Source: exp3_luck_skill.json  (+ exp13_parametric_sweep.json for group panel)

    Figures
    -------
    luck_skill_stacked
        100%-stacked bar chart (one bar per variant).  Each bar is divided into
        three coloured segments: agent skill (blue), seat position (orange), and
        residual/luck (grey).  Gives the variance-decomposition breakdown at a
        glance for every variant.

    Tables
    ------
    variance_decomp  (tab:anova)
        Three-column LaTeX table listing agent-skill %, seat %, and residual %
        for every variant — the numerical complement to the stacked bar chart.
    """
    _header('Luck vs Skill', 3, 16)
    d = LD('exp3_luck_skill.json')
    if d:
        all_vd = d['all_vd']
        fig_luck_skill_stacked(all_vd, F('luck_skill_stacked'))
        _save_tex(tex_anova(all_vd, 'Variance decomposition: agent skill vs seat position vs luck.', 'tab:anova'), T('variance_decomp'))


def run_exp4_comeback_snowball(F, T, LD):
    """
    Experiment 4 — Comeback / Snowball Dynamics
    ============================================
    Source: exp4_comeback.json  (+ exp13_parametric_sweep.json for round stability)

    Figures
    -------
    comeback_score_margins
        Box-and-whisker plot of the winning margin (winner score minus runner-up)
        per variant.  Wide distributions suggest frequent blowouts; tight
        distributions indicate close, competitive finishes.

    comeback_tight_games
        Bar chart of the percentage of games with a margin ≤ 1 ("tight games")
        for each variant.  Higher values mean more contested endgames.

    comeback_score_variance
        Bar chart of total score variance per variant.  High variance is a proxy
        for snowball effects — once a player leads, scores diverge.

    comeback_game_lengths
        Box plot of game length (turns) per variant.  Relevant because shorter
        games reduce comeback opportunities.

    sprint_vs_marathon  (if both Sprint and Marathon variants exist)
        Scatter plot comparing each agent's win rate in 3-round (Sprint) vs
        10-round (Marathon) configurations.  Agents above the y = x diagonal
        perform better over longer games, suggesting snowball synergy.

    round_stability  (requires exp13_parametric_sweep.json)
        Line chart of agent win rates as a function of round count.  Stable
        lines indicate consistent skill expression; diverging lines indicate
        snowball sensitivity.

    Tables
    ------
    score_margins  (tab:score_margins)
        LaTeX table with mean margin, median, tight-game %, and score variance
        for each variant — numerical summary of comeback potential.
    """
    _header('Comeback / Snowball', 4, 16)
    d = LD('exp4_comeback.json')
    if d:
        vdata   = d['variants']
        sm_data = {vn: vdata[vn]['sm'] for vn in vdata}
        gl_data = {vn: vdata[vn]['gl'] for vn in vdata}
        ag_data = {vn: vdata[vn]['ag'] for vn in vdata}
        fig_score_margins(sm_data, F('comeback_score_margins'))
        fig_tight_games(sm_data,   F('comeback_tight_games'))
        fig_score_variance(sm_data, F('comeback_score_variance'))
        fig_game_lengths(gl_data,  F('comeback_game_lengths'))
        if 'Sprint' in ag_data and 'Marathon' in ag_data:
            fig_short_vs_long(ag_data['Sprint'], ag_data['Marathon'],
                               F('sprint_vs_marathon'))
        d13 = LD('exp13_parametric_sweep.json')
        if d13 and 'Round Count' in d13['groups']:
            rc = d13['groups']['Round Count']
            fig_round_stability({vn: rc[vn]['ag'] for vn in rc}, F('round_stability'))
        _save_tex(tex_score_margins(sm_data,
            'Score margin statistics by variant.', 'tab:score_margins'),
            T('score_margins'))


def run_exp5_agent_winrates(F, T, LD):
    """
    Experiment 5 — Heuristic Agent Analysis
    =========================================
    Source: exp5_agent_winrates.json

    Figures
    -------
    agent_winrates_<variant>  (one per variant)
        Bar chart of each agent's win rate with 95% Wilson confidence-interval
        error bars for the named variant.  The dashed horizontal line marks the
        fair expected rate (1/N).

    agent_radar_<variant>  (one per variant)
        Polar radar chart showing four normalised metrics per agent: win rate,
        mean score, score stability (inverse std), and median score.  Useful for
        comparing multi-dimensional agent profiles at a glance.

    h2h_baseline
        Colour-coded NxN heatmap of pairwise head-to-head win rates on the
        Baseline variant.  Each cell (row i, col j) gives the fraction of games
        that agent i won against agent j.  Cell colours range from red (0%) to
        green (100%), with white near 50%.

    agent_across_variants
        Grouped bar chart with one cluster per variant and one bar per agent.
        Makes it easy to see whether agent rankings are consistent across rule
        configurations or flip between variants.

    Tables
    ------
    agent_winrates_<variant>  (tab:wr_<variant>) — one per variant
        Win rate, 95% CI, mean score, and std score for each agent in that
        variant.

    h2h_baseline  (tab:h2h_baseline)
        Full pairwise win-rate matrix in LaTeX tabular form for the Baseline
        variant.

    ttest_baseline  (tab:ttest)  (requires SciPy)
        Two-sample t-tests and Cohen's d for all agent-pair score comparisons on
        the Baseline variant.  Significant pairs are starred.
    """
    _header('Agent Win Rates', 5, 16)
    d = LD('exp5_agent_winrates.json')
    if d:
        variants = d['variants']
        for vn, vd in variants.items():
            safe = vn.replace(' ', '_').replace("'", '').replace('&', 'and')
            fig_agent_winrates(vd['ag'], F(f'agent_winrates_{safe}'),
                               title=f'Agent Win Rates — {vn}')
            fig_agent_radar(vd['ag'], F(f'agent_radar_{safe}'))
            _save_tex(tex_agent_winrates(vd['ag'],
                f'Agent win rates — {vn}.', f'tab:wr_{safe}'),
                T(f'agent_winrates_{safe}'))
        if 'Baseline' in variants and 'h2h' in variants['Baseline']:
            fig_heatmap(variants['Baseline']['h2h'], F('h2h_baseline'))
            _save_tex(tex_headtohead(variants['Baseline']['h2h'],
                'Head-to-head win rates (\\%) — Baseline variant.',
                'tab:h2h_baseline'), T('h2h_baseline'))
        fig_agent_across_variants(
            {vn: variants[vn]['ag'] for vn in variants}, F('agent_across_variants'))
        if 'Baseline' in variants and HAS_SCIPY:
            ag_bl = variants['Baseline']['ag']
            names_s = sorted(ag_bl.keys()); t_rows = []
            for i, a1 in enumerate(names_s):
                for a2 in names_s[i+1:]:
                    n1,n2 = ag_bl[a1]['total'], ag_bl[a2]['total']
                    m1,m2 = ag_bl[a1]['mean_score'], ag_bl[a2]['mean_score']
                    s1,s2 = ag_bl[a1]['std_score'],  ag_bl[a2]['std_score']
                    if n1>1 and n2>1:
                        se  = np.sqrt(s1**2/n1 + s2**2/n2)
                        t   = (m1-m2)/se if se else 0.0
                        p   = float(2*scipy_stats.t.sf(abs(t), n1+n2-2))
                        d_v = (m1-m2)/np.sqrt((s1**2+s2**2)/2) if (s1**2+s2**2) else 0.0
                        ps  = '< 0.001' if p < 0.001 else f'{p:.4f}'
                        sig = r'$^*$' if p < 0.05 else ''
                        t_rows.append(
                            f'  {a1} vs {a2} & {t:.2f} & {ps}{sig} & {d_v:.3f} \\\\')
            if t_rows:
                _save_tex(tex_ttest_table(t_rows,
                    'Two-sample $t$-tests comparing mean scores (Baseline).',
                    'tab:ttest'), T('ttest_baseline'))


def run_exp6_number_of_players(F, T, LD):
    """
    Experiment 6 — Number of Players
    ==================================
    Source: exp6_nplayers.json

    Figures
    -------
    player_count_positional
        Side-by-side bar panels — one panel per player count (2P, 3P, 4P …) —
        each showing seat win rates with confidence intervals on the Baseline
        variant.  The dashed line marks the per-count fair rate (1/N).

    player_count_lengths
        Box plots of game lengths grouped by player count.  Illustrates how
        adding players changes expected game duration and variance.

    player_count_fairness
        Dual-axis line chart: Gini coefficient (left axis, blue) and maximum
        seat deviation (right axis, orange) both plotted against player count.
        Reveals whether positional bias grows or shrinks as the game scales.

    Tables
    ------
    player_count  (tab:player_count)
        Seat win rates (%), Gini, and chi-square p-value for each player count
        on the Baseline variant.

    nplayer_game_lengths  (tab:nplayer_gl)
        Mean, median, std, min, and max game length for every (player-count ×
        variant) combination.
    """
    _header('Number of Players', 6, 16)
    d = LD('exp6_nplayers.json')
    if d:
        ndata   = d['nplayer_data']
        base_pc = {npl: ndata[npl]['Baseline']
                   for npl in ndata if 'Baseline' in ndata[npl]}
        if base_pc:
            fig_player_count_positional(base_pc, F('player_count_positional'))
            fig_player_count_lengths(base_pc,    F('player_count_lengths'))
            fig_player_count_fairness(base_pc,   F('player_count_fairness'))
            _save_tex(tex_player_count(base_pc,
                'Positional win rates by player count (Baseline variant).',
                'tab:player_count'), T('player_count'))
        all_gl = {f'{npl}P-{vn}': ndata[npl][vn]['gl']
                  for npl in ndata for vn in ndata[npl]}
        _save_tex(tex_game_lengths(all_gl,
            'Game length by player count and variant.', 'tab:nplayer_gl'),
            T('nplayer_game_lengths'))


def run_exp7_variant_fairness(F, T, LD):
    """
    Experiment 7 — Variant Fairness
    =================================
    Source: exp7_fairness.json

    Figures
    -------
    fairness_scatter
        Scatter plot: Gini coefficient (x) vs maximum seat deviation % (y), one
        point per variant.  Orange/red points are statistically biased (p < 0.05);
        blue points are not.  Both axes measure unfairness, so variants should
        cluster near the origin for ideal fairness.

    fairness_ranking
        Horizontal bar chart ranking every variant by Gini coefficient.  Bars are
        colour-coded: green (Gini < 0.03, fair), orange (0.03–0.07, moderate), red
        (> 0.07, unfair).

    game_length_distributions
        Box plots of game lengths for every variant — the fairness experiment's
        companion to the comeback analysis in Exp 4.

    score_variance_all
        Bar chart of score variance across all variants, providing an aggregate
        snowball overview for the full 33-variant set.

    group_fairness_summary  (if group_fairness present)
        Bar chart of mean Gini (± std) per rule-family group, showing which
        parameter dimensions introduce the most positional bias on average.

    group_game_lengths  (if group_fairness present)
        Faceted bar chart of mean game length (± std) per variant within each
        rule-family group.

    fairness_heatmap_<group>  (one per group: Dice Probability, Round Count,
                                Scoring Mode, Bad Effect)
        Positional win-rate heatmap (same format as Exp 1) restricted to a
        single parameter-family sweep, making it easy to trace how one rule
        dimension affects seat bias as its value varies.

    Tables
    ------
    fairness_ranking  (tab:fairness_ranking)
        Variants ranked by Gini, including max deviation, chi-square statistic,
        and p-value.  This is the primary summary table for the fairness chapter.

    game_lengths  (tab:game_lengths)
        Mean, median, std, min, and max game length for all 33 variants.
    """
    _header('Variant Fairness', 7, 16)
    d = LD('exp7_fairness.json')
    if d:
        fairness = d['fairness']
        fig_fairness_scatter(fairness,      F('fairness_scatter'))
        fig_fairness_ranking_bar(fairness,  F('fairness_ranking'))
        gl_data = {vn: fairness[vn]['gl'] for vn in fairness}
        fig_game_lengths(gl_data,           F('game_length_distributions'))
        fig_score_variance({vn: fairness[vn]['sm'] for vn in fairness},
                            F('score_variance_all'))
        _save_tex(tex_fairness_ranking(fairness,
            'Positional fairness ranking — all 3 variants.', 'tab:fairness_ranking'),
            T('fairness_ranking'))
        _save_tex(tex_game_lengths(gl_data,
            'Game length statistics — all 33 variants.', 'tab:game_lengths'),
            T('game_lengths'))
        if 'group_fairness' in d:
            gf = d['group_fairness']
            fig_group_fairness_summary(gf, F('group_fairness_summary'))
            fig_group_game_lengths(gf,     F('group_game_lengths'))
            for gname in ['Dice Probability', 'Round Count', 'Scoring Mode', 'Bad Effect']:
                if gname in gf:
                    fig_group_heatmap(gf[gname], gname,
                                      F(f'fairness_heatmap_{gname.replace(" ","_")}'))


def run_exp8_diagnostics_and_information(F, T, LD, root):
    """
    Experiment 8 — System Diagnostics + Information Reveal
    ========================================================
    Sources: training_log.json (RL),  exp8_information.json,  exp7_fairness.json

    Figures
    -------
    rl_learning_curve  (requires training_log.json)
        Line plot of the RL agent's win rate vs heuristics over training
        episodes.  The dashed 25% line marks the fair random baseline.  Upward
        trend confirms the agent is learning; flat or declining curves indicate
        a training bug.

    info_sweep_winrate  (requires exp8_information.json)
        Two-panel figure: left panel shows each agent's win rate vs the
        probability that hand cards are revealed; right panel shows mean game
        length vs the same probability.  Quantifies how information transparency
        shifts power towards more strategic agents.

    score_variance_all_variants  (requires exp7_fairness.json)
        Bar chart of score variance across all 33 named variants — the same
        chart as in Exp 7 but reproduced here in the diagnostics context.

    Tables
    ------
    info_variants  (tab:info_variants)
        Agent win rates (%) on information-reveal variants — the tabular
        companion to the info-sweep line chart.
    """
    _header('Diagnostics + Information Reveal', 8, 16)
    for cand in ['rl_agent/models/training_log.json',
                 'models/training_log.json', 'simulation/training_log.json']:
        full = os.path.join(root, cand)
        if os.path.exists(full):
            fig_rl_learning_curve(full, F('rl_learning_curve')); break
    d8 = LD('exp8_information.json')
    if d8:
        fig_info_sweep(d8.get('info_sweep', {}), F('info_sweep_winrate'))
        _save_tex(tex_information_analysis(
            d8.get('info_variants', {}),
            'Agent win rates on information-reveal variants.',
            'tab:info_variants'), T('info_variants'))
    d7 = LD('exp7_fairness.json')
    if d7:
        fig_score_variance({vn: d7['fairness'][vn]['sm']
                            for vn in d7['fairness']},
                            F('score_variance_all_variants'))


def run_exp9_double_play(F, T, LD):
    """
    Experiment 9 — Double Play Deep Dive
    ======================================
    Source: exp9_double_play.json

    Figures
    -------
    dp_prob_sweep
        Two-panel figure: left panel traces each agent's win rate as the
        probability of a double-play event increases; right panel shows mean
        game length vs the same probability.  Together they reveal whether
        double play accelerates games and whether it advantages particular
        strategies.

    dp_penalty_sweep
        Grouped bar chart comparing agent win rates across different pass-penalty
        magnitudes for the Double Play variant.  Shows the sensitivity of agent
        rankings to the cost of voluntarily passing.

    dp_variant_winrates
        Grouped bar chart of agent win rates across all Double Play sub-variants
        (the full variant-comparison counterpart to the probability sweep).
    """
    _header('Double Play Deep Dive', 9, 16)
    d = LD('exp9_double_play.json')
    if d:
        sw = d.get('dp_prob_sweep', {})
        if sw:
            probs  = sorted(sw.keys())
            pvals  = [float(k.split('=')[1]) for k in probs]
            agents = sorted(sw[probs[0]]['ag'].keys())
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for nm in agents:
                wrs = [sw[k]['ag'].get(nm,{}).get('win_rate',0)*100 for k in probs]
                axes[0].plot(pvals, wrs, 'o-', label=nm, color=_agent_colour(nm), lw=1.8)
            axes[0].axhline(25, color='gray', ls='--', lw=0.7)
            axes[0].set_xlabel('p(Double Play)'); axes[0].set_ylabel('Win Rate (%)')
            axes[0].set_title('Win Rate vs Double Play Probability'); axes[0].legend()
            gl = [sw[k]['gl']['mean'] for k in probs]
            axes[1].plot(pvals, gl, 's-', color='#D55E00', lw=1.8)
            axes[1].set_xlabel('p(Double Play)'); axes[1].set_ylabel('Mean Turns')
            axes[1].set_title('Game Length vs Double Play Probability')
            fig.tight_layout(); _save(fig, F('dp_prob_sweep'))
        pen_sw = d.get('dp_penalty_sweep', {})
        if pen_sw:
            pens = sorted(pen_sw.keys(), key=lambda k: int(k.split('=')[1]))
            agents = sorted(pen_sw[pens[0]].keys()) if pens else []
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(pens)); w = 0.8 / max(len(agents), 1)
            for i, nm in enumerate(agents):
                wrs = [pen_sw[k].get(nm,{}).get('win_rate',0)*100 for k in pens]
                ax.bar(x+(i-len(agents)/2)*w, wrs, w, label=nm,
                       color=_agent_colour(nm), alpha=0.85)
            ax.axhline(25, color='black', ls='--', lw=0.7)
            ax.set_xticks(x); ax.set_xticklabels(pens)
            ax.set_ylabel('Win Rate (%)'); ax.set_title('Win Rate vs Pass Penalty (DP)')
            ax.legend(); fig.tight_layout(); _save(fig, F('dp_penalty_sweep'))
        dp_v = d.get('dp_variants', {})
        if dp_v:
            fig_agent_across_variants(
                {vn: dp_v[vn]['ag'] for vn in dp_v}, F('dp_variant_winrates'))


def run_exp10_forced_pass(F, T, LD):
    """
    Experiment 10 — Forced Pass Dynamics
    ======================================
    Source: exp10_forced_pass.json

    Figures
    -------
    fp_prob_sweep
        Two-panel figure.  Left panel: each agent's win rate vs p(Forced Pass).
        Right panel: mean game length (left y-axis, red) and Gini coefficient
        (right y-axis, green) vs the same probability.  A rising Gini as forced
        pass increases would indicate that the mechanic introduces positional
        bias.

    fp_comparison
        Grouped bar chart of agent win rates across all Forced Pass variant
        configurations (None, Low, Medium, High probability).
    """
    _header('Forced Pass Dynamics', 10, 16)
    d = LD('exp10_forced_pass.json')
    if d:
        fp_sw = d.get('fp_prob_sweep', {})
        if fp_sw:
            probs  = sorted(fp_sw.keys())
            pvals  = [float(k.split('=')[1]) for k in probs]
            agents = sorted(fp_sw[probs[0]]['ag'].keys())
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for nm in agents:
                wrs = [fp_sw[k]['ag'].get(nm,{}).get('win_rate',0)*100 for k in probs]
                axes[0].plot(pvals, wrs, 'o-', label=nm, color=_agent_colour(nm), lw=1.8)
            axes[0].axhline(25, color='gray', ls='--', lw=0.7)
            axes[0].set_xlabel('p(Forced Pass)'); axes[0].set_ylabel('Win Rate (%)')
            axes[0].set_title('Win Rate vs Forced Pass Probability'); axes[0].legend()
            ginis = [fp_sw[k]['pos']['gini'] for k in probs]
            gl_m  = [fp_sw[k]['gl']['mean']  for k in probs]
            axes[1].plot(pvals, gl_m, 's-', color='#D55E00', lw=1.8, label='Mean turns')
            ax2 = axes[1].twinx()
            ax2.plot(pvals, ginis, 'D--', color='#009E73', lw=1.5, label='Gini')
            axes[1].set_xlabel('p(Forced Pass)')
            axes[1].set_ylabel('Mean Turns', color='#D55E00')
            ax2.set_ylabel('Gini', color='#009E73')
            axes[1].set_title('Dynamics vs Forced Pass Probability')
            axes[1].legend(loc='upper left'); ax2.legend(loc='upper right')
            fig.tight_layout(); _save(fig, F('fp_prob_sweep'))
        comp = d.get('comparison', {})
        if comp:
            fig_agent_across_variants(
                {vn: comp[vn]['ag'] for vn in comp}, F('fp_comparison'))


def run_exp11_knowledge_advantage(F, T, LD):
    """
    Experiment 11 — Knowledge Advantage
    =====================================
    Source: exp11_knowledge_advantage.json

    Figures
    -------
    knowledge_advantage
        Horizontal bar chart of Cohen's d for each agent, measuring the score
        gain obtained when playing in an information-reveal variant versus a
        non-information variant.  Reference lines at d = 0.2 (small effect) and
        d = 0.5 (medium effect) help contextualise the magnitude.  Agents with
        higher d benefit more from knowing opponents' cards, implying their
        strategy is more information-sensitive.
    """
    _header('Knowledge Advantage', 11, 16)
    d = LD('exp11_knowledge_advantage.json')
    if d:
        fig_knowledge_advantage(d.get('cohens_d_info_vs_light', {}),
                                 F('knowledge_advantage'))


def run_exp12_all_variant_tournament(F, T, LD):
    """
    Experiment 12 — All-Variant Tournament
    ========================================
    Source: exp12_all_variant_tournament.json

    Figures
    -------
    tournament_heatmap
        Large heatmap (variants × agents).  Each cell shows the agent's win
        rate (%) in that variant.  Green cells indicate above-fair performance;
        red cells below-fair.  Provides a full cross-tabulation of agent ×
        variant outcomes in a single image.

    tournament_ranking
        Bar chart of each agent's mean win rate averaged across all 33 variants,
        ranked highest to lowest.  The definitive overall agent-ranking figure
        for the dissertation.

    Tables
    ------
    tournament_rankings  (tab:tournament)
        Full win-rate matrix in LaTeX tabular form — every agent's win rate for
        every variant, sorted alphabetically by variant name.
    """
    _header('All-Variant Tournament', 12, 16)
    d = LD('exp12_all_variant_tournament.json')
    if d:
        rankings    = d['rankings']
        agent_names = d.get('agent_names', ['Random','Aggressive','Defensive','Balanced'])
        fig_tournament_heatmap(rankings, agent_names, F('tournament_heatmap'))
        mean_wrs = {nm: np.mean([rankings[vn].get(nm,0) for vn in rankings])
                    for nm in agent_names}
        sorted_a = sorted(mean_wrs, key=lambda a: -mean_wrs[a])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(sorted_a, [mean_wrs[a]*100 for a in sorted_a],
               color=[_agent_colour(a) for a in sorted_a], alpha=0.85)
        ax.axhline(100/len(agent_names), color='black', ls='--', lw=0.7,
           label=f'Fair ({100/len(agent_names):.1f}%)')
        ax.set_ylabel('Mean Win Rate Across All Variants (%)')
        ax.set_title(f'Overall Agent Ranking — {len(rankings)}-Variant Tournament')
        ax.legend(); fig.tight_layout(); _save(fig, F('tournament_ranking'))
        hdr  = ' & '.join(f'\\textbf{{{nm}}}' for nm in agent_names)
        body = [f'\\begin{{tabular}}{{l{"c"*len(agent_names)}}}', r'\toprule',
                f'\\textbf{{Variant}} & {hdr} \\\\', r'\midrule']
        for vn in sorted(rankings):
            cells = ' & '.join(f"{rankings[vn].get(nm,0)*100:.1f}" for nm in agent_names)
            body.append(f'  {vn} & {cells} \\\\')
        body += [r'\bottomrule', r'\end{tabular}']
        _save_tex(_tex_wrap(body,
            'Agent win rates (\\%) across all 33 variants.', 'tab:tournament'),
            T('tournament_rankings'))


def run_exp13_parametric_sweep(F, T, LD):
    """
    Experiment 13 — Parametric Sweep Analysis
    ===========================================
    Source: exp13_parametric_sweep.json

    Figures
    -------
    parametric_group_overview
        Faceted 2-column panel.  Each subplot covers one parameter group (Dice
        Probability, Round Count, Scoring Mode, Bad Effect …) and plots
        Balanced, Aggressive, and Defensive agent win rates across that group's
        configurations.  The 25% reference line marks the fair baseline.

    parametric_heatmap_Dice_Probability  \
    parametric_heatmap_Round_Count        >  one per group (if present)
    parametric_heatmap_Scoring_Mode      /
        Positional win-rate heatmaps restricted to a single parameter family,
        identical in format to the group heatmaps in Exp 7 but driven
        specifically by the parametric sweep data.
    """
    _header('Parametric Sweep Analysis', 13, 16)
    d = LD('exp13_parametric_sweep.json')
    if d:
        groups = d['groups']
        fig_parametric_group_overview(groups, F('parametric_group_overview'))
        for gname in ['Dice Probability', 'Round Count', 'Scoring Mode']:
            if gname in groups:
                safe_g = gname.replace(' ', '_')
                fig_group_heatmap(groups[gname], gname,
                                  F(f'parametric_heatmap_{safe_g}'))


def run_exp14_end_condition(F, T, LD):
    """
    Experiment 14 — Target Score vs Fixed Rounds
    ==============================================
    Source: exp14_end_condition.json

    Figures
    -------
    end_condition_lengths
        Violin plot comparing game-length distributions for points-based (target
        score) vs rounds-based (fixed round count) end conditions.  The median
        line inside each violin highlights the typical game length; the body
        width shows the density.  Helps answer whether an open-ended scoring
        game naturally converges to a similar length as a capped-round game.
    """
    _header('Target Score vs Fixed Rounds', 14, 16)
    d = LD('exp14_end_condition.json')
    if d:
        pts_d   = d.get('pts_based', {})
        round_d = d.get('round_based', {})
        pts_l   = [l for v in pts_d.values()   for l in v['gl'].get('lengths', [])]
        round_l = [l for v in round_d.values() for l in v['gl'].get('lengths', [])]
        if pts_l and round_l:
            fig, ax = plt.subplots(figsize=(6, 5))
            parts = ax.violinplot([pts_l, round_l], showmedians=True)
            for pc, c in zip(parts['bodies'], ['#CC79A7', '#56B4E9']):
                pc.set_facecolor(c); pc.set_alpha(0.7)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Points-based', 'Rounds-based'])
            ax.set_ylabel('Number of Turns')
            ax.set_title('Game Length: Target Score vs Fixed Rounds')
            fig.tight_layout(); _save(fig, F('end_condition_lengths'))


def run_exp15_transfer_effect(F, T, LD):
    """
    Experiment 15 — Transfer Effect
    =================================
    Source: exp15_transfer.json

    Figures
    -------
    transfer_variant_winrates
        Grouped bar chart of agent win rates across all Transfer-Effect
        sub-variants (No Transfer, TAKE_CARDS, FORCED_PASS, REVEAL_HAND at
        various probabilities).  Establishes baseline win-rate context before
        the probability sweep.

    transfer_prob_sweep
        Line chart with one line per bad-effect type (TAKE_CARDS, FORCED_PASS,
        REVEAL_HAND) plotting mean game length as a function of the transfer or
        give-card probability.  Reveals which mechanism most dramatically
        extends or shortens games as its probability rises.
    """
    _header('Transfer Effect', 15, 16)
    d = LD('exp15_transfer.json')
    if d:
        tv = d.get('transfer_variants', {})
        ts = d.get('transfer_sweep', {})
        if tv:
            fig_agent_across_variants(
                {vn: tv[vn]['ag'] for vn in tv}, F('transfer_variant_winrates'))
        if ts:
            fig, ax = plt.subplots(figsize=(8, 5))
            for bt, label in [('take','TAKE\\_CARDS'),('fp','FORCED\\_PASS'),('rh','REVEAL\\_HAND')]:
                keys = sorted([k for k in ts if k.endswith('_'+bt)],
                              key=lambda k: float(k.split('=')[1].split('_')[0]))
                if not keys: continue
                p_vals = [float(k.split('=')[1].split('_')[0]) for k in keys]
                gl_m   = [ts[k]['gl']['mean'] for k in keys]
                ax.plot(p_vals, gl_m, 'o-', label=label, lw=1.8)
            ax.set_xlabel('p(Transfer/Give-Card)'); ax.set_ylabel('Mean Turns')
            ax.set_title('Game Length vs Transfer Probability by Bad Effect Type')
            ax.legend(); fig.tight_layout(); _save(fig, F('transfer_prob_sweep'))


def run_exp16_mcts_benchmark(F, T, LD):
    """
    Experiment 16 — MCTS Benchmark
    ================================
    Source: exp16_mcts_benchmark.json

    Figures
    -------
    mcts_h2h_heatmap
        Colour-coded heatmap (MCTS variants × benchmark variants).  Each cell
        shows the MCTS variant's win rate in that rule configuration.  Enables
        direct comparison of MCTS-SuperFast / Fast / Standard / Deep across
        different game contexts.

    mcts_vs_heuristics_winrate
        Grouped bar chart.  One cluster per MCTS variant; bars within the
        cluster correspond to different benchmark variants.  95% CI error bars
        are included.  Reveals whether a specific variant configuration gives
        heuristics an edge over MCTS.

    mcts_mean_winrate_vs_heuristics
        Horizontal bar chart ranking MCTS variants by their mean win rate
        against heuristic opponents, averaged across all benchmark variants.
        The 25% dashed line marks the fair baseline.

    mcts_time_per_move
        Bar chart of mean computation time per move (ms) for each MCTS variant,
        with ± std error bars and p95 latency shown as red diamonds.
        Quantifies the computational cost of increasing the iteration budget.

    mcts_tradeoff_scatter
        Scatter plot: mean time per move (x) vs mean win rate vs heuristics (y),
        one labelled point per MCTS variant.  Identifies the "efficiency
        frontier" — the variant that maximises win rate per millisecond.

    Tables
    ------
    mcts_benchmark  (tab:mcts_benchmark)
        Four-column LaTeX table: MCTS variant name, iteration budget, time
        limit, mean move time (ms), and mean win rate vs heuristics (%).
        Ready to drop into the dissertation's MCTS evaluation chapter.
    """
    _header('MCTS Benchmark — Win Rate & Time per Move', 16, 16)
    d = LD('exp16_mcts_benchmark.json')
    if d:
        mcts_names   = d.get('mcts_variants', [])
        bench_vars   = d.get('bench_variants', [])
        h2h_data     = d.get('head_to_head', {})
        vs_h_data    = d.get('vs_heuristics', {})
        timing_sum   = d.get('timing_summary', {})
        mean_wr_vs_h = d.get('mean_wr_vs_heuristics', {})
        heuristic_names = ['Aggressive', 'Defensive', 'Balanced']

        # ── 16a. Head-to-head win rates: heatmap (MCTS × variant) ────────
        if h2h_data and mcts_names:
            wr_matrix = np.zeros((len(mcts_names), len(bench_vars)))
            for vi, vn in enumerate(bench_vars):
                ag = h2h_data.get(vn, {}).get('ag', {})
                for mi, mn in enumerate(mcts_names):
                    wr_matrix[mi, vi] = ag.get(mn, {}).get('win_rate', 0.0) * 100

            fig, ax = plt.subplots(figsize=(8, 4))
            im = ax.imshow(wr_matrix, aspect='auto', cmap='RdYlGn',
                           vmin=0, vmax=50)
            ax.set_xticks(range(len(bench_vars)))
            ax.set_xticklabels([v.replace(' ', '\n') for v in bench_vars],
                               fontsize=9)
            ax.set_yticks(range(len(mcts_names)))
            ax.set_yticklabels(mcts_names)
            for mi in range(len(mcts_names)):
                for vi in range(len(bench_vars)):
                    ax.text(vi, mi, f'{wr_matrix[mi, vi]:.1f}',
                            ha='center', va='center', fontsize=8,
                            color='black')
            fig.colorbar(im, ax=ax, label='Win Rate (%)')
            ax.set_title('MCTS Head-to-Head: Win Rate by Variant (%)')
            ax.set_xlabel('Variant'); ax.set_ylabel('MCTS Variant')
            fig.tight_layout(); _save(fig, F('mcts_h2h_heatmap'))

        # ── 16b. MCTS vs heuristics: grouped bar chart ────────────────
        if vs_h_data and mcts_names:
            fig, ax = plt.subplots(figsize=(9, 5))
            x = np.arange(len(mcts_names))
            w = 0.2
            offsets = np.linspace(-(len(bench_vars)-1)/2*w,
                                   (len(bench_vars)-1)/2*w, len(bench_vars))
            cmap = _get_cmap('tab10', len(bench_vars))
            for bi, vn in enumerate(bench_vars):
                wrs = []
                for mn in mcts_names:
                    ag = vs_h_data.get(mn, {}).get(vn, {}).get('ag', {})
                    wr = ag.get(mn, {}).get('win_rate', 0.0) * 100
                    ci_lo = ag.get(mn, {}).get('ci_low', 0.0) * 100
                    ci_hi = ag.get(mn, {}).get('ci_high', 0.0) * 100
                    wrs.append((wr, wr - ci_lo, ci_hi - wr))
                bars = ax.bar(x + offsets[bi], [v[0] for v in wrs], w,
                              label=vn, color=cmap(bi), alpha=0.82)
                ax.errorbar(x + offsets[bi], [v[0] for v in wrs],
                            yerr=[[v[1] for v in wrs], [v[2] for v in wrs]],
                            fmt='none', color='black', capsize=2, lw=0.8)
            ax.axhline(25, color='black', ls='--', lw=0.8, label='Fair (25%)')
            ax.set_xticks(x)
            ax.set_xticklabels(mcts_names, rotation=15, ha='right')
            ax.set_ylabel('Win Rate vs Heuristics (%)')
            ax.set_title('MCTS Win Rate vs Heuristic Opponents by Variant')
            ax.legend(title='Variant', fontsize=8, title_fontsize=8,
                      loc='upper left', ncol=2)
            fig.tight_layout(); _save(fig, F('mcts_vs_heuristics_winrate'))

        # ── 16c. Mean win rate vs heuristics — single bar ────────────
        if mean_wr_vs_h and mcts_names:
            fig, ax = plt.subplots(figsize=(6, 4))
            sorted_m = sorted(mcts_names,
                              key=lambda n: mean_wr_vs_h.get(n, 0))
            vals  = [mean_wr_vs_h.get(n, 0) * 100 for n in sorted_m]
            bars  = ax.barh(sorted_m, vals,
                            color=[_agent_colour(n) for n in sorted_m],
                            alpha=0.85, edgecolor='black', lw=0.5)
            ax.axvline(25, color='black', ls='--', lw=0.8, label='Fair (25%)')
            for bar, v in zip(bars, vals):
                ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                        f'{v:.1f}%', va='center', fontsize=9)
            ax.set_xlabel('Mean Win Rate vs Heuristics (%)')
            ax.set_title('Overall MCTS Performance vs Heuristic Opponents')
            ax.legend(); fig.tight_layout()
            _save(fig, F('mcts_mean_winrate_vs_heuristics'))

        # ── 16d. Time per move — bar chart ────────────────────────────
        if timing_sum and mcts_names:
            present = [mn for mn in mcts_names if mn in timing_sum]
            means   = [timing_sum[mn]['mean_ms'] for mn in present]
            p95s    = [timing_sum[mn]['p95_ms']  for mn in present]
            stds    = [timing_sum[mn]['std_ms']  for mn in present]

            fig, ax = plt.subplots(figsize=(7, 4.5))
            x = np.arange(len(present))
            ax.bar(x, means, color=[_agent_colour(n) for n in present],
                   alpha=0.85, edgecolor='black', lw=0.5, label='Mean')
            ax.errorbar(x, means, yerr=stds, fmt='none',
                        color='black', capsize=4, lw=1.0)
            ax.scatter(x, p95s, marker='D', color='crimson', zorder=5,
                       s=40, label='p95')
            for xi, (m, p) in enumerate(zip(means, p95s)):
                ax.text(xi, m + stds[xi] + 0.5, f'{m:.1f}',
                        ha='center', va='bottom', fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(present, rotation=15, ha='right')
            ax.set_ylabel('Time per Move (ms)')
            ax.set_title('MCTS Computation Time per Move\n(mean ± std, p95 shown as ◆)')
            ax.legend()
            fig.tight_layout(); _save(fig, F('mcts_time_per_move'))

        # ── 16e. Win-rate vs time trade-off scatter ───────────────────
        if timing_sum and mean_wr_vs_h and mcts_names:
            xs = [timing_sum.get(mn, {}).get('mean_ms', 0) for mn in mcts_names]
            ys = [mean_wr_vs_h.get(mn, 0) * 100              for mn in mcts_names]
            fig, ax = plt.subplots(figsize=(6, 5))
            for mn, xi, yi in zip(mcts_names, xs, ys):
                ax.scatter(xi, yi, color=_agent_colour(mn), s=120,
                           zorder=5, edgecolors='black', lw=0.5)
                ax.annotate(mn, (xi, yi),
                            textcoords='offset points', xytext=(6, 3),
                            fontsize=8)
            ax.axhline(25, color='black', ls='--', lw=0.7, label='Fair (25%)')
            ax.set_xlabel('Mean Time per Move (ms)')
            ax.set_ylabel('Mean Win Rate vs Heuristics (%)')
            ax.set_title('MCTS Trade-off: Computation Time vs Win Rate')
            ax.legend()
            fig.tight_layout(); _save(fig, F('mcts_tradeoff_scatter'))

        # ── 16f. LaTeX summary table ──────────────────────────────────
        iter_map   = {'MCTS-SuperFast': 100,  'MCTS-Fast': 500,
                      'MCTS-Standard': 1000,  'MCTS-Deep': 2000}
        tlimit_map = {'MCTS-SuperFast': 0.10, 'MCTS-Fast': 0.20,
                      'MCTS-Standard': 0.80,  'MCTS-Deep': 2.00}
        rows_tex = [
            r'\begin{tabular}{lrrrr}', r'\toprule',
            (r'\textbf{Variant} & \textbf{Iterations} & \textbf{Time Limit (s)}'
             r' & \textbf{Mean Move (ms)} & \textbf{Win Rate vs Heur. (\%)} \\'),
            r'\midrule',
        ]
        for mn in mcts_names:
            it = iter_map.get(mn, '–')
            tl = tlimit_map.get(mn, '–')
            mm = f"{timing_sum.get(mn, {}).get('mean_ms', 0.0):.1f}"
            wr = f"{mean_wr_vs_h.get(mn, 0.0)*100:.1f}"
            rows_tex.append(f'  {mn} & {it} & {tl:.2f} & {mm} & {wr} \\\\')
        rows_tex += [r'\bottomrule', r'\end{tabular}']
        caption = ('MCTS variant comparison: configuration parameters, mean '
                   'wall-clock time per \\texttt{choose\\_move()} call, and mean '
                   'win rate against heuristic opponents across four representative variants.')
        _save_tex(_tex_wrap(rows_tex, caption, 'tab:mcts_benchmark'),
                  T('mcts_benchmark'))


# ===========================================================================
# MAIN
# ===========================================================================

# Maps experiment number → runner function.  Functions that need `root` receive
# it via a lambda so the dispatch table stays uniform.
_EXPERIMENT_RUNNERS = {
    1:  lambda F, T, LD, root: run_exp1_first_player_advantage(F, T, LD),
    2:  lambda F, T, LD, root: run_exp2_dice_probability(F, T, LD),
    3:  lambda F, T, LD, root: run_exp3_luck_vs_skill(F, T, LD),
    4:  lambda F, T, LD, root: run_exp4_comeback_snowball(F, T, LD),
    5:  lambda F, T, LD, root: run_exp5_agent_winrates(F, T, LD),
    6:  lambda F, T, LD, root: run_exp6_number_of_players(F, T, LD),
    7:  lambda F, T, LD, root: run_exp7_variant_fairness(F, T, LD),
    8:  lambda F, T, LD, root: run_exp8_diagnostics_and_information(F, T, LD, root),
    9:  lambda F, T, LD, root: run_exp9_double_play(F, T, LD),
    10: lambda F, T, LD, root: run_exp10_forced_pass(F, T, LD),
    11: lambda F, T, LD, root: run_exp11_knowledge_advantage(F, T, LD),
    12: lambda F, T, LD, root: run_exp12_all_variant_tournament(F, T, LD),
    13: lambda F, T, LD, root: run_exp13_parametric_sweep(F, T, LD),
    14: lambda F, T, LD, root: run_exp14_end_condition(F, T, LD),
    15: lambda F, T, LD, root: run_exp15_transfer_effect(F, T, LD),
    16: lambda F, T, LD, root: run_exp16_mcts_benchmark(F, T, LD),
}


def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description='Cinquillo 2.0 — Visualisation')
    parser.add_argument('--exp',    nargs='+', type=int)
    parser.add_argument('--format', choices=['pdf', 'png'], default='pdf')
    args = parser.parse_args()

    ext     = f'.{args.format}'
    run_exp = set(args.exp) if args.exp else set(range(1, 17))

    root   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fdir   = os.path.join(root, 'output', 'figures')
    tdir   = os.path.join(root, 'output', 'tables')
    ddir_e = os.path.join(root, 'output', 'data', 'experiments')
    for d in (fdir, tdir): os.makedirs(d, exist_ok=True)

    # Path-builder helpers passed into every runner
    F  = lambda n: os.path.join(fdir, n + ext)
    T  = lambda n: os.path.join(tdir, n + '.tex')
    LD = lambda n: _load(os.path.join(ddir_e, n))

    _header('Cinquillo 2.0 — Visualisation & Report Generator', total=len(run_exp))

    for exp_num in sorted(run_exp):
        runner = _EXPERIMENT_RUNNERS.get(exp_num)
        if runner:
            runner(F, T, LD, root)
        else:
            _warn(f'No runner registered for experiment {exp_num}')

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f'\n{"━"*72}')
    print(f'  VISUALISATION COMPLETE  —  elapsed: {elapsed:.1f} s')
    print(f'{"━"*72}')
    for label, d in [('Figures', fdir), ('Tables', tdir)]:
        if os.path.isdir(d):
            files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
            kb = sum(os.path.getsize(os.path.join(d, f)) for f in files) // 1024
            print(f'  {label:<10} {len(files):>3} files  ({kb} KB)  → {d}/')
    print('\n  LaTeX usage:')
    print(r'    \graphicspath{{output/figures/}}')
    print(r'    \input{output/tables/fairness_ranking}')
    print()


if __name__ == '__main__':
    main()