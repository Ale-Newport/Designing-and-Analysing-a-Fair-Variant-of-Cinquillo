"""
Test script for the trained RL agent.

Loads models/rl_agent.pkl by default and runs a 5-variant tournament
against three heuristic opponents (Aggressive, Defensive, Balanced).

Variant configurations tested
  V1  Baseline        Wild / Take 2        p=0.50  Pen=1  WTA   5 rounds
  V2  Combo Rush      Double Play / FPass  p=0.60  Pen=2  DP    5 rounds
  V3  Fortune's Wheel Wild / FPass         p=1.00  Pen=1  WTA   5 rounds
  V4  Lucky Draw      Wild / −1 pt         p=0.80  Pen=1  WTA   5 rounds
  V5  Open Book       Info / Exposed       p=0.50  Pen=0  WTA   8 rounds

Usage
  python rl_agent/test_agent.py
  python rl_agent/test_agent.py --weights path/to/rl_agent.pkl
  python rl_agent/test_agent.py --games 2000
"""
import sys
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import random
from typing import List, Dict, Tuple

from tqdm import tqdm

from agents.rl_agent import RLAgent, StateEncoder
from agents.base_agents import (
    Agent,
    create_aggressive_heuristic,
    create_defensive_heuristic,
    create_balanced_heuristic,
)
from game.entities import (
    VariantConfig, MatchEndMode, ScoringMode,
    GoodDiceEffect, BadDiceEffect,
)
from game.rules import Rules


# Default weights path
DEFAULT_WEIGHTS = SCRIPT_DIR / "models" / "rl_agent.pkl"


# 5 variant configurations
VARIANTS: List[Tuple[str, VariantConfig]] = [
    (
        "V1 — Baseline  (Wild / Take 2)",
        VariantConfig(
            match_end_mode=MatchEndMode.FIXED_ROUNDS,
            fixed_rounds_count=5,
            scoring_mode=ScoringMode.WINNER_TAKES_ALL,
            points_per_card=1,
            voluntary_pass_penalty=1,
            dice_good_probability=0.50,
            dice_good_effect=GoodDiceEffect.WILD,
            dice_bad_effect=BadDiceEffect.TAKE_CARDS,
            dice_bad_cards_count=2,
        ),
    ),
    (
        "V2 — Combo Rush  (Double Play / Forced Pass)",
        VariantConfig(
            match_end_mode=MatchEndMode.FIXED_ROUNDS,
            fixed_rounds_count=5,
            scoring_mode=ScoringMode.DOUBLE_PENALTY,
            points_per_card=1,
            voluntary_pass_penalty=2,
            dice_good_probability=0.60,
            dice_good_effect=GoodDiceEffect.DOUBLE_PLAY,
            dice_bad_effect=BadDiceEffect.FORCED_PASS,
        ),
    ),
    (
        "V3 — Fortune's Wheel  (Wild / Forced Pass  p=1.00)",
        VariantConfig(
            match_end_mode=MatchEndMode.FIXED_ROUNDS,
            fixed_rounds_count=5,
            scoring_mode=ScoringMode.WINNER_TAKES_ALL,
            points_per_card=1,
            voluntary_pass_penalty=1,
            dice_good_probability=1.00,
            dice_good_effect=GoodDiceEffect.WILD,
            dice_bad_effect=BadDiceEffect.FORCED_PASS,  # never triggered
        ),
    ),
    (
        "V4 — Lucky Draw  (Wild / −1 pt  p=0.80)",
        VariantConfig(
            match_end_mode=MatchEndMode.FIXED_ROUNDS,
            fixed_rounds_count=5,
            scoring_mode=ScoringMode.WINNER_TAKES_ALL,
            points_per_card=1,
            voluntary_pass_penalty=1,
            dice_good_probability=0.80,
            dice_good_effect=GoodDiceEffect.WILD,
            dice_bad_effect=BadDiceEffect.NEGATIVE_POINTS,
            dice_bad_penalty_points=1,
        ),
    ),
    (
        "V5 — Open Book  (Information / Exposed  8R)",
        VariantConfig(
            match_end_mode=MatchEndMode.FIXED_ROUNDS,
            fixed_rounds_count=8,
            scoring_mode=ScoringMode.WINNER_TAKES_ALL,
            points_per_card=1,
            voluntary_pass_penalty=0,
            dice_good_probability=0.50,
            dice_good_effect=GoodDiceEffect.INFO_REVEAL,
            dice_bad_effect=BadDiceEffect.REVEAL_HAND,
        ),
    ),
]


# Opponents
def make_opponents() -> List[Agent]:
    """Aggressive + Defensive + Balanced heuristic agents."""
    return [
        create_aggressive_heuristic(),
        create_defensive_heuristic(),
        create_balanced_heuristic(),
    ]


# Core game runner
def play_one_game(agents: List[Agent], variant: VariantConfig) -> Dict:
    state = Rules.initialize_game(len(agents), variant)
    turns = 0
    while not Rules.is_terminal(state) and turns < 1000:
        legal = Rules.get_legal_moves(state)
        if not legal:
            break
        action = agents[state.current_player].choose_move(state, legal)
        state  = Rules.apply_move(state, action)
        turns += 1
    Rules.compute_round_scores(state)
    scores = [p.match_score for p in state.players]
    max_s  = max(scores)
    return {
        'scores':         scores,
        'winner_indices': [i for i, s in enumerate(scores) if s == max_s],
    }


def run_block(
    rl_agent: RLAgent,
    opponents: List[Agent],
    variant: VariantConfig,
    num_games: int,
    desc: str = "",
) -> Dict:
    """
    Run num_games games cycling the RL agent through all 4 seats
    Returns per-agent win rates, avg scores, and per-seat win rates
    """
    all_agents = [rl_agent] + opponents
    n          = len(all_agents)
    names      = [a.name for a in all_agents]

    wins      = {name: 0.0 for name in names}
    score_sum = {name: 0.0 for name in names}
    pos_wins  = [0.0] * n
    rl_name   = rl_agent.name

    for g in tqdm(range(num_games), desc=desc, leave=False):
        pos     = g % n
        rotated = all_agents[pos:] + all_agents[:pos]
        result  = play_one_game(rotated, variant)
        share   = 1.0 / len(result['winner_indices'])

        for idx, agent in enumerate(rotated):
            score_sum[agent.name] += result['scores'][idx]
            if idx in result['winner_indices']:
                wins[agent.name] += share
                # only track positional wins for the RL agent itself
                if agent.name == rl_name:
                    pos_wins[pos] += share

    games_per_seat = num_games / n
    return {
        'agent_names':   names,
        'win_rates':     {name: wins[name]      / num_games * 100 for name in names},
        'avg_scores':    {name: score_sum[name] / num_games       for name in names},
        'pos_win_rates': [w / games_per_seat * 100 for w in pos_wins],
        'games':         num_games,
    }


# Output formatting
W = 66

def print_block(v_label: str, result: Dict, rl_name: str) -> None:
    print("\n" + "=" * W)
    print(f"  {v_label}")
    print(f"  Opponents: Aggressive + Defensive + Balanced  |  Games: {result['games']}")
    print("=" * W)

    print(f"\n  {'Agent':<26} {'Win %':>7}  {'Avg Score':>10}")
    print("  " + "-" * 47)
    for name in sorted(result['agent_names'],
                       key=lambda n: result['win_rates'][n], reverse=True):
        tag = " ◀ RL" if name == rl_name else ""
        print(f"  {name:<26} {result['win_rates'][name]:>6.1f}%"
              f"  {result['avg_scores'][name]:>10.2f}{tag}")

    print(f"\n  Seat win rates (positional fairness check):")
    for i, pwr in enumerate(result['pos_win_rates']):
        bar = "█" * int(pwr / 2)
        print(f"    Seat {i + 1}: {pwr:5.1f}%  {bar}")
    print("=" * W)


def print_summary(results: Dict[str, Dict], rl_name: str) -> None:
    print("\n" + "=" * W)
    print("  SUMMARY  —  RL agent win rate vs 3 heuristics")
    print("  (random baseline ≈ 25.0%)")
    print("=" * W)
    print(f"  {'Variant':<44}  {'Win %':>7}")
    print("  " + "-" * 54)

    win_rates = []
    for v_label, _ in VARIANTS:
        wr = results[v_label]['win_rates'].get(rl_name, float('nan'))
        win_rates.append(wr)
        marker = "▲" if wr > 30 else ("—" if wr > 25 else "▼")
        print(f"  {v_label[:44]:<44}  {wr:>6.1f}%  {marker}")

    print("  " + "-" * 54)
    valid = [w for w in win_rates if not (w != w)] # filter NaN
    mean  = sum(valid) / len(valid) if valid else 0.0
    print(f"  {'Mean':<44}  {mean:>6.1f}%")
    print("=" * W)

    print()
    verdict = ("outperforms baseline ✓" if mean > 30
               else "above random" if mean > 25
               else "needs more training ✗")
    print(f"  Result: {mean:.1f}% mean win rate  =>  {verdict}")
    print("=" * W)



# Dimension guard
def check_dim(rl_agent: RLAgent) -> int:
    probe    = Rules.initialize_game(4, VARIANTS[0][1])
    live_dim = len(StateEncoder.encode(probe, 0))
    print(f"  Encoder feature dim:  {live_dim}")

    if rl_agent.q_network is not None:
        saved_dim = rl_agent.q_network.state_dim
        print(f"  Loaded model dim:     {saved_dim}")
        if saved_dim != live_dim:
            print(
                f"\n  [ERROR] Dimension mismatch — model has {saved_dim} inputs "
                f"but encoder produces {live_dim}.\n"
                f"  Retrain with train_agent.py to produce a compatible model."
            )
            sys.exit(1)
    else:
        print(f"  Loaded model dim:     N/A  (no weights loaded)")
        print(f"  Run train_agent.py first."
              f"Continuing with uninitialised network for demo.")

    return live_dim


# Entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test trained RL agent, 5 variants vs 3 heuristic opponents"
    )
    parser.add_argument(
        "--weights", type=str, default=str(DEFAULT_WEIGHTS),
        help=f"Path to trained weights  [default: models/rl_agent.pkl]",
    )
    parser.add_argument(
        "--games", type=int, default=1000,
        help="Games per variant block  [default: 1000]",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path

    print("=" * W)
    print("  RL AGENT EVALUATION")
    print("=" * W)
    print(f"  Weights:      {weights_path}")
    print(f"  Games/block:  {args.games}")
    print(f"  Variants:     {len(VARIANTS)}")
    print(f"  Opponents:    Aggressive + Defensive + Balanced")
    print(f"  Total games:  {args.games * len(VARIANTS)}")
    print()

    # Load agent
    rl_agent = RLAgent(
        name="RL",
        epsilon=0.0,
        learning_rate=0.0,
        discount_factor=0.95,
        use_heuristics=True,
    )
    rl_agent.load_weights(str(weights_path))
    check_dim(rl_agent)
    print()

    # Run tournament
    opponents = make_opponents()
    results: Dict[str, Dict] = {}

    for block_num, (v_label, variant) in enumerate(VARIANTS, 1):
        print(f"[{block_num}/{len(VARIANTS)}]  {v_label}")
        result = run_block(
            rl_agent=rl_agent,
            opponents=opponents,
            variant=variant,
            num_games=args.games,
            desc=f"  Block {block_num}/{len(VARIANTS)}",
        )
        print_block(v_label, result, rl_agent.name)
        results[v_label] = result

    # Summary
    print_summary(results, rl_agent.name)