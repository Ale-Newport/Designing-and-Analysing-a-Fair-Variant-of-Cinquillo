"""
RL agent training  (fast, full-coverage variant pool).

Designed to complete in <= 30 minutes while exposing the agent to every
combination of dice effect and scoring mode in the Cinquillo 2.0 variant table.

Speed budget (4-player games, 3 heuristic opponents)
------------------------------------------------------
  Stage 0   Ep    0-400   Randoms x 3            base only    2 updates/ep
  Stage 1   Ep  400-1500  1 Heuristic + Randoms   base only    3 updates/ep
  Stage 2   Ep 1500-5000  Full heuristics         all variants  5 updates/ep
  Estimated wall time at ~4 it/s average: ~21 min
  Evaluation overhead (4 evals x 40 games): ~3 min
  Total: ~24 min

Variant coverage
-----------------
  Every combination of Good x Bad effect (4x4 = 16 configs) with realistic
  parameters drawn from the named-variant table.

Output
------
  models/rl_agent.pkl        -- overwritten whenever a new best is reached
  models/training_log.json   -- metrics after each evaluation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import time
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

# Save inside the script's own directory so path is CWD-independent
_SCRIPT_DIR = Path(os.path.abspath(__file__)).parent
_DEFAULT_SAVE_DIR = str(_SCRIPT_DIR / 'models')

from game.entities import (
    VariantConfig, MatchEndMode, ScoringMode,
    GoodDiceEffect, BadDiceEffect,
)
from game.rules import Rules
from agents.rl_agent import RLAgent, StateEncoder
from agents.base_agents import Agent, RandomAgent

try:
    from agents.base_agents import (
        create_aggressive_heuristic,
        create_defensive_heuristic,
        create_balanced_heuristic,
    )
    _NAMED_HEURISTICS = True
except ImportError:
    from agents.base_agents import HeuristicAgent
    _NAMED_HEURISTICS = False


# ---------------------------------------------------------------------------
# Compact VariantConfig factory
# ---------------------------------------------------------------------------

def _vc(good, bad, p=0.5, pen=1, ppc=1, scoring=None, rounds=5,
        target_mult=10, take_n=2, neg_pts=1):
    if scoring is None:
        scoring = ScoringMode.WINNER_TAKES_ALL
    return VariantConfig(
        dice_good_effect=good,
        dice_bad_effect=bad,
        dice_good_probability=p,
        voluntary_pass_penalty=pen,
        points_per_card=ppc,
        scoring_mode=scoring,
        match_end_mode=MatchEndMode.FIXED_ROUNDS,
        fixed_rounds_count=rounds,
        dice_bad_cards_count=take_n,
        dice_bad_penalty_points=neg_pts,
    )

W   = GoodDiceEffect.WILD
DP2 = GoodDiceEffect.DOUBLE_PLAY
INF = GoodDiceEffect.INFO_REVEAL
GIV = GoodDiceEffect.GIVE_CARD
TC  = BadDiceEffect.TAKE_CARDS
FP  = BadDiceEffect.FORCED_PASS
NP  = BadDiceEffect.NEGATIVE_POINTS
RH  = BadDiceEffect.REVEAL_HAND
WTA = ScoringMode.WINNER_TAKES_ALL
DP  = ScoringMode.DOUBLE_PENALTY


# ---------------------------------------------------------------------------
# Variant pools
# ---------------------------------------------------------------------------

BASE_ONLY: List[VariantConfig] = [
    _vc(W, TC, p=0.50, pen=1, scoring=WTA, rounds=5, take_n=2),
]

# Stage 2 pool: base variants + DoublePlay family so the mechanic is
# introduced before full-difficulty stage 3
STAGE2_VARIANTS: List[VariantConfig] = [
    _vc(W,   TC,  p=0.50, pen=1, scoring=WTA, rounds=5, take_n=2),  # Baseline
    _vc(W,   TC,  p=0.50, pen=1, scoring=WTA, rounds=5, take_n=2),  # extra weight
    _vc(DP2, FP,  p=0.60, pen=2, scoring=DP,  rounds=5),             # Combo Rush
    _vc(DP2, TC,  p=0.50, pen=0, scoring=WTA, rounds=5, take_n=2),  # Double Edge
]

# 16 configs — one per (good_effect x bad_effect) combination.
# Parameters sampled from the named-variant table in the UI.
ALL_VARIANTS: List[VariantConfig] = [
    # WILD x *
    _vc(W,   TC,  p=0.50, pen=1, scoring=WTA, rounds=5, take_n=2),   # Baseline
    _vc(W,   FP,  p=1.00, pen=1, scoring=WTA, rounds=5),              # Fortune's Wheel
    _vc(W,   NP,  p=0.80, pen=1, scoring=WTA, rounds=5, neg_pts=1),   # Lucky Draw
    _vc(W,   RH,  p=0.50, pen=1, scoring=DP,  rounds=5),              # Safe Harbour-style
    # DOUBLE_PLAY x *
    _vc(DP2, TC,  p=0.50, pen=0, scoring=WTA, rounds=5, take_n=2),   # Double Edge
    _vc(DP2, FP,  p=0.60, pen=2, scoring=DP,  rounds=5),              # Combo Rush
    _vc(DP2, NP,  p=0.70, pen=2, scoring=DP,  rounds=5, neg_pts=3),   # Power Play
    _vc(DP2, RH,  p=0.60, pen=1, scoring=WTA, rounds=5),              # Reveal Rush
    # INFO_REVEAL x *
    _vc(INF, TC,  p=0.45, pen=2, scoring=WTA, rounds=5, take_n=2),   # Pass & Peek
    _vc(INF, FP,  p=0.50, pen=1, scoring=WTA, rounds=5),              # Pure Strategy-style
    _vc(INF, NP,  p=0.50, pen=1, scoring=DP,  rounds=5, neg_pts=1),   # Intel War-style
    _vc(INF, RH,  p=0.50, pen=0, scoring=WTA, rounds=8),              # Open Book
    # GIVE_CARD x *
    _vc(GIV, TC,  p=0.55, pen=1, scoring=WTA, rounds=5, take_n=2),   # Hand Swap
    _vc(GIV, FP,  p=0.50, pen=1, scoring=WTA, rounds=5),              # Mirror Match
    _vc(GIV, NP,  p=0.50, pen=1, scoring=DP,  rounds=5, neg_pts=2),   # Card Exchange
    _vc(GIV, RH,  p=0.40, pen=2, scoring=DP,  rounds=5),              # Ghost Hand
]

NUM_EVAL_GAMES = 200   # per checkpoint — fast but representative


# ---------------------------------------------------------------------------
# Opponents
# ---------------------------------------------------------------------------

def make_heuristic_agents() -> List[Agent]:
    """Exactly 3 opponents so all games are 4-player (feature dim = 202)."""
    if _NAMED_HEURISTICS:
        return [
            create_aggressive_heuristic(),
            create_defensive_heuristic(),
            create_balanced_heuristic(),
        ]
    return [
        HeuristicAgent(name="Aggressive",
                       prefer_reduce_hand=1.5, prefer_balance_suits=0.3,
                       prefer_open_suits=0.2,  prefer_block=0.2,
                       avoid_voluntary_pass=3.0, dice_risk_tolerance=0.5),
        HeuristicAgent(name="Defensive",
                       prefer_reduce_hand=0.8, prefer_balance_suits=0.8,
                       prefer_open_suits=0.4,  prefer_block=0.6,
                       avoid_voluntary_pass=1.5, dice_risk_tolerance=0.2),
        HeuristicAgent(name="Balanced",
                       prefer_reduce_hand=1.0, prefer_balance_suits=0.5,
                       prefer_open_suits=0.3,  prefer_block=0.4,
                       avoid_voluntary_pass=2.0, dice_risk_tolerance=0.3),
    ]


# ---------------------------------------------------------------------------
# Game / eval helpers
# ---------------------------------------------------------------------------

def play_single_game(agents: List[Agent], variant: VariantConfig) -> dict:
    state = Rules.initialize_game(len(agents), variant)
    turns = 0
    while not Rules.is_terminal(state) and turns < 500:
        legal = Rules.get_legal_moves(state)
        if not legal:
            break
        state = Rules.apply_move(
            state, agents[state.current_player].choose_move(state, legal)
        )
        turns += 1
    Rules.compute_round_scores(state)
    scores = [p.match_score for p in state.players]
    mx = max(scores)
    return {
        'scores':         scores,
        'winner_indices': [i for i, s in enumerate(scores) if s == mx],
    }



def evaluate_agent(
    agent: RLAgent,
    opponents: List[Agent],
    num_games: int = NUM_EVAL_GAMES,
    variants: Optional[List[VariantConfig]] = None,
) -> dict:
    """Evaluate with epsilon = 0, cycling RL through all 4 seats."""
    if variants is None:
        variants = ALL_VARIANTS
    orig_eps      = agent.epsilon
    agent.epsilon = 0.0

    wins = 0.0
    total_score = 0.0
    played = 0

    games_per_seat = max(1, num_games // 4)
    n_agents = 1 + len(opponents)
    for seat in range(4):
        rl_idx = (n_agents - seat) % n_agents
        for g in range(games_per_seat):
            variant  = variants[g % len(variants)]
            all_agts = [agent] + opponents
            rotated  = all_agts[seat:] + all_agts[:seat]
            result   = play_single_game(rotated, variant)
            if rl_idx in result['winner_indices']:
                wins += 1.0 / len(result['winner_indices'])
            total_score += result['scores'][rl_idx]
            played += 1

    agent.epsilon = orig_eps
    return {
        'win_rate':  wins / played * 100,
        'avg_score': total_score / played,
        'games':     played,
    }


class TrainingLogger:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {
            'episodes': [], 'avg_reward': [], 'win_rate': [],
            'avg_score': [], 'epsilon': [],
            'best_win_rate': 0.0, 'best_episode': 0, 'feature_dim': None,
        }

    def log(self, episode: int, metrics: dict):
        self.stats['episodes'].append(episode)
        for k, v in metrics.items():
            if k in self.stats:
                self.stats[k].append(v)

    def save(self):
        with open(self.save_dir / 'training_log.json', 'w') as f:
            json.dump(self.stats, f, indent=2)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_rl_agent(
    rl_agent: RLAgent,
    opponents: List[Agent],
    num_episodes: int = 5000,
    eval_interval: int = 1000,
    batch_size: int = 64,
    save_dir: str = _DEFAULT_SAVE_DIR,
    curriculum_learning: bool = True,
) -> dict:
    """
    Three-stage curriculum designed for <= 30 min wall time.

    Stage 0  (0-400):    Randoms,              base only,   2 updates/ep
    Stage 1  (400-1500): 1 Heuristic + Randoms, base only,  3 updates/ep
    Stage 2  (1500+):    Full heuristics,       all 16 vars, 5 updates/ep
    """
    save_path       = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    best_model_path = str(save_path / "rl_agent.pkl")
    logger          = TrainingLogger(save_dir)

    probe    = Rules.initialize_game(4, BASE_ONLY[0])
    feat_dim = len(StateEncoder.encode(probe, 0))
    logger.stats['feature_dim'] = feat_dim

    randoms = [RandomAgent(f"R{i}") for i in range(3)]

    if curriculum_learning:
        stages = [
            # Stage 0: randoms only — learn basic card play (fast)
            (0,    400,          randoms,                          BASE_ONLY,    2),
            # Stage 1: 1 heuristic + randoms — introduce moderate opposition
            (400,  1500,         opponents[:1] + randoms[:2],      BASE_ONLY,    3),
            # Stage 2 (NEW): 2 heuristics + 1 random, still base variant only
            #   Smooth transition — full heuristic strength without variant shock
            (1500, 3000,         opponents[:2] + randoms[:1],      BASE_ONLY,    4),
            # Stage 3: full heuristics + all 16 variants — full difficulty
            (3000, num_episodes, opponents,                        ALL_VARIANTS, 5),
        ]
    else:
        stages = [(0, num_episodes, opponents, ALL_VARIANTS, 5)]

    print("=" * 62)
    print("RL AGENT TRAINING  v6  (fast full-coverage)")
    print("=" * 62)
    print(f"  Agent:         {rl_agent.name}")
    print(f"  Episodes:      {num_episodes}")
    print(f"  Eval interval: {eval_interval}  ({NUM_EVAL_GAMES} games each)")
    print(f"  Batch size:    {batch_size}")
    print(f"  Feature dim:   {feat_dim}")
    print(f"  Variants:      {len(ALL_VARIANTS)}  (all 16 effect combos)")
    print(f"  Save path:     {best_model_path}")
    print(f"  Curriculum:    {curriculum_learning}")
    if curriculum_learning:
        print()
        print("  Curriculum stages:")
        for s, e, opps, vpool, nu in stages:
            print(f"    Ep {s:>5}-{e:<5}  opp={[o.name for o in opps]}"
                  f"  variants={len(vpool)}  updates/ep={nu}")
    print("=" * 62)

    start_time    = time.time()
    best_win_rate = 0.0

    for episode in tqdm(range(num_episodes), desc="Training"):

        # Resolve curriculum stage
        cur_opps, cur_variants, cur_updates = opponents, ALL_VARIANTS, 5
        for s, e, opps, vpool, nu in stages:
            if s <= episode < e:
                cur_opps, cur_variants, cur_updates = opps, vpool, nu
                break

        variant    = random.choice(cur_variants)
        all_agents = [rl_agent] + cur_opps
        position   = episode % 4
        rotated    = all_agents[position:] + all_agents[:position]
        n          = len(rotated)

        state  = Rules.initialize_game(n, variant)
        # RL agent's actual game-player index depends on which seat it occupies.
        # With all_agents=[RL, o1, o2, o3] and position=p:
        #   rotated = all_agents[p:] + all_agents[:p]
        #   RL ends up at index (n - p) % n in the rotated list.
        # Without this, rl_pos=0 wrongly treats the first opponent as RL for p>0,
        # causing 75% of training experiences to be stored from opponents' perspectives.
        rl_pos = (n - position) % n

        prev_rl_state        = None   # s_t (RL turn state)
        prev_rl_action       = None   # a_t
        prev_rl_reward       = 0.0    # r_t (immediate, action-only)
        prev_rl_next_imm     = None   # s immediately after a_t (before opponents)
        rl_won               = False
        scores_computed      = False
        turns                = 0

        # ── Experience storage strategy ───────────────────────────────────
        #
        # We store (s_t, a_t, r_t, s_{t+1}) where:
        #   s_t      = state at RL turn t              (current_player == rl_pos)
        #   a_t      = RL's chosen action
        #   r_t      = IMMEDIATE reward: only the direct effect of a_t
        #              (computed from s_t → apply_move(s_t, a_t))
        #              This avoids penalising RL for opponent TAKE_CARDS effects.
        #   s_{t+1}  = state at RL turn t+1            (current_player == rl_pos)
        #              Used for Q-bootstrap — semantically correct.
        #
        # The reward for what happened between turns (opponents reducing/growing
        # RL's hand) is NOT added — it would misattribute opponent effects to
        # RL's action.  The terminal bonus (win/loss) is captured when the game
        # ends and RL either played the winning card (immediate) or we detect
        # an opponent win.

        while not Rules.is_terminal(state) and turns < 500:
            cur_player = state.current_player
            legal = Rules.get_legal_moves(state)
            if not legal:
                break

            if cur_player == rl_pos:
                # ── RL's turn ──────────────────────────────────────────────
                action = rotated[rl_pos].choose_move(state, legal)

                # Immediate next state (before any opponent moves)
                next_imm = Rules.apply_move(state, action)
                done     = next_imm.game_over

                if done:
                    Rules.compute_round_scores(next_imm)
                    scores_computed = True
                    rl_won = True
                    imm_reward = rl_agent.compute_reward(state, action, next_imm, rl_pos)
                    rl_agent.store_experience(state, action, imm_reward, next_imm, True)
                else:
                    # Compute immediate reward now; store when we reach RL turn t+1
                    # so we have the correct bootstrap state (s_{t+1}).
                    imm_reward = rl_agent.compute_reward(state, action, next_imm, rl_pos)

                    # Flush the PREVIOUS RL transition now that s_{t+1} = state
                    if prev_rl_state is not None:
                        rl_agent.store_experience(
                            prev_rl_state, prev_rl_action,
                            prev_rl_reward, state, False
                        )

                    prev_rl_state    = state
                    prev_rl_action   = action
                    prev_rl_reward   = imm_reward

                state = next_imm
            else:
                # ── Opponent's turn ─────────────────────────────────────────
                action = rotated[cur_player].choose_move(state, legal)
                next_s = Rules.apply_move(state, action)

                if next_s.game_over and not rl_won:
                    # Opponent won — give RL a terminal loss signal
                    Rules.compute_round_scores(next_s)
                    scores_computed = True
                    if prev_rl_state is not None:
                        loss_reward = rl_agent.compute_reward(
                            prev_rl_state, prev_rl_action, next_s, rl_pos
                        )
                        rl_agent.store_experience(
                            prev_rl_state, prev_rl_action, loss_reward, next_s, True
                        )

                state = next_s

            turns += 1

        if not scores_computed:
            Rules.compute_round_scores(state)

        rl_agent.end_episode()

        # Flush replay buffer at stage-3 transition so stale experiences
        # from random/partial-heuristic stages don't contaminate gradients.
        if episode == 3000:
            rl_agent.replay_buffer._buffer.clear()
            rl_agent.replay_buffer._priorities.clear()
            rl_agent.replay_buffer._pos = 0
            rl_agent.replay_buffer._max_priority = 1.0

        # Once the buffer is full (ep ~10000), cut updates to 2/ep.
        # More updates on a full buffer just causes over-fitting to PER
        # high-priority samples and drives the post-peak WR decline.
        effective_updates = cur_updates if episode < 10000 else 2

        if len(rl_agent.replay_buffer) >= batch_size:
            for _ in range(effective_updates):
                rl_agent.train_from_replay(batch_size)

        # Epsilon decay
        if episode > 0 and episode % 50 == 0:
            if episode < 1500:
                rl_agent.decay_epsilon(decay_rate=0.990, min_epsilon=0.15)
            else:
                rl_agent.decay_epsilon(decay_rate=0.993, min_epsilon=0.05)

        # Learning rate schedule — reduce on stage transitions to prevent
        # late-stage oscillation once the agent is close to convergence.
        if episode == 3000:    # entering stage 3 (full difficulty)
            rl_agent.learning_rate = max(rl_agent.learning_rate * 0.5, 5e-5)
        elif episode == 8000:   # buffer filling, reduce oscillation
            rl_agent.learning_rate = max(rl_agent.learning_rate * 0.5, 1e-5)
        elif episode == 15000:  # late fine-tuning on stable policy
            rl_agent.learning_rate = max(rl_agent.learning_rate * 0.5, 5e-6)

        # Evaluation checkpoint
        if (episode > 0 and episode % eval_interval == 0) or episode == num_episodes - 1:
            eval_stats = evaluate_agent(rl_agent, opponents)
            recent     = rl_agent.episode_rewards[-eval_interval:]
            avg_r      = float(np.mean(recent)) if recent else 0.0
            elapsed    = time.time() - start_time

            stage_idx = next(
                (i for i, (s, e, *_) in enumerate(stages) if s <= episode < e),
                len(stages) - 1
            )

            logger.log(episode, {
                'avg_reward': avg_r,
                'win_rate':   eval_stats['win_rate'],
                'avg_score':  eval_stats['avg_score'],
                'epsilon':    rl_agent.epsilon,
            })

            improved = eval_stats['win_rate'] > best_win_rate
            if improved:
                best_win_rate = eval_stats['win_rate']
                logger.stats['best_win_rate'] = best_win_rate
                logger.stats['best_episode']  = episode
                rl_agent.save_weights(best_model_path)

            logger.save()

            marker = " ✓ NEW BEST" if improved else ""
            print(f"\n  [S{stage_idx}] Ep {episode:>5}  "
                  f"WR={eval_stats['win_rate']:.1f}%  "
                  f"best={best_win_rate:.1f}%  "
                  f"e={rl_agent.epsilon:.3f}  "
                  f"buf={len(rl_agent.replay_buffer)}  "
                  f"t={elapsed:.0f}s{marker}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 62)
    print("TRAINING COMPLETE")
    print(f"  Time:          {elapsed:.0f}s  ({elapsed / 60:.1f} min)")
    print(f"  Total updates: {rl_agent.total_updates}")
    print(f"  Final epsilon: {rl_agent.epsilon:.3f}")
    print(f"  Best win rate: {best_win_rate:.1f}%  (Ep {logger.stats['best_episode']})")
    print(f"  Model:         {best_model_path}")
    print("=" * 62)

    logger.save()
    return logger.stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train RL agent — all variant configs in <= 30 min"
    )
    parser.add_argument("--episodes",      type=int,   default=25000)
    parser.add_argument("--eval-interval", type=int,   default=1000)
    parser.add_argument("--epsilon",       type=float, default=0.65)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--save-dir",      type=str,   default=_DEFAULT_SAVE_DIR,
                        help="Output directory  [default: <script_dir>/models]")
    parser.add_argument("--load",          type=str,   default=None,
                        help="Warm-start from an existing .pkl file")
    parser.add_argument("--batch-size",    type=int,   default=64)
    parser.add_argument("--no-curriculum", action="store_true")
    args = parser.parse_args()

    rl_agent = RLAgent(
        name="RL-Agent-v6",
        epsilon=args.epsilon,
        learning_rate=args.lr,
        discount_factor=0.95,
        use_heuristics=True,
    )

    if args.load:
        rl_agent.load_weights(args.load)
        print(f"Warm-started from {args.load}")

    opponents = make_heuristic_agents()   # exactly 3 -> 4-player games, dim=202

    train_rl_agent(
        rl_agent=rl_agent,
        opponents=opponents,
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        curriculum_learning=not args.no_curriculum,
    )

    print(f"\n  Test: python rl_agent/test_agent.py --weights {args.save_dir}/rl_agent.pkl")