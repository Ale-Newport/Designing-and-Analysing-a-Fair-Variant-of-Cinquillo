#!/usr/bin/env python3
"""
Cinquillo 2.0 Markdown debug visualiser.

Purpose
-------
Run one or more simulations and write a Markdown report that traces the game
flow and the decision logic used by each agent. This is intended for debugging
agent behaviour, not for presentation.

Highlights
----------
- same variant/agent registry style as visualise_flow.py
- outputs Markdown instead of LaTeX
- traces every turn with pre/post state summaries
- logs legal moves, chosen move, and agent-specific reasoning
- supports multiple runs in one file for reproducible debugging
- can optionally include full hands for all players every turn

Examples
--------
python simulation/debug_agents_md.py
python simulation/debug_agents_md.py "combo rush"
python simulation/debug_agents_md.py "intel war" --agents mcts rl balanced aggressive
python simulation/debug_agents_md.py "open book" --runs 3 --seed 7
python simulation/debug_agents_md.py --list-variants
python simulation/debug_agents_md.py --list-agents
"""

import argparse
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.entities import (  # type: ignore
    BadDiceEffect,
    Card,
    Deck,
    GameState,
    GoodDiceEffect,
    MatchEndMode,
    ScoringMode,
    Suit,
    VariantConfig,
)
from game.rules import Pass, PlayCard, RollDice, Rules  # type: ignore
from agents.base_agents import (  # type: ignore
    Agent,
    HeuristicAgent,
    RandomAgent,
    create_aggressive_heuristic,
    create_balanced_heuristic,
    create_defensive_heuristic,
)
from agents.mcts_agent import (  # type: ignore
    MCTSAgent,
    MCTSAgentDeep,
    MCTSAgentFast,
    MCTSAgentStandard,
    MCTSAgentSuperFast,
)
from agents.rl_agent import RLAgent, StateEncoder  # type: ignore


# ============================================================================
# Variant registry (mirrors visualise_flow.py)
# ============================================================================

@dataclass(frozen=True)
class VariantSpec:
    name: str
    good: str
    bad: str
    p_good: float
    penalty: int
    scoring: str
    end: str


VARIANT_SPECS: List[VariantSpec] = [
    VariantSpec("Baseline",         "Wild",         "Take 2",      0.50, 1, "WTA", "5R"),
    VariantSpec("Blitz",            "Wild",         "Take 2",      0.50, 2, "WTA", "1R"),
    VariantSpec("Card Exchange",    "Transfer",     "-2 pts",      0.50, 1, "DP",  "10 pts"),
    VariantSpec("Card Flood",       "Wild",         "Take 4",      0.50, 1, "WTA", "5R"),
    VariantSpec("Chaos Mode",       "Wild",         "-5 pts",      0.90, 3, "DP",  "20 pts"),
    VariantSpec("Combo Rush",       "Double Play",  "Forced Pass", 0.50, 2, "DP",  "5R"),
    VariantSpec("Double Edge",      "Double Play",  "Take 2",      0.50, 0, "WTA", "5R"),
    VariantSpec("Double Spy",       "Double Play",  "Skip",        0.60, 1, "WTA", "5R"),
    VariantSpec("Endurance",        "Wild",         "Take 2",      0.50, 1, "WTA", "15R"),
    VariantSpec("Fortune's Wheel",  "Wild",         "Forced Pass", 1.00, 1, "WTA", "5R"),
    VariantSpec("Gambler's Run",    "Double Play",  "-2 pts",      0.45, 2, "DP",  "20 pts"),
    VariantSpec("Ghost Hand",       "Transfer",     "Exposed",     0.40, 2, "DP",  "15 pts"),
    VariantSpec("Glass Cannon",     "Double Play",  "-5 pts",      0.30, 2, "DP",  "5R"),
    VariantSpec("Hand Swap",        "Transfer",     "Take 2",      0.55, 1, "WTA", "5R"),
    VariantSpec("Heavy Toll",       "Wild",         "-3 pts",      0.40, 3, "DP",  "20 pts"),
    VariantSpec("High Roller",      "Wild",         "-2 pts",      0.60, 2, "DP",  "15 pts"),
    VariantSpec("Intel War",        "Information",  "Exposed",     0.50, 1, "DP",  "15 pts"),
    VariantSpec("Lucky Draw",       "Wild",         "-1 pt",       0.80, 1, "WTA", "5R"),
    VariantSpec("Marathon",         "Wild",         "Take 2",      0.50, 1, "WTA", "10R"),
    VariantSpec("Mirror Match",     "Transfer",     "Skip",        0.50, 1, "WTA", "5R"),
    VariantSpec("Open Book",        "Information",  "Exposed",     0.50, 0, "WTA", "8R"),
    VariantSpec("Pass & Peek",      "Information",  "Take 2",      0.45, 2, "WTA", "5R"),
    VariantSpec("Point Race",       "Wild",         "-1 pt",       0.50, 1, "WTA", "20 pts"),
    VariantSpec("Power Play",       "Double Play",  "-3 pts",      0.70, 2, "DP",  "5R"),
    VariantSpec("Pure Strategy",    "Wild",         "Forced Pass", 0.00, 1, "WTA", "5R"),
    VariantSpec("Reveal Rush",      "Double Play",  "Exposed",     0.60, 1, "WTA", "5R"),
    VariantSpec("Risk & Reward",    "Wild",         "-4 pts",      0.50, 3, "DP",  "25 pts"),
    VariantSpec("Safe Harbour",     "Wild",         "Forced Pass", 0.50, 0, "WTA", "5R"),
    VariantSpec("Score Doubler",    "Wild",         "Take 2",      0.50, 1, "DP",  "15 pts"),
    VariantSpec("Scout's Edge",     "Information",  "Forced Pass", 0.70, 0, "WTA", "5R"),
    VariantSpec("Slow Burn",        "Wild",         "Take 2",      0.50, 0, "WTA", "20R"),
    VariantSpec("Sprint",           "Wild",         "Take 2",      0.50, 1, "WTA", "3R"),
    VariantSpec("Spy Game",         "Information",  "Skip",        0.65, 1, "WTA", "10 pts"),
]


def slugify(text: str) -> str:
    return (
        text.strip()
        .lower()
        .replace("&", "and")
        .replace("'", "")
        .replace(".", "")
        .replace("—", "-")
        .replace("–", "-")
        .replace("_", "-")
        .replace(" ", "-")
    )


VARIANT_REGISTRY: Dict[str, VariantSpec] = {slugify(spec.name): spec for spec in VARIANT_SPECS}


def parse_good_effect(label: str) -> GoodDiceEffect:
    key = slugify(label)
    mapping = {
        "wild": GoodDiceEffect.WILD,
        "double-play": GoodDiceEffect.DOUBLE_PLAY,
        "transfer": GoodDiceEffect.GIVE_CARD,
        "information": GoodDiceEffect.INFO_REVEAL,
    }
    if key not in mapping:
        raise ValueError(f"Unknown good effect label: {label}")
    return mapping[key]


def parse_bad_effect(label: str) -> Tuple[BadDiceEffect, int]:
    raw = label.strip().lower()

    if raw.startswith("take"):
        count = int(raw.replace("take", "").strip().split()[0])
        return (BadDiceEffect.TAKE_CARDS, count)

    if raw.startswith("-") and "pt" in raw:
        number = raw.replace("pts", "").replace("pt", "").strip().lstrip("-")
        return (BadDiceEffect.NEGATIVE_POINTS, int(float(number)))

    if raw in {"forced pass", "skip"}:
        return (BadDiceEffect.FORCED_PASS, 1)

    if raw == "exposed":
        return (BadDiceEffect.REVEAL_HAND, 1)

    raise ValueError(f"Unknown bad effect label: {label}")


def parse_scoring(label: str) -> ScoringMode:
    key = label.strip().upper()
    if key == "WTA":
        return ScoringMode.WINNER_TAKES_ALL
    if key == "DP":
        return ScoringMode.DOUBLE_PENALTY
    raise ValueError(f"Unknown scoring label: {label}")


def parse_end(label: str) -> Tuple[MatchEndMode, int, int]:
    raw = label.strip().lower()
    if raw.endswith("r"):
        rounds = int(raw[:-1].strip())
        return (MatchEndMode.FIXED_ROUNDS, 10, rounds)
    if raw.endswith("pts"):
        points = int(float(raw[:-3].strip()))
        multiplier = max(1, points // 4)
        return (MatchEndMode.TARGET_SCORE, multiplier, 5)
    raise ValueError(f"Unknown end label: {label}")


def build_variant(spec: VariantSpec) -> VariantConfig:
    good_effect = parse_good_effect(spec.good)
    bad_effect, magnitude = parse_bad_effect(spec.bad)
    match_end_mode, target_score_multiplier, fixed_rounds_count = parse_end(spec.end)

    kwargs = {
        "dice_good_probability": spec.p_good,
        "dice_good_effect": good_effect,
        "dice_bad_effect": bad_effect,
        "scoring_mode": parse_scoring(spec.scoring),
        "points_per_card": 1,
        "voluntary_pass_penalty": spec.penalty,
        "match_end_mode": match_end_mode,
        "target_score_multiplier": target_score_multiplier,
        "fixed_rounds_count": fixed_rounds_count,
        "dice_bad_cards_count": 2,
        "dice_bad_penalty_points": 1,
    }

    if bad_effect == BadDiceEffect.TAKE_CARDS:
        kwargs["dice_bad_cards_count"] = magnitude
    elif bad_effect == BadDiceEffect.NEGATIVE_POINTS:
        kwargs["dice_bad_penalty_points"] = magnitude

    return VariantConfig(**kwargs)


# ============================================================================
# Agent builders
# ============================================================================

AGENT_BUILDERS = {
    "mcts-deep": lambda: MCTSAgentDeep(name="MCTS-Deep"),
    "mcts": lambda: MCTSAgentStandard(name="MCTS"),
    "mcts-fast": lambda: MCTSAgentFast(name="MCTS-Fast"),
    "mcts-superfast": lambda: MCTSAgentSuperFast(name="MCTS-SuperFast"),
    "rl": lambda: RLAgent(name="RL"),
    "balanced": create_balanced_heuristic,
    "aggressive": create_aggressive_heuristic,
    "defensive": create_defensive_heuristic,
    "random": lambda: RandomAgent(name="Random"),
    "random-safe": lambda: RandomAgent(name="Random-Safe", avoid_bad_moves=True),
}

DEFAULT_AGENT_KEYS = ["mcts", "rl", "balanced", "aggressive"]


def build_agents(agent_keys: Sequence[str]) -> List[Agent]:
    if len(agent_keys) != 4:
        raise ValueError("You must provide exactly 4 agents.")
    agents: List[Agent] = []
    for key in agent_keys:
        norm = slugify(key)
        if norm not in AGENT_BUILDERS:
            raise ValueError(f"Unknown agent '{key}'. Available: {', '.join(sorted(AGENT_BUILDERS.keys()))}")
        agents.append(AGENT_BUILDERS[norm]())
    return agents


# ============================================================================
# Formatting helpers
# ============================================================================

SUIT_SHORT = {
    Suit.OROS: "O",
    Suit.COPAS: "C",
    Suit.ESPADAS: "E",
    Suit.BASTOS: "B",
}

SUIT_ORDER = [Suit.OROS, Suit.COPAS, Suit.ESPADAS, Suit.BASTOS]
RANK_ORDER = Deck.RANKS


def fmt_card(card: Card) -> str:
    return f"{card.rank}{SUIT_SHORT[card.suit]}"


def sort_cards(cards: List[Card]) -> List[Card]:
    return sorted(cards, key=lambda c: (SUIT_ORDER.index(c.suit), Deck.RANK_INDEX[c.rank]))


def fmt_cards(cards: List[Card]) -> str:
    ordered = sort_cards(list(cards))
    return "[" + ", ".join(fmt_card(c) for c in ordered) + "]"


def fmt_move(move) -> str:
    if isinstance(move, PlayCard):
        return f"PlayCard({fmt_card(move.card)})"
    if isinstance(move, RollDice):
        return "RollDice()"
    if isinstance(move, Pass):
        return f"Pass(voluntary={move.voluntary})"
    return str(move)


def fmt_legal_moves(moves: List) -> str:
    return ", ".join(fmt_move(m) for m in moves)


def board_summary(state: GameState) -> str:
    parts = []
    for suit in SUIT_ORDER:
        ranks = sorted(state.board.suit_cards[suit], key=lambda r: Deck.RANK_INDEX[r])
        label = SUIT_SHORT[suit]
        parts.append(f"{label}: [{' '.join(map(str, ranks))}]" if ranks else f"{label}: []")
    return " | ".join(parts)


def hands_summary(state: GameState) -> str:
    return " | ".join(f"P{p.index} {p.hand_size()} cards {fmt_cards(p.hand)}" for p in state.players)


def scores_summary(state: GameState) -> str:
    return " | ".join(
        f"P{p.index}: round={p.round_score}, match={p.match_score}, hand={p.hand_size()}"
        for p in state.players
    )


def dice_summary(state: GameState) -> str:
    reveal_bits = [f"P{viewer}->P{target}" for viewer, target in sorted(state.dice_state.revealed_hands.items())]
    return (
        f"wild={state.dice_state.wild_active}, "
        f"double_play={state.dice_state.double_play_active}, "
        f"reveals={reveal_bits if reveal_bits else '[]'}"
    )


def variant_summary(variant: VariantConfig, num_players: int) -> List[str]:
    target_score = variant.get_target_score(num_players)
    return [
        f"good effect: `{variant.dice_good_effect.value}` with probability `{variant.dice_good_probability:.2f}`",
        f"bad effect: `{variant.dice_bad_effect.value}`",
        f"bad cards count: `{variant.dice_bad_cards_count}`",
        f"bad points penalty: `{variant.dice_bad_penalty_points}`",
        f"voluntary pass penalty: `{variant.voluntary_pass_penalty}`",
        f"scoring mode: `{variant.scoring_mode.value}`",
        f"match end mode: `{variant.match_end_mode.value}`",
        f"fixed rounds: `{variant.fixed_rounds_count}`",
        f"target score ({num_players} players): `{target_score}`",
    ]


# ============================================================================
# Agent reasoning tracer
# ============================================================================

class AgentTracer:
    def snapshot(self, agent: Agent, state: GameState, legal_moves: List):
        if isinstance(agent, RLAgent):
            return self._snapshot_rl(agent, state, legal_moves)
        if isinstance(agent, MCTSAgent):
            return self._snapshot_mcts(agent, state, legal_moves)
        return None

    def trace(self, agent: Agent, state: GameState, legal_moves: List, chosen_move, snapshot=None) -> List[str]:
        if isinstance(agent, RLAgent):
            return self._trace_rl(agent, state, legal_moves, chosen_move, snapshot=snapshot)
        if isinstance(agent, HeuristicAgent):
            return self._trace_heuristic(agent, state, legal_moves, chosen_move)
        if isinstance(agent, MCTSAgent):
            return self._trace_mcts(agent, state, legal_moves, chosen_move, snapshot=snapshot)
        if isinstance(agent, RandomAgent):
            return self._trace_random(agent, state, legal_moves, chosen_move)
        return [f"No specialised tracer for `{agent.__class__.__name__}`."]


    def _snapshot_mcts(self, agent: MCTSAgent, state: GameState, legal_moves: List) -> dict:
        filtered_moves = [m for m in legal_moves if not (isinstance(m, Pass) and getattr(m, 'voluntary', False))]
        effective_moves = filtered_moves if filtered_moves else legal_moves

        dice_can_advance = True
        if hasattr(agent, '_root_filter_moves'):
            root_filtered = agent._root_filter_moves(state, legal_moves)
            dice_can_advance = any(isinstance(m, RollDice) for m in root_filtered)
        else:
            root_filtered = effective_moves

        return {
            'filtered_moves': filtered_moves,
            'effective_moves': effective_moves,
            'root_filtered_moves': root_filtered,
            'root_dice_available': any(isinstance(m, RollDice) for m in effective_moves),
            'root_dice_survived_filter': any(isinstance(m, RollDice) for m in root_filtered),
            'dice_can_advance': dice_can_advance,
        }

    def _snapshot_rl(self, agent: RLAgent, state: GameState, legal_moves: List) -> dict:
        prev_pass = agent._consecutive_voluntary_passes
        prev_roll = agent._consecutive_rolls
        agent._sync_action_streak_tracking(state)
        synced_pass = agent._consecutive_voluntary_passes
        synced_roll = agent._consecutive_rolls
        constrained_moves = list(agent._apply_action_constraints(legal_moves))
        state_features = StateEncoder.encode(state, state.current_player)
        qnet_was_none = agent.q_network is None
        q_values = None
        if agent.q_network is not None:
            q_values, _ = agent.q_network.forward(state_features)

        agent._consecutive_voluntary_passes = prev_pass
        agent._consecutive_rolls = prev_roll

        return {
            'synced_pass': synced_pass,
            'synced_roll': synced_roll,
            'constrained_moves': constrained_moves,
            'state_features_len': len(state_features),
            'qnet_was_none': qnet_was_none,
            'q_values': q_values,
        }

    def _trace_random(self, agent: RandomAgent, state: GameState, legal_moves: List, chosen_move) -> List[str]:
        lines = [f"Policy: random choice, avoid_bad_moves={agent.avoid_bad_moves}."]
        if agent.avoid_bad_moves:
            filtered = [m for m in legal_moves if not isinstance(m, Pass)]
            pool = filtered if filtered else legal_moves
            lines.append(f"Candidate pool after pass filtering: {fmt_legal_moves(pool)}")
        else:
            lines.append(f"Candidate pool: {fmt_legal_moves(legal_moves)}")
        lines.append(f"Chosen move: {fmt_move(chosen_move)}")
        return lines

    def _trace_heuristic(self, agent: HeuristicAgent, state: GameState, legal_moves: List, chosen_move) -> List[str]:
        lines = [
            "Policy: weighted heuristic evaluation.",
            (
                f"Weights: reduce_hand={agent.prefer_reduce_hand}, balance={agent.prefer_balance_suits}, "
                f"open_suits={agent.prefer_open_suits}, block={agent.prefer_block}, "
                f"avoid_pass={agent.avoid_voluntary_pass}, dice_risk={agent.dice_risk_tolerance}"
            ),
        ]
        scored = []
        for move in legal_moves:
            score = agent._evaluate_move(state, move)
            scored.append((move, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        lines.append("Move scores:")
        for move, score in scored:
            lines.append(f"- {fmt_move(move)} -> {score:.4f}")
        lines.append(f"Chosen move: {fmt_move(chosen_move)}")
        return lines

    def _trace_mcts(self, agent: MCTSAgent, state: GameState, legal_moves: List, chosen_move, snapshot=None) -> List[str]:
        snapshot = snapshot or self._snapshot_mcts(agent, state, legal_moves)
        filtered_moves = snapshot['filtered_moves']
        effective_moves = snapshot['effective_moves']
        root_filtered_moves = snapshot['root_filtered_moves']
        lines = [
            "Policy: Monte Carlo Tree Search.",
            (
                f"Config: iterations={agent.num_iterations}, exploration={agent.exploration_weight}, "
                f"rollout_depth={agent.max_rollout_depth}, time_limit={agent.time_limit}s"
            ),
            f"Legal moves: {fmt_legal_moves(legal_moves)}",
        ]
        if len(filtered_moves) != len(legal_moves):
            lines.append(f"Voluntary passes filtered before search: {fmt_legal_moves(filtered_moves)}")
        if root_filtered_moves != effective_moves:
            lines.append(f"Root-level non-progress dice filtered out: {fmt_legal_moves(root_filtered_moves)}")
        lines.append(
            "Rollout policy note: if card plays are available, non-progress dice loops are avoided during rollouts."
        )
        revealed = state.dice_state.get_revealed_target(state.current_player)
        if revealed is not None:
            lines.append(f"Revealed information active: current player can inspect P{revealed}'s hand.")
        lines.append(f"Search returned: {fmt_move(chosen_move)} from pool {{{fmt_legal_moves(root_filtered_moves)}}}")
        return lines

    def _trace_rl(self, agent: RLAgent, state: GameState, legal_moves: List, chosen_move, snapshot=None) -> List[str]:
        snapshot = snapshot or self._snapshot_rl(agent, state, legal_moves)
        lines = [
            "Policy: epsilon-greedy Deep RL with heuristic guidance.",
            (
                f"Params: epsilon={agent.epsilon}, lr={agent.learning_rate}, gamma={agent.discount_factor}, "
                f"heuristics={agent.use_heuristics}"
            ),
        ]

        lines.append(
            f"Action streaks before choice: consecutive voluntary passes={snapshot['synced_pass']}, "
            f"consecutive rolls={snapshot['synced_roll']}"
        )

        constrained_moves = snapshot['constrained_moves']
        if constrained_moves != legal_moves:
            lines.append(f"Constraints removed some optional actions. Remaining: {fmt_legal_moves(constrained_moves)}")
        else:
            lines.append("No streak constraint removed any move.")

        lines.append(f"Encoded state feature length: {snapshot['state_features_len']}")
        q_values = snapshot['q_values']
        if snapshot['qnet_was_none'] and q_values is None:
            lines.append("Q-network not initialised before this choice; choose_move initialised it internally.")
        else:
            if q_values is None and agent.q_network is not None:
                state_features = StateEncoder.encode(state, state.current_player)
                q_values, _ = agent.q_network.forward(state_features)
                lines.append("Q-network was initialised during choose_move; values below use that freshly-created network.")
            scored = []
            for move in constrained_moves:
                idx = agent._move_to_idx(move)
                q_val = float(q_values[idx])
                heuristic_bonus = agent._heuristic_value(state, move) * 0.10 if agent.use_heuristics else 0.0
                total = q_val + heuristic_bonus
                scored.append((move, idx, q_val, heuristic_bonus, total))
            scored.sort(key=lambda x: x[4], reverse=True)
            lines.append("Move values (Q + tiny heuristic tiebreak):")
            for move, idx, q_val, h_bonus, total in scored:
                lines.append(
                    f"- {fmt_move(move)} -> idx={idx}, Q={q_val:.4f}, heuristic_bonus={h_bonus:.4f}, total={total:.4f}"
                )

        plays = [m for m in constrained_moves if isinstance(m, PlayCard)]
        if plays:
            fives = [m for m in plays if m.card.rank == 5]
            if fives:
                lines.append(f"Heuristic priority candidates (5s): {fmt_legal_moves(fives)}")
            revealed_target = state.dice_state.get_revealed_target(state.current_player)
            if revealed_target is not None:
                blocking = []
                rev_hand = state.players[revealed_target].hand
                for m in plays:
                    our_idx = Deck.RANK_INDEX[m.card.rank]
                    for opp_c in rev_hand:
                        if opp_c.suit == m.card.suit and abs(our_idx - Deck.RANK_INDEX[opp_c.rank]) == 1:
                            blocking.append(m)
                            break
                if blocking:
                    lines.append(f"Heuristic blocking candidates vs revealed P{revealed_target}: {fmt_legal_moves(blocking)}")

        lines.append(f"Chosen move: {fmt_move(chosen_move)}")
        return lines


# ============================================================================
# Markdown visualiser
# ============================================================================

class MarkdownDebugVisualizer:
    def __init__(
        self,
        agents: List[Agent],
        variant: VariantConfig,
        variant_name: str,
        output_file: Path,
        max_turns: int = 1000,
        show_all_hands: bool = True,
    ):
        self.agents = agents
        self.variant = variant
        self.variant_name = variant_name
        self.output_file = output_file
        self.max_turns = max_turns
        self.show_all_hands = show_all_hands
        self.tracer = AgentTracer()

    def run_match(self, run_index: int, seed: int) -> str:
        random.seed(seed)
        state = Rules.initialize_game(len(self.agents), self.variant)
        lines: List[str] = []

        lines.append(f"# Cinquillo agent debug trace")
        lines.append("")
        lines.append(f"- generated: `{datetime.now().isoformat(timespec='seconds')}`")
        lines.append(f"- run: `{run_index}`")
        lines.append(f"- seed: `{seed}`")
        lines.append(f"- variant: `{self.variant_name}`")
        lines.append(f"- agents: {', '.join(f'P{i}={a.name} ({a.__class__.__name__})' for i, a in enumerate(self.agents))}")
        lines.append("")
        lines.append("## Variant configuration")
        lines.append("")
        for item in variant_summary(self.variant, len(self.agents)):
            lines.append(f"- {item}")
        lines.append("")
        lines.append("## Initial state")
        lines.append("")
        lines.append(f"- current player: `P{state.current_player}`")
        lines.append(f"- board: `{board_summary(state)}`")
        lines.append(f"- dice state: `{dice_summary(state)}`")
        lines.append(f"- scores: `{scores_summary(state)}`")
        if self.show_all_hands:
            lines.append(f"- hands: `{hands_summary(state)}`")
        lines.append("")

        turn_count = 0
        while not Rules.is_terminal(state) and turn_count < self.max_turns:
            turn_count += 1
            pre_state = state.copy()
            current_player_idx = pre_state.current_player
            agent = self.agents[current_player_idx]
            legal_moves = Rules.get_legal_moves(pre_state)
            trace_snapshot = self.tracer.snapshot(agent, pre_state, legal_moves)
            chosen_move = agent.choose_move(pre_state, legal_moves)
            trace_lines = self.tracer.trace(agent, pre_state, legal_moves, chosen_move, snapshot=trace_snapshot)
            state = chosen_move.apply(state)
            post_state = state.copy()

            lines.append(f"## Turn {turn_count} — P{current_player_idx} ({agent.name})")
            lines.append("")
            lines.append("### Pre-state")
            lines.append("")
            lines.append(f"- turn number: `{pre_state.turn_number}`")
            lines.append(f"- board: `{board_summary(pre_state)}`")
            lines.append(f"- dice state: `{dice_summary(pre_state)}`")
            lines.append(f"- scores: `{scores_summary(pre_state)}`")
            lines.append(f"- current hand: `{fmt_cards(pre_state.players[current_player_idx].hand)}`")
            if self.show_all_hands:
                lines.append(f"- all hands: `{hands_summary(pre_state)}`")
            lines.append(f"- legal moves: `{fmt_legal_moves(legal_moves)}`")
            lines.append("")
            lines.append("### Agent reasoning")
            lines.append("")
            for item in trace_lines:
                lines.append(f"- {item}")
            lines.append("")
            lines.append("### Applied move")
            lines.append("")
            lines.append(f"- chosen move: `{fmt_move(chosen_move)}`")
            lines.append("")
            lines.append("### Post-state")
            lines.append("")
            lines.append(f"- next current player: `P{post_state.current_player}`")
            lines.append(f"- board: `{board_summary(post_state)}`")
            lines.append(f"- dice state: `{dice_summary(post_state)}`")
            lines.append(f"- scores: `{scores_summary(post_state)}`")
            if self.show_all_hands:
                lines.append(f"- all hands: `{hands_summary(post_state)}`")
            if post_state.game_over:
                lines.append(f"- game_over: `True`, winner=`P{post_state.winner}`")
            lines.append("")

        lines.append("## Final result")
        lines.append("")
        if Rules.is_terminal(state):
            Rules.compute_round_scores(state)
            lines.append(f"- terminal: `True`")
            lines.append(f"- winner: `P{state.winner}`")
        else:
            lines.append(f"- terminal: `False`")
            lines.append(f"- stopped because max_turns=`{self.max_turns}` was reached")
        lines.append(f"- final board: `{board_summary(state)}`")
        lines.append(f"- final dice state: `{dice_summary(state)}`")
        lines.append(f"- final scores: `{scores_summary(state)}`")
        if self.show_all_hands:
            lines.append(f"- final hands: `{hands_summary(state)}`")
        lines.append("")

        return "\n".join(lines)

    def run(self, runs: int, base_seed: int) -> Path:
        chunks = []
        for run_idx in range(1, runs + 1):
            seed = base_seed + run_idx - 1
            chunks.append(self.run_match(run_idx, seed))
        self.output_file.write_text("\n\n---\n\n".join(chunks), encoding="utf-8")
        return self.output_file


# ============================================================================
# CLI
# ============================================================================


def build_output_path(variant_name: str, output: Optional[str]) -> Path:
    if output:
        return Path(output)
    stem = slugify(variant_name) if variant_name else "baseline"
    return Path(f"debug_trace_{stem}.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Markdown debug trace for Cinquillo agents.")
    parser.add_argument("variant", nargs="?", default="Baseline", help="Variant name. Defaults to Baseline.")
    parser.add_argument("--agents", nargs=4, default=DEFAULT_AGENT_KEYS, help="Exactly 4 agent keys.")
    parser.add_argument("--runs", type=int, default=1, help="Number of simulations to append to the same Markdown file.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed. Run k uses seed+(k-1).")
    parser.add_argument("--max-turns", type=int, default=1000, help="Safety cap on number of turns per run.")
    parser.add_argument("--hide-all-hands", action="store_true", help="Only show the current player's hand each turn.")
    parser.add_argument("--output", type=str, default=None, help="Output Markdown path.")
    parser.add_argument("--list-variants", action="store_true", help="List available variants and exit.")
    parser.add_argument("--list-agents", action="store_true", help="List available agents and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_variants:
        for spec in VARIANT_SPECS:
            print(spec.name)
        return

    if args.list_agents:
        for key in sorted(AGENT_BUILDERS.keys()):
            print(key)
        return

    variant_key = slugify(args.variant)
    if variant_key not in VARIANT_REGISTRY:
        raise SystemExit(
            f"Unknown variant '{args.variant}'. Available: {', '.join(spec.name for spec in VARIANT_SPECS)}"
        )

    spec = VARIANT_REGISTRY[variant_key]
    variant = build_variant(spec)
    agents = build_agents(args.agents)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / "flows"
    output_dir.mkdir(parents=True, exist_ok=True)

    default_filename = f"{slugify(spec.name)}_{timestamp}.md"

    if args.output:
        candidate = Path(args.output)
        if candidate.exists() and candidate.is_dir():
            output_path = candidate / default_filename
        elif str(args.output).endswith("/") or str(args.output).endswith("\\"):
            candidate.mkdir(parents=True, exist_ok=True)
            output_path = candidate / default_filename
        else:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            output_path = candidate
    else:
        output_path = output_dir / default_filename

    visualizer = MarkdownDebugVisualizer(
        agents=agents,
        variant=variant,
        variant_name=spec.name,
        output_file=output_path,
        max_turns=args.max_turns,
        show_all_hands=not args.hide_all_hands,
    )
    saved = visualizer.run(runs=args.runs, base_seed=args.seed)
    print(f"✓ Markdown debug trace saved to: {saved}")

if __name__ == "__main__":
    main()
