"""
Tests for agents/base_agents.py

Covers: Agent ABC contract, RandomAgent, HeuristicAgent,
        factory functions, and edge-case move selection.
"""
import pytest
import random
from game.entities import Card, Suit, VariantConfig, ScoringMode
from game.rules import Rules, PlayCard, RollDice, Pass
from agents.base_agents import (
    RandomAgent, HeuristicAgent,
    create_aggressive_heuristic,
    create_defensive_heuristic,
    create_balanced_heuristic,
    create_risky_heuristic,
)


# ══════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════

def _get_legal(state):
    return Rules.get_legal_moves(state)


# ══════════════════════════════════════════════
# RandomAgent
# ══════════════════════════════════════════════

class TestRandomAgent:

    def test_name(self):
        assert RandomAgent("Bob").name == "Bob"

    def test_str(self):
        assert "RandomAgent" in str(RandomAgent("Bob"))

    def test_returns_legal_move(self, state_with_5s_on_board):
        agent = RandomAgent()
        moves = _get_legal(state_with_5s_on_board)
        chosen = agent.choose_move(state_with_5s_on_board, moves)
        assert chosen in moves

    def test_raises_when_no_moves(self, fresh_state):
        with pytest.raises(ValueError):
            RandomAgent().choose_move(fresh_state, [])

    def test_single_move_returned(self, fresh_state):
        moves = [Pass(voluntary=False)]
        chosen = RandomAgent().choose_move(fresh_state, moves)
        assert isinstance(chosen, Pass)

    def test_avoid_bad_moves_prefers_non_pass(self, state_with_5s_on_board):
        """With avoid_bad_moves=True the agent should prefer non-Pass moves."""
        agent = RandomAgent(avoid_bad_moves=True)
        state = state_with_5s_on_board
        state.current_player = 0
        moves = _get_legal(state)
        card_moves = [m for m in moves if isinstance(m, PlayCard)]
        if not card_moves:
            pytest.skip("No card moves available")
        # Run several times and verify no voluntary pass is chosen
        for _ in range(20):
            chosen = agent.choose_move(state, moves)
            if card_moves:
                assert not (isinstance(chosen, Pass) and chosen.voluntary)

    def test_avoid_bad_moves_falls_back_to_pass(self, fresh_state):
        """When only a Pass is available, it must still be returned."""
        agent = RandomAgent(avoid_bad_moves=True)
        moves = [Pass(voluntary=False)]
        chosen = agent.choose_move(fresh_state, moves)
        assert isinstance(chosen, Pass)

    def test_deterministic_with_seed(self, state_with_5s_on_board):
        moves = _get_legal(state_with_5s_on_board)
        random.seed(42)
        a = RandomAgent().choose_move(state_with_5s_on_board, moves)
        random.seed(42)
        b = RandomAgent().choose_move(state_with_5s_on_board, moves)
        assert a == b


# ══════════════════════════════════════════════
# HeuristicAgent
# ══════════════════════════════════════════════

class TestHeuristicAgent:

    def test_returns_legal_move(self, state_with_5s_on_board):
        agent = HeuristicAgent()
        moves = _get_legal(state_with_5s_on_board)
        chosen = agent.choose_move(state_with_5s_on_board, moves)
        assert chosen in moves

    def test_raises_when_no_moves(self, fresh_state):
        with pytest.raises(ValueError):
            HeuristicAgent().choose_move(fresh_state, [])

    def test_single_move_no_crash(self, fresh_state):
        moves = [Pass(voluntary=False)]
        chosen = HeuristicAgent().choose_move(fresh_state, moves)
        assert isinstance(chosen, Pass)

    def test_prefers_playing_over_passing(self, state_with_5s_on_board):
        """HeuristicAgent should strongly prefer playing a card over passing."""
        agent = HeuristicAgent(avoid_voluntary_pass=10.0)
        state = state_with_5s_on_board
        state.current_player = 0
        moves = _get_legal(state)
        card_moves = [m for m in moves if isinstance(m, PlayCard)]
        if not card_moves:
            pytest.skip("No card moves")
        chosen = agent.choose_move(state, moves)
        assert isinstance(chosen, PlayCard)

    def test_avoids_voluntary_pass_when_cards_available(self, state_with_5s_on_board):
        agent = HeuristicAgent(avoid_voluntary_pass=100.0)
        state = state_with_5s_on_board
        state.current_player = 0
        moves = _get_legal(state)
        card_moves = [m for m in moves if isinstance(m, PlayCard)]
        if not card_moves:
            pytest.skip("No card moves")
        chosen = agent.choose_move(state, moves)
        assert not (isinstance(chosen, Pass) and chosen.voluntary)

    def test_evaluate_move_play_card_positive(self, state_with_5s_on_board):
        agent = HeuristicAgent()
        state = state_with_5s_on_board
        state.current_player = 0
        for card in state.players[0].hand:
            if state.board.is_adjacent(card):
                score = agent._evaluate_move(state, PlayCard(card))
                assert score > 0
                return
        pytest.skip("No adjacent card")

    def test_evaluate_move_voluntary_pass_negative(self, state_with_5s_on_board):
        agent = HeuristicAgent(avoid_voluntary_pass=2.0)
        state = state_with_5s_on_board
        state.current_player = 0
        score = agent._evaluate_move(state, Pass(voluntary=True))
        assert score < 0

    def test_compute_suit_balance_empty_hand(self):
        agent = HeuristicAgent()
        assert agent._compute_suit_balance([]) == 1.0

    def test_compute_suit_balance_single_suit(self):
        agent = HeuristicAgent()
        hand = [Card(Suit.OROS, r) for r in [1, 2, 3, 4]]
        balance = agent._compute_suit_balance(hand)
        # All cards in one suit → worst balance
        assert balance < 0.5

    def test_compute_suit_balance_four_suits(self):
        agent = HeuristicAgent()
        hand = [Card(s, 1) for s in Suit]
        balance = agent._compute_suit_balance(hand)
        # Perfectly balanced
        assert balance > 0.9

    def test_compute_blocking_value_range(self):
        agent = HeuristicAgent()
        for rank in [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]:
            card = Card(Suit.OROS, rank)
            val = agent._compute_blocking_value(None, card)
            assert 0.0 <= val <= 1.0


# ══════════════════════════════════════════════
# Factory functions
# ══════════════════════════════════════════════

class TestFactories:

    def test_aggressive_name(self):
        assert create_aggressive_heuristic().name == "Aggressive"

    def test_defensive_name(self):
        assert create_defensive_heuristic().name == "Defensive"

    def test_balanced_name(self):
        assert create_balanced_heuristic().name == "Balanced"

    def test_risky_name(self):
        assert create_risky_heuristic().name == "Risky"

    def test_aggressive_higher_dice_than_defensive(self):
        agg = create_aggressive_heuristic()
        defs = create_defensive_heuristic()
        assert agg.dice_risk_tolerance > defs.dice_risk_tolerance

    def test_aggressive_higher_pass_penalty(self):
        agg = create_aggressive_heuristic()
        defs = create_defensive_heuristic()
        assert agg.avoid_voluntary_pass >= defs.avoid_voluntary_pass

    def test_all_factories_return_legal_move(self, state_with_5s_on_board):
        state = state_with_5s_on_board
        state.current_player = 0
        moves = _get_legal(state)
        for factory in [
            create_aggressive_heuristic,
            create_defensive_heuristic,
            create_balanced_heuristic,
            create_risky_heuristic,
        ]:
            agent = factory()
            chosen = agent.choose_move(state, moves)
            assert chosen in moves, f"{agent.name} returned illegal move"


# ══════════════════════════════════════════════
# Full-game smoke test
# ══════════════════════════════════════════════

class TestAgentFullGame:

    @pytest.mark.parametrize("AgentFactory,kwargs", [
        (RandomAgent, {}),
        (HeuristicAgent, {}),
        (lambda **kw: create_aggressive_heuristic(), {}),
    ])
    def test_agent_completes_game(self, AgentFactory, kwargs, default_variant):
        """Each agent type must be able to complete a full game without error."""
        random.seed(123)
        state = Rules.initialize_game(4, default_variant)
        agents = [AgentFactory(**kwargs) for _ in range(4)]
        max_turns = 2000
        turns = 0
        while not Rules.is_terminal(state) and turns < max_turns:
            moves = Rules.get_legal_moves(state)
            agent = agents[state.current_player]
            move = agent.choose_move(state, moves)
            assert move in moves
            state = Rules.apply_move(state, move)
            turns += 1
        assert Rules.is_terminal(state), "Game did not complete"
