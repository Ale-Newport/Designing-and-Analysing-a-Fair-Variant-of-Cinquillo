"""
Tests for agents/mcts_agent.py

Covers: MCTSNode (UCT, expansion, terminal detection),
        MCTSAgent (choose_move, rollout, evaluation),
        all MCTSAgent variants.
"""
import math
import random
import pytest

from game.entities import Card, Suit, VariantConfig, ScoringMode
from game.rules import Rules, PlayCard, RollDice, Pass
from agents.mcts_agent import (
    MCTSNode, MCTSAgent,
    MCTSAgentSuperFast, MCTSAgentFast,
    MCTSAgentStandard, MCTSAgentDeep,
)


# ══════════════════════════════════════════════
# MCTSNode
# ══════════════════════════════════════════════

class TestMCTSNode:

    def test_initial_visits_zero(self, fresh_state):
        node = MCTSNode(state=fresh_state)
        assert node.visits == 0

    def test_initial_total_reward_zero(self, fresh_state):
        node = MCTSNode(state=fresh_state)
        assert node.total_reward == 0.0

    def test_untried_moves_populated(self, state_with_5s_on_board):
        node = MCTSNode(state=state_with_5s_on_board)
        assert len(node.untried_moves) > 0

    def test_is_not_fully_expanded_initially(self, state_with_5s_on_board):
        node = MCTSNode(state=state_with_5s_on_board)
        assert not node.is_fully_expanded()

    def test_is_fully_expanded_when_no_untried(self, fresh_state):
        node = MCTSNode(state=fresh_state)
        node.untried_moves = []
        assert node.is_fully_expanded()

    def test_is_not_terminal_fresh(self, fresh_state):
        node = MCTSNode(state=fresh_state)
        assert not node.is_terminal()

    def test_is_terminal_after_game_over(self, fresh_state):
        fresh_state.game_over = True
        node = MCTSNode(state=fresh_state)
        assert node.is_terminal()

    def test_is_terminal_cached(self, fresh_state):
        """Terminal result should be cached after first call."""
        node = MCTSNode(state=fresh_state)
        result1 = node.is_terminal()
        result2 = node.is_terminal()
        assert result1 == result2

    def test_add_child_creates_child(self, state_with_5s_on_board):
        node = MCTSNode(state=state_with_5s_on_board)
        move = node.untried_moves[0]
        new_state = Rules.apply_move(state_with_5s_on_board.copy(), move)
        child = node.add_child(move, new_state)
        assert child in node.children
        assert child.parent is node

    def test_add_child_removes_from_untried(self, state_with_5s_on_board):
        node = MCTSNode(state=state_with_5s_on_board)
        move = node.untried_moves[0]
        before_count = len(node.untried_moves)
        new_state = Rules.apply_move(state_with_5s_on_board.copy(), move)
        node.add_child(move, new_state)
        assert len(node.untried_moves) == before_count - 1

    def test_best_child_returns_highest_uct(self, fresh_state):
        parent = MCTSNode(state=fresh_state)
        parent.visits = 10

        child_a = MCTSNode(state=fresh_state, parent=parent)
        child_a.visits = 5
        child_a.total_reward = 4.0   # exploitation = 0.8

        child_b = MCTSNode(state=fresh_state, parent=parent)
        child_b.visits = 1
        child_b.total_reward = 0.5   # exploitation = 0.5, but high exploration

        parent.children = [child_a, child_b]
        best = parent.best_child(exploration_weight=1.414)
        # child_b has lower exploitation but much higher exploration term
        # with 10 parent visits: exploration_b = 1.414 * sqrt(ln10/1) ≈ 3.26
        # vs exploration_a = 1.414 * sqrt(ln10/5) ≈ 1.46
        assert best is child_b

    def test_best_child_returns_none_no_children(self, fresh_state):
        node = MCTSNode(state=fresh_state)
        assert node.best_child() is None

    def test_most_visited_child(self, fresh_state):
        parent = MCTSNode(state=fresh_state)
        c1 = MCTSNode(state=fresh_state, parent=parent)
        c1.visits = 3
        c2 = MCTSNode(state=fresh_state, parent=parent)
        c2.visits = 10
        parent.children = [c1, c2]
        assert parent.most_visited_child() is c2

    def test_most_visited_child_none_when_no_children(self, fresh_state):
        node = MCTSNode(state=fresh_state)
        assert node.most_visited_child() is None


# ══════════════════════════════════════════════
# MCTSAgent – core behaviour
# ══════════════════════════════════════════════

class TestMCTSAgent:

    def _make_agent(self, iterations=50):
        return MCTSAgent(num_iterations=iterations)

    def test_returns_legal_move(self, state_with_5s_on_board):
        agent = self._make_agent()
        state = state_with_5s_on_board
        state.current_player = 0
        moves = Rules.get_legal_moves(state)
        random.seed(1)
        chosen = agent.choose_move(state, moves)
        assert chosen in moves

    def test_raises_when_no_moves(self, fresh_state):
        agent = self._make_agent()
        with pytest.raises(ValueError):
            agent.choose_move(fresh_state, [])

    def test_single_move_returned_immediately(self, fresh_state):
        agent = self._make_agent()
        moves = [Pass(voluntary=False)]
        chosen = agent.choose_move(fresh_state, moves)
        assert isinstance(chosen, Pass)

    def test_avoids_voluntary_pass(self, state_with_5s_on_board):
        """MCTS should filter out voluntary passes by default."""
        agent = self._make_agent()
        state = state_with_5s_on_board
        state.current_player = 0
        moves = Rules.get_legal_moves(state)
        card_moves = [m for m in moves if isinstance(m, PlayCard)]
        if not card_moves:
            pytest.skip("No card moves")
        random.seed(2)
        chosen = agent.choose_move(state, moves)
        assert not (isinstance(chosen, Pass) and chosen.voluntary)

    def test_rollout_returns_float_in_0_1(self, state_with_5s_on_board):
        agent = self._make_agent()
        state = state_with_5s_on_board
        random.seed(3)
        reward = agent._rollout(state.copy(), player_index=0)
        assert 0.0 <= reward <= 1.0

    def test_rollout_terminal_returns_valid(self, near_win_state):
        """Rollout from a near-terminal state should still return valid value."""
        agent = self._make_agent()
        state = near_win_state
        random.seed(4)
        reward = agent._rollout(state.copy(), player_index=0)
        assert 0.0 <= reward <= 1.0

    def test_backpropagate_updates_visits(self, fresh_state):
        agent = self._make_agent()
        root = MCTSNode(state=fresh_state)
        child = MCTSNode(state=fresh_state, parent=root)
        root.children = [child]
        agent._backpropagate(child, 1.0)
        assert child.visits == 1
        assert root.visits == 1

    def test_backpropagate_accumulates_reward(self, fresh_state):
        agent = self._make_agent()
        root = MCTSNode(state=fresh_state)
        child = MCTSNode(state=fresh_state, parent=root)
        root.children = [child]
        agent._backpropagate(child, 0.7)
        agent._backpropagate(child, 0.3)
        assert abs(child.total_reward - 1.0) < 1e-9

    def test_should_terminate_early_when_extreme(self, fresh_state):
        """Terminate early when someone is nearly done and others hold many cards."""
        agent = self._make_agent()
        fresh_state.players[0].hand = [Card(Suit.OROS, 5)]
        for i in range(1, 4):
            fresh_state.players[i].hand = [Card(Suit.OROS, r) for r in [1, 2, 3, 4, 6, 7, 10, 11]]
        assert agent._should_terminate_early(fresh_state)

    def test_should_not_terminate_early_balanced(self, fresh_state):
        """Should not terminate early when hands are balanced."""
        agent = self._make_agent()
        # All players have 5 cards each
        for p in fresh_state.players:
            p.hand = p.hand[:5]
        assert not agent._should_terminate_early(fresh_state)

    def test_fast_card_selection_prefers_5s(self, fresh_state):
        """_fast_card_selection should prefer playing 5s on empty suits."""
        agent = self._make_agent()
        fresh_state.current_player = 0
        hand = [
            Card(Suit.OROS, 5),    # playable (empty suit, is 5)
            Card(Suit.COPAS, 1),   # not a 5
        ]
        fresh_state.players[0].hand = hand
        moves = [PlayCard(hand[0]), PlayCard(hand[1])]
        # Filter to only truly legal moves
        legal = [m for m in moves if m.is_legal(fresh_state, 0)]
        if not legal:
            pytest.skip("No legal card moves")
        chosen = agent._fast_card_selection(fresh_state, legal)
        assert chosen.card.rank == 5

    def test_evaluate_terminal_state_winner_gets_high(self, near_win_state):
        """The player who won should receive a high evaluation."""
        agent = self._make_agent()
        state = near_win_state
        state.game_over = True
        state.winner = 0
        state.players[0].hand = []
        val = agent._evaluate_terminal_state(state, player_index=0)
        assert val > 0.5

    def test_evaluate_terminal_state_loser_gets_low(self, near_win_state):
        """A non-winner should receive a lower evaluation."""
        agent = self._make_agent()
        state = near_win_state
        state.game_over = True
        state.winner = 1
        state.players[1].hand = []
        state.players[0].hand = [Card(Suit.OROS, 1)] * 3
        val = agent._evaluate_terminal_state(state, player_index=0)
        # May not be below 0.5 in all scoring modes, but should be ≤ winner's val
        winner_val = agent._evaluate_terminal_state(state, player_index=1)
        assert val <= winner_val

    def test_mcts_iterations_tree_grows(self, state_with_5s_on_board):
        """After running MCTS, the root should have children."""
        agent = self._make_agent(iterations=20)
        state = state_with_5s_on_board
        state.current_player = 0
        moves = Rules.get_legal_moves(state)
        non_vol = [m for m in moves if not (isinstance(m, Pass) and m.voluntary)]
        root = MCTSNode(state=state.copy())
        root.untried_moves = non_vol
        random.seed(5)
        for _ in range(20):
            agent._mcts_iteration(root, state, 0)
        assert len(root.children) > 0


# ══════════════════════════════════════════════
# MCTSAgent variants
# ══════════════════════════════════════════════

class TestMCTSVariants:

    @pytest.mark.parametrize("AgentClass,expected_iter", [
        (MCTSAgentSuperFast, 100),
        (MCTSAgentFast, 500),
        (MCTSAgentStandard, 1000),
        (MCTSAgentDeep, 2000),
    ])
    def test_iteration_count(self, AgentClass, expected_iter):
        agent = AgentClass()
        assert agent.num_iterations == expected_iter

    @pytest.mark.parametrize("AgentClass", [
        MCTSAgentSuperFast, MCTSAgentFast,
    ])
    def test_variant_returns_legal_move(self, AgentClass, state_with_5s_on_board):
        agent = AgentClass()
        state = state_with_5s_on_board
        state.current_player = 0
        moves = Rules.get_legal_moves(state)
        random.seed(6)
        chosen = agent.choose_move(state, moves)
        assert chosen in moves

    def test_super_fast_name(self):
        assert MCTSAgentSuperFast().name == "MCTS-SuperFast"

    def test_fast_name(self):
        assert MCTSAgentFast().name == "MCTS-Fast"

    def test_standard_name(self):
        assert MCTSAgentStandard().name == "MCTS"

    def test_deep_name(self):
        assert MCTSAgentDeep().name == "MCTS-Deep"


# ══════════════════════════════════════════════
# Full game with MCTS (smoke test)
# ══════════════════════════════════════════════

class TestMCTSFullGame:

    def test_super_fast_completes_game(self, default_variant):
        """MCTSAgentSuperFast must complete a 4-player game without crashing."""
        random.seed(77)
        state = Rules.initialize_game(4, default_variant)
        agents = [MCTSAgentSuperFast() for _ in range(4)]
        max_turns = 2000
        turns = 0
        while not Rules.is_terminal(state) and turns < max_turns:
            moves = Rules.get_legal_moves(state)
            move = agents[state.current_player].choose_move(state, moves)
            assert move in moves
            state = Rules.apply_move(state, move)
            turns += 1
        assert Rules.is_terminal(state)

    def test_mcts_vs_random_winner_exists(self, default_variant):
        """Mixed MCTS/Random game must produce a winner."""
        from agents.base_agents import RandomAgent
        random.seed(88)
        state = Rules.initialize_game(4, default_variant)
        agents = [MCTSAgentSuperFast(), RandomAgent(), RandomAgent(), RandomAgent()]
        max_turns = 2000
        turns = 0
        while not Rules.is_terminal(state) and turns < max_turns:
            moves = Rules.get_legal_moves(state)
            move = agents[state.current_player].choose_move(state, moves)
            state = Rules.apply_move(state, move)
            turns += 1
        assert state.winner is not None
