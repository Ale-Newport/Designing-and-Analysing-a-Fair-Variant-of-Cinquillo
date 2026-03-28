"""
Tests for agents/rl_agent.py

Covers: StateEncoder (feature vector structure & correctness),
        ImprovedQNetwork (forward pass, weight update, gradient clipping),
        RLAgent (choose_move, reward computation, experience replay,
                 epsilon decay, save/load),
        RLAgent variants.
"""
import os
import random
import tempfile
import numpy as np
import pytest

from game.entities import (
    Card, Suit, VariantConfig, DiceState,
    ScoringMode, GoodDiceEffect, BadDiceEffect, MatchEndMode,
)
from game.rules import Rules, PlayCard, RollDice, Pass
from agents.rl_agent import (
    StateEncoder, ImprovedQNetwork, RLAgent,
    RLAgentExplore, RLAgentExploit, RLAgentPure,
)


# ══════════════════════════════════════════════
# StateEncoder
# ══════════════════════════════════════════════

class TestStateEncoder:

    def test_output_is_ndarray(self, fresh_state):
        vec = StateEncoder.encode(fresh_state, 0)
        assert isinstance(vec, np.ndarray)

    def test_output_dtype_float32(self, fresh_state):
        vec = StateEncoder.encode(fresh_state, 0)
        assert vec.dtype == np.float32

    def test_output_length_consistent(self, fresh_state):
        """Feature vector length must be the same across two calls."""
        v1 = StateEncoder.encode(fresh_state, 0)
        v2 = StateEncoder.encode(fresh_state, 1)
        assert len(v1) == len(v2)

    def test_hand_encoding_changes_with_hand(self, fresh_state):
        """Different hands should produce different feature vectors."""
        v1 = StateEncoder.encode(fresh_state, 0)
        # Modify player 0's hand
        fresh_state.players[0].hand = fresh_state.players[0].hand[:3]
        v2 = StateEncoder.encode(fresh_state, 0)
        assert not np.array_equal(v1, v2)

    def test_all_values_finite(self, fresh_state):
        vec = StateEncoder.encode(fresh_state, 0)
        assert np.all(np.isfinite(vec))

    def test_hand_encoding_binary(self, fresh_state):
        """The first 40 features (hand encoding) must be binary."""
        vec = StateEncoder.encode(fresh_state, 0)
        hand_feats = vec[:40]
        assert np.all((hand_feats == 0.0) | (hand_feats == 1.0))

    def test_board_encoding_binary(self, fresh_state):
        """Board encoding (features 40-79) must be binary."""
        vec = StateEncoder.encode(fresh_state, 0)
        board_feats = vec[40:80]
        assert np.all((board_feats == 0.0) | (board_feats == 1.0))

    def test_board_encoding_reflects_board(self, state_with_5s_on_board):
        """After placing 5s, board encoding must show them as 1."""
        vec = StateEncoder.encode(state_with_5s_on_board, 0)
        # Feature 44 = OROS * 10 + rank_idx(5) = 0*10 + 4 = 4 (in 40-79 range → offset 40)
        board_feats = vec[40:80]
        # At least one board feature should be 1 (5s are placed)
        assert board_feats.sum() >= 4

    def test_playable_encoding_binary(self, fresh_state):
        """Playable cards encoding (features 80-119) must be binary."""
        vec = StateEncoder.encode(fresh_state, 0)
        playable_feats = vec[80:120]
        assert np.all((playable_feats == 0.0) | (playable_feats == 1.0))

    def test_wild_active_reflected(self, fresh_state):
        """Wild active state must change the feature vector."""
        v1 = StateEncoder.encode(fresh_state, 0)
        fresh_state.dice_state.wild_active = True
        v2 = StateEncoder.encode(fresh_state, 0)
        assert not np.array_equal(v1, v2)

    def test_variant_features_present(self, fresh_state):
        """Variant configuration features must be non-trivially present."""
        fresh_state.variant = VariantConfig(
            scoring_mode=ScoringMode.WINNER_TAKES_ALL,
            dice_good_effect=GoodDiceEffect.DOUBLE_PLAY,
        )
        vec = StateEncoder.encode(fresh_state, 0)
        assert np.any(vec != 0)

    def test_encode_cards_all_zeros_for_empty(self):
        enc = StateEncoder._encode_cards([])
        assert all(v == 0.0 for v in enc)
        assert len(enc) == 40

    def test_encode_cards_marks_correct_index(self):
        # OROS=0, rank 5 → idx 4
        enc = StateEncoder._encode_cards([Card(Suit.OROS, 5)])
        assert enc[4] == 1.0
        assert sum(enc) == 1.0

    def test_encode_board_all_zeros_initial(self, empty_board):
        enc = StateEncoder._encode_board(empty_board)
        assert all(v == 0.0 for v in enc)

    def test_encode_board_marks_card(self, empty_board):
        empty_board.add_card(Card(Suit.COPAS, 10))
        enc = StateEncoder._encode_board(empty_board)
        # COPAS=1, rank 10 → rank_idx 7 → idx = 1*10+7 = 17
        assert enc[17] == 1.0

    def test_extension_cards_list(self, state_with_5s_on_board):
        state = state_with_5s_on_board
        player = state.players[0]
        extensions = StateEncoder._get_extension_cards(state, player)
        # All returned cards must be adjacent on the board and non-5
        for card in extensions:
            assert card.rank != 5
            assert state.board.is_adjacent(card)


# ══════════════════════════════════════════════
# ImprovedQNetwork
# ══════════════════════════════════════════════

class TestImprovedQNetwork:

    def _make_net(self, state_dim=50):
        return ImprovedQNetwork(state_dim=state_dim, action_dim=3, hidden_dims=[32, 16])

    def test_forward_output_shape(self):
        net = self._make_net()
        x = np.random.randn(50).astype(np.float32)
        q_values, activations = net.forward(x)
        assert q_values.shape == (3,)

    def test_activations_length(self):
        net = self._make_net()
        x = np.random.randn(50).astype(np.float32)
        _, activations = net.forward(x)
        # One entry per layer input: input + 2 hidden
        assert len(activations) == 3  # input + 2 hidden layers

    def test_q_values_clipped(self):
        """Q-values must be clipped to [-100, 100]."""
        net = self._make_net()
        # Set huge weights to force extreme output
        for w in net.weights:
            w[:] = 1000.0
        for b in net.biases:
            b[:] = 1000.0
        x = np.ones(50, dtype=np.float32)
        q_values, _ = net.forward(x)
        assert np.all(q_values <= 100.0)
        assert np.all(q_values >= -100.0)

    def test_update_changes_weights(self):
        np.random.seed(42)
        net = self._make_net()
        x = np.random.randn(50).astype(np.float32)
        old_w = [w.copy() for w in net.weights]
        net.update(x, action_idx=0, target=5.0, lr=0.01)
        changed = any(not np.allclose(w, old) for w, old in zip(net.weights, old_w))
        assert changed

    def test_update_with_zero_lr_no_change(self):
        np.random.seed(1)
        net = self._make_net()
        x = np.random.randn(50).astype(np.float32)
        old_w = [w.copy() for w in net.weights]
        net.update(x, action_idx=1, target=0.0, lr=0.0)
        for w, old in zip(net.weights, old_w):
            assert np.allclose(w, old)

    def test_weights_clipped_after_update(self):
        """Weights must never exceed [-10, 10]."""
        np.random.seed(2)
        net = self._make_net()
        for w in net.weights:
            w[:] = 9.9
        x = np.ones(50, dtype=np.float32)
        for _ in range(10):
            net.update(x, action_idx=0, target=100.0, lr=1.0)
        for w in net.weights:
            assert np.all(w <= 10.0)
            assert np.all(w >= -10.0)

    def test_xavier_init_reasonable_scale(self):
        """Xavier-initialised weights should have small mean values."""
        net = ImprovedQNetwork(state_dim=100, action_dim=3, hidden_dims=[64, 32])
        for w in net.weights:
            assert abs(np.mean(w)) < 0.5
            assert np.std(w) < 0.5


# ══════════════════════════════════════════════
# RLAgent
# ══════════════════════════════════════════════

class TestRLAgent:

    def test_choose_move_returns_legal(self, state_with_5s_on_board):
        agent = RLAgent(epsilon=0.0)   # pure exploitation
        state = state_with_5s_on_board
        state.current_player = 0
        moves = Rules.get_legal_moves(state)
        random.seed(1)
        chosen = agent.choose_move(state, moves)
        assert chosen in moves

    def test_choose_move_explore_returns_legal(self, state_with_5s_on_board):
        agent = RLAgent(epsilon=1.0)   # pure exploration
        state = state_with_5s_on_board
        state.current_player = 0
        moves = Rules.get_legal_moves(state)
        random.seed(2)
        chosen = agent.choose_move(state, moves)
        assert chosen in moves

    def test_raises_when_no_moves(self, fresh_state):
        agent = RLAgent()
        with pytest.raises(ValueError):
            agent.choose_move(fresh_state, [])

    def test_single_move_returned_directly(self, fresh_state):
        agent = RLAgent()
        moves = [Pass(voluntary=False)]
        chosen = agent.choose_move(fresh_state, moves)
        assert isinstance(chosen, Pass)

    def test_network_initialised_after_first_call(self, fresh_state):
        agent = RLAgent()
        assert agent.q_network is None
        fresh_state.current_player = 0
        moves = Rules.get_legal_moves(fresh_state)
        agent.choose_move(fresh_state, moves)
        assert agent.q_network is not None

    def test_heuristic_move_prefers_5s(self, fresh_state):
        """_heuristic_move must prefer 5s when available."""
        agent = RLAgent(use_heuristics=True)
        fresh_state.current_player = 0
        fresh_state.players[0].hand = [Card(Suit.OROS, 5), Card(Suit.COPAS, 1)]
        moves = [PlayCard(Card(Suit.OROS, 5))]   # only 5O is legal
        chosen = agent._heuristic_move(fresh_state, moves)
        assert chosen.card.rank == 5

    def test_heuristic_value_pass_is_lowest(self, fresh_state):
        agent = RLAgent()
        fresh_state.current_player = 0
        val_pass = agent._heuristic_value(fresh_state, Pass(voluntary=True))
        val_play = agent._heuristic_value(fresh_state, PlayCard(Card(Suit.OROS, 5)))
        val_roll = agent._heuristic_value(fresh_state, RollDice())
        assert val_pass < val_play
        assert val_pass < val_roll

    def test_card_value_adjustment_5_bonus(self, fresh_state):
        """Playing a 5 early should give a positive adjustment."""
        agent = RLAgent()
        fresh_state.current_player = 0
        # Board is empty → few cards played
        adj = agent._card_value_adjustment(fresh_state, Card(Suit.OROS, 5))
        assert adj > 0

    def test_card_value_adjustment_high_rank_penalty(self, fresh_state):
        agent = RLAgent()
        fresh_state.current_player = 0
        adj_12 = agent._card_value_adjustment(fresh_state, Card(Suit.OROS, 12))
        adj_3 = agent._card_value_adjustment(fresh_state, Card(Suit.OROS, 3))
        assert adj_12 < adj_3

    # ── compute_reward ────────────────────────

    def test_reward_for_playing_card(self, state_with_5s_on_board):
        agent = RLAgent()
        state = state_with_5s_on_board
        state.current_player = 0
        for card in state.players[0].hand:
            if state.board.is_adjacent(card):
                move = PlayCard(card)
                next_state = Rules.apply_move(state.copy(), move)
                reward = agent.compute_reward(state, move, next_state, 0)
                assert reward > 0
                return
        pytest.skip("No adjacent card found")

    def test_reward_penalty_for_voluntary_pass(self, state_with_5s_on_board):
        agent = RLAgent()
        state = state_with_5s_on_board
        state.current_player = 0
        move = Pass(voluntary=True)
        next_state = Rules.apply_move(state.copy(), move)
        reward = agent.compute_reward(state, move, next_state, 0)
        assert reward < 0

    def test_reward_large_bonus_for_winning(self, near_win_state):
        agent = RLAgent()
        state = near_win_state
        state.current_player = 0
        card = state.players[0].hand[0]
        move = PlayCard(card)
        next_state = Rules.apply_move(state.copy(), move)
        reward = agent.compute_reward(state, move, next_state, 0)
        assert reward > 10   # winning should give large reward

    # ── experience replay ─────────────────────

    def test_store_experience_adds_to_buffer(self, state_with_5s_on_board):
        agent = RLAgent()
        state = state_with_5s_on_board
        state.current_player = 0
        # Initialise network
        moves = Rules.get_legal_moves(state)
        agent.choose_move(state, moves)
        move = Pass(voluntary=False)
        next_state = Rules.apply_move(state.copy(), move)
        before = len(agent.replay_buffer)
        agent.store_experience(state, move, 0.5, next_state, False)
        assert len(agent.replay_buffer) == before + 1

    def test_train_from_replay_no_crash_when_buffer_small(self, fresh_state):
        agent = RLAgent()
        # Should silently do nothing when buffer is too small
        agent.train_from_replay(batch_size=64)

    def test_train_from_replay_updates_network(self, state_with_5s_on_board):
        agent = RLAgent(epsilon=0.0)
        state = state_with_5s_on_board
        state.current_player = 0
        # Initialise network
        moves = Rules.get_legal_moves(state)
        agent.choose_move(state, moves)
        # Fill buffer
        for _ in range(64):
            move = Pass(voluntary=False)
            ns = Rules.apply_move(state.copy(), move)
            agent.store_experience(state, move, 0.1, ns, False)
        old_weights = [w.copy() for w in agent.q_network.weights]
        agent.train_from_replay(batch_size=32)
        changed = any(
            not np.allclose(w, old)
            for w, old in zip(agent.q_network.weights, old_weights)
        )
        assert changed

    # ── epsilon decay ─────────────────────────

    def test_epsilon_decay(self):
        agent = RLAgent(epsilon=0.5)
        agent.decay_epsilon(decay_rate=0.9, min_epsilon=0.01)
        assert abs(agent.epsilon - 0.45) < 1e-6

    def test_epsilon_not_below_min(self):
        agent = RLAgent(epsilon=0.05)
        agent.decay_epsilon(decay_rate=0.1, min_epsilon=0.05)
        assert agent.epsilon >= 0.05

    # ── end_episode ───────────────────────────

    def test_end_episode_resets_counters(self):
        agent = RLAgent()
        agent.current_episode_reward = 42.0
        agent.episode_step_count = 10
        agent.end_episode()
        assert agent.current_episode_reward == 0.0
        assert agent.episode_step_count == 0

    def test_end_episode_stores_avg_reward(self):
        agent = RLAgent()
        agent.current_episode_reward = 30.0
        agent.episode_step_count = 10
        agent.end_episode()
        assert len(agent.episode_rewards) == 1
        assert abs(agent.episode_rewards[0] - 3.0) < 1e-6

    # ── save / load weights ───────────────────

    def test_save_load_roundtrip(self, state_with_5s_on_board):
        agent = RLAgent(epsilon=0.0)
        state = state_with_5s_on_board
        state.current_player = 0
        moves = Rules.get_legal_moves(state)
        # Initialise network by calling choose_move
        agent.choose_move(state, moves)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "weights.pkl")
            agent.save_weights(path)
            assert os.path.exists(path)

            # Load into a fresh agent
            agent2 = RLAgent()
            agent2.load_weights(path)
            assert agent2.q_network is not None
            # Weights should match
            for w1, w2 in zip(agent.q_network.weights, agent2.q_network.weights):
                assert np.allclose(w1, w2)

    def test_load_nonexistent_file_no_crash(self):
        agent = RLAgent()
        agent.load_weights("/tmp/nonexistent_rl_weights_12345.pkl")
        assert agent.q_network is None   # should stay None, not crash

    def test_save_without_network_no_crash(self):
        agent = RLAgent()
        with tempfile.TemporaryDirectory() as tmpdir:
            agent.save_weights(os.path.join(tmpdir, "w.pkl"))  # should just print warning


# ══════════════════════════════════════════════
# RLAgent variants
# ══════════════════════════════════════════════

class TestRLVariants:

    def test_explore_high_epsilon(self):
        assert RLAgentExplore().epsilon >= 0.2

    def test_exploit_low_epsilon(self):
        assert RLAgentExploit().epsilon <= 0.1

    def test_pure_no_heuristics(self):
        assert not RLAgentPure().use_heuristics

    def test_explore_name(self):
        assert RLAgentExplore().name == "RL-Explore"

    def test_exploit_name(self):
        assert RLAgentExploit().name == "RL-Exploit"

    def test_pure_name(self):
        assert RLAgentPure().name == "RL-Pure"

    @pytest.mark.parametrize("AgentClass", [RLAgentExplore, RLAgentExploit, RLAgentPure])
    def test_variant_returns_legal_move(self, AgentClass, state_with_5s_on_board):
        agent = AgentClass()
        state = state_with_5s_on_board
        state.current_player = 0
        moves = Rules.get_legal_moves(state)
        random.seed(9)
        chosen = agent.choose_move(state, moves)
        assert chosen in moves


# ══════════════════════════════════════════════
# Full game smoke test
# ══════════════════════════════════════════════

class TestRLFullGame:

    def test_rl_agent_completes_game(self, default_variant):
        random.seed(55)
        np.random.seed(55)
        state = Rules.initialize_game(4, default_variant)
        agents = [RLAgent(epsilon=0.5) for _ in range(4)]
        max_turns = 2000
        turns = 0
        while not Rules.is_terminal(state) and turns < max_turns:
            moves = Rules.get_legal_moves(state)
            agent = agents[state.current_player]
            move = agent.choose_move(state, moves)
            assert move in moves
            state = Rules.apply_move(state, move)
            turns += 1
        assert Rules.is_terminal(state)
