"""
Tests for game/rules.py

Covers: PlayCard, RollDice, Pass (legality & apply),
        Rules (get_legal_moves, apply_move, compute_round_scores, initialize_game).
"""
import random
import pytest
from game.entities import (
    Card, Player, Board, GameState, VariantConfig, DiceState,
    Suit, ScoringMode, GoodDiceEffect, BadDiceEffect, MatchEndMode,
)
from game.rules import Rules, PlayCard, RollDice, Pass


# ══════════════════════════════════════════════
# PlayCard
# ══════════════════════════════════════════════

class TestPlayCard:

    # ── legality ──────────────────────────────

    def test_play_5_on_empty_suit_is_legal(self, fresh_state):
        """Playing a 5 on an empty suit must always be legal."""
        # Find a player who holds a 5
        for p_idx, p in enumerate(fresh_state.players):
            for card in p.hand:
                if card.rank == 5:
                    fresh_state.current_player = p_idx
                    move = PlayCard(card)
                    assert move.is_legal(fresh_state, p_idx)
                    return
        pytest.skip("No 5 found in any hand (unexpected)")

    def test_play_non_5_on_empty_suit_is_illegal(self, fresh_state):
        """Playing a non-5 when suit is not on board must be illegal."""
        p_idx = 0
        fresh_state.current_player = p_idx
        player = fresh_state.players[p_idx]
        for card in player.hand:
            if card.rank != 5 and fresh_state.board.is_empty(card.suit):
                move = PlayCard(card)
                assert not move.is_legal(fresh_state, p_idx)
                return
        pytest.skip("No non-5 card on empty suit found")

    def test_play_adjacent_card_is_legal(self, state_with_5s_on_board):
        state = state_with_5s_on_board
        p_idx = 0
        state.current_player = p_idx
        player = state.players[p_idx]
        for card in player.hand:
            if card.rank != 5 and state.board.is_adjacent(card):
                move = PlayCard(card)
                assert move.is_legal(state, p_idx)
                return
        pytest.skip("No adjacent card found")

    def test_play_card_not_in_hand_is_illegal(self, fresh_state):
        p_idx = 0
        fresh_state.current_player = p_idx
        ghost = Card(Suit.OROS, 5)
        # Ensure ghost is NOT in hand
        if ghost in fresh_state.players[p_idx].hand:
            fresh_state.players[p_idx].hand.remove(ghost)
        move = PlayCard(ghost)
        assert not move.is_legal(fresh_state, p_idx)

    def test_play_duplicate_card_is_illegal(self, state_with_5s_on_board):
        """A 5 already on the board cannot be played again."""
        state = state_with_5s_on_board
        p_idx = 0
        state.current_player = p_idx
        # Force player 0 to hold a 5O despite it being on board
        state.players[p_idx].hand.append(Card(Suit.OROS, 5))
        move = PlayCard(Card(Suit.OROS, 5))
        assert not move.is_legal(state, p_idx)

    def test_wild_allows_any_card(self, fresh_state):
        """With wild active, any card in hand should be legal."""
        fresh_state.dice_state.wild_active = True
        p_idx = 0
        fresh_state.current_player = p_idx
        card = fresh_state.players[p_idx].hand[0]
        assert PlayCard(card).is_legal(fresh_state, p_idx)

    # ── apply ─────────────────────────────────

    def test_apply_removes_card_from_hand(self, state_with_5s_on_board):
        state = state_with_5s_on_board
        p_idx = 0
        state.current_player = p_idx
        player = state.players[p_idx]
        for card in player.hand:
            if state.board.is_adjacent(card) and card.rank != 5:
                before = len(player.hand)
                new_state = PlayCard(card).apply(state)
                assert len(new_state.players[p_idx].hand) == before - 1
                return
        pytest.skip("No adjacent card found")

    def test_apply_adds_card_to_board(self, state_with_5s_on_board):
        state = state_with_5s_on_board
        p_idx = 0
        state.current_player = p_idx
        for card in state.players[p_idx].hand:
            if state.board.is_adjacent(card):
                new_state = PlayCard(card).apply(state)
                assert new_state.board.has_rank(card.suit, card.rank)
                return
        pytest.skip("No adjacent card found")

    def test_apply_advances_turn(self, state_with_5s_on_board):
        state = state_with_5s_on_board
        p_idx = 0
        state.current_player = p_idx
        for card in state.players[p_idx].hand:
            if state.board.is_adjacent(card):
                new_state = PlayCard(card).apply(state)
                assert new_state.current_player == 1
                return
        pytest.skip("No adjacent card found")

    def test_apply_sets_game_over_when_hand_empty(self, near_win_state):
        state = near_win_state
        state.current_player = 0
        card = state.players[0].hand[0]   # only card left
        new_state = PlayCard(card).apply(state)
        assert new_state.game_over
        assert new_state.winner == 0

    def test_apply_clears_wild(self, state_with_5s_on_board):
        state = state_with_5s_on_board
        state.dice_state.wild_active = True
        p_idx = 0
        state.current_player = p_idx
        card = state.players[p_idx].hand[0]
        new_state = PlayCard(card).apply(state)
        assert not new_state.dice_state.wild_active

    def test_str(self):
        assert "PlayCard" in str(PlayCard(Card(Suit.OROS, 5)))


# ══════════════════════════════════════════════
# RollDice
# ══════════════════════════════════════════════

class TestRollDice:

    def test_legal_only_when_playable_cards_exist(self, state_with_5s_on_board):
        state = state_with_5s_on_board
        p_idx = 0
        state.current_player = p_idx
        roll = RollDice()
        # Player should have playable cards after 5s are on board
        has_playable = any(
            PlayCard(c).is_legal(state, p_idx)
            for c in state.players[p_idx].hand
        )
        assert roll.is_legal(state, p_idx) == has_playable

    def test_illegal_when_no_playable_cards(self, fresh_state):
        """If player has no playable cards, RollDice is illegal."""
        p_idx = 0
        fresh_state.current_player = p_idx
        # Replace hand with unplayable cards (non-5s on empty suits)
        fresh_state.players[p_idx].hand = [Card(Suit.OROS, 1)]
        roll = RollDice()
        assert not roll.is_legal(fresh_state, p_idx)

    def test_apply_good_wild(self, state_with_5s_on_board):
        """Good outcome WILD should set wild_active."""
        state = state_with_5s_on_board
        state.variant = VariantConfig(
            dice_good_probability=1.0,   # always good
            dice_good_effect=GoodDiceEffect.WILD,
        )
        random.seed(0)
        new_state = RollDice().apply(state)
        assert new_state.dice_state.wild_active

    def test_apply_good_double_play(self, state_with_5s_on_board):
        state = state_with_5s_on_board
        state.variant = VariantConfig(
            dice_good_probability=1.0,
            dice_good_effect=GoodDiceEffect.DOUBLE_PLAY,
        )
        random.seed(0)
        new_state = RollDice().apply(state)
        assert new_state.dice_state.double_play_active

    def test_apply_bad_take_cards(self, state_with_5s_on_board):
        """Bad outcome TAKE_CARDS should increase current player's hand."""
        state = state_with_5s_on_board
        state.variant = VariantConfig(
            dice_good_probability=0.0,   # always bad
            dice_bad_effect=BadDiceEffect.TAKE_CARDS,
            dice_bad_cards_count=2,
        )
        p_idx = 0
        state.current_player = p_idx
        before = len(state.players[p_idx].hand)
        random.seed(0)
        new_state = RollDice().apply(state)
        after = len(new_state.players[p_idx].hand)
        assert after >= before  # may receive 0 if donors have no cards

    def test_apply_bad_negative_points(self, state_with_5s_on_board):
        state = state_with_5s_on_board
        state.variant = VariantConfig(
            dice_good_probability=0.0,
            dice_bad_effect=BadDiceEffect.NEGATIVE_POINTS,
            dice_bad_penalty_points=3,
        )
        p_idx = 0
        state.current_player = p_idx
        before_score = state.players[p_idx].match_score
        random.seed(0)
        new_state = RollDice().apply(state)
        assert new_state.players[p_idx].match_score == before_score - 3

    def test_apply_bad_forced_pass(self, state_with_5s_on_board):
        state = state_with_5s_on_board
        state.variant = VariantConfig(
            dice_good_probability=0.0,
            dice_bad_effect=BadDiceEffect.FORCED_PASS,
        )
        p_idx = 0
        state.current_player = p_idx
        random.seed(0)
        new_state = RollDice().apply(state)
        assert new_state.current_player != p_idx

    def test_str(self):
        assert "RollDice" in str(RollDice())


# ══════════════════════════════════════════════
# Pass
# ══════════════════════════════════════════════

class TestPass:

    def test_always_legal(self, fresh_state):
        p_idx = 0
        fresh_state.current_player = p_idx
        assert Pass(voluntary=False).is_legal(fresh_state, p_idx)
        assert Pass(voluntary=True).is_legal(fresh_state, p_idx)

    def test_voluntary_pass_applies_penalty(self, fresh_state):
        p_idx = 0
        fresh_state.current_player = p_idx
        penalty = fresh_state.variant.voluntary_pass_penalty
        before = fresh_state.players[p_idx].round_score
        new_state = Pass(voluntary=True).apply(fresh_state)
        assert new_state.players[p_idx].round_score == before - penalty

    def test_forced_pass_no_penalty(self, fresh_state):
        p_idx = 0
        fresh_state.current_player = p_idx
        before = fresh_state.players[p_idx].round_score
        new_state = Pass(voluntary=False).apply(fresh_state)
        assert new_state.players[p_idx].round_score == before

    def test_pass_advances_turn(self, fresh_state):
        p_idx = 0
        fresh_state.current_player = p_idx
        new_state = Pass(voluntary=False).apply(fresh_state)
        assert new_state.current_player == 1

    def test_pass_clears_dice_state(self, fresh_state):
        fresh_state.dice_state.wild_active = True
        new_state = Pass(voluntary=False).apply(fresh_state)
        assert not new_state.dice_state.wild_active

    def test_str(self):
        assert "Pass" in str(Pass(voluntary=True))


# ══════════════════════════════════════════════
# Rules
# ══════════════════════════════════════════════

class TestRules:

    # ── get_legal_moves ───────────────────────

    def test_legal_moves_not_empty(self, fresh_state):
        moves = Rules.get_legal_moves(fresh_state)
        assert len(moves) > 0

    def test_legal_moves_empty_when_game_over(self, fresh_state):
        fresh_state.game_over = True
        assert Rules.get_legal_moves(fresh_state) == []

    def test_forced_pass_when_no_playable(self, fresh_state):
        """With no playable cards, only a forced Pass should appear."""
        p_idx = 0
        fresh_state.current_player = p_idx
        # Give player cards that are definitely not playable
        fresh_state.players[p_idx].hand = [Card(Suit.OROS, 3)]
        # All suits empty, 3 ≠ 5 → not playable
        moves = Rules.get_legal_moves(fresh_state)
        assert len(moves) == 1
        assert isinstance(moves[0], Pass)
        assert not moves[0].voluntary

    def test_voluntary_pass_available_when_has_moves(self, state_with_5s_on_board):
        """If player has playable cards, a voluntary Pass should be in legal moves."""
        state = state_with_5s_on_board
        p_idx = 0
        state.current_player = p_idx
        moves = Rules.get_legal_moves(state)
        voluntary_passes = [m for m in moves if isinstance(m, Pass) and m.voluntary]
        assert len(voluntary_passes) == 1

    def test_roll_dice_in_legal_moves_when_playable(self, state_with_5s_on_board):
        state = state_with_5s_on_board
        p_idx = 0
        state.current_player = p_idx
        moves = Rules.get_legal_moves(state)
        has_playable = any(
            PlayCard(c).is_legal(state, p_idx)
            for c in state.players[p_idx].hand
        )
        roll_moves = [m for m in moves if isinstance(m, RollDice)]
        if has_playable:
            assert len(roll_moves) == 1

    # ── apply_move ────────────────────────────

    def test_apply_legal_move_returns_state(self, state_with_5s_on_board):
        state = state_with_5s_on_board
        p_idx = 0
        state.current_player = p_idx
        move = Pass(voluntary=False)
        new_state = Rules.apply_move(state, move)
        assert isinstance(new_state, GameState)

    def test_apply_illegal_move_raises(self, fresh_state):
        p_idx = 0
        fresh_state.current_player = p_idx
        # A non-5 card on empty suit is illegal
        ghost_card = None
        for card in fresh_state.players[p_idx].hand:
            if card.rank != 5 and fresh_state.board.is_empty(card.suit):
                ghost_card = card
                break
        if ghost_card is None:
            pytest.skip("No illegal card found")
        with pytest.raises(ValueError):
            Rules.apply_move(fresh_state, PlayCard(ghost_card))

    # ── is_terminal / get_winner ──────────────

    def test_is_terminal_false_initially(self, fresh_state):
        assert not Rules.is_terminal(fresh_state)

    def test_is_terminal_true_when_game_over(self, fresh_state):
        fresh_state.game_over = True
        assert Rules.is_terminal(fresh_state)

    def test_get_winner_none_initially(self, fresh_state):
        assert Rules.get_winner(fresh_state) is None

    def test_get_winner_after_game_over(self, fresh_state):
        fresh_state.game_over = True
        fresh_state.winner = 2
        assert Rules.get_winner(fresh_state) == 2

    # ── compute_round_scores ──────────────────

    def test_compute_round_scores_winner_takes_all(self, near_win_state):
        state = near_win_state
        state.variant = VariantConfig(scoring_mode=ScoringMode.WINNER_TAKES_ALL)
        # Manually trigger game over for player 0
        state.game_over = True
        state.winner = 0
        state.players[0].hand = []
        # Other players have cards
        total_opp = sum(len(p.hand) for p in state.players if p.index != 0)
        Rules.compute_round_scores(state)
        assert state.players[0].round_score == total_opp
        # Losers should not gain or lose
        for p in state.players:
            if p.index != 0:
                assert p.round_score == 0

    def test_compute_round_scores_double_penalty(self, near_win_state):
        state = near_win_state
        state.variant = VariantConfig(scoring_mode=ScoringMode.DOUBLE_PENALTY)
        state.game_over = True
        state.winner = 0
        state.players[0].hand = []
        for p in state.players:
            if p.index != 0:
                p.hand = [Card(Suit.OROS, 1), Card(Suit.OROS, 2)]  # 2 cards each
        Rules.compute_round_scores(state)
        # Losers should have negative round scores
        for p in state.players:
            if p.index != 0:
                assert p.round_score < 0

    def test_compute_round_scores_updates_match_score(self, near_win_state):
        state = near_win_state
        state.game_over = True
        state.winner = 0
        state.players[0].hand = []
        Rules.compute_round_scores(state)
        for p in state.players:
            assert p.match_score == p.round_score

    # ── initialize_game ───────────────────────

    def test_initialize_game_4_players(self, default_variant):
        state = Rules.initialize_game(4, default_variant)
        assert len(state.players) == 4
        assert all(len(p.hand) == 10 for p in state.players)

    def test_initialize_game_2_players(self, default_variant):
        state = Rules.initialize_game(2, default_variant)
        assert len(state.players) == 2
        assert all(len(p.hand) == 20 for p in state.players)

    def test_initialize_game_board_empty(self, default_variant):
        state = Rules.initialize_game(4, default_variant)
        for suit in Suit:
            assert state.board.is_empty(suit)

    def test_initialize_game_first_player_is_0(self, default_variant):
        state = Rules.initialize_game(4, default_variant)
        assert state.current_player == 0

    def test_initialize_game_no_duplicate_cards(self, default_variant):
        state = Rules.initialize_game(4, default_variant)
        all_cards = [c for p in state.players for c in p.hand]
        assert len(set(all_cards)) == len(all_cards)

    # ── full round simulation ─────────────────

    def test_full_round_terminates(self, default_variant):
        """A randomly played game must eventually terminate."""
        random.seed(99)
        state = Rules.initialize_game(4, default_variant)
        max_turns = 2000
        turns = 0
        while not Rules.is_terminal(state) and turns < max_turns:
            moves = Rules.get_legal_moves(state)
            # Avoid voluntary passes to keep game moving
            non_vol = [m for m in moves if not (isinstance(m, Pass) and m.voluntary)]
            move = random.choice(non_vol if non_vol else moves)
            state = Rules.apply_move(state, move)
            turns += 1
        assert Rules.is_terminal(state), "Game did not terminate within turn limit"
