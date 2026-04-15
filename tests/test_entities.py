"""
Tests for game/entities.py

Covers: Card, Deck, Player, Board, DiceState,
        VariantConfig, GameState (and their copy/helper methods).
"""
import pytest
from game.entities import (
    Card, Deck, Player, Board, GameState, VariantConfig, DiceState,
    Suit, GoodDiceEffect, BadDiceEffect, ScoringMode, MatchEndMode,
)


# ══════════════════════════════════════════════
# Card
# ══════════════════════════════════════════════

class TestCard:
    def test_str_representation(self):
        c = Card(Suit.OROS, 5)
        assert "5" in str(c)
        assert "O" in str(c)          # First letter of "Oros"

    def test_repr(self):
        c = Card(Suit.COPAS, 12)
        assert "Copas" in repr(c)
        assert "12" in repr(c)

    def test_equality_same(self):
        assert Card(Suit.OROS, 5) == Card(Suit.OROS, 5)

    def test_equality_different_suit(self):
        assert Card(Suit.OROS, 5) != Card(Suit.COPAS, 5)

    def test_equality_different_rank(self):
        assert Card(Suit.OROS, 5) != Card(Suit.OROS, 6)

    def test_hashable(self):
        """Cards must be usable as dict keys / set members."""
        s = {Card(Suit.OROS, 5), Card(Suit.OROS, 5), Card(Suit.COPAS, 7)}
        assert len(s) == 2

    def test_frozen(self):
        """Cards are frozen dataclasses – mutation should raise."""
        c = Card(Suit.OROS, 5)
        with pytest.raises((AttributeError, TypeError)):
            c.rank = 6


# ══════════════════════════════════════════════
# Deck
# ══════════════════════════════════════════════

class TestDeck:
    def test_deck_size(self):
        assert len(Deck().cards) == 40

    def test_all_suits_present(self):
        deck = Deck()
        for suit in Suit:
            assert any(c.suit == suit for c in deck.cards)

    def test_all_ranks_present(self):
        deck = Deck()
        for rank in Deck.RANKS:
            assert any(c.rank == rank for c in deck.cards)

    def test_no_8_or_9(self):
        """Spanish deck skips 8 and 9."""
        deck = Deck()
        for c in deck.cards:
            assert c.rank not in (8, 9)

    def test_rank_index_covers_all_ranks(self):
        for rank in Deck.RANKS:
            assert rank in Deck.RANK_INDEX

    def test_rank_index_is_contiguous(self):
        """RANK_INDEX values must be 0-9 with no gaps (adjacency logic depends on this)."""
        indices = sorted(Deck.RANK_INDEX.values())
        assert indices == list(range(10))

    def test_shuffle_changes_order(self):
        import random
        deck = Deck()
        original = list(deck.cards)
        random.seed(1)
        deck.shuffle()
        assert deck.cards != original   # overwhelmingly likely

    def test_deal_4_players(self):
        deck = Deck()
        hands = deck.deal(4)
        assert len(hands) == 4
        assert all(len(h) == 10 for h in hands)

    def test_deal_2_players(self):
        deck = Deck()
        hands = deck.deal(2)
        assert len(hands) == 2
        assert all(len(h) == 20 for h in hands)

    def test_deal_no_duplicates(self):
        deck = Deck()
        hands = deck.deal(4)
        all_cards = [c for h in hands for c in h]
        assert len(set(all_cards)) == len(all_cards)


# ══════════════════════════════════════════════
# Player
# ══════════════════════════════════════════════

class TestPlayer:
    def test_has_card_true(self):
        p = Player(index=0, hand=[Card(Suit.OROS, 5)])
        assert p.has_card(Card(Suit.OROS, 5))

    def test_has_card_false(self):
        p = Player(index=0, hand=[])
        assert not p.has_card(Card(Suit.OROS, 5))

    def test_remove_card(self):
        c = Card(Suit.OROS, 5)
        p = Player(index=0, hand=[c, Card(Suit.COPAS, 7)])
        p.remove_card(c)
        assert c not in p.hand
        assert len(p.hand) == 1

    def test_add_card(self):
        p = Player(index=0, hand=[])
        c = Card(Suit.BASTOS, 10)
        p.add_card(c)
        assert c in p.hand

    def test_hand_size(self):
        p = Player(index=0, hand=[Card(Suit.OROS, 1), Card(Suit.OROS, 2)])
        assert p.hand_size() == 2

    def test_default_scores_zero(self):
        p = Player(index=0)
        assert p.round_score == 0
        assert p.match_score == 0


# ══════════════════════════════════════════════
# Board
# ══════════════════════════════════════════════

class TestBoard:
    def test_initially_empty(self, empty_board):
        for suit in Suit:
            assert empty_board.is_empty(suit)

    def test_add_card_not_empty(self, empty_board):
        empty_board.add_card(Card(Suit.OROS, 5))
        assert not empty_board.is_empty(Suit.OROS)

    def test_has_rank(self, empty_board):
        empty_board.add_card(Card(Suit.OROS, 5))
        assert empty_board.has_rank(Suit.OROS, 5)
        assert not empty_board.has_rank(Suit.OROS, 6)

    def test_is_adjacent_after_5(self, empty_board):
        empty_board.add_card(Card(Suit.OROS, 5))
        # 6 is adjacent to 5
        assert empty_board.is_adjacent(Card(Suit.OROS, 6))
        # 4 is adjacent to 5
        assert empty_board.is_adjacent(Card(Suit.OROS, 4))

    def test_is_not_adjacent_when_gap(self, empty_board):
        empty_board.add_card(Card(Suit.OROS, 5))
        # 7 is rank-adjacent to 6, not to 5
        assert not empty_board.is_adjacent(Card(Suit.OROS, 7))

    def test_adjacency_across_gap_7_to_10(self, empty_board):
        """In Spanish deck 7 and 10 are adjacent (indices differ by 1)."""
        empty_board.add_card(Card(Suit.OROS, 5))
        empty_board.add_card(Card(Suit.OROS, 6))
        empty_board.add_card(Card(Suit.OROS, 7))
        assert empty_board.is_adjacent(Card(Suit.OROS, 10))  # 10 is next after 7

    def test_is_adjacent_empty_suit_returns_false(self, empty_board):
        """Board.is_adjacent returns False on an empty suit.
        The special case that a 5 can be played on an empty suit is handled
        in PlayCard.is_legal, not in Board.is_adjacent."""
        assert not empty_board.is_adjacent(Card(Suit.OROS, 5))
        assert not empty_board.is_adjacent(Card(Suit.OROS, 3))

    def test_get_min_max_empty(self, empty_board):
        lo, hi = empty_board.get_min_max(Suit.OROS)
        assert lo is None and hi is None

    def test_get_min_max(self, empty_board):
        for rank in [5, 6, 4]:
            empty_board.add_card(Card(Suit.OROS, rank))
        lo, hi = empty_board.get_min_max(Suit.OROS)
        assert lo == 4
        assert hi == 6

    def test_copy_is_independent(self, empty_board):
        empty_board.add_card(Card(Suit.OROS, 5))
        copy = empty_board.copy()
        copy.add_card(Card(Suit.OROS, 6))
        assert not empty_board.has_rank(Suit.OROS, 6)


# ══════════════════════════════════════════════
# DiceState
# ══════════════════════════════════════════════

class TestDiceState:
    """
    DiceState.revealed_hands is a Dict[int, int] (viewer → target).
    It is initialised as an empty dict and cleared back to an empty dict
    by DiceState.clear().  Helper methods reveal_to(), can_see_hand(),
    get_revealed_target(), and clear_viewer() all operate on this dict.
    """

    def test_defaults(self):
        ds = DiceState()
        assert not ds.wild_active
        assert not ds.double_play_active
        # revealed_hands defaults to an empty dict, not None
        assert ds.revealed_hands == {}
        assert isinstance(ds.revealed_hands, dict)

    def test_reveal_to_stores_mapping(self):
        ds = DiceState()
        ds.reveal_to(viewer=0, target=2)
        assert ds.revealed_hands[0] == 2

    def test_can_see_hand_true(self):
        ds = DiceState()
        ds.reveal_to(viewer=1, target=3)
        assert ds.can_see_hand(viewer=1, target=3)

    def test_can_see_hand_false_wrong_target(self):
        ds = DiceState()
        ds.reveal_to(viewer=1, target=3)
        assert not ds.can_see_hand(viewer=1, target=2)

    def test_can_see_hand_false_no_reveal(self):
        ds = DiceState()
        assert not ds.can_see_hand(viewer=0, target=1)

    def test_get_revealed_target_present(self):
        ds = DiceState()
        ds.reveal_to(viewer=2, target=0)
        assert ds.get_revealed_target(viewer=2) == 0

    def test_get_revealed_target_absent(self):
        ds = DiceState()
        assert ds.get_revealed_target(viewer=0) is None

    def test_clear_viewer_removes_entry(self):
        ds = DiceState()
        ds.reveal_to(0, 1)
        ds.reveal_to(2, 3)
        ds.clear_viewer(0)
        assert ds.get_revealed_target(0) is None
        # Other entries unaffected
        assert ds.get_revealed_target(2) == 3

    def test_clear_viewer_noop_when_absent(self):
        """clear_viewer on a non-existent key must not raise."""
        ds = DiceState()
        ds.clear_viewer(99)  # should not raise

    def test_clear_resets_all_effects(self):
        ds = DiceState()
        ds.wild_active = True
        ds.double_play_active = True
        ds.reveal_to(0, 1)
        ds.reveal_to(2, 3)
        ds.clear()
        assert not ds.wild_active
        assert not ds.double_play_active
        assert ds.revealed_hands == {}

    def test_multiple_reveals_independent(self):
        """REVEAL_HAND makes multiple opponents see the roller's hand."""
        ds = DiceState()
        # Simulate BadDiceEffect.REVEAL_HAND for player 1: opponents 0, 2, 3 can see p1
        for viewer in [0, 2, 3]:
            ds.reveal_to(viewer=viewer, target=1)
        assert ds.can_see_hand(0, 1)
        assert ds.can_see_hand(2, 1)
        assert ds.can_see_hand(3, 1)
        assert not ds.can_see_hand(1, 0)

    def test_info_reveal_good_effect(self):
        """INFO_REVEAL: roller p sees opponent u with fewest cards."""
        ds = DiceState()
        ds.reveal_to(viewer=0, target=3)   # player 0 sees player 3
        assert ds.get_revealed_target(0) == 3
        # Other viewers unaffected
        assert ds.get_revealed_target(1) is None


# ══════════════════════════════════════════════
# VariantConfig
# ══════════════════════════════════════════════

class TestVariantConfig:
    def test_default_values(self):
        vc = VariantConfig()
        assert vc.dice_good_probability == 0.5
        assert vc.scoring_mode == ScoringMode.DOUBLE_PENALTY
        assert vc.match_end_mode == MatchEndMode.FIXED_ROUNDS

    def test_get_target_score(self):
        vc = VariantConfig(target_score_multiplier=10)
        assert vc.get_target_score(4) == 40

    def test_str(self):
        vc = VariantConfig()
        s = str(vc)
        assert "scoring" in s.lower() or "double_penalty" in s.lower()

    def test_custom_variant(self):
        vc = VariantConfig(
            dice_good_effect=GoodDiceEffect.DOUBLE_PLAY,
            dice_bad_effect=BadDiceEffect.NEGATIVE_POINTS,
            scoring_mode=ScoringMode.WINNER_TAKES_ALL,
            match_end_mode=MatchEndMode.TARGET_SCORE,
        )
        assert vc.dice_good_effect == GoodDiceEffect.DOUBLE_PLAY
        assert vc.dice_bad_effect == BadDiceEffect.NEGATIVE_POINTS

    def test_all_good_effects_constructable(self):
        for eff in GoodDiceEffect:
            vc = VariantConfig(dice_good_effect=eff)
            assert vc.dice_good_effect == eff

    def test_all_bad_effects_constructable(self):
        for eff in BadDiceEffect:
            vc = VariantConfig(dice_bad_effect=eff)
            assert vc.dice_bad_effect == eff


# ══════════════════════════════════════════════
# GameState
# ══════════════════════════════════════════════

class TestGameState:
    def test_get_current_player(self, fresh_state):
        p = fresh_state.get_current_player()
        assert p.index == fresh_state.current_player

    def test_next_player_index_wraps(self, fresh_state):
        fresh_state.current_player = 3
        assert fresh_state.next_player_index() == 0

    def test_copy_independence(self, fresh_state):
        copy = fresh_state.copy()
        copy.players[0].hand.append(Card(Suit.OROS, 5))
        # Original should be unchanged
        assert len(fresh_state.players[0].hand) != len(copy.players[0].hand)

    def test_copy_board_independence(self, fresh_state):
        copy = fresh_state.copy()
        copy.board.add_card(Card(Suit.OROS, 5))
        assert not fresh_state.board.has_rank(Suit.OROS, 5)

    def test_copy_dice_state_independence(self, fresh_state):
        """Copying state must deep-copy revealed_hands dict."""
        copy = fresh_state.copy()
        copy.dice_state.reveal_to(0, 1)
        assert fresh_state.dice_state.get_revealed_target(0) is None

    def test_get_player_with_fewest_cards(self, fresh_state):
        # Give player 2 fewer cards
        stolen = fresh_state.players[2].hand.pop()
        fresh_state.players[0].hand.append(stolen)
        idx = fresh_state.get_player_with_fewest_cards()
        assert idx == 2

    def test_get_player_with_most_cards(self, fresh_state):
        # Give player 1 an extra card (move from player 3)
        extra = fresh_state.players[3].hand.pop()
        fresh_state.players[1].hand.append(extra)
        idx = fresh_state.get_player_with_most_cards()
        assert idx == 1

    def test_get_player_with_fewest_cards_exclude(self, fresh_state):
        fresh_state.players[2].hand = fresh_state.players[2].hand[:1]
        idx = fresh_state.get_player_with_fewest_cards(exclude=2)
        assert idx != 2

    def test_game_over_defaults_false(self, fresh_state):
        assert not fresh_state.game_over
        assert fresh_state.winner is None
