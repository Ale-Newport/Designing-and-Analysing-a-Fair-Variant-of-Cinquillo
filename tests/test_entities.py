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
        assert not empty_board.is_adjacent(Card(Suit.OROS, 5))

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
    def test_defaults(self):
        ds = DiceState()
        assert not ds.wild_active
        assert not ds.double_play_active
        assert ds.revealed_player is None

    def test_clear(self):
        ds = DiceState(wild_active=True, double_play_active=True, revealed_player=2)
        ds.clear()
        assert not ds.wild_active
        assert not ds.double_play_active
        assert ds.revealed_player is None


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
