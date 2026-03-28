"""
Shared pytest fixtures for Cinquillo 2.0 test suite.

Place this file in PRJ/tests/ alongside the other test files.
Run from PRJ root with: pytest tests/
"""
import sys
import os

# Ensure PRJ root is on the path so imports like `from game.entities import ...` work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from game.entities import (
    Card, Deck, Player, Board, GameState, VariantConfig, DiceState,
    Suit, GoodDiceEffect, BadDiceEffect, ScoringMode, MatchEndMode,
)
from game.rules import Rules, PlayCard, RollDice, Pass


# ──────────────────────────────────────────────
# Basic object builders
# ──────────────────────────────────────────────

@pytest.fixture
def default_variant():
    return VariantConfig()


@pytest.fixture
def empty_board():
    return Board()


@pytest.fixture
def board_with_5s():
    """Board with all four 5s already placed."""
    board = Board()
    for suit in Suit:
        board.add_card(Card(suit, 5))
    return board


@pytest.fixture
def sample_hand():
    """A small, deterministic hand of 5 cards."""
    return [
        Card(Suit.OROS, 5),
        Card(Suit.OROS, 6),
        Card(Suit.COPAS, 5),
        Card(Suit.ESPADAS, 3),
        Card(Suit.BASTOS, 10),
    ]


@pytest.fixture
def player_with_hand(sample_hand):
    p = Player(index=0, hand=list(sample_hand))
    return p


# ──────────────────────────────────────────────
# Full game-state builders
# ──────────────────────────────────────────────

def _make_state(num_players=4, variant=None, seed=42):
    """Helper: initialise a deterministic game state."""
    import random
    random.seed(seed)
    if variant is None:
        variant = VariantConfig()
    return Rules.initialize_game(num_players, variant)


@pytest.fixture
def fresh_state():
    return _make_state()


@pytest.fixture
def two_player_state():
    return _make_state(num_players=2)


@pytest.fixture
def state_with_5s_on_board(default_variant):
    """4-player state where all 5s have been placed on the board."""
    import random
    random.seed(0)
    state = Rules.initialize_game(4, default_variant)
    # Place all 5s onto the board directly (bypass rules for fixture setup)
    for suit in Suit:
        card = Card(suit, 5)
        # Remove from whatever player holds it
        for p in state.players:
            if card in p.hand:
                p.hand.remove(card)
                break
        state.board.add_card(card)
    return state


@pytest.fixture
def near_win_state(default_variant):
    """State where player 0 has exactly one card left (the 5O, already on board)."""
    import random
    random.seed(7)
    state = Rules.initialize_game(4, default_variant)
    # Give player 0 a single playable card
    state.players[0].hand = [Card(Suit.OROS, 6)]
    # Make sure 5O is on the board so the 6O is adjacent
    state.board.add_card(Card(Suit.OROS, 5))
    return state
