"""
Core entities for Cinquillo 2.0 game.
Defines Card, Deck, Player, Board, and GameState.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Set, Dict, Optional, Tuple
import random


class Suit(Enum):
    """Spanish deck suits."""
    OROS = "Oros"
    COPAS = "Copas"
    ESPADAS = "Espadas"
    BASTOS = "Bastos"


@dataclass(frozen=True)
class Card:
    """Immutable card with suit and rank."""
    suit: Suit
    rank: int  # 1-7, 10-12
    
    def __str__(self):
        return f"{self.rank}{self.suit.value[0]}"
    
    def __repr__(self):
        return f"Card({self.suit.value}, {self.rank})"


class Deck:
    """40-card Spanish deck."""
    RANKS = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
    
    # Rank index mapping for adjacency checks
    RANK_INDEX = {rank: idx for idx, rank in enumerate(RANKS)}
    
    def __init__(self):
        self.cards: List[Card] = [
            Card(suit, rank) 
            for suit in Suit 
            for rank in self.RANKS
        ]
    
    def shuffle(self):
        """Shuffle the deck in place."""
        random.shuffle(self.cards)
    
    def deal(self, num_players: int) -> List[List[Card]]:
        """
        Deal cards evenly to num_players.
        Returns list of hands, one per player.
        """
        cards_per_player = len(self.cards) // num_players
        hands = []
        for i in range(num_players):
            start = i * cards_per_player
            end = start + cards_per_player
            hands.append(self.cards[start:end])
        return hands


@dataclass
class Player:
    """Player state."""
    index: int
    hand: List[Card] = field(default_factory=list)
    round_score: int = 0
    match_score: int = 0
    
    def has_card(self, card: Card) -> bool:
        """Check if player has a specific card."""
        return card in self.hand
    
    def remove_card(self, card: Card):
        """Remove a card from hand."""
        self.hand.remove(card)
    
    def add_card(self, card: Card):
        """Add a card to hand."""
        self.hand.append(card)
    
    def hand_size(self) -> int:
        """Get number of cards in hand."""
        return len(self.hand)


@dataclass
class Board:
    """
    Game board tracking played cards per suit.
    For each suit, tracks set of ranks currently present.
    """
    suit_cards: Dict[Suit, Set[int]] = field(default_factory=lambda: {
        suit: set() for suit in Suit
    })
    
    def is_empty(self, suit: Suit) -> bool:
        """Check if no cards of a suit are on the board."""
        return len(self.suit_cards[suit]) == 0
    
    def has_rank(self, suit: Suit, rank: int) -> bool:
        """Check if a specific rank is on the board."""
        return rank in self.suit_cards[suit]
    
    def add_card(self, card: Card):
        """Add a card to the board."""
        self.suit_cards[card.suit].add(card.rank)
    
    def is_adjacent(self, card: Card) -> bool:
        """
        Check if card is adjacent to any existing card of same suit.
        A card is adjacent if it's next to a placed card in the rank sequence.
        Handles Spanish deck where 7 is followed by 10 (not 8).
        """
        if self.is_empty(card.suit):
            return False
        
        from game.entities import Deck
        rank_index = Deck.RANK_INDEX
        
        card_idx = rank_index[card.rank]
        
        for rank in self.suit_cards[card.suit]:
            placed_idx = rank_index[rank]
            # Check if indices differ by exactly 1 (adjacent in sequence)
            if abs(card_idx - placed_idx) == 1:
                return True
        return False
    
    def get_min_max(self, suit: Suit) -> Tuple[Optional[int], Optional[int]]:
        """Get min and max ranks for a suit (returns None if empty)."""
        if self.is_empty(suit):
            return None, None
        ranks = self.suit_cards[suit]
        return min(ranks), max(ranks)
    
    def copy(self) -> 'Board':
        """Create a deep copy of the board."""
        new_board = Board()
        new_board.suit_cards = {
            suit: ranks.copy() 
            for suit, ranks in self.suit_cards.items()
        }
        return new_board


class GoodDiceEffect(Enum):
    """Good dice outcomes."""
    WILD = "wild"  # Play any card
    DOUBLE_PLAY = "double_play"  # Play two cards this turn
    GIVE_CARD = "give_card"  # Give one card to another player
    INFO_REVEAL = "info_reveal"  # See another player's hand


class BadDiceEffect(Enum):
    """Bad dice outcomes."""
    TAKE_CARDS = "take_cards"  # Receive cards from another player
    NEGATIVE_POINTS = "negative_points"  # Lose points immediately
    FORCED_PASS = "forced_pass"  # Must pass this turn
    REVEAL_HAND = "reveal_hand"  # Your hand is revealed to others


class ScoringMode(Enum):
    """Scoring system modes."""
    WINNER_TAKES_ALL = "winner_takes_all"  # Winner gets sum of all opponent cards
    DOUBLE_PENALTY = "double_penalty"  # Winner gains, losers lose per card


class MatchEndMode(Enum):
    """Match ending conditions."""
    TARGET_SCORE = "target_score"  # First to reach score target
    FIXED_ROUNDS = "fixed_rounds"  # Play fixed number of rounds


@dataclass
class DiceState:
    """Transient state from dice effects."""
    wild_active: bool = False
    double_play_active: bool = False
    revealed_player: Optional[int] = None  # Which player's hand was revealed
    
    def clear(self):
        """Clear all one-shot effects."""
        self.wild_active = False
        self.double_play_active = False
        self.revealed_player = None


@dataclass
class VariantConfig:
    """Configuration for rule variants - flexible system for experimentation."""
    
    # === DICE SYSTEM (2 outcomes: good/bad) ===
    dice_good_probability: float = 0.5  # Probability of good outcome
    dice_good_effect: GoodDiceEffect = GoodDiceEffect.WILD
    dice_bad_effect: BadDiceEffect = BadDiceEffect.TAKE_CARDS
    dice_bad_cards_count: int = 2  # For TAKE_CARDS effect
    dice_bad_penalty_points: int = 1  # For NEGATIVE_POINTS effect
    
    # === SCORING SYSTEM ===
    scoring_mode: ScoringMode = ScoringMode.DOUBLE_PENALTY
    points_per_card: int = 1  # Multiplier for card values
    voluntary_pass_penalty: int = 1  # Penalty for passing when you have moves
    
    # === MATCH END CONDITION ===
    match_end_mode: MatchEndMode = MatchEndMode.FIXED_ROUNDS
    target_score_multiplier: int = 10  # For TARGET_SCORE: target = n_players * multiplier
    fixed_rounds_count: int = 5  # For FIXED_ROUNDS
    
    def get_target_score(self, num_players: int) -> int:
        """Calculate target score based on number of players."""
        return num_players * self.target_score_multiplier
    
    def __str__(self):
        return (f"VariantConfig(scoring={self.scoring_mode.value}, "
                f"dice={self.dice_good_effect.value}/{self.dice_bad_effect.value}, "
                f"match_end={self.match_end_mode.value})")


@dataclass
class GameState:
    """Complete game state."""
    board: Board
    players: List[Player]
    current_player: int
    variant: VariantConfig
    dice_state: DiceState = field(default_factory=DiceState)
    round_number: int = 0
    turn_number: int = 0
    game_over: bool = False
    winner: Optional[int] = None
    
    def get_current_player(self) -> Player:
        """Get the current player object."""
        return self.players[self.current_player]
    
    def next_player_index(self) -> int:
        """Get index of next player in turn order."""
        return (self.current_player + 1) % len(self.players)
    
    def get_player_with_fewest_cards(self, exclude: Optional[int] = None) -> int:
        """Get index of player with fewest cards (excluding specified player)."""
        candidates = [
            (i, p.hand_size()) 
            for i, p in enumerate(self.players)
            if exclude is None or i != exclude
        ]
        return min(candidates, key=lambda x: x[1])[0]
    
    def get_player_with_most_cards(self, exclude: Optional[int] = None) -> int:
        """Get index of player with most cards (excluding specified player)."""
        candidates = [
            (i, p.hand_size()) 
            for i, p in enumerate(self.players)
            if exclude is None or i != exclude
        ]
        return max(candidates, key=lambda x: x[1])[0]
    
    def copy(self) -> 'GameState':
        """Create a deep copy of the game state."""
        new_players = [
            Player(
                index=p.index,
                hand=p.hand.copy(),
                round_score=p.round_score,
                match_score=p.match_score
            )
            for p in self.players
        ]
        
        new_dice_state = DiceState(
            wild_active=self.dice_state.wild_active,
            double_play_active=self.dice_state.double_play_active,
            revealed_player=self.dice_state.revealed_player
        )
        
        return GameState(
            board=self.board.copy(),
            players=new_players,
            current_player=self.current_player,
            variant=self.variant,  # Config is immutable, can share
            dice_state=new_dice_state,
            round_number=self.round_number,
            turn_number=self.turn_number,
            game_over=self.game_over,
            winner=self.winner
        )
