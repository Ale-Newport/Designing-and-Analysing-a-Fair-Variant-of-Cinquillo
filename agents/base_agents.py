"""
AI agent interface and implementations for Cinquillo 2.0.
"""
from abc import ABC, abstractmethod
from typing import List
import random

from game.entities import GameState
from game.rules import Move, Rules


class Agent(ABC):
    """Abstract base class for all AI agents."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def choose_move(self, state: GameState, legal_moves: List[Move]) -> Move:
        """
        Choose a move from the list of legal moves.
        
        Args:
            state: Current game state
            legal_moves: List of legal moves to choose from
            
        Returns:
            Selected move
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


class RandomAgent(Agent):
    """
    Random agent that selects uniformly from legal moves.
    Can optionally avoid obviously bad moves.
    """
    
    def __init__(self, name: str = "Random", avoid_bad_moves: bool = False):
        super().__init__(name)
        self.avoid_bad_moves = avoid_bad_moves
    
    def choose_move(self, state: GameState, legal_moves: List[Move]) -> Move:
        """Choose a random move, optionally filtering bad ones."""
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        if not self.avoid_bad_moves:
            return random.choice(legal_moves)
        
        # Try to avoid voluntary passes if we have other options
        from game.rules import Pass
        non_pass_moves = [m for m in legal_moves if not isinstance(m, Pass)]
        
        if non_pass_moves:
            # Prefer non-pass moves
            return random.choice(non_pass_moves)
        else:
            return random.choice(legal_moves)


class HeuristicAgent(Agent):
    """
    Heuristic agent that evaluates moves based on hand-crafted features.
    Different play styles can be achieved by varying feature weights.
    """
    
    def __init__(self, name: str = "Heuristic", 
                 prefer_reduce_hand: float = 1.0,
                 prefer_balance_suits: float = 0.5,
                 prefer_open_suits: float = 0.3,
                 prefer_block: float = 0.4,
                 avoid_voluntary_pass: float = 2.0,
                 dice_risk_tolerance: float = 0.3):
        """
        Initialize heuristic agent with feature weights.
        
        Args:
            prefer_reduce_hand: Weight for reducing hand size
            prefer_balance_suits: Weight for maintaining balanced suit distribution
            prefer_open_suits: Weight for opening new suits
            prefer_block: Weight for blocking opponents
            avoid_voluntary_pass: Penalty weight for voluntary passes
            dice_risk_tolerance: Threshold for rolling dice
        """
        super().__init__(name)
        self.prefer_reduce_hand = prefer_reduce_hand
        self.prefer_balance_suits = prefer_balance_suits
        self.prefer_open_suits = prefer_open_suits
        self.prefer_block = prefer_block
        self.avoid_voluntary_pass = avoid_voluntary_pass
        self.dice_risk_tolerance = dice_risk_tolerance
    
    def choose_move(self, state: GameState, legal_moves: List[Move]) -> Move:
        """Choose move with highest heuristic score."""
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Evaluate each move
        move_scores = []
        for move in legal_moves:
            score = self._evaluate_move(state, move)
            move_scores.append((move, score))
        
        # Return move with highest score
        best_move = max(move_scores, key=lambda x: x[1])[0]
        return best_move
    
    def _evaluate_move(self, state: GameState, move: Move) -> float:
        """Evaluate a move based on heuristic features."""
        from game.rules import PlayCard, RollDice, Pass
        
        score = 0.0
        player = state.get_current_player()
        current_hand_size = player.hand_size()
        
        if isinstance(move, PlayCard):
            # Playing a card reduces hand size (good)
            score += self.prefer_reduce_hand
            
            # Check if this opens a new suit
            if state.board.is_empty(move.card.suit):
                score += self.prefer_open_suits
            
            # Simulate playing the card to check suit balance
            remaining_hand = [c for c in player.hand if c != move.card]
            suit_balance = self._compute_suit_balance(remaining_hand)
            score += self.prefer_balance_suits * suit_balance
            
            # Check if this blocks opponents (playing high/low cards)
            blocking_value = self._compute_blocking_value(state, move.card)
            score += self.prefer_block * blocking_value
            
        elif isinstance(move, RollDice):
            # Rolling dice is risky but can be beneficial when stuck
            # More attractive when hand is large or unbalanced
            hand_pressure = current_hand_size / 10.0  # Normalize
            
            suit_balance = self._compute_suit_balance(player.hand)
            imbalance = 1.0 - suit_balance
            
            dice_score = (hand_pressure + imbalance) * self.dice_risk_tolerance
            score += dice_score
            
        elif isinstance(move, Pass):
            # Passing is generally bad, especially voluntary
            if move.voluntary:
                score -= self.avoid_voluntary_pass
            else:
                # Forced pass is neutral (no choice)
                score = 0.0
        
        return score
    
    def _compute_suit_balance(self, hand: List) -> float:
        """
        Compute how balanced the suit distribution is.
        Returns value in [0, 1] where 1 is perfectly balanced.
        """
        if not hand:
            return 1.0
        
        from game.entities import Suit
        suit_counts = {suit: 0 for suit in Suit}
        
        for card in hand:
            suit_counts[card.suit] += 1
        
        counts = list(suit_counts.values())
        ideal = len(hand) / 4.0
        
        # Compute variance from ideal
        variance = sum((c - ideal) ** 2 for c in counts) / 4.0
        
        # Normalize to [0, 1] where 0 is worst (all cards one suit), 1 is best
        max_variance = (len(hand) ** 2) / 4.0
        balance = 1.0 - (variance / max(max_variance, 0.01))
        
        return max(0.0, min(1.0, balance))
    
    def _compute_blocking_value(self, state: GameState, card) -> float:
        """
        Compute how much this card blocks opponents.
        Higher values for cards that limit opponent options.
        """
        # Cards at extremes (1, 12) block more
        # Cards in middle (5, 6, 7) block less
        
        # Simplified: distance from middle rank
        middle_rank = 6.5
        distance = abs(card.rank - middle_rank)
        
        # Normalize to [0, 1]
        max_distance = 5.5  # Distance from 1 to 6.5
        blocking = distance / max_distance
        
        return blocking


def create_aggressive_heuristic() -> HeuristicAgent:
    """Create an aggressive heuristic agent."""
    return HeuristicAgent(
        name="Aggressive",
        prefer_reduce_hand=1.5,
        prefer_balance_suits=0.2,
        prefer_open_suits=0.5,
        prefer_block=0.1,
        avoid_voluntary_pass=3.0,
        dice_risk_tolerance=0.5
    )


def create_defensive_heuristic() -> HeuristicAgent:
    """Create a defensive heuristic agent."""
    return HeuristicAgent(
        name="Defensive",
        prefer_reduce_hand=0.8,
        prefer_balance_suits=0.8,
        prefer_open_suits=0.2,
        prefer_block=0.7,
        avoid_voluntary_pass=2.0,
        dice_risk_tolerance=0.2
    )


def create_balanced_heuristic() -> HeuristicAgent:
    """Create a balanced heuristic agent."""
    return HeuristicAgent(
        name="Balanced",
        prefer_reduce_hand=1.0,
        prefer_balance_suits=0.5,
        prefer_open_suits=0.3,
        prefer_block=0.4,
        avoid_voluntary_pass=2.0,
        dice_risk_tolerance=0.3
    )


def create_risky_heuristic() -> HeuristicAgent:
    """Create a risky heuristic agent that likes dice rolls."""
    return HeuristicAgent(
        name="Risky",
        prefer_reduce_hand=0.9,
        prefer_balance_suits=0.3,
        prefer_open_suits=0.4,
        prefer_block=0.3,
        avoid_voluntary_pass=2.0,
        dice_risk_tolerance=0.7
    )
