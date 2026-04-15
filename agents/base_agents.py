"""
AI agent interface and implementations for Cinquillo 2.0.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import random

from game.entities import GameState, Card, Deck
from game.rules import Move, Rules


class Agent(ABC):
    """Abstract base class for all AI agents"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def choose_move(self, state: GameState, legal_moves: List[Move]) -> Move:
        """
        Choose a move from the list of legal moves
        
        args:
            state: Current game state
            legal_moves: List of legal moves to choose from
            
        returns:
            Selected move
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"


class RandomAgent(Agent):
    """
    Random agent that selects uniformly from legal moves
    """
    
    def __init__(self, name: str = "Random", avoid_bad_moves: bool = False):
        super().__init__(name)
        self.avoid_bad_moves = avoid_bad_moves
    
    def choose_move(self, state: GameState, legal_moves: List[Move]) -> Move:
        """choose a random move, optionally filtering bad ones"""
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        if not self.avoid_bad_moves:
            return random.choice(legal_moves)
        
        # try to avoid voluntary passes if we have other options
        from game.rules import Pass
        non_pass_moves = [m for m in legal_moves if not isinstance(m, Pass)]
        
        if non_pass_moves:
            return random.choice(non_pass_moves)
        else:
            return random.choice(legal_moves)


class HeuristicAgent(Agent):
    """
    Heuristic agent that evaluates moves based on hand-crafted features
    Multiple play styles can be achieved by varying feature weights
    """
    
    def __init__(self, name: str = "Heuristic", 
                 prefer_reduce_hand: float = 1.0,
                 prefer_balance_suits: float = 0.5,
                 prefer_open_suits: float = 0.3,
                 prefer_block: float = 0.4,
                 avoid_voluntary_pass: float = 2.0,
                 dice_risk_tolerance: float = 0.3):
        """
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
        """choose move with highest heuristic score"""
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Evaluate each move
        move_scores = []
        for move in legal_moves:
            score = self._evaluate_move(state, move)
            move_scores.append((move, score))
        
        best_move = max(move_scores, key=lambda x: x[1])[0]
        return best_move
    
    def _get_revealed_hand(self, state: GameState) -> Optional[List[Card]]:
        """
        return the hand of the opponent revealed to the current player, or None
        """
        target = state.dice_state.get_revealed_target(state.current_player)
        if target is None:
            return None
        return state.players[target].hand

    def _evaluate_move(self, state: GameState, move: Move) -> float:
        """evaluate a move based on heuristic features"""
        from game.rules import PlayCard, RollDice, Pass
        
        score = 0.0
        player = state.get_current_player()
        current_hand_size = player.hand_size()

        revealed_hand = self._get_revealed_hand(state)
        
        if isinstance(move, PlayCard):
            score += self.prefer_reduce_hand
            
            if state.board.is_empty(move.card.suit):
                score += self.prefer_open_suits
            
            remaining_hand = [c for c in player.hand if c != move.card]
            suit_balance = self._compute_suit_balance(remaining_hand)
            score += self.prefer_balance_suits * suit_balance
            
            blocking_value = self._compute_blocking_value(
                state, move.card, revealed_hand=revealed_hand
            )
            score += self.prefer_block * blocking_value

            if self._opponents_see_my_hand(state):
                score += 0.3

        elif isinstance(move, RollDice):
            hand_pressure = current_hand_size / 10.0
            suit_balance = self._compute_suit_balance(player.hand)
            imbalance = 1.0 - suit_balance
            dice_score = (hand_pressure + imbalance) * self.dice_risk_tolerance
            score += dice_score
            
        elif isinstance(move, Pass):
            if move.voluntary:
                score -= self.avoid_voluntary_pass
            else:
                score = 0.0
        
        return score

    def _opponents_see_my_hand(self, state: GameState) -> bool:
        p = state.current_player
        return any(
            state.dice_state.get_revealed_target(i) == p
            for i in range(len(state.players))
            if i != p
        )
    
    def _compute_suit_balance(self, hand: List) -> float:
        if not hand:
            return 1.0
        
        from game.entities import Suit
        suit_counts = {suit: 0 for suit in Suit}
        
        for card in hand:
            suit_counts[card.suit] += 1
        
        counts = list(suit_counts.values())
        ideal = len(hand) / 4.0
        
        variance = sum((c - ideal) ** 2 for c in counts) / 4.0
        
        max_variance = (len(hand) ** 2) / 4.0
        balance = 1.0 - (variance / max(max_variance, 0.01))
        
        return max(0.0, min(1.0, balance))
    
    def _compute_blocking_value(self, state: GameState, card: Card, revealed_hand: Optional[List[Card]] = None) -> float:
        """
        compute how much this card blocks opponents
        """
        if revealed_hand is not None:
            block_count = 0
            for opp_card in revealed_hand:
                if opp_card.suit == card.suit:
                    our_idx = Deck.RANK_INDEX[card.rank]
                    their_idx = Deck.RANK_INDEX[opp_card.rank]
                    if abs(our_idx - their_idx) == 1:
                        block_count += 1
            return min(1.0, block_count / 3.0)
        
        middle_rank = 6.5
        distance = abs(card.rank - middle_rank)
        max_distance = 5.5
        return distance / max_distance


def create_aggressive_heuristic() -> HeuristicAgent:
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
    return HeuristicAgent(
        name="Risky",
        prefer_reduce_hand=0.9,
        prefer_balance_suits=0.3,
        prefer_open_suits=0.4,
        prefer_block=0.3,
        avoid_voluntary_pass=2.0,
        dice_risk_tolerance=0.7
    )