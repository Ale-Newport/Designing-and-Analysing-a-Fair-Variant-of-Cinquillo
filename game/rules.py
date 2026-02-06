"""
Move types and game rules for Cinquillo 2.0.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from game.entities import Card, GameState, Player, VariantConfig
import random


class Move(ABC):
    """Abstract base class for all moves."""
    
    @abstractmethod
    def is_legal(self, state: GameState, player_index: int) -> bool:
        """Check if this move is legal in the given state."""
        pass
    
    @abstractmethod
    def apply(self, state: GameState) -> GameState:
        """Apply this move to the state and return new state."""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass
class PlayCard(Move):
    """Play a card from hand to the board."""
    card: Card
    
    def is_legal(self, state: GameState, player_index: int) -> bool:
        """Check if playing this card is legal."""
        player = state.players[player_index]
        
        # Must have the card
        if not player.has_card(self.card):
            return False
        
        # If wild is active, any card can be played
        if state.dice_state.wild_active:
            return True
        
        # If suit not on board, must be a 5
        if state.board.is_empty(self.card.suit):
            return self.card.rank == 5
        
        # If suit is on board, must be adjacent or a 5
        if self.card.rank == 5:
            return not state.board.has_rank(self.card.suit, 5)
        
        return (state.board.is_adjacent(self.card) and 
                not state.board.has_rank(self.card.suit, self.card.rank))
    
    def apply(self, state: GameState) -> GameState:
        """Apply the card play."""
        new_state = state.copy()
        player = new_state.get_current_player()
        
        # Remove card from hand and add to board
        player.remove_card(self.card)
        new_state.board.add_card(self.card)
        
        # Check if player won
        if player.hand_size() == 0:
            new_state.game_over = True
            new_state.winner = new_state.current_player
        
        # Clear wild if it was active
        if new_state.dice_state.wild_active:
            new_state.dice_state.wild_active = False
        
        # If double play is active, keep it for one more card
        # Otherwise, advance turn
        if not new_state.dice_state.double_play_active:
            if not new_state.game_over:
                new_state.current_player = new_state.next_player_index()
                new_state.turn_number += 1
        else:
            # First card of double play used
            new_state.dice_state.double_play_active = False
        
        return new_state
    
    def __str__(self) -> str:
        return f"PlayCard({self.card})"


class RollDice(Move):
    """Roll dice to trigger one of two outcomes (good or bad)."""
    
    def is_legal(self, state: GameState, player_index: int) -> bool:
        """Can only roll if player has at least one legal card to play."""
        player = state.players[player_index]
        
        # Check if any card in hand is playable
        for card in player.hand:
            play_move = PlayCard(card)
            if play_move.is_legal(state, player_index):
                return True
        return False
    
    def apply(self, state: GameState) -> GameState:
        """Apply dice roll with configurable good/bad outcomes."""
        new_state = state.copy()
        variant = new_state.variant
        
        # Determine outcome: good or bad
        is_good_outcome = random.random() < variant.dice_good_probability
        
        if is_good_outcome:
            self._apply_good_effect(new_state, variant.dice_good_effect)
        else:
            self._apply_bad_effect(new_state, variant.dice_bad_effect, variant)
        
        return new_state
    
    def _apply_good_effect(self, state: GameState, effect):
        """Apply the good dice effect."""
        from game.entities import GoodDiceEffect
        
        if effect == GoodDiceEffect.WILD:
            # Player can play any card
            state.dice_state.wild_active = True
            
        elif effect == GoodDiceEffect.DOUBLE_PLAY:
            # Player can play two cards this turn
            state.dice_state.double_play_active = True
            
        elif effect == GoodDiceEffect.GIVE_CARD:
            # Give one card to another player (strategic choice)
            current_player = state.get_current_player()
            if current_player.hand_size() > 0:
                # Give to player with most cards (helps them less)
                target_idx = state.get_player_with_most_cards(exclude=state.current_player)
                card = current_player.hand.pop()
                state.players[target_idx].add_card(card)
                
        elif effect == GoodDiceEffect.INFO_REVEAL:
            # See hand of player with most cards (most threatening)
            target_idx = state.get_player_with_most_cards(exclude=state.current_player)
            state.dice_state.revealed_player = target_idx
    
    def _apply_bad_effect(self, state: GameState, effect, variant):
        """Apply the bad dice effect."""
        from game.entities import BadDiceEffect
        
        current_player = state.get_current_player()
        
        if effect == BadDiceEffect.TAKE_CARDS:
            # Receive cards from player with most cards
            donor_idx = state.get_player_with_most_cards(exclude=state.current_player)
            donor = state.players[donor_idx]
            
            cards_to_take = min(variant.dice_bad_cards_count, donor.hand_size())
            
            # Transfer worst cards (farthest from playable)
            cards_to_transfer = self._choose_worst_cards(donor, cards_to_take, state)
            for card in cards_to_transfer:
                donor.remove_card(card)
                current_player.add_card(card)
                
        elif effect == BadDiceEffect.NEGATIVE_POINTS:
            # Lose points immediately
            current_player.match_score -= variant.dice_bad_penalty_points
            
        elif effect == BadDiceEffect.FORCED_PASS:
            # Must pass this turn (no penalty)
            # Advance to next player
            state.current_player = state.next_player_index()
            state.turn_number += 1
            
        elif effect == BadDiceEffect.REVEAL_HAND:
            # Your hand is revealed to all players
            state.dice_state.revealed_player = state.current_player
    
    def _choose_worst_cards(self, player, count: int, state: GameState):
        """
        Choose worst cards to give away (farthest from being playable).
        Similar to HTML implementation's strategic card selection.
        """
        scored_cards = []
        
        for card in player.hand:
            score = 0
            board = state.board
            
            # If suit is empty and card is not 5, very bad to give away (hard to play)
            if board.is_empty(card.suit) and card.rank != 5:
                score = 100
            
            # If suit has cards, score based on distance from playable positions
            elif not board.is_empty(card.suit):
                min_max = board.get_min_max(card.suit)
                if min_max[0] is not None and min_max[1] is not None:
                    min_rank, max_rank = min_max[0], min_max[1]
                    
                    # Distance from either end of the sequence
                    dist_to_min = abs(card.rank - (min_rank - 1)) if min_rank > 1 else 999
                    dist_to_max = abs(card.rank - (max_rank + 1)) if max_rank < 12 else 999
                    
                    # Farther = worse card = give it away
                    score = min(dist_to_min, dist_to_max)
            
            scored_cards.append((card, score))
        
        # Sort by score descending (worst cards first)
        scored_cards.sort(key=lambda x: x[1], reverse=True)
        
        return [card for card, _ in scored_cards[:count]]
    
    def __str__(self) -> str:
        return "RollDice"



@dataclass
class Pass(Move):
    """Pass the turn (voluntary or forced)."""
    voluntary: bool = False
    
    def is_legal(self, state: GameState, player_index: int) -> bool:
        """
        Passing is always legal.
        Voluntary pass requires having at least one legal card.
        """
        return True
    
    def apply(self, state: GameState) -> GameState:
        """Apply the pass."""
        new_state = state.copy()
        
        # Apply penalty if voluntary pass
        if self.voluntary:
            player = new_state.get_current_player()
            player.round_score -= new_state.variant.voluntary_pass_penalty
        
        # Advance to next player
        new_state.current_player = new_state.next_player_index()
        new_state.turn_number += 1
        
        # Clear any one-shot dice effects
        new_state.dice_state.clear()
        
        return new_state
    
    def __str__(self) -> str:
        return f"Pass(voluntary={self.voluntary})"


class Rules:
    """Game rules engine."""
    
    @staticmethod
    def get_legal_moves(state: GameState) -> List[Move]:
        """Get all legal moves for the current player."""
        if state.game_over:
            return []
        
        player_index = state.current_player
        player = state.players[player_index]
        legal_moves = []
        
        # Check which cards can be played
        playable_cards = []
        for card in player.hand:
            play_move = PlayCard(card)
            if play_move.is_legal(state, player_index):
                legal_moves.append(play_move)
                playable_cards.append(card)
        
        # If player has playable cards, they can also roll dice
        if len(playable_cards) > 0:
            roll_move = RollDice()
            if roll_move.is_legal(state, player_index):
                legal_moves.append(roll_move)
            
            # Voluntary pass (with penalty)
            legal_moves.append(Pass(voluntary=True))
        else:
            # Forced pass (no penalty)
            legal_moves.append(Pass(voluntary=False))
        
        return legal_moves
    
    @staticmethod
    def apply_move(state: GameState, move: Move) -> GameState:
        """Apply a move and return new state."""
        if not move.is_legal(state, state.current_player):
            raise ValueError(f"Illegal move: {move}")
        
        return move.apply(state)
    
    @staticmethod
    def is_terminal(state: GameState) -> bool:
        """Check if the game is over."""
        return state.game_over
    
    @staticmethod
    def get_winner(state: GameState) -> Optional[int]:
        """Get the winner if game is over."""
        return state.winner if state.game_over else None
    
    @staticmethod
    def compute_round_scores(state: GameState):
        """
        Compute round scores when a player goes out.
        Handles both WINNER_TAKES_ALL and DOUBLE_PENALTY scoring modes.
        Modifies player round_scores and match_scores in place.
        """
        from game.entities import ScoringMode
        
        if not state.game_over or state.winner is None:
            return
        
        winner = state.players[state.winner]
        variant = state.variant
        
        # Calculate total cards in opponent hands
        total_opponent_cards = sum(
            p.hand_size() 
            for p in state.players 
            if p.index != state.winner
        )
        
        if variant.scoring_mode == ScoringMode.WINNER_TAKES_ALL:
            # Winner gets points equal to sum of all opponent cards
            winner_bonus = total_opponent_cards * variant.points_per_card
            winner.round_score += winner_bonus
            # Losers get nothing (no penalty)
            
        elif variant.scoring_mode == ScoringMode.DOUBLE_PENALTY:
            # Winner gets points for opponent cards
            winner_bonus = total_opponent_cards * variant.points_per_card
            winner.round_score += winner_bonus
            
            # Each loser loses points for their own cards
            for player in state.players:
                if player.index != state.winner:
                    penalty = player.hand_size() * variant.points_per_card
                    player.round_score -= penalty
        
        # Update match scores for all players
        for player in state.players:
            player.match_score += player.round_score
    
    @staticmethod
    def initialize_game(num_players: int, variant: VariantConfig) -> GameState:
        """Initialize a new game state."""
        from game.entities import Deck, Board, Player
        
        # Create and shuffle deck
        deck = Deck()
        deck.shuffle()
        
        # Deal cards
        hands = deck.deal(num_players)
        
        # Create players
        players = [
            Player(index=i, hand=hand)
            for i, hand in enumerate(hands)
        ]
        
        # Create initial state
        state = GameState(
            board=Board(),
            players=players,
            current_player=0,
            variant=variant
        )
        
        return state
