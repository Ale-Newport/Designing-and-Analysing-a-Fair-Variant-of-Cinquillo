"""
Move types and game rules for Cinquillo 2.0.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from game.entities import Card, GameState, Player, VariantConfig
import random


class Move(ABC):
    """Abstract base class for all moves"""
    
    @abstractmethod
    def is_legal(self, state: GameState, player_index: int) -> bool:
        """check if this move is legal in the given state"""
        pass
    
    @abstractmethod
    def apply(self, state: GameState) -> GameState:
        """apply this move to the state and return new state"""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass
class PlayCard(Move):
    """play a card from hand to the board"""
    card: Card
    
    def is_legal(self, state: GameState, player_index: int) -> bool:
        """check if playing this card is legal"""
        player = state.players[player_index]
        
        #must have the card
        if not player.has_card(self.card):
            return False
        
        #if wild is active, any card can be played
        if state.dice_state.wild_active:
            return True
        
        #if suit not on board, must be a 5
        if state.board.is_empty(self.card.suit):
            return self.card.rank == 5
        
        #if suit is on board, must be adjacent or a 5
        if self.card.rank == 5:
            return not state.board.has_rank(self.card.suit, 5)
        
        return (state.board.is_adjacent(self.card) and 
                not state.board.has_rank(self.card.suit, self.card.rank))
    
    def apply(self, state: GameState) -> GameState:
        """apply the card play"""
        new_state = state.copy()
        outgoing_player = new_state.current_player
        player = new_state.get_current_player()
        
        #remove card from hand and add to board
        player.remove_card(self.card)
        new_state.board.add_card(self.card)
        
        #check if player won
        if player.hand_size() == 0:
            new_state.game_over = True
            new_state.winner = new_state.current_player
        
        #clear wild if it was active
        if new_state.dice_state.wild_active:
            new_state.dice_state.wild_active = False
        
        #if double play is active, keep it for one more card this same turn, otherwise advance to next player and clear the outgoing player's reveal.
        if not new_state.dice_state.double_play_active:
            if not new_state.game_over:
                new_state.current_player = new_state.next_player_index()
                new_state.turn_number += 1
                new_state.dice_state.clear_viewer(outgoing_player)
        else:
            new_state.dice_state.double_play_active = False
        
        return new_state
    
    def __str__(self) -> str:
        return f"PlayCard({self.card})"


class RollDice(Move):
    """roll dice to trigger one of two outcomes (good or bad)"""
    
    def is_legal(self, state: GameState, player_index: int) -> bool:
        """can only roll if player has at least one legal card to play"""
        player = state.players[player_index]
        
        # check if any card in hand is playable
        for card in player.hand:
            play_move = PlayCard(card)
            if play_move.is_legal(state, player_index):
                return True
        return False
    
    def apply(self, state: GameState) -> GameState:
        """apply dice roll with configurable good/bad outcomes"""
        new_state = state.copy()
        variant = new_state.variant
        
        # determine outcome: good or bad
        is_good_outcome = random.random() < variant.dice_good_probability
        
        if is_good_outcome:
            self._apply_good_effect(new_state, variant.dice_good_effect)
        else:
            self._apply_bad_effect(new_state, variant.dice_bad_effect, variant)
        
        return new_state
    
    def _apply_good_effect(self, state: GameState, effect):
        """apply the good dice effect"""
        from game.entities import GoodDiceEffect
        
        if effect == GoodDiceEffect.WILD:
            # player can play any card ignoring adjacency rules
            state.dice_state.wild_active = True
            
        elif effect == GoodDiceEffect.DOUBLE_PLAY:
            # player can play up to two individually-legal cards this turn
            state.dice_state.double_play_active = True
            
        elif effect == GoodDiceEffect.GIVE_CARD:
            # give one card to the next player in turn order (q), reducing own hand
            current_player = state.get_current_player()
            if current_player.hand_size() > 0:
                target_idx = state.next_player_index()
                card = current_player.hand.pop()
                state.players[target_idx].add_card(card)

            # if the giver gave away their last card they've gone out, end the game.
            if current_player.hand_size() == 0:
                state.game_over = True
                state.winner = state.current_player
                
        elif effect == GoodDiceEffect.INFO_REVEAL:
            # player p sees the full hand of the opponent with the FEWEST cards
            target_idx = state.get_player_with_fewest_cards(exclude=state.current_player)
            state.dice_state.reveal_to(state.current_player, target_idx)
    
    def _apply_bad_effect(self, state: GameState, effect, variant):
        """apply the bad dice effect"""
        from game.entities import BadDiceEffect
        
        current_player = state.get_current_player()
        
        if effect == BadDiceEffect.TAKE_CARDS:
            # receive cards from player k, the opponent with the MOST cards
            donor_idx = state.get_player_with_most_cards(exclude=state.current_player)
            donor = state.players[donor_idx]
            
            cards_to_take = min(variant.dice_bad_cards_count, donor.hand_size())
            
            # transfer worst cards (farthest from playable)
            cards_to_transfer = self._choose_worst_cards(donor, cards_to_take, state)
            for card in cards_to_transfer:
                donor.remove_card(card)
                current_player.add_card(card)

            # if the donor was drained to 0 cards they've gone out, end the game.
            if donor.hand_size() == 0:
                state.game_over = True
                state.winner = donor_idx
                
        elif effect == BadDiceEffect.NEGATIVE_POINTS:
            # lose points immediately
            current_player.match_score -= variant.dice_bad_penalty_points
            
        elif effect == BadDiceEffect.FORCED_PASS:
            # turn ends immediately — involuntary, no penalty applied
            outgoing = state.current_player
            state.current_player = state.next_player_index()
            state.turn_number += 1
            state.dice_state.clear_viewer(outgoing)
            
        elif effect == BadDiceEffect.REVEAL_HAND:
            # all opponents can see the current player's (p's) full hand.
            # each opponent i gets: revealed_hands[i] = p
            # they will use this information on their own turns.
            p = state.current_player
            for i in range(len(state.players)):
                if i != p:
                    state.dice_state.reveal_to(i, p)
    
    def _choose_worst_cards(self, player, count: int, state: GameState):
        """
        choose worst cards to give away (farthest from being playable)
        """
        scored_cards = []
        
        for card in player.hand:
            score = 0
            board = state.board
            
            # if suit is empty and card is not 5, very bad to give away (hard to play)
            if board.is_empty(card.suit) and card.rank != 5:
                score = 100
            
            # if suit has cards, score based on distance from playable positions
            elif not board.is_empty(card.suit):
                min_max = board.get_min_max(card.suit)
                if min_max[0] is not None and min_max[1] is not None:
                    min_rank, max_rank = min_max[0], min_max[1]
                    
                    # distance from either end of the sequence
                    dist_to_min = abs(card.rank - (min_rank - 1)) if min_rank > 1 else 999
                    dist_to_max = abs(card.rank - (max_rank + 1)) if max_rank < 12 else 999
                    
                    # farther = worse card = give it away
                    score = min(dist_to_min, dist_to_max)
            
            scored_cards.append((card, score))
        
        # sort by score descending (worst cards first)
        scored_cards.sort(key=lambda x: x[1], reverse=True)
        
        return [card for card, _ in scored_cards[:count]]
    
    def __str__(self) -> str:
        return "RollDice"


@dataclass
class Pass(Move):
    """Pass the turn (voluntary or forced)"""
    voluntary: bool = False
    
    def is_legal(self, state: GameState, player_index: int) -> bool:
        """passing is always legal"""
        return True
    
    def apply(self, state: GameState) -> GameState:
        """apply the pass"""
        new_state = state.copy()
        
        # Apply penalty if voluntary pass
        if self.voluntary:
            player = new_state.get_current_player()
            player.round_score -= new_state.variant.voluntary_pass_penalty
        
        # Advance to next player
        new_state.current_player = new_state.next_player_index()
        new_state.turn_number += 1
        
        # Clear all one-shot dice effects (wild, double-play, and all reveal entries)
        new_state.dice_state.clear()
        
        return new_state
    
    def __str__(self) -> str:
        return f"Pass(voluntary={self.voluntary})"


class Rules:
    """Game rules engine"""
    
    @staticmethod
    def get_legal_moves(state: GameState) -> List[Move]:
        """get all legal moves for the current player"""
        if state.game_over:
            return []
        
        player_index = state.current_player
        player = state.players[player_index]
        legal_moves = []
        
        # check which cards can be played
        playable_cards = []
        for card in player.hand:
            play_move = PlayCard(card)
            if play_move.is_legal(state, player_index):
                legal_moves.append(play_move)
                playable_cards.append(card)
        
        # if player has playable cards, they can also roll dice
        if len(playable_cards) > 0:
            roll_move = RollDice()
            if roll_move.is_legal(state, player_index):
                legal_moves.append(roll_move)
            
            # voluntary pass (with penalty)
            legal_moves.append(Pass(voluntary=True))
        else:
            # forced pass (no penalty)
            legal_moves.append(Pass(voluntary=False))
        
        return legal_moves
    
    @staticmethod
    def apply_move(state: GameState, move: Move) -> GameState:
        """apply a move and return new state"""
        if not move.is_legal(state, state.current_player):
            raise ValueError(f"Illegal move: {move}")
        
        return move.apply(state)
    
    @staticmethod
    def is_terminal(state: GameState) -> bool:
        """check if the game is over"""
        return state.game_over
    
    @staticmethod
    def get_winner(state: GameState) -> Optional[int]:
        """get the winner if game is over"""
        return state.winner if state.game_over else None
    
    @staticmethod
    def compute_round_scores(state: GameState):
        """compute round scores when a player goes out"""
        from game.entities import ScoringMode
        
        if not state.game_over or state.winner is None:
            return
        
        winner = state.players[state.winner]
        variant = state.variant
        
        # calculate total cards in opponent hands
        total_opponent_cards = sum(
            p.hand_size() 
            for p in state.players 
            if p.index != state.winner
        )
        
        if variant.scoring_mode == ScoringMode.WINNER_TAKES_ALL:
            # winner gets points equal to sum of all opponent cards
            winner_bonus = total_opponent_cards * variant.points_per_card
            winner.round_score += winner_bonus
            # losers get nothing (no penalty)
            
        elif variant.scoring_mode == ScoringMode.DOUBLE_PENALTY:
            # winner gets points for opponent cards
            winner_bonus = total_opponent_cards * variant.points_per_card
            winner.round_score += winner_bonus
            
            # each loser loses points for their own cards
            for player in state.players:
                if player.index != state.winner:
                    penalty = player.hand_size() * variant.points_per_card
                    player.round_score -= penalty
        
        # update match scores for all players
        for player in state.players:
            player.match_score += player.round_score
    
    @staticmethod
    def initialize_game(num_players: int, variant: VariantConfig) -> GameState:
        """initialize a new game state"""
        from game.entities import Deck, Board, Player
        
        # Create and shuffle deck
        deck = Deck()
        deck.shuffle()
        
        # deal cards
        hands = deck.deal(num_players)
        
        # create players
        players = [
            Player(index=i, hand=hand)
            for i, hand in enumerate(hands)
        ]
        
        # create initial state
        state = GameState(
            board=Board(),
            players=players,
            current_player=0,
            variant=variant
        )
        
        return state