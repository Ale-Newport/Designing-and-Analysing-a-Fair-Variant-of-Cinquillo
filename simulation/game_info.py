#!/usr/bin/env python3
"""
Game visualization tool for Cinquillo 2.0.
Provides detailed step-by-step output of game progression for debugging.

Place this file in your simulation/ folder.
Make sure your project structure has:
- game/entities.py
- game/rules.py
- agents/base_agents.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List
from game.entities import GameState, VariantConfig, Card, Suit
from game.rules import Rules, PlayCard, RollDice, Pass
from agents.base_agents import Agent, create_balanced_heuristic, create_aggressive_heuristic, create_defensive_heuristic, RandomAgent


class GameVisualizer:
    """Visualizes game progression with detailed output."""
    
    def __init__(self, agents: List[Agent], variant: VariantConfig, output_file: str = None):
        self.agents = agents
        self.variant = variant
        self.num_players = len(agents)
        self.output_file = output_file
        self.file_handle = None
    
    def _write(self, text: str = ""):
        """Write text to file or stdout."""
        if self.file_handle:
            self.file_handle.write(text + "\n")
        else:
            print(text)
    
    def play_visualized_game(self):
        """Play a single game with detailed visualization."""
        # Open output file if specified
        if self.output_file:
            self.file_handle = open(self.output_file, 'w', encoding='utf-8')
        
        try:
            self._write("=" * 80)
            self._write("CINQUILLO 2.0 - GAME VISUALIZATION")
            self._write("=" * 80)
            
            # Initialize game
            state = Rules.initialize_game(self.num_players, self.variant)
            
            # Show initial state
            self._print_game_header(state)
            self._print_initial_hands(state)
            
            # Play until game over
            turn_count = 0
            max_turns = 1000
            
            while not Rules.is_terminal(state) and turn_count < max_turns:
                turn_count += 1
                
                self._write()
                self._write("=" * 80)
                self._write(f"TURN {turn_count}")
                self._write("=" * 80)
                
                # Show current state
                current_player_idx = state.current_player
                agent = self.agents[current_player_idx]
                
                self._print_turn_header(state, agent)
                self._print_board_state(state)
                self._print_player_hand(state, current_player_idx)
                
                # Get and display legal moves
                legal_moves = Rules.get_legal_moves(state)
                self._print_legal_moves(legal_moves)
                
                # Agent chooses move
                chosen_move = agent.choose_move(state, legal_moves)
                self._print_chosen_move(agent, chosen_move)
                
                # Apply move and show outcome
                new_state = chosen_move.apply(state)
                self._print_move_outcome(state, new_state, chosen_move)
                
                state = new_state
            
            # Game over
            self._print_game_over(state, turn_count)
            
            if self.output_file:
                self._write()
                self._write(f"Output saved to: {self.output_file}")
        
        finally:
            # Close file if it was opened
            if self.file_handle:
                self.file_handle.close()
                self.file_handle = None
    
    def _print_game_header(self, state: GameState):
        """Print game configuration header."""
        self._write()
        self._write(f"Players: {self.num_players}")
        self._write(f"Agents: {', '.join([f'P{i}: {agent.name}' for i, agent in enumerate(self.agents)])}")
        self._write(f"Variant: {state.variant}")
        self._write()
    
    def _print_initial_hands(self, state: GameState):
        """Print all players' initial hands."""
        self._write("INITIAL HANDS")
        self._write("-" * 80)
        for i, player in enumerate(state.players):
            agent_name = self.agents[i].name
            hand_str = self._format_hand(player.hand)
            self._write(f"Player {i} ({agent_name}): {hand_str} [{len(player.hand)} cards]")
        self._write()
    
    def _print_turn_header(self, state: GameState, agent: Agent):
        """Print turn header with current player info."""
        player = state.get_current_player()
        self._write()
        self._write(f"Current Player: P{state.current_player} ({agent.name})")
        self._write(f"Hand Size: {player.hand_size()} cards")
        self._write(f"Match Score: {player.match_score}")
        
        # Show dice state if active
        if state.dice_state.wild_active:
            self._write("🎲 WILD ACTIVE - Can play any card!")
        if state.dice_state.double_play_active:
            self._write("🎲 DOUBLE PLAY ACTIVE - Can play two cards!")
        if state.dice_state.revealed_player is not None:
            self._write(f"🎲 Player {state.dice_state.revealed_player}'s hand is revealed")
        self._write()
    
    def _print_board_state(self, state: GameState):
        """Print current board state."""
        self._write("BOARD STATE")
        self._write("-" * 80)
        
        for suit in Suit:
            cards = state.board.suit_cards[suit]
            if cards:
                sorted_ranks = sorted(cards)
                cards_str = ", ".join([f"{rank}" for rank in sorted_ranks])
                min_max = state.board.get_min_max(suit)
                self._write(f"{suit.value:8s}: [{cards_str}] (range: {min_max[0]}-{min_max[1]})")
            else:
                self._write(f"{suit.value:8s}: [empty]")
        self._write()
    
    def _print_player_hand(self, state: GameState, player_idx: int):
        """Print current player's hand organized by suit."""
        player = state.players[player_idx]
        self._write("CURRENT HAND")
        self._write("-" * 80)
        
        # Organize by suit
        by_suit = {suit: [] for suit in Suit}
        for card in player.hand:
            by_suit[card.suit].append(card.rank)
        
        for suit in Suit:
            if by_suit[suit]:
                sorted_ranks = sorted(by_suit[suit])
                cards_str = ", ".join([str(r) for r in sorted_ranks])
                self._write(f"{suit.value:8s}: {cards_str}")
            else:
                self._write(f"{suit.value:8s}: -")
        self._write()
    
    def _print_legal_moves(self, legal_moves: List):
        """Print all legal moves."""
        self._write("LEGAL MOVES")
        self._write("-" * 80)
        
        play_cards = []
        other_moves = []
        
        for move in legal_moves:
            if isinstance(move, PlayCard):
                play_cards.append(move)
            else:
                other_moves.append(move)
        
        if play_cards:
            cards_str = ", ".join([f"{m.card}" for m in play_cards])
            self._write(f"Play Cards: {cards_str}")
        
        for move in other_moves:
            self._write(f"Other: {move}")
        
        self._write(f"Total: {len(legal_moves)} legal moves")
        self._write()
    
    def _print_chosen_move(self, agent: Agent, move):
        """Print the move chosen by the agent."""
        self._write("CHOSEN MOVE")
        self._write("-" * 80)
        self._write(f"{agent.name} chooses: {move}")
        self._write()
    
    def _print_move_outcome(self, old_state: GameState, new_state: GameState, move):
        """Print the outcome of the move."""
        self._write("MOVE OUTCOME")
        self._write("-" * 80)
        
        if isinstance(move, PlayCard):
            self._write(f"✓ Card {move.card} played to board")
            self._write(f"✓ Hand size: {old_state.get_current_player().hand_size()} → {new_state.players[old_state.current_player].hand_size()}")
            
            # Check if player won
            if new_state.game_over:
                self._write(f"🏆 Player {new_state.winner} wins the round!")
        
        elif isinstance(move, RollDice):
            self._write("🎲 Dice rolled!")
            
            # Detect what changed
            if new_state.dice_state.wild_active and not old_state.dice_state.wild_active:
                self._write("✓ GOOD OUTCOME: Wild card activated - can play any card!")
            elif new_state.dice_state.double_play_active and not old_state.dice_state.double_play_active:
                self._write("✓ GOOD OUTCOME: Double play activated - can play two cards!")
            elif new_state.dice_state.revealed_player != old_state.dice_state.revealed_player:
                if new_state.dice_state.revealed_player == old_state.current_player:
                    self._write("✗ BAD OUTCOME: Your hand is revealed!")
                else:
                    self._write(f"✓ GOOD OUTCOME: Player {new_state.dice_state.revealed_player}'s hand revealed!")
            else:
                # Check for card transfers
                old_hand = old_state.get_current_player().hand_size()
                new_hand = new_state.players[old_state.current_player].hand_size()
                
                if new_hand > old_hand:
                    self._write(f"✗ BAD OUTCOME: Received {new_hand - old_hand} cards (hand: {old_hand} → {new_hand})")
                elif new_hand < old_hand:
                    self._write(f"✓ GOOD OUTCOME: Gave away {old_hand - new_hand} cards (hand: {old_hand} → {new_hand})")
                else:
                    # Check score change
                    old_score = old_state.get_current_player().match_score
                    new_score = new_state.players[old_state.current_player].match_score
                    if new_score < old_score:
                        self._write(f"✗ BAD OUTCOME: Lost {old_score - new_score} points (score: {old_score} → {new_score})")
                    elif old_state.current_player != new_state.current_player:
                        self._write("✗ BAD OUTCOME: Forced to pass turn")
        
        elif isinstance(move, Pass):
            if move.voluntary:
                penalty = old_state.variant.voluntary_pass_penalty
                self._write(f"⚠ Voluntary pass - penalty of {penalty} points applied")
                old_score = old_state.get_current_player().round_score
                new_score = new_state.players[old_state.current_player].round_score
                self._write(f"  Round score: {old_score} → {new_score}")
            else:
                self._write("⚠ Forced pass - no playable cards, no penalty")
        
        # Show turn progression
        if old_state.current_player != new_state.current_player:
            self._write(f"→ Turn passes to Player {new_state.current_player} ({self.agents[new_state.current_player].name})")
        else:
            self._write(f"→ Player {new_state.current_player} continues (double play)")
        self._write()
    
    def _print_game_over(self, state: GameState, turn_count: int):
        """Print game over summary."""
        self._write()
        self._write("=" * 80)
        self._write("GAME OVER")
        self._write("=" * 80)
        
        # Compute final scores
        Rules.compute_round_scores(state)
        
        self._write()
        self._write(f"🏆 Winner: Player {state.winner} ({self.agents[state.winner].name})")
        self._write(f"Total Turns: {turn_count}")
        self._write()
        
        self._write("FINAL SCORES")
        self._write("-" * 80)
        for i, player in enumerate(state.players):
            agent_name = self.agents[i].name
            cards_left = player.hand_size()
            round_score = player.round_score
            match_score = player.match_score
            
            status = "🏆 WINNER" if i == state.winner else f"{cards_left} cards left"
            self._write(f"P{i} ({agent_name:12s}): Round Score: {round_score:+4d}, Match Score: {match_score:+4d} - {status}")
        self._write()
    
    def _format_hand(self, hand: List[Card]) -> str:
        """Format a hand of cards as a string."""
        if not hand:
            return "[empty]"
        
        # Group by suit
        by_suit = {suit: [] for suit in Suit}
        for card in hand:
            by_suit[card.suit].append(card.rank)
        
        parts = []
        for suit in Suit:
            if by_suit[suit]:
                sorted_ranks = sorted(by_suit[suit])
                suit_str = f"{suit.value[0]}:{','.join([str(r) for r in sorted_ranks])}"
                parts.append(suit_str)
        
        return " | ".join(parts)


def main():
    """Run a visualized game."""
    from game.entities import ScoringMode, MatchEndMode
    from datetime import datetime
    
    # Create agents
    agents = [
        create_balanced_heuristic(),
        create_aggressive_heuristic(),
        create_defensive_heuristic(),
        RandomAgent("Random", avoid_bad_moves=True)
    ]
    
    # Configure variant
    variant = VariantConfig(
        scoring_mode=ScoringMode.DOUBLE_PENALTY,
        match_end_mode=MatchEndMode.FIXED_ROUNDS,
        fixed_rounds_count=1,  # Just one round for visualization
        points_per_card=1,
        voluntary_pass_penalty=1
    )
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"game_visualization_{timestamp}.txt"
    
    # Run visualized game
    visualizer = GameVisualizer(agents, variant, output_file=output_file)
    visualizer.play_visualized_game()
    
    print(f"✓ Game visualization saved to: {output_file}")


if __name__ == "__main__":
    main()