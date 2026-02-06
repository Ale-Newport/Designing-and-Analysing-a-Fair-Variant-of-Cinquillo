#!/usr/bin/env python3
"""
Match simulator for Cinquillo 2.0.
Handles multi-round matches with configurable end conditions.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from typing import List, Optional
from game.entities import GameState, VariantConfig, MatchEndMode
from game.rules import Rules
from agents.base_agents import Agent


@dataclass
class RoundResult:
    """Result of a single round in a match."""
    winner: int
    scores_after_round: List[int]  # Match scores after this round
    num_turns: int
    cards_remaining: List[int]  # Cards in each player's hand


@dataclass
class MatchResult:
    """Complete match result."""
    winner: int
    final_scores: List[int]
    rounds_played: int
    round_results: List[RoundResult] = field(default_factory=list)
    agent_names: List[str] = field(default_factory=list)


class MatchSimulator:
    """Simulates complete multi-round matches."""
    
    def __init__(self, agents: List[Agent], variant: VariantConfig, verbose: bool = False):
        """
        Initialize match simulator.
        
        Args:
            agents: List of agents (one per player)
            variant: Game variant configuration
            verbose: If True, print match progress
        """
        self.agents = agents
        self.variant = variant
        self.verbose = verbose
        self.num_players = len(agents)
    
    def play_match(self) -> MatchResult:
        """
        Play a complete match according to variant settings.
        
        Returns:
            MatchResult with winner and statistics
        """
        # Initialize match scores
        match_scores = [0] * self.num_players
        round_results = []
        round_number = 0
        
        # Determine match end condition
        if self.variant.match_end_mode == MatchEndMode.TARGET_SCORE:
            target = self.variant.get_target_score(self.num_players)
            
            # Play rounds until someone reaches target
            while max(match_scores) < target:
                round_number += 1
                round_result = self._play_round(round_number, match_scores)
                round_results.append(round_result)
                
                # Update match scores
                match_scores = round_result.scores_after_round.copy()
                
                if self.verbose:
                    print(f"Round {round_number}: Winner P{round_result.winner}, "
                          f"Scores: {match_scores}, Target: {target}")
            
        elif self.variant.match_end_mode == MatchEndMode.FIXED_ROUNDS:
            # Play fixed number of rounds
            for round_num in range(1, self.variant.fixed_rounds_count + 1):
                round_result = self._play_round(round_num, match_scores)
                round_results.append(round_result)
                
                # Update match scores
                match_scores = round_result.scores_after_round.copy()
                
                if self.verbose:
                    print(f"Round {round_num}/{self.variant.fixed_rounds_count}: "
                          f"Winner P{round_result.winner}, Scores: {match_scores}")
        
        # Determine match winner (highest score)
        match_winner = match_scores.index(max(match_scores))
        
        return MatchResult(
            winner=match_winner,
            final_scores=match_scores,
            rounds_played=len(round_results),
            round_results=round_results,
            agent_names=[agent.name for agent in self.agents]
        )
    
    def _play_round(self, round_number: int, current_match_scores: List[int]) -> RoundResult:
        """
        Play a single round.
        
        Args:
            round_number: Which round this is (1-indexed)
            current_match_scores: Current match scores before this round
        
        Returns:
            RoundResult with round winner and updated scores
        """
        # Initialize game state
        state = Rules.initialize_game(self.num_players, self.variant)
        state.round_number = round_number
        
        # Set match scores
        for i, player in enumerate(state.players):
            player.match_score = current_match_scores[i]
        
        # Play until someone wins the round
        turn_count = 0
        max_turns = 1000  # Safety limit
        
        while not Rules.is_terminal(state) and turn_count < max_turns:
            current_player_idx = state.current_player
            agent = self.agents[current_player_idx]
            
            # Get legal moves
            legal_moves = Rules.get_legal_moves(state)
            
            if not legal_moves:
                break  # Game stuck (shouldn't happen with proper rules)
            
            # Agent chooses move
            move = agent.choose_move(state, legal_moves)
            
            # Apply move
            state = move.apply(state)
            turn_count += 1
        
        # Compute scores for this round
        Rules.compute_round_scores(state)
        
        # Get updated match scores
        match_scores_after = [p.match_score for p in state.players]
        cards_remaining = [p.hand_size() for p in state.players]
        
        return RoundResult(
            winner=state.winner,
            scores_after_round=match_scores_after,
            num_turns=turn_count,
            cards_remaining=cards_remaining
        )


def quick_match_demo():
    """Quick demo of match simulation."""
    from agents.base_agents import create_balanced_heuristic, create_aggressive_heuristic, create_defensive_heuristic
    from agents.mcts_agent import MCTSAgentDeep
    from agents.rl_agent import RLAgent
    from game.entities import ScoringMode
    
    # Create agents
    agents = [
        MCTSAgentDeep(),
        create_balanced_heuristic(),
        create_defensive_heuristic(),
        create_aggressive_heuristic()
    ]
    
    # Test TARGET_SCORE mode
    print("=== TARGET SCORE MODE (20 points) ===")
    
    variant_target = VariantConfig(
        match_end_mode=MatchEndMode.TARGET_SCORE,
        target_score_multiplier=5,  # 4 players * 5 = 20 target
        scoring_mode=ScoringMode.WINNER_TAKES_ALL,
        points_per_card=1
    )
    
    simulator = MatchSimulator(agents, variant_target, verbose=True)
    result = simulator.play_match()
    
    print(f"\nMatch Winner: {result.agent_names[result.winner]}")
    print(f"Final Scores: {result.final_scores}")
    print(f"Rounds Played: {result.rounds_played}")
    
    # Test FIXED_ROUNDS mode
    print("\n=== FIXED ROUNDS MODE (5 rounds) ===")
    variant_fixed = VariantConfig(
        match_end_mode=MatchEndMode.FIXED_ROUNDS,
        fixed_rounds_count=5,
        scoring_mode=ScoringMode.DOUBLE_PENALTY,
        points_per_card=1
    )
    
    simulator = MatchSimulator(agents, variant_fixed, verbose=True)
    result = simulator.play_match()
    
    print(f"\nMatch Winner: {result.agent_names[result.winner]}")
    print(f"Final Scores: {result.final_scores}")
    print(f"Rounds Played: {result.rounds_played}")


if __name__ == "__main__":
    quick_match_demo()
