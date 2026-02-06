#!/usr/bin/env python3
"""
Tournament and simulation framework for Cinquillo 2.0.
Runs games between agents and collects statistics for analysis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import random
from collections import defaultdict
import time

from game.entities import GameState, VariantConfig
from game.rules import Rules, Move
from agents.base_agents import Agent


@dataclass
class GameResult:
    """Result of a single game."""
    winner: int
    final_scores: List[int]
    num_turns: int
    player_names: List[str]
    starting_positions: List[int]  # Maps agent to starting position
    round_number: int = 0
    
    def get_winner_name(self) -> str:
        """Get name of winning player."""
        return self.player_names[self.winner]
    
    def get_position_of_agent(self, agent_index: int) -> int:
        """Get starting position of an agent."""
        return self.starting_positions[agent_index]


@dataclass
class TournamentResult:
    """Results from a tournament."""
    games: List[GameResult] = field(default_factory=list)
    agent_names: List[str] = field(default_factory=list)
    variant_config: Optional[VariantConfig] = None
    total_games: int = 0
    total_time: float = 0.0
    
    def add_game(self, result: GameResult):
        """Add a game result."""
        self.games.append(result)
        self.total_games += 1
    
    def compute_statistics(self) -> Dict:
        """Compute summary statistics from tournament."""
        if not self.games:
            return {}
        
        num_agents = len(self.agent_names)
        
        # Win rates by agent
        wins_by_agent = defaultdict(int)
        games_by_agent = defaultdict(int)
        
        # Win rates by position
        wins_by_position = defaultdict(int)
        games_by_position = defaultdict(int)
        
        # Scores
        scores_by_agent = defaultdict(list)
        
        # Game lengths
        game_lengths = []
        
        for game in self.games:
            game_lengths.append(game.num_turns)
            
            # Track wins by agent
            winner_name = game.get_winner_name()
            wins_by_agent[winner_name] += 1
            
            # Track games played by each agent
            for name in game.player_names:
                games_by_agent[name] += 1
            
            # Track wins by starting position
            winner_position = game.get_position_of_agent(game.winner)
            wins_by_position[winner_position] += 1
            
            # IMPORTANT: Count games for ALL positions, not just winner
            for pos in range(len(game.player_names)):
                games_by_position[pos] += 1
            
            # Track scores
            for i, (name, score) in enumerate(zip(game.player_names, game.final_scores)):
                scores_by_agent[name].append(score)
        
        # Compute win rates
        agent_win_rates = {
            name: wins_by_agent[name] / games_by_agent[name]
            if games_by_agent[name] > 0 else 0.0
            for name in self.agent_names
        }
        
        position_win_rates = {
            pos: wins_by_position[pos] / games_by_position[pos]
            if games_by_position[pos] > 0 else 0.0
            for pos in range(num_agents)
        }
        
        # Compute average scores
        avg_scores = {
            name: sum(scores_by_agent[name]) / len(scores_by_agent[name])
            if len(scores_by_agent[name]) > 0 else 0.0
            for name in self.agent_names
        }
        
        # Game length statistics
        avg_game_length = sum(game_lengths) / len(game_lengths)
        min_game_length = min(game_lengths)
        max_game_length = max(game_lengths)
        
        return {
            'agent_win_rates': agent_win_rates,
            'position_win_rates': position_win_rates,
            'avg_scores': avg_scores,
            'wins_by_agent': dict(wins_by_agent),
            'games_by_agent': dict(games_by_agent),
            'avg_game_length': avg_game_length,
            'min_game_length': min_game_length,
            'max_game_length': max_game_length,
            'total_games': self.total_games,
            'total_time': self.total_time
        }
    
    def print_summary(self):
        """Print tournament summary."""
        stats = self.compute_statistics()
        
        print("\n" + "="*60)
        print("TOURNAMENT SUMMARY")
        print("="*60)
        print(f"Total games: {stats['total_games']}")
        print(f"Total time: {stats['total_time']:.2f}s")
        print(f"Avg game length: {stats['avg_game_length']:.1f} turns")
        print(f"Game length range: {stats['min_game_length']}-{stats['max_game_length']} turns")
        
        print("\n" + "-"*60)
        print("WIN RATES BY AGENT")
        print("-"*60)
        for name in self.agent_names:
            win_rate = stats['agent_win_rates'][name]
            wins = stats['wins_by_agent'].get(name, 0)
            games = stats['games_by_agent'].get(name, 0)
            print(f"{name:20s}: {win_rate:6.2%} ({wins}/{games} wins)")
        
        print("\n" + "-"*60)
        print("WIN RATES BY POSITION")
        print("-"*60)
        num_positions = len(stats['position_win_rates'])
        for pos in range(num_positions):
            win_rate = stats['position_win_rates'][pos]
            print(f"Position {pos} (Seat {pos+1}): {win_rate:6.2%}")
        
        print("\n" + "-"*60)
        print("AVERAGE SCORES")
        print("-"*60)
        for name in self.agent_names:
            avg_score = stats['avg_scores'][name]
            print(f"{name:20s}: {avg_score:8.2f}")
        
        print("="*60 + "\n")


class GameSimulator:
    """Simulates individual games."""
    
    @staticmethod
    def simulate_game(agents: List[Agent], 
                     variant: VariantConfig,
                     verbose: bool = False) -> GameResult:
        """
        Simulate a single game.
        
        Args:
            agents: List of agents (one per player)
            variant: Game variant configuration
            verbose: Print game progress
            
        Returns:
            GameResult with outcome
        """
        # Initialize game
        num_players = len(agents)
        state = Rules.initialize_game(num_players, variant)
        
        turn_count = 0
        max_turns = 1000  # Safety limit
        
        if verbose:
            print(f"\nStarting game with {num_players} players")
            for i, agent in enumerate(agents):
                print(f"  Player {i}: {agent.name}")
        
        # Game loop
        consecutive_passes = 0  # Track consecutive passes to detect deadlock
        while not Rules.is_terminal(state) and turn_count < max_turns:
            current_player_idx = state.current_player
            agent = agents[current_player_idx]
            
            # Get legal moves
            legal_moves = Rules.get_legal_moves(state)
            
            if not legal_moves:
                if verbose:
                    print(f"Turn {turn_count}: Player {current_player_idx} has no moves!")
                break
            
            # Agent chooses move
            move = agent.choose_move(state, legal_moves)
            
            # Track consecutive passes to detect deadlock
            from game.rules import Pass
            if isinstance(move, Pass):
                consecutive_passes += 1
                # If all players passed consecutively, game is deadlocked
                if consecutive_passes >= num_players:
                    if verbose:
                        print(f"Turn {turn_count}: Deadlock detected (all players passed)")
                    break
            else:
                consecutive_passes = 0
            
            if verbose:
                print(f"Turn {turn_count}: Player {current_player_idx} ({agent.name}) plays {move}")
            
            # Apply move
            state = Rules.apply_move(state, move)
            turn_count += 1
        
        # Compute final scores
        if Rules.is_terminal(state):
            Rules.compute_round_scores(state)
        else:
            # Game ended without normal termination (deadlock or max turns)
            # Compute scores based on remaining cards
            for player in state.players:
                cards_left = player.hand_size()
                player.round_score -= cards_left  # Penalty for cards in hand
                player.match_score += player.round_score
        
        winner = Rules.get_winner(state)
        if winner is None:
            # If no winner, find player with highest score
            scores = [p.match_score for p in state.players]
            max_score = max(scores)
            
            # Handle ties fairly - collect all players with max score
            tied_players = [i for i, score in enumerate(scores) if score == max_score]
            
            if len(tied_players) == 1:
                winner = tied_players[0]
            else:
                # Fair tiebreaker: player with fewest cards left wins
                cards_left = [state.players[i].hand_size() for i in tied_players]
                min_cards = min(cards_left)
                finalists = [tied_players[i] for i in range(len(tied_players)) 
                           if cards_left[i] == min_cards]
                
                # If still tied, random selection
                import random as rnd
                winner = rnd.choice(finalists)
        
        final_scores = [p.match_score for p in state.players]
        player_names = [agent.name for agent in agents]
        starting_positions = list(range(num_players))
        
        if verbose:
            print(f"\nGame ended after {turn_count} turns")
            print(f"Winner: Player {winner} ({player_names[winner]})")
            print(f"Final scores: {final_scores}")
        
        return GameResult(
            winner=winner,
            final_scores=final_scores,
            num_turns=turn_count,
            player_names=player_names,
            starting_positions=starting_positions
        )


class Tournament:
    """Runs tournaments between multiple agents."""
    
    def __init__(self, agents: List[Agent], variant: VariantConfig):
        """
        Initialize tournament.
        
        Args:
            agents: List of agents to compete
            variant: Game variant configuration
        """
        self.agents = agents
        self.variant = variant
        self.agent_names = [agent.name for agent in agents]
    
    def run_round_robin(self, 
                    games_per_matchup: int = 100,
                    rotate_positions: bool = True,
                    verbose: bool = False) -> TournamentResult:
        """
        Run round-robin tournament where each agent plays against others.
        
        Args:
            games_per_matchup: Number of games per unique agent combination
            rotate_positions: Whether to rotate starting positions
            verbose: Print progress
            
        Returns:
            TournamentResult with all game outcomes
        """
        result = TournamentResult(
            agent_names=self.agent_names,
            variant_config=self.variant
        )
        
        num_agents = len(self.agents)
        
        if num_agents < 2:
            raise ValueError("Need at least 2 agents for tournament")
        
        start_time = time.time()
        games_played = 0
        
        # Generate all matchups
        from itertools import combinations
        
        if num_agents == 2:
            matchups = [list(range(num_agents))]
        elif num_agents == 3:
            matchups = [list(range(num_agents))]
        elif num_agents == 4:
            matchups = [list(range(num_agents))]
        else:
            target_players = 4
            if num_agents >= target_players:
                matchups = list(combinations(range(num_agents), target_players))
            else:
                matchups = [list(range(num_agents))]
        
        total_matchups = len(matchups)
        total_games_to_play = total_matchups * games_per_matchup
        
        # Try to import tqdm for progress bar
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            print(f"\nNote: Install tqdm for better progress bars: pip install tqdm")
            print(f"Starting tournament: {total_games_to_play} games to play\n")
        
        # Create progress bar
        if use_tqdm:
            pbar = tqdm(total=total_games_to_play, 
                    desc="Tournament Progress",
                    unit="games",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for matchup_idx, agent_indices in enumerate(matchups):
            # For this matchup, play multiple games with position rotation
            for game_num in range(games_per_matchup):
                # Select agents for this game
                game_agents = [self.agents[i] for i in agent_indices]
                
                # Rotate positions if enabled
                if rotate_positions:
                    rotation = game_num % len(game_agents)
                    game_agents = game_agents[rotation:] + game_agents[:rotation]
                
                # Simulate game
                game_result = GameSimulator.simulate_game(
                    game_agents, 
                    self.variant,
                    verbose=False
                )
                
                result.add_game(game_result)
                games_played += 1
                
                # Update progress bar
                if use_tqdm:
                    pbar.update(1)
                elif games_played % 10 == 0 or games_played == total_games_to_play:
                    # Fallback: simple text progress every 10 games
                    elapsed = time.time() - start_time
                    games_per_sec = games_played / elapsed if elapsed > 0 else 0
                    progress_pct = (games_played / total_games_to_play) * 100
                    remaining = total_games_to_play - games_played
                    eta = remaining / games_per_sec if games_per_sec > 0 else 0
                    print(f"\r[{games_played}/{total_games_to_play}] {progress_pct:.1f}% | "
                        f"{games_per_sec:.2f} games/s | ETA: {eta:.0f}s", end='', flush=True)
        
        if use_tqdm:
            pbar.close()
        else:
            print()  # New line after progress
        
        result.total_time = time.time() - start_time
        
        print(f"\n✓ Tournament complete!")
        print(f"  Total games: {games_played}")
        print(f"  Total time: {result.total_time:.2f}s ({result.total_time/60:.1f}m)")
        print(f"  Games/sec: {games_played / result.total_time:.2f}\n")
        
        return result

    def run_fixed_opponents(self,
                        target_agent_index: int,
                        num_games: int = 1000,
                        rotate_positions: bool = True,
                        verbose: bool = False) -> TournamentResult:
        """
        Run tournament with one target agent against fixed opponents.
        
        Args:
            target_agent_index: Index of agent to evaluate
            num_games: Number of games to run
            rotate_positions: Whether to rotate starting positions
            verbose: Print progress
            
        Returns:
            TournamentResult
        """
        result = TournamentResult(
            agent_names=self.agent_names,
            variant_config=self.variant
        )
        
        start_time = time.time()
        
        # Try to import tqdm for progress bar
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            print(f"\nNote: Install tqdm for better progress bars: pip install tqdm")
            print(f"Starting tournament: {num_games} games to play\n")
        
        # Create progress bar
        if use_tqdm:
            pbar = tqdm(total=num_games, 
                    desc="Tournament Progress",
                    unit="games",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for game_num in range(num_games):
            # Use all agents
            game_agents = self.agents.copy()
            
            # Rotate positions if enabled
            if rotate_positions:
                rotation = game_num % len(game_agents)
                game_agents = game_agents[rotation:] + game_agents[:rotation]
            
            # Simulate game
            game_result = GameSimulator.simulate_game(
                game_agents,
                self.variant,
                verbose=False
            )
            
            result.add_game(game_result)
            games_played = game_num + 1
            
            # Update progress bar
            if use_tqdm:
                pbar.update(1)
            elif games_played % 10 == 0 or games_played == num_games:
                # Fallback: simple text progress every 10 games
                elapsed = time.time() - start_time
                games_per_sec = games_played / elapsed if elapsed > 0 else 0
                progress_pct = (games_played / num_games) * 100
                remaining = num_games - games_played
                eta = remaining / games_per_sec if games_per_sec > 0 else 0
                print(f"\r[{games_played}/{num_games}] {progress_pct:.1f}% | "
                    f"{games_per_sec:.2f} games/s | ETA: {eta:.0f}s", end='', flush=True)
        
        if use_tqdm:
            pbar.close()
        else:
            print()  # New line after progress
        
        result.total_time = time.time() - start_time
        
        print(f"\n✓ Tournament complete!")
        print(f"  Total time: {result.total_time:.2f}s ({result.total_time/60:.1f}m)")
        print(f"  Games/sec: {num_games / result.total_time:.2f}\n")
        
        return result


def main():
    """Main entry point for running tournaments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run Cinquillo tournaments and simulations'
    )
    parser.add_argument(
        '--games', 
        type=int, 
        default=1000,
        help='Number of games to simulate (default: 1000)'
    )
    parser.add_argument(
        '--mode',
        choices=['round-robin', 'fixed'],
        default='round-robin',
        help='Tournament mode (default: round-robin)'
    )
    parser.add_argument(
        '--rotate',
        action='store_true',
        default=True,
        help='Rotate starting positions (default: True)'
    )
    parser.add_argument(
        '--no-rotate',
        action='store_true',
        help='Disable position rotation'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress'
    )
    parser.add_argument(
        '--variant',
        choices=['classic', 'wild', 'reverse'],
        default='classic',
        help='Game variant to use (default: classic)'
    )
    
    args = parser.parse_args()
    
    # Import agents
    from agents.base_agents import RandomAgent
    from agents.base_agents import HeuristicAgent
    from agents.mcts_agent import MCTSAgentStandard, MCTSAgentSuperFast
    from agents.rl_agent import RLAgent

    
    # Setup variant configuration
    if args.variant == 'classic':
        variant = VariantConfig()  # Default classic rules
    elif args.variant == 'wild':
        variant = VariantConfig(use_wild_cards=True)
    elif args.variant == 'reverse':
        variant = VariantConfig(reverse_scoring=True)
    else:
        variant = VariantConfig()


    rlagent = RLAgent(name="RL-v3", epsilon=0.0)
    rlagent.load_weights("models/rl_agent_v3.pkl")

    # Create agents
    agents = [
        HeuristicAgent("Heuristic-1"),
        HeuristicAgent("Heuristic-2"),
        MCTSAgentStandard("mcts-standard"),
        rlagent,
    ]


    print(f"\n{'='*60}")
    print(f"CINQUILLO TOURNAMENT SIMULATION")
    print(f"{'='*60}")
    print(f"Variant: {args.variant}")
    print(f"Mode: {args.mode}")
    print(f"Games: {args.games}")
    print(f"Agents: {', '.join(agent.name for agent in agents)}")
    
    # Create and run tournament
    tournament = Tournament(agents, variant)
    
    rotate_positions = args.rotate and not args.no_rotate
    
    if args.mode == 'round-robin':
        result = tournament.run_round_robin(
            games_per_matchup=args.games,
            rotate_positions=rotate_positions,
            verbose=args.verbose
        )
    else:  # fixed mode
        result = tournament.run_fixed_opponents(
            target_agent_index=0,  # Evaluate first agent
            num_games=args.games,
            rotate_positions=rotate_positions,
            verbose=args.verbose
        )
    
    # Print results
    result.print_summary()
    
    # Calculate and print games per second
    if result.total_time > 0:
        games_per_sec = result.total_games / result.total_time
        print(f"Performance: {games_per_sec:.1f} games/second\n")


if __name__ == "__main__":
    main()