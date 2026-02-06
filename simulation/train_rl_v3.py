"""
Enhanced RL training v3 with curriculum learning and better checkpointing.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from typing import List, Optional
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path

from game.entities import GameState, VariantConfig, MatchEndMode, ScoringMode
from game.rules import Rules
from agents.rl_agent import RLAgent
from agents.base_agents import Agent, HeuristicAgent, RandomAgent


def get_heuristic_agents():
    """Get heuristic agents."""
    return [
        HeuristicAgent(
            name="Heuristic-1",
            prefer_reduce_hand=1.0,
            prefer_balance_suits=0.5,
            prefer_open_suits=0.3,
            prefer_block=0.4,
            avoid_voluntary_pass=2.0,
            dice_risk_tolerance=0.3
        ),
        HeuristicAgent(
            name="Heuristic-2",
            prefer_reduce_hand=1.2,
            prefer_balance_suits=0.4,
            prefer_open_suits=0.5,
            prefer_block=0.3,
            avoid_voluntary_pass=2.5,
            dice_risk_tolerance=0.4
        )
    ]


def play_single_game(agents: List[Agent], variant: VariantConfig) -> dict:
    """Play a single game and return results."""
    state = Rules.initialize_game(len(agents), variant)
    
    turn_count = 0
    max_turns = 1000
    
    while not Rules.is_terminal(state) and turn_count < max_turns:
        legal_moves = Rules.get_legal_moves(state)
        if not legal_moves:
            break
        
        current_player = state.current_player
        action = agents[current_player].choose_move(state, legal_moves)
        state = Rules.apply_move(state, action)
        turn_count += 1
    
    # Compute final scores
    Rules.compute_round_scores(state)
    
    # Get results
    scores = [(p.match_score, i) for i, p in enumerate(state.players)]
    max_score = max(s[0] for s in scores)
    
    return {
        'scores': [p.match_score for p in state.players],
        'winner_indices': [i for score, i in scores if score == max_score],
        'turns': turn_count
    }


def evaluate_agent_proper(
    agent: Agent,
    opponents: List[Agent],
    num_games: int = 100,
    variant: VariantConfig = None,
    verbose: bool = False
) -> dict:
    """Properly evaluate agent against opponents."""
    if variant is None:
        variant = VariantConfig(
            match_end_mode=MatchEndMode.FIXED_ROUNDS,
            fixed_rounds_count=5,
            scoring_mode=ScoringMode.DOUBLE_PENALTY,
            points_per_card=1
        )
    
    # Temporarily disable exploration
    original_epsilon = None
    if isinstance(agent, RLAgent):
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0
    
    wins = 0
    total_score = 0
    games_played = 0
    position_wins = [0, 0, 0, 0]
    
    # Test in each position
    for position in range(4):
        for _ in range(num_games // 4):
            agents = [agent] + opponents
            # Rotate so agent is at 'position'
            rotated = agents[position:] + agents[:position]
            
            result = play_single_game(rotated, variant)
            
            # Check if agent won (agent is at position 0 after rotation)
            agent_score = result['scores'][0]
            if 0 in result['winner_indices']:
                win_share = 1.0 / len(result['winner_indices'])
                wins += win_share
                position_wins[position] += win_share
            
            total_score += agent_score
            games_played += 1
    
    # Restore exploration
    if original_epsilon is not None:
        agent.epsilon = original_epsilon
    
    result = {
        'win_rate': (wins / games_played) * 100,
        'avg_score': total_score / games_played,
        'games': games_played,
        'position_wins': position_wins
    }
    
    if verbose:
        print(f"  Position win rates: {[f'{100*w/games_played*4:.1f}%' for w in position_wins]}")
    
    return result


class TrainingLogger:
    """Logger for training statistics."""
    
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'episodes': [],
            'avg_reward': [],
            'win_rate': [],
            'avg_score': [],
            'epsilon': [],
            'learning_rate': [],
            'eval_games': [],
            'best_win_rate': 0.0,
            'best_episode': 0,
            'checkpoints': []
        }
    
    def log(self, episode: int, metrics: dict):
        """Log metrics for an episode."""
        self.stats['episodes'].append(episode)
        for key, value in metrics.items():
            if key in self.stats:
                self.stats[key].append(value)
    
    def save(self, filename: str = 'training_log.json'):
        """Save training log to file."""
        filepath = self.save_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def load(self, filename: str = 'training_log.json'):
        """Load training log from file."""
        filepath = self.save_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                self.stats = json.load(f)


def train_rl_agent(
    rl_agent: RLAgent,
    opponents: List[Agent],
    num_episodes: int = 10000,
    eval_interval: int = 500,
    batch_size: int = 64,
    variant: VariantConfig = None,
    save_dir: str = "models",
    curriculum_learning: bool = True,
    save_all_checkpoints: bool = False,
    min_improvement: float = 0.5
):
    """
    Train RL agent with enhanced features.
    
    Args:
        rl_agent: The RL agent to train
        opponents: List of opponent agents
        num_episodes: Number of training episodes
        eval_interval: Episodes between evaluations
        batch_size: Batch size for replay
        variant: Game variant config
        save_dir: Directory to save models and logs
        curriculum_learning: Whether to use curriculum learning
        save_all_checkpoints: Save model at every evaluation
        min_improvement: Minimum improvement to save checkpoint (%)
    """
    if variant is None:
        variant = VariantConfig(
            match_end_mode=MatchEndMode.FIXED_ROUNDS,
            fixed_rounds_count=5,
            scoring_mode=ScoringMode.DOUBLE_PENALTY,
            points_per_card=1
        )
    
    # Setup
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(save_dir)
    
    print("="*60)
    print("RL AGENT TRAINING V3 (ENHANCED)")
    print("="*60)
    print(f"Agent: {rl_agent.name}")
    print(f"Opponents: {[a.name for a in opponents]}")
    print(f"Episodes: {num_episodes}")
    print(f"Eval Interval: {eval_interval}")
    print(f"Batch Size: {batch_size}")
    print(f"Curriculum Learning: {curriculum_learning}")
    print(f"Save Directory: {save_dir}")
    print(f"Variant: {variant}")
    print(f"Initial epsilon: {rl_agent.epsilon:.3f}")
    print(f"Learning rate: {rl_agent.learning_rate}")
    print("="*60)
    
    all_agents = [rl_agent] + opponents
    
    # Curriculum: Start with easier opponents (more random agents)
    if curriculum_learning:
        curriculum_stages = [
            (0, 2000, [RandomAgent("R1"), RandomAgent("R2"), RandomAgent("R3")]),  # Easy
            (2000, 5000, opponents[:1] + [RandomAgent("R1"), RandomAgent("R2")]),  # Medium
            (5000, num_episodes, opponents)  # Hard
        ]
        print("\nCurriculum Learning Enabled:")
        for start, end, opps in curriculum_stages:
            print(f"  Episodes {start}-{end}: {[o.name for o in opps]}")
        print()
    
    start_time = time.time()
    best_win_rate = 0.0
    last_saved_win_rate = 0.0
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        
        # Curriculum learning: adjust opponents
        if curriculum_learning:
            current_opponents = opponents  # Default
            for start, end, stage_opponents in curriculum_stages:
                if start <= episode < end:
                    current_opponents = stage_opponents
                    break
        else:
            current_opponents = opponents
        
        # Setup game
        agents_pool = [rl_agent] + current_opponents
        position = episode % 4
        rotated = agents_pool[position:] + agents_pool[:position]
        
        # Play training game
        state = Rules.initialize_game(len(rotated), variant)
        rl_position = 0
        
        prev_state = None
        prev_action = None
        turn_count = 0
        max_turns = 1000
        
        while not Rules.is_terminal(state) and turn_count < max_turns:
            current_state = state.copy()
            current_player = state.current_player
            
            # Store experience for RL agent
            if current_player == rl_position and prev_state is not None:
                reward = rl_agent.compute_reward(
                    prev_state, prev_action, current_state, rl_position
                )
                rl_agent.store_experience(
                    prev_state, prev_action, reward, current_state, False
                )
            
            # Get action
            legal_moves = Rules.get_legal_moves(state)
            if not legal_moves:
                break
            
            action = rotated[current_player].choose_move(state, legal_moves)
            
            if current_player == rl_position:
                prev_state = current_state
                prev_action = action
            
            state = Rules.apply_move(state, action)
            turn_count += 1
        
        # Final experience
        Rules.compute_round_scores(state)
        
        if prev_state is not None:
            final_reward = rl_agent.compute_reward(
                prev_state, prev_action, state, rl_position
            )
            rl_agent.store_experience(
                prev_state, prev_action, final_reward, state, True
            )
        
        rl_agent.end_episode()
        
        # Train from replay
        if len(rl_agent.replay_buffer) >= batch_size:
            # Progressive training intensity
            if episode < 2000:
                num_updates = 3
            elif episode < 5000:
                num_updates = 5
            else:
                num_updates = 7
            
            for _ in range(num_updates):
                rl_agent.train_from_replay(batch_size)
        
        # Decay exploration
        if episode > 0 and episode % 100 == 0:
            decay_rate = 0.995 if episode < 5000 else 0.998
            rl_agent.decay_epsilon(decay_rate=decay_rate, min_epsilon=0.05)
        
        # Evaluation
        if (episode > 0 and episode % eval_interval == 0) or episode == num_episodes - 1:
            # Evaluate against final opponents (not curriculum)
            eval_stats = evaluate_agent_proper(
                rl_agent, opponents, num_games=100, variant=variant, verbose=True
            )
            
            # Calculate metrics
            recent_rewards = rl_agent.episode_rewards[-eval_interval:]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            
            # Log metrics
            logger.log(episode, {
                'avg_reward': avg_reward,
                'win_rate': eval_stats['win_rate'],
                'avg_score': eval_stats['avg_score'],
                'epsilon': rl_agent.epsilon,
                'learning_rate': rl_agent.learning_rate,
                'eval_games': eval_stats['games']
            })
            
            # Save checkpoints
            improved = eval_stats['win_rate'] > best_win_rate
            significant_improvement = eval_stats['win_rate'] >= last_saved_win_rate + min_improvement
            
            if improved:
                best_win_rate = eval_stats['win_rate']
                logger.stats['best_win_rate'] = best_win_rate
                logger.stats['best_episode'] = episode
                
                # Save best model
                best_path = save_path / "rl_agent_best.pkl"
                rl_agent.save_weights(str(best_path))
                print(f"\n✓ New best model! Win rate: {best_win_rate:.1f}%")
            
            if save_all_checkpoints or significant_improvement:
                checkpoint_path = save_path / f"rl_agent_ep{episode}.pkl"
                rl_agent.save_weights(str(checkpoint_path))
                logger.stats['checkpoints'].append({
                    'episode': episode,
                    'win_rate': eval_stats['win_rate'],
                    'path': str(checkpoint_path)
                })
                last_saved_win_rate = eval_stats['win_rate']
                if not improved:
                    print(f"\n✓ Checkpoint saved (improved by {eval_stats['win_rate'] - (last_saved_win_rate - min_improvement):.1f}%)")
            
            # Print evaluation
            print(f"\n--- Evaluation at Episode {episode} ---")
            print(f"Avg Reward: {avg_reward:.2f}")
            print(f"Win Rate: {eval_stats['win_rate']:.1f}%")
            print(f"Avg Score: {eval_stats['avg_score']:.2f}")
            print(f"Epsilon: {rl_agent.epsilon:.3f}")
            print(f"Buffer: {len(rl_agent.replay_buffer)}/{rl_agent.replay_buffer.maxlen}")
            print(f"Updates: {rl_agent.total_updates}")
            print(f"Best: {best_win_rate:.1f}% (Ep {logger.stats['best_episode']})")
            
            # Save log
            logger.save()
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")
    print(f"Total updates: {rl_agent.total_updates}")
    print(f"Final epsilon: {rl_agent.epsilon:.3f}")
    print(f"Best performance: {best_win_rate:.1f}% at episode {logger.stats['best_episode']}")
    print(f"Checkpoints saved: {len(logger.stats['checkpoints'])}")
    
    # Save final model
    final_path = save_path / "rl_agent_v3_final.pkl"
    rl_agent.save_weights(str(final_path))
    
    # Final comprehensive evaluation
    print("\nRunning final comprehensive evaluation (500 games)...")
    final_eval = evaluate_agent_proper(rl_agent, opponents, num_games=500, variant=variant, verbose=True)
    print(f"\nFinal Win Rate: {final_eval['win_rate']:.1f}%")
    print(f"Final Avg Score: {final_eval['avg_score']:.2f}")
    
    logger.stats['final_win_rate'] = final_eval['win_rate']
    logger.stats['final_avg_score'] = final_eval['avg_score']
    logger.save()
    
    return logger.stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RL agent v3 (enhanced)')
    parser.add_argument('--episodes', type=int, default=15000, help='Number of episodes')
    parser.add_argument('--eval-interval', type=int, default=500, help='Evaluation interval')
    parser.add_argument('--epsilon', type=float, default=0.6, help='Initial exploration rate')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='models/rl_v3', help='Save directory')
    parser.add_argument('--no-curriculum', action='store_true', help='Disable curriculum learning')
    parser.add_argument('--save-all', action='store_true', help='Save all checkpoints')
    parser.add_argument('--load', type=str, default=None, help='Load from checkpoint')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    
    args = parser.parse_args()
    
    # Create RL agent
    rl_agent = RLAgent(
        name="RL-Agent-v3",
        epsilon=args.epsilon,
        learning_rate=args.lr,
        discount_factor=0.95,
        use_heuristics=True
    )
    
    # Load checkpoint if specified
    if args.load:
        rl_agent.load_weights(args.load)
        print(f"Loaded checkpoint from {args.load}")
    
    # Get opponents
    opponents = get_heuristic_agents() + [RandomAgent("Random")]
    
    # Training variant
    variant = VariantConfig(
        match_end_mode=MatchEndMode.FIXED_ROUNDS,
        fixed_rounds_count=5,
        scoring_mode=ScoringMode.DOUBLE_PENALTY,
        points_per_card=1
    )
    
    # Train
    stats = train_rl_agent(
        rl_agent=rl_agent,
        opponents=opponents,
        num_episodes=args.episodes,
        eval_interval=args.eval_interval,
        batch_size=args.batch_size,
        variant=variant,
        save_dir=args.save_dir,
        curriculum_learning=not args.no_curriculum,
        save_all_checkpoints=args.save_all,
        min_improvement=1.0
    )
    
    print("\n" + "="*60)
    print("RESULTS SAVED TO:")
    print(f"  Best model: {args.save_dir}/rl_agent.pkl")
    print(f"  Final model: {args.save_dir}/rl_agent_v3_final.pkl")
    print(f"  Training log: {args.save_dir}/training_log.json")
    print(f"  Checkpoints: {len(stats['checkpoints'])}")
    print("\nTest with:")
    print(f"  python simulation/test_rl_agent.py --weights {args.save_dir}/rl_agent_v3.pkl")
    print("="*60)