"""
Improved training script with proper evaluation and debugging.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from typing import List
from tqdm import tqdm
import numpy as np

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
    variant: VariantConfig = None
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
                wins += 1.0 / len(result['winner_indices'])
            
            total_score += agent_score
            games_played += 1
    
    # Restore exploration
    if original_epsilon is not None:
        agent.epsilon = original_epsilon
    
    return {
        'win_rate': (wins / games_played) * 100,
        'avg_score': total_score / games_played,
        'games': games_played
    }


def train_rl_agent(
    rl_agent: RLAgent,
    opponents: List[Agent],
    num_episodes: int = 10000,
    eval_interval: int = 500,
    batch_size: int = 64,
    variant: VariantConfig = None,
    save_path: str = "models/rl_agent.pkl"
):
    """Train RL agent with proper evaluation."""
    if variant is None:
        variant = VariantConfig(
            match_end_mode=MatchEndMode.FIXED_ROUNDS,
            fixed_rounds_count=5,  # Use full matches for training
            scoring_mode=ScoringMode.DOUBLE_PENALTY,
            points_per_card=1
        )
    
    print("="*60)
    print("RL AGENT TRAINING V2 (FIXED)")
    print("="*60)
    print(f"Agent: {rl_agent.name}")
    print(f"Opponents: {[a.name for a in opponents]}")
    print(f"Episodes: {num_episodes}")
    print(f"Eval Interval: {eval_interval}")
    print(f"Batch Size: {batch_size}")
    print(f"Variant: {variant}")
    print(f"Initial epsilon: {rl_agent.epsilon:.3f}")
    print(f"Learning rate: {rl_agent.learning_rate}")
    print("="*60)
    
    all_agents = [rl_agent] + opponents
    
    training_stats = {
        'episodes': [],
        'avg_reward': [],
        'win_rate': [],
        'avg_score': [],
        'epsilon': [],
        'best_win_rate': 0.0,
        'best_episode': 0
    }
    
    start_time = time.time()
    
    # Training loop
    for episode in tqdm(range(num_episodes), desc="Training Progress"):
        # Rotate RL agent position
        position = episode % 4
        agents = all_agents[:]
        rotated = agents[position:] + agents[:position]
        
        # Play training game
        state = Rules.initialize_game(len(rotated), variant)
        
        # Track RL agent's position in rotated list (always 0)
        rl_position = 0
        
        prev_state = None
        prev_action = None
        
        turn_count = 0
        max_turns = 1000
        
        while not Rules.is_terminal(state) and turn_count < max_turns:
            current_state = state.copy()
            current_player = state.current_player
            
            # Store experience only when it's RL agent's turn
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
            
            # Track RL agent's state and action
            if current_player == rl_position:
                prev_state = current_state
                prev_action = action
            
            # Execute move
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
            num_updates = 3 if episode < 2000 else 5
            for _ in range(num_updates):
                rl_agent.train_from_replay(batch_size)
        
        # Decay exploration
        if episode > 0 and episode % 100 == 0:
            rl_agent.decay_epsilon(decay_rate=0.995, min_epsilon=0.05)
        
        # Evaluation
        if (episode > 0 and episode % eval_interval == 0) or episode == num_episodes - 1:
            eval_stats = evaluate_agent_proper(rl_agent, opponents, num_games=100, variant=variant)
            
            training_stats['episodes'].append(episode)
            
            recent_rewards = rl_agent.episode_rewards[-eval_interval:]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
            training_stats['avg_reward'].append(avg_reward)
            
            training_stats['win_rate'].append(eval_stats['win_rate'])
            training_stats['avg_score'].append(eval_stats['avg_score'])
            training_stats['epsilon'].append(rl_agent.epsilon)
            
            # Save best model
            if eval_stats['win_rate'] > training_stats['best_win_rate']:
                training_stats['best_win_rate'] = eval_stats['win_rate']
                training_stats['best_episode'] = episode
                rl_agent.save_weights(save_path.replace('.pkl', '_best.pkl'))
                print(f"\n✓ New best model saved! Win rate: {eval_stats['win_rate']:.1f}%")
            
            print(f"\n--- Evaluation at Episode {episode} ---")
            print(f"Avg Reward: {avg_reward:.2f}")
            print(f"Win Rate: {eval_stats['win_rate']:.1f}%")
            print(f"Avg Score: {eval_stats['avg_score']:.2f}")
            print(f"Epsilon: {rl_agent.epsilon:.3f}")
            print(f"Best: {training_stats['best_win_rate']:.1f}% (Ep {training_stats['best_episode']})")
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")
    print(f"Total updates: {rl_agent.total_updates}")
    print(f"Final epsilon: {rl_agent.epsilon:.3f}")
    print(f"Best: {training_stats['best_win_rate']:.1f}% at episode {training_stats['best_episode']}")
    
    rl_agent.save_weights(save_path)
    
    return training_stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RL agent (fixed version)')
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--eval-interval', type=int, default=500)
    parser.add_argument('--epsilon', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--save', type=str, default='models/rl_agent_v2.pkl')
    
    args = parser.parse_args()
    
    # Create RL agent
    rl_agent = RLAgent(
        name="RL-Agent",
        epsilon=args.epsilon,
        learning_rate=args.lr,
        discount_factor=0.95,
        use_heuristics=True
    )
    
    # Get opponents
    opponents = get_heuristic_agents() + [RandomAgent("Random")]
    
    # Training variant - use full 5-round matches
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
        batch_size=64,
        variant=variant,
        save_path=args.save
    )
    
    print("\n" + "="*60)
    print(f"Weights saved to: {args.save}")
    print(f"Best weights: {args.save.replace('.pkl', '_best.pkl')}")
    print("\nTest with:")
    print(f"  python simulation/test_rl_agent.py --weights {args.save.replace('.pkl', '_best.pkl')}")
    print("="*60)