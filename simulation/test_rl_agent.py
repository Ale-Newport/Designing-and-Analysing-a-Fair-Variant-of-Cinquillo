"""
Test trained RL agent in tournaments.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.rl_agent import RLAgent
from agents.base_agents import HeuristicAgent, RandomAgent
from game.entities import VariantConfig, MatchEndMode, ScoringMode
from game.rules import Rules


def test_trained_agent(weights_path: str = "models/rl_agent_best.pkl"):
    """Test trained RL agent against other agents."""
    
    print("="*60)
    print("TESTING TRAINED RL AGENT")
    print("="*60)
    
    # Load trained agent
    rl_agent = RLAgent(
        name="RL-Trained",
        epsilon=0.0,  # No exploration for testing
        learning_rate=0.0005,
        discount_factor=0.95,
        use_heuristics=True
    )
    rl_agent.load_weights(weights_path)
    
    # Create opponents
    heuristic1 = HeuristicAgent(
        name="Heuristic-1",
        prefer_reduce_hand=1.0,
        prefer_balance_suits=0.5,
        prefer_open_suits=0.3,
        prefer_block=0.4,
        avoid_voluntary_pass=2.0,
        dice_risk_tolerance=0.3
    )
    
    heuristic2 = HeuristicAgent(
        name="Heuristic-2",
        prefer_reduce_hand=1.2,
        prefer_balance_suits=0.4,
        prefer_open_suits=0.5,
        prefer_block=0.3,
        avoid_voluntary_pass=2.5,
        dice_risk_tolerance=0.4
    )
    
    random_agent = RandomAgent("Random")
    
    opponents = [heuristic1, heuristic2, random_agent]
    
    # Test configuration
    variant = VariantConfig(
        match_end_mode=MatchEndMode.FIXED_ROUNDS,
        fixed_rounds_count=5,
        scoring_mode=ScoringMode.DOUBLE_PENALTY,
        points_per_card=1
    )
    
    num_games = 10000
    print(f"\nRunning {num_games} games...")
    print(f"Agents: {rl_agent.name}, {', '.join([a.name for a in opponents])}")
    print(f"Variant: {variant}\n")
    
    # Track results
    results = {
        'RL-Trained': {'wins': 0, 'total_score': 0},
        'Heuristic-1': {'wins': 0, 'total_score': 0},
        'Heuristic-2': {'wins': 0, 'total_score': 0},
        'Random': {'wins': 0, 'total_score': 0}
    }
    
    position_wins = [0, 0, 0, 0]
    
    from tqdm import tqdm
    
    for game_num in tqdm(range(num_games), desc="Testing"):
        # Rotate positions
        position = game_num % 4
        agents = [rl_agent] + opponents
        
        # Rotate so RL agent is at different positions
        rotated_agents = agents[position:] + agents[:position]
        
        # Play game
        state = Rules.initialize_game(len(rotated_agents), variant)
        
        turn_count = 0
        max_turns = 1000
        
        while not Rules.is_terminal(state) and turn_count < max_turns:
            legal_moves = Rules.get_legal_moves(state)
            if not legal_moves:
                break
            
            current_player = state.current_player
            action = rotated_agents[current_player].choose_move(state, legal_moves)
            state = Rules.apply_move(state, action)
            turn_count += 1
        
        # Compute final scores
        Rules.compute_round_scores(state)
        
        # Record results
        scores = [(p.match_score, i, rotated_agents[i].name) 
                  for i, p in enumerate(state.players)]
        scores.sort(reverse=True)
        
        # Winner is the one with highest score
        max_score = scores[0][0]
        winners = [s for s in scores if s[0] == max_score]
        
        # Count wins (handle ties)
        for score, pos, name in winners:
            results[name]['wins'] += 1.0 / len(winners)
            position_wins[pos] += 1.0 / len(winners)
        
        # Record scores
        for score, pos, name in scores:
            results[name]['total_score'] += score
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total games: {num_games}\n")
    
    print("WIN RATES BY AGENT:")
    print("-" * 60)
    agent_results = sorted(results.items(), 
                          key=lambda x: x[1]['wins'], 
                          reverse=True)
    
    for agent_name, stats in agent_results:
        win_rate = (stats['wins'] / num_games) * 100
        avg_score = stats['total_score'] / num_games
        print(f"{agent_name:20s}: {win_rate:5.1f}% wins | Avg Score: {avg_score:6.2f}")
    
    print("\n" + "WIN RATES BY POSITION:")
    print("-" * 60)
    for i, wins in enumerate(position_wins):
        win_rate = (wins / num_games) * 100
        print(f"Position {i} (Seat {i+1}): {win_rate:5.1f}%")
    
    print("\n" + "="*60)
    
    # Compare to baseline
    rl_win_rate = (results['RL-Trained']['wins'] / num_games) * 100
    h1_win_rate = (results['Heuristic-1']['wins'] / num_games) * 100
    h2_win_rate = (results['Heuristic-2']['wins'] / num_games) * 100
    
    print("\nPERFORMANCE SUMMARY:")
    if rl_win_rate > max(h1_win_rate, h2_win_rate):
        print(f"✓ RL agent OUTPERFORMS heuristics!")
        print(f"  RL: {rl_win_rate:.1f}% vs Best Heuristic: {max(h1_win_rate, h2_win_rate):.1f}%")
    elif rl_win_rate >= min(h1_win_rate, h2_win_rate):
        print(f"~ RL agent is COMPETITIVE with heuristics")
        print(f"  RL: {rl_win_rate:.1f}% vs Heuristics: {h1_win_rate:.1f}%, {h2_win_rate:.1f}%")
    else:
        print(f"✗ RL agent needs more training")
        print(f"  RL: {rl_win_rate:.1f}% vs Heuristics: {h1_win_rate:.1f}%, {h2_win_rate:.1f}%")
    
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='models/rl_agent_best.pkl',
                       help='Path to trained weights')
    args = parser.parse_args()
    
    test_trained_agent(args.weights)