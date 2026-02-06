"""
Flask web server for Cinquillo 2.0.
Provides a web UI that uses the Python game engine.
"""
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import json
from typing import Dict, List, Optional

from game.entities import GameState, VariantConfig, GoodDiceEffect, BadDiceEffect, ScoringMode, MatchEndMode, Suit, Card
from game.rules import Rules, PlayCard, Pass, RollDice
from agents.base_agents import RandomAgent, create_balanced_heuristic, create_aggressive_heuristic

app = Flask(__name__)
CORS(app)

# Store active games (in production, use Redis or database)
active_games: Dict[str, GameState] = {}
match_states: Dict[str, Dict] = {}  # Match-level state


@app.route('/')
def index():
    """Serve the game UI."""
    return render_template('cinquillo.html')


@app.route('/api/new_match', methods=['POST'])
def new_match():
    """Start a new match."""
    data = request.json
    
    # Extract configuration
    num_players = data.get('num_players', 4)
    human_player = data.get('human_player', 0)  # Which position is human
    debug_mode = data.get('debug_mode', False)  # Debug mode toggle
    starting_player = data.get('starting_player', 0)  # Who starts first match
    
    # Agent selection for each player
    agent_types = data.get('agent_types', {})
    
    # Variant configuration
    variant_data = data.get('variant', {})
    variant = VariantConfig(
        # Dice settings
        dice_good_probability=variant_data.get('dice_good_probability', 0.5),
        dice_good_effect=GoodDiceEffect[variant_data.get('dice_good_effect', 'WILD')],
        dice_bad_effect=BadDiceEffect[variant_data.get('dice_bad_effect', 'TAKE_CARDS')],
        dice_bad_cards_count=variant_data.get('dice_bad_cards_count', 2),
        dice_bad_penalty_points=variant_data.get('dice_bad_penalty_points', 1),
        
        # Scoring settings
        scoring_mode=ScoringMode[variant_data.get('scoring_mode', 'WINNER_TAKES_ALL')],
        points_per_card=variant_data.get('points_per_card', 1),
        voluntary_pass_penalty=variant_data.get('voluntary_pass_penalty', 1),
        
        # Match end settings
        match_end_mode=MatchEndMode[variant_data.get('match_end_mode', 'TARGET_SCORE')],
        target_score_multiplier=variant_data.get('target_score_multiplier', 10),
        fixed_rounds_count=variant_data.get('fixed_rounds_count', 5)
    )
    
    # Create match state
    match_id = f"match_{len(match_states)}"
    match_states[match_id] = {
        'num_players': num_players,
        'human_player': human_player,
        'variant': variant,
        'match_scores': [0] * num_players,
        'round_number': 0,
        'agents': create_agents(num_players, human_player, agent_types),
        'agent_types': agent_types,
        'game_history': [],
        'debug_mode': debug_mode,
        'starting_player': starting_player,  # Track starting player for round robin
        'current_round_starter': starting_player  # Who starts current round
    }
    
    # Start first round
    round_state = start_new_round(match_id)
    
    return jsonify({
        'match_id': match_id,
        'state': serialize_state(round_state, match_id),
        'match_info': {
            'round': 1,
            'match_scores': [0] * num_players,
            'target': variant.get_target_score(num_players) if variant.match_end_mode == MatchEndMode.TARGET_SCORE else None
        }
    })


@app.route('/api/make_move', methods=['POST'])
def make_move():
    """Make a move in the current game."""
    data = request.json
    match_id = data['match_id']
    move_type = data['move_type']
    move_data = data.get('move_data', {})
    
    if match_id not in active_games:
        return jsonify({'error': 'Game not found'}), 404
    
    state = active_games[match_id]
    match_state = match_states[match_id]
    
    # Validate it's human player's turn
    if state.current_player != match_state['human_player']:
        return jsonify({'error': 'Not your turn'}), 400
    
    # Create move object
    move = None
    if move_type == 'play_card':
        suit = Suit[move_data['suit'].upper()]
        rank = move_data['rank']
        card = Card(suit, rank)
        move = PlayCard(card)
    elif move_type == 'pass':
        # Check if this is a voluntary pass (has playable cards) or forced pass (no playable cards)
        player = state.players[state.current_player]
        has_playable_cards = False
        for card in player.hand:
            play_move = PlayCard(card)
            if play_move.is_legal(state, state.current_player):
                has_playable_cards = True
                break
        
        # Voluntary pass = has playable cards but chooses to pass (penalty applies)
        # Forced pass = no playable cards, must pass (no penalty)
        move = Pass(voluntary=has_playable_cards)
    elif move_type == 'roll_dice':
        move = RollDice()
    
    if not move or not move.is_legal(state, state.current_player):
        return jsonify({'error': 'Illegal move'}), 400
    
    # Apply move
    state = move.apply(state)
    active_games[match_id] = state
    
    # Check if round is over
    if state.game_over:
        return handle_round_end(match_id, state)
    
    # Don't process bot turns automatically - frontend will handle with delays
    return jsonify({
        'state': serialize_state(state, match_id),
        'match_info': get_match_info(match_id)
    })


@app.route('/api/bot_move', methods=['POST'])
def bot_move():
    """Process a single bot move (for step-by-step mode)."""
    data = request.json
    match_id = data['match_id']
    
    if match_id not in active_games:
        return jsonify({'error': 'Game not found'}), 404
    
    state = active_games[match_id]
    match_state = match_states[match_id]
    
    if state.current_player == match_state['human_player']:
        return jsonify({'error': 'Human player turn'}), 400
    
    # Make one bot move
    agent = match_state['agents'][state.current_player]
    legal_moves = Rules.get_legal_moves(state)
    
    move_info = None
    if legal_moves:
        move = agent.choose_move(state, legal_moves)
        
        # In debug mode, capture detailed move information
        if match_state['debug_mode']:
            agent_type = match_state['agent_types'].get(str(state.current_player), 'Unknown')
            move_description = format_move_for_debug(move, state.current_player)
            
            move_info = {
                'player': state.current_player,
                'agent_type': agent_type,
                'move': str(move),
                'move_description': move_description,
                'legal_moves_count': len(legal_moves)
            }
        
        state = move.apply(state)
        active_games[match_id] = state
    
    if state.game_over:
        return handle_round_end(match_id, state)
    
    response = {
        'state': serialize_state(state, match_id),
        'match_info': get_match_info(match_id)
    }
    
    if move_info:
        response['move_info'] = move_info
    
    return jsonify(response)


def format_move_for_debug(move, player_index: int) -> str:
    """Format a move for debug display."""
    from game.rules import PlayCard, Pass, RollDice
    
    if isinstance(move, PlayCard):
        suit_symbols = {'Oros': '♦', 'Copas': '♥', 'Espadas': '♠', 'Bastos': '♣'}
        suit_symbol = suit_symbols.get(move.card.suit.value, move.card.suit.value)
        return f"Player {player_index + 1} played {move.card.rank}{suit_symbol}"
    elif isinstance(move, Pass):
        if move.voluntary:
            return f"Player {player_index + 1} passed (voluntary, -1 penalty)"
        else:
            return f"Player {player_index + 1} passed (forced, no penalty)"
    elif isinstance(move, RollDice):
        return f"Player {player_index + 1} rolled dice"
    else:
        return f"Player {player_index + 1}: {str(move)}"


def create_agents(num_players: int, human_player: int, agent_types: Dict = None):
    """Create AI agents for non-human players."""
    from agents.base_agents import create_defensive_heuristic
    
    if agent_types is None:
        agent_types = {}
    
    agents = []
    for i in range(num_players):
        if i == human_player:
            agents.append(None)  # Human player
        else:
            # Get agent type from selection or default
            agent_type = agent_types.get(str(i), 'balanced')
            
            if agent_type == 'random':
                agents.append(RandomAgent())
            elif agent_type == 'aggressive':
                agents.append(create_aggressive_heuristic())
            elif agent_type == 'balanced':
                agents.append(create_balanced_heuristic())
            elif agent_type == 'defensive':
                try:
                    agents.append(create_defensive_heuristic())
                except (ImportError, AttributeError) as e:
                    print(f"Defensive heuristic not available for player {i}, using balanced: {e}")
                    agents.append(create_balanced_heuristic())
            elif agent_type == 'mcts':
                # Import MCTS agent if available
                try:
                    from agents.mcts_agent import MCTSAgent
                    # Try to instantiate without parameters first
                    try:
                        agents.append(MCTSAgent())
                    except TypeError:
                        # If that fails, try with common parameter names
                        try:
                            agents.append(MCTSAgent(num_iterations=100))
                        except TypeError:
                            try:
                                agents.append(MCTSAgent(iterations=100))
                            except TypeError:
                                # Just use the default constructor
                                print(f"Could not determine MCTSAgent constructor for player {i}, using default")
                                agents.append(MCTSAgent())
                except ImportError as e:
                    print(f"MCTS agent not available for player {i}, using balanced heuristic: {e}")
                    agents.append(create_balanced_heuristic())
                except Exception as e:
                    print(f"Error creating MCTS agent for player {i}, using balanced heuristic: {e}")
                    agents.append(create_balanced_heuristic())
            elif agent_type == 'rl':
                # Import RL agent if available
                try:
                    from agents.rl_agent import RLAgent
                    agents.append(RLAgent())
                except ImportError as e:
                    print(f"RL agent not available for player {i}, using balanced heuristic: {e}")
                    agents.append(create_balanced_heuristic())
                except Exception as e:
                    print(f"Error creating RL agent for player {i}, using balanced heuristic: {e}")
                    agents.append(create_balanced_heuristic())
            else:
                # Default to balanced
                agents.append(create_balanced_heuristic())
    return agents


def start_new_round(match_id: str) -> GameState:
    """Start a new round in the match."""
    match_state = match_states[match_id]
    match_state['round_number'] += 1
    
    # Initialize new game
    state = Rules.initialize_game(
        match_state['num_players'],
        match_state['variant']
    )
    
    # Set match scores from previous rounds
    for i, player in enumerate(state.players):
        player.match_score = match_state['match_scores'][i]
    
    state.round_number = match_state['round_number']
    
    # Round robin: rotate starting player each round
    # First round uses starting_player, subsequent rounds rotate
    if match_state['round_number'] == 1:
        starting_player = match_state['starting_player']
    else:
        # Rotate to next player
        current_starter = match_state['current_round_starter']
        starting_player = (current_starter + 1) % match_state['num_players']
    
    state.current_player = starting_player
    match_state['current_round_starter'] = starting_player
    
    active_games[match_id] = state
    return state


def process_bot_turns(match_id: str, state: GameState) -> GameState:
    """Process all bot turns until human player's turn or game over."""
    match_state = match_states[match_id]
    
    max_iterations = 100  # Safety limit
    iterations = 0
    
    while (not state.game_over and 
           state.current_player != match_state['human_player'] and
           iterations < max_iterations):
        
        agent = match_state['agents'][state.current_player]
        legal_moves = Rules.get_legal_moves(state)
        
        if not legal_moves:
            break
        
        move = agent.choose_move(state, legal_moves)
        state = move.apply(state)
        iterations += 1
    
    return state


def handle_round_end(match_id: str, state: GameState):
    """Handle end of round and check if match is over."""
    match_state = match_states[match_id]
    
    # Compute scores
    Rules.compute_round_scores(state)
    
    # Update match scores
    for i, player in enumerate(state.players):
        match_state['match_scores'][i] = player.match_score
    
    # Check if match is over
    variant = match_state['variant']
    match_over = False
    
    if variant.match_end_mode == MatchEndMode.TARGET_SCORE:
        target = variant.get_target_score(match_state['num_players'])
        if max(match_state['match_scores']) >= target:
            match_over = True
    elif variant.match_end_mode == MatchEndMode.FIXED_ROUNDS:
        if match_state['round_number'] >= variant.fixed_rounds_count:
            match_over = True
    
    response = {
        'round_over': True,
        'match_over': match_over,
        'state': serialize_state(state, match_id),
        'match_info': get_match_info(match_id)
    }
    
    if match_over:
        winner_idx = match_state['match_scores'].index(max(match_state['match_scores']))
        response['match_winner'] = winner_idx
        response['final_scores'] = match_state['match_scores']
    
    return jsonify(response)


@app.route('/api/next_round', methods=['POST'])
def next_round():
    """Start the next round after current round ends."""
    data = request.json
    match_id = data['match_id']
    
    if match_id not in match_states:
        return jsonify({'error': 'Match not found'}), 404
    
    # Start new round
    state = start_new_round(match_id)
    
    # Don't process bot turns automatically - frontend will handle with delays
    active_games[match_id] = state
    
    return jsonify({
        'state': serialize_state(state, match_id),
        'match_info': get_match_info(match_id)
    })


def get_match_info(match_id: str) -> Dict:
    """Get current match information."""
    match_state = match_states[match_id]
    variant = match_state['variant']
    
    return {
        'round': match_state['round_number'],
        'match_scores': match_state['match_scores'],
        'target': variant.get_target_score(match_state['num_players']) if variant.match_end_mode == MatchEndMode.TARGET_SCORE else None,
        'total_rounds': variant.fixed_rounds_count if variant.match_end_mode == MatchEndMode.FIXED_ROUNDS else None
    }


def serialize_state(state: GameState, match_id: str = None) -> Dict:
    """Convert GameState to JSON-serializable dict."""
    # Check if debug mode is active
    debug_mode = False
    if match_id and match_id in match_states:
        debug_mode = match_states[match_id].get('debug_mode', False)
    
    players_data = []
    for p in state.players:
        player_data = {
            'index': p.index,
            'hand_size': p.hand_size(),
            'round_score': p.round_score,
            'match_score': p.match_score
        }
        
        # In debug mode, reveal all hands
        # Otherwise only show current player's hand (will be filtered on frontend)
        if debug_mode:
            player_data['hand'] = [{'suit': card.suit.value, 'rank': card.rank} for card in p.hand]
            player_data['debug_visible'] = True
        else:
            player_data['hand'] = [{'suit': card.suit.value, 'rank': card.rank} for card in p.hand]
        
        players_data.append(player_data)
    
    return {
        'current_player': state.current_player,
        'turn_number': state.turn_number,
        'round_number': state.round_number,
        'game_over': state.game_over,
        'winner': state.winner,
        'players': players_data,
        'board': {
            suit.value: list(state.board.suit_cards[suit])
            for suit in Suit
        },
        'dice_state': {
            'wild_active': state.dice_state.wild_active,
            'double_play_active': state.dice_state.double_play_active,
            'revealed_player': state.dice_state.revealed_player
        },
        'debug_mode': debug_mode
    }


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)