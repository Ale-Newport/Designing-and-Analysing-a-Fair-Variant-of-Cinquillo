"""
Flask web server for Cinquillo 2.0.
Provides a web UI that uses the Python game engine.
"""
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import io
import json
from typing import Dict, List, Optional
from dataclasses import dataclass

from game.entities import GameState, VariantConfig, GoodDiceEffect, BadDiceEffect, ScoringMode, MatchEndMode, Suit, Card
from game.rules import Rules, PlayCard, Pass, RollDice
from agents.base_agents import RandomAgent, create_balanced_heuristic, create_aggressive_heuristic

app = Flask(__name__)
CORS(app)

# Store active games (in production, use Redis or database)
active_games: Dict[str, GameState] = {}
match_states: Dict[str, Dict] = {}  # Match-level state


@dataclass
class SimpleAgent:
    """Lightweight agent-like object for visualizer naming."""
    name: str

    def choose_move(self, state, legal_moves):
        return legal_moves[0]  # fallback — never called for replay


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
    variant_name = data.get('variant_name', 'Custom')
    
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
        'variant_name': variant_name,
        'match_scores': [0] * num_players,
        'round_number': 0,
        'agents': create_agents(num_players, human_player, agent_types),
        'agent_types': agent_types,
        'game_history': [],
        'round_turns': [],          # per-round turn recording for LaTeX export
        'debug_mode': debug_mode,
        'starting_player': starting_player,
        'current_round_starter': starting_player
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
        player = state.players[state.current_player]
        has_playable_cards = any(
            PlayCard(card).is_legal(state, state.current_player)
            for card in player.hand
        )
        move = Pass(voluntary=has_playable_cards)
    elif move_type == 'roll_dice':
        move = RollDice()
    
    if not move or not move.is_legal(state, state.current_player):
        return jsonify({'error': 'Illegal move'}), 400
    
    # Capture pre-move data for turn recording
    pre_state = state
    legal_moves_recorded = Rules.get_legal_moves(state)

    # Apply move
    state = move.apply(state)
    active_games[match_id] = state

    # Record this turn
    match_state['round_turns'].append({
        "turn": len(match_state['round_turns']) + 1,
        "state": pre_state,
        "post_state": state,
        "agent": SimpleAgent(name="Human"),
        "move": move,
        "legal_moves": legal_moves_recorded,
    })
    
    # Check if round is over
    if state.game_over:
        return handle_round_end(match_id, state)
    
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
        
        # Capture pre-move state for turn recording
        pre_state = state

        # In debug mode, capture detailed move information
        if match_state['debug_mode']:
            agent_type = match_state['agent_types'].get(str(state.current_player), 'Unknown')
            move_description = format_move_for_debug(move, state.current_player)
            
            # Annotate if agent is using revealed hand info
            revealed_target = state.dice_state.get_revealed_target(state.current_player)
            if revealed_target is not None:
                move_description += f" [👁 saw P{revealed_target + 1}'s hand]"
            
            move_info = {
                'player': state.current_player,
                'agent_type': agent_type,
                'move': str(move),
                'move_description': move_description,
                'legal_moves_count': len(legal_moves),
                'had_revealed_info': revealed_target is not None
            }
        
        state = move.apply(state)
        active_games[match_id] = state

        # Record this turn
        match_state['round_turns'].append({
            "turn": len(match_state['round_turns']) + 1,
            "state": pre_state,
            "post_state": state,
            "agent": agent,
            "move": move,
            "legal_moves": legal_moves,
        })
    
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
                try:
                    from agents.mcts_agent import MCTSAgent
                    try:
                        agents.append(MCTSAgent())
                    except TypeError:
                        try:
                            agents.append(MCTSAgent(num_iterations=100))
                        except TypeError:
                            agents.append(MCTSAgent())
                except ImportError as e:
                    print(f"MCTS agent not available for player {i}, using balanced heuristic: {e}")
                    agents.append(create_balanced_heuristic())
                except Exception as e:
                    print(f"Error creating MCTS agent for player {i}, using balanced heuristic: {e}")
                    agents.append(create_balanced_heuristic())
            elif agent_type == 'rl':
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
                agents.append(create_balanced_heuristic())
    return agents


def start_new_round(match_id: str) -> GameState:
    """Start a new round in the match."""
    match_state = match_states[match_id]
    match_state['round_number'] += 1
    match_state['round_turns'] = []   # reset turn recording for this round
    
    state = Rules.initialize_game(
        match_state['num_players'],
        match_state['variant']
    )
    
    for i, player in enumerate(state.players):
        player.match_score = match_state['match_scores'][i]
    
    state.round_number = match_state['round_number']
    
    if match_state['round_number'] == 1:
        starting_player = match_state['starting_player']
    else:
        current_starter = match_state['current_round_starter']
        starting_player = (current_starter + 1) % match_state['num_players']
    
    state.current_player = starting_player
    match_state['current_round_starter'] = starting_player
    
    active_games[match_id] = state
    return state


def process_bot_turns(match_id: str, state: GameState) -> GameState:
    """Process all bot turns until human player's turn or game over."""
    match_state = match_states[match_id]
    
    max_iterations = 100
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
    
    Rules.compute_round_scores(state)
    
    for i, player in enumerate(state.players):
        match_state['match_scores'][i] = player.match_score

    # Append the final-state sentinel so LaTeX generation can read end scores
    match_state['round_turns'].append({
        "turn": len(match_state['round_turns']) + 1,
        "state": state,
        "agent": None,
        "move": None,
        "legal_moves": [],
        "post_state": None,
        "final": True,
    })
    
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
    
    state = start_new_round(match_id)
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
    """
    Convert GameState to JSON-serialisable dict.

    Information-reveal fields added:
      dice_state.revealed_hands   — dict mapping viewer_index (str) → target_index
                                    (all active reveal relationships)
      visible_hands               — dict mapping player_index (str) → card list,
                                    containing only hands visible to the human player
      opponents_see_my_hand       — bool: True if any opponent currently has
                                    visibility of the human player's hand
                                    (BadDiceEffect.REVEAL_HAND was rolled)
    """
    debug_mode = False
    human_player = 0
    if match_id and match_id in match_states:
        debug_mode = match_states[match_id].get('debug_mode', False)
        human_player = match_states[match_id].get('human_player', 0)
    
    players_data = []
    for p in state.players:
        player_data = {
            'index': p.index,
            'hand_size': p.hand_size(),
            'round_score': p.round_score,
            'match_score': p.match_score,
            'hand': [{'suit': card.suit.value, 'rank': card.rank} for card in p.hand]
        }
        if debug_mode:
            player_data['debug_visible'] = True
        players_data.append(player_data)
    
    # -----------------------------------------------------------------------
    # Build visible_hands: cards that the human player is entitled to see.
    # 1. Always include their own hand (index == human_player).
    # 2. Include any opponent hand revealed by INFO_REVEAL (good dice effect).
    # -----------------------------------------------------------------------
    visible_hands: Dict[str, list] = {}
    
    # Own hand always visible
    visible_hands[str(human_player)] = [
        {'suit': c.suit.value, 'rank': c.rank}
        for c in state.players[human_player].hand
    ]
    
    # Opponent hand revealed to human player (INFO_REVEAL good effect)
    revealed_target = state.dice_state.get_revealed_target(human_player)
    if revealed_target is not None:
        visible_hands[str(revealed_target)] = [
            {'suit': c.suit.value, 'rank': c.rank}
            for c in state.players[revealed_target].hand
        ]
    
    # -----------------------------------------------------------------------
    # Check if human player's hand is exposed to opponents.
    # This happens when BadDiceEffect.REVEAL_HAND was rolled by another player:
    #   revealed_hands[opp_i] == human_player  for each opponent opp_i
    # -----------------------------------------------------------------------
    opponents_see_my_hand = any(
        state.dice_state.get_revealed_target(i) == human_player
        for i in range(len(state.players))
        if i != human_player
    )

    # -----------------------------------------------------------------------
    # Serialise the full revealed_hands dict so the frontend can display
    # appropriate notifications for every active reveal relationship.
    # Keys and values are converted to strings for JSON compatibility.
    # -----------------------------------------------------------------------
    revealed_hands_serialized = {
        str(viewer): target
        for viewer, target in state.dice_state.revealed_hands.items()
    }
    
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
            # Full reveal map (viewer_str -> target_int)
            'revealed_hands': revealed_hands_serialized,
        },
        # Convenience fields for the human-player view
        'visible_hands': visible_hands,
        'opponents_see_my_hand': opponents_see_my_hand,
        'debug_mode': debug_mode
    }


@app.route('/api/generate_round_latex', methods=['POST'])
def generate_round_latex():
    """Generate a LaTeX flow diagram for the most recently completed round."""
    data = request.json
    match_id = data.get('match_id')

    if match_id not in match_states:
        return jsonify({'error': 'Match not found'}), 404

    match_state = match_states[match_id]
    round_turns = match_state.get('round_turns', [])

    if not round_turns:
        return jsonify({'error': 'No turn data for this round'}), 400

    try:
        from simulation.visualise_flow import LaTeXGridVisualizer

        # Build named agent list (None slots → "Human")
        agents_list = []
        for i in range(match_state['num_players']):
            raw = match_state['agents'][i]
            agents_list.append(raw if raw is not None else SimpleAgent(name="Human"))

        # Prepend the mandatory turn-0 sentinel
        turn_states = [{
            "turn": 0,
            "state": round_turns[0]["state"].copy(),
            "agent": None,
            "move": None,
            "legal_moves": [],
            "post_state": None,
        }] + round_turns

        variant_name = match_state.get('variant_name', 'Custom')

        class _NonClosingStringIO(io.StringIO):
            def close(self):
                # Prevent parent code from closing the buffer before we read it
                pass

        class _BufferVis(LaTeXGridVisualizer):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self._buf = _NonClosingStringIO()
                self.file_handle = self._buf

            def get_latex(self):
                return self._buf.getvalue()

        vis = _BufferVis(
            agents=agents_list,
            variant=match_state['variant'],
            variant_name=variant_name,
        )
        vis.turn_states = turn_states
        vis._generate_latex()

        latex_text = vis.get_latex()

        return jsonify({
            'latex': latex_text,
            'round': match_state['round_number']
        })

    except Exception as exc:
        import traceback
        return jsonify({
            'error': str(exc),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)