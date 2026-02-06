"""
Reinforcement Learning agent for Cinquillo 2.0.
Uses Q-learning with function approximation.
"""
import random
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
from collections import deque
from pathlib import Path

from game.entities import GameState, Suit, Card
from game.rules import Move, Rules, PlayCard, RollDice, Pass
from agents.base_agents import Agent


class StateEncoder:
    """Encodes game state into a rich feature vector."""
    
    @staticmethod
    def encode(state: GameState, player_index: int) -> np.ndarray:
        """
        Encode state into comprehensive feature vector.
        
        Enhanced features:
        - Hand cards (40 binary)
        - Board state (40 binary)
        - Playable cards (40 binary)
        - Hand size metrics
        - Opponent hand sizes
        - Round/turn info
        - Strategic features (5s, sequence extensions, etc.)
        
        Total: ~150 features
        """
        features = []
        
        player = state.players[player_index]
        
        # Encode player's hand (40 binary features)
        hand_encoding = StateEncoder._encode_cards(player.hand)
        features.extend(hand_encoding)
        
        # Encode board state (40 binary features)
        board_encoding = StateEncoder._encode_board(state.board)
        features.extend(board_encoding)
        
        # Encode playable cards (40 binary features)
        playable_encoding = StateEncoder._encode_playable_cards(state, player_index)
        features.extend(playable_encoding)
        
        # Hand size (normalized)
        hand_size_norm = player.hand_size() / 10.0
        features.append(hand_size_norm)
        
        # Opponent hand sizes (normalized)
        for p in state.players:
            if p.index != player_index:
                opp_hand_size = p.hand_size() / 10.0
                features.append(opp_hand_size)
        
        # Current player indicator
        is_current = 1.0 if state.current_player == player_index else 0.0
        features.append(is_current)
        
        # Dice state
        features.append(1.0 if state.dice_state.wild_active else 0.0)
        features.append(1.0 if state.dice_state.double_play_active else 0.0)
        
        # Round and turn information (normalized)
        features.append(state.round_number / 10.0)
        features.append(state.turn_number / 100.0)
        
        # Score information
        features.append(np.clip(player.round_score / 50.0, -1.0, 1.0))
        features.append(np.clip(player.match_score / 50.0, -1.0, 1.0))
        
        # Strategic features
        # Has 5s (important to play early)
        has_5s = sum(1 for c in player.hand if c.rank == 5) / 4.0
        features.append(has_5s)
        
        # Can extend sequences
        can_extend = sum(1 for _ in StateEncoder._get_extension_cards(state, player)) / 10.0
        features.append(can_extend)
        
        # Relative position (normalized)
        if len(state.players) > 1:
            hand_sizes = [p.hand_size() for p in state.players]
            max_hand = max(hand_sizes)
            relative_pos = 1.0 - (player.hand_size() / max(max_hand, 1))
            features.append(relative_pos)
        else:
            features.append(0.5)
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def _encode_cards(cards: List[Card]) -> List[float]:
        """Encode a list of cards as binary features."""
        encoding = [0.0] * 40
        
        suit_order = [Suit.OROS, Suit.COPAS, Suit.ESPADAS, Suit.BASTOS]
        rank_to_idx = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 10: 7, 11: 8, 12: 9}
        
        for card in cards:
            suit_idx = suit_order.index(card.suit)
            rank_idx = rank_to_idx[card.rank]
            idx = suit_idx * 10 + rank_idx
            encoding[idx] = 1.0
        
        return encoding
    
    @staticmethod
    def _encode_board(board) -> List[float]:
        """Encode board state as binary features."""
        encoding = [0.0] * 40
        
        suit_order = [Suit.OROS, Suit.COPAS, Suit.ESPADAS, Suit.BASTOS]
        rank_to_idx = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 10: 7, 11: 8, 12: 9}
        
        for suit in suit_order:
            suit_idx = suit_order.index(suit)
            for rank in board.suit_cards[suit]:
                rank_idx = rank_to_idx[rank]
                idx = suit_idx * 10 + rank_idx
                encoding[idx] = 1.0
        
        return encoding
    
    @staticmethod
    def _encode_playable_cards(state: GameState, player_index: int) -> List[float]:
        """Encode which cards are currently playable."""
        encoding = [0.0] * 40
        
        player = state.players[player_index]
        suit_order = [Suit.OROS, Suit.COPAS, Suit.ESPADAS, Suit.BASTOS]
        rank_to_idx = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 10: 7, 11: 8, 12: 9}
        
        for card in player.hand:
            # Check if card can be played using the actual game logic
            can_play = False
            
            # If wild is active, any card can be played
            if state.dice_state.wild_active:
                can_play = True
            # If suit not on board, must be a 5
            elif state.board.is_empty(card.suit):
                can_play = (card.rank == 5)
            # If suit is on board, must be adjacent or a 5
            elif card.rank == 5:
                can_play = not state.board.has_rank(card.suit, 5)
            else:
                can_play = (state.board.is_adjacent(card) and 
                           not state.board.has_rank(card.suit, card.rank))
            
            if can_play:
                suit_idx = suit_order.index(card.suit)
                rank_idx = rank_to_idx[card.rank]
                idx = suit_idx * 10 + rank_idx
                encoding[idx] = 1.0
        
        return encoding
    
    @staticmethod
    def _get_extension_cards(state: GameState, player) -> List[Card]:
        """Get cards that extend existing sequences."""
        extensions = []
        for card in player.hand:
            # Check if card can be played and is not a 5
            if card.rank != 5:
                # Card must be adjacent to existing cards
                if not state.board.is_empty(card.suit):
                    if state.board.is_adjacent(card) and not state.board.has_rank(card.suit, card.rank):
                        extensions.append(card)
        return extensions


class ImprovedQNetwork:
    """Improved neural network with gradient clipping."""
    
    def __init__(self, state_dim: int, action_dim: int = 3, hidden_dims: List[int] = [256, 128]):
        """
        Initialize Q-network with multiple hidden layers.
        
        Args:
            state_dim: Dimension of state features
            action_dim: Number of action types
            hidden_dims: List of hidden layer dimensions
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # Initialize weights with Xavier initialization
        self.weights = []
        self.biases = []
        
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            # Xavier initialization
            w = np.random.randn(prev_dim, hidden_dim) * np.sqrt(2.0 / prev_dim)
            b = np.zeros(hidden_dim)
            self.weights.append(w)
            self.biases.append(b)
            prev_dim = hidden_dim
        
        # Output layer
        w_out = np.random.randn(prev_dim, action_dim) * np.sqrt(2.0 / prev_dim)
        b_out = np.zeros(action_dim)
        self.weights.append(w_out)
        self.biases.append(b_out)
    
    def forward(self, state: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Forward pass through network, returning Q-values and activations."""
        activations = [state]
        
        # Forward through hidden layers
        for i in range(len(self.hidden_dims)):
            h = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            h = np.maximum(0, h)  # ReLU
            activations.append(h)
        
        # Output layer
        q_values = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        
        # Clip Q-values to prevent explosion
        q_values = np.clip(q_values, -100, 100)
        
        return q_values, activations
    
    def update(self, state: np.ndarray, action_idx: int, target: float, lr: float = 0.001):
        """Update weights using gradient descent with gradient clipping."""
        # Clip target to prevent explosion
        target = np.clip(target, -100, 100)
        
        # Forward pass
        q_values, activations = self.forward(state)
        
        # Compute loss gradient
        q_pred = q_values[action_idx]
        error = q_pred - target
        
        # Clip error
        error = np.clip(error, -10, 10)
        
        # Backprop through network
        dq = np.zeros_like(q_values)
        dq[action_idx] = error
        
        # Backprop through each layer
        gradients = []
        d_activation = dq
        
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradient w.r.t. weights and biases
            dW = np.outer(activations[i], d_activation)
            db = d_activation
            
            # Gradient clipping
            dW = np.clip(dW, -1.0, 1.0)
            db = np.clip(db, -1.0, 1.0)
            
            gradients.append((dW, db))
            
            # Gradient w.r.t. previous activation
            if i > 0:
                d_activation = np.dot(d_activation, self.weights[i].T)
                # ReLU gradient
                d_activation[activations[i] <= 0] = 0
                # Clip gradient
                d_activation = np.clip(d_activation, -1.0, 1.0)
        
        # Update weights (reverse order)
        gradients = gradients[::-1]
        for i, (dW, db) in enumerate(gradients):
            self.weights[i] -= lr * dW
            self.biases[i] -= lr * db
            
            # Weight clipping to prevent explosion
            self.weights[i] = np.clip(self.weights[i], -10, 10)
            self.biases[i] = np.clip(self.biases[i], -10, 10)


class RLAgent(Agent):
    """
    Improved Reinforcement Learning agent using Q-learning.
    Includes proper action representation and reward shaping.
    """
    
    def __init__(self, 
                 name: str = "RL",
                 epsilon: float = 0.1,
                 learning_rate: float = 0.0005,
                 discount_factor: float = 0.95,
                 use_heuristics: bool = True):
        """
        Initialize RL agent.
        
        Args:
            name: Agent name
            epsilon: Exploration rate (epsilon-greedy)
            learning_rate: Learning rate for Q-network updates
            discount_factor: Discount factor for future rewards
            use_heuristics: Whether to use heuristic guidance for action selection
        """
        super().__init__(name)
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.use_heuristics = use_heuristics
        
        # Initialize Q-network
        self.q_network: Optional[ImprovedQNetwork] = None
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=50000)
        
        # Statistics
        self.total_updates = 0
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        self.episode_step_count = 0
    
    def choose_move(self, state: GameState, legal_moves: List[Move]) -> Move:
        """Choose move using epsilon-greedy policy with heuristic guidance."""
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Encode state
        state_features = StateEncoder.encode(state, state.current_player)
        
        # Initialize network if needed
        if self.q_network is None:
            self.q_network = ImprovedQNetwork(len(state_features))
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: use heuristic or random
            if self.use_heuristics:
                return self._heuristic_move(state, legal_moves)
            else:
                return random.choice(legal_moves)
        else:
            # Exploit: choose best move according to Q-values
            return self._choose_best_move(state, state_features, legal_moves)
    
    def _choose_best_move(self, state: GameState, state_features: np.ndarray, 
                         legal_moves: List[Move]) -> Move:
        """Choose move with highest Q-value, with heuristic tiebreaking."""
        move_values = []
        
        for move in legal_moves:
            # Compute Q-value for this move
            q_value = self._evaluate_move(state, state_features, move)
            
            # Add small heuristic bonus for tiebreaking
            heuristic_bonus = 0.0
            if self.use_heuristics:
                heuristic_bonus = self._heuristic_value(state, move) * 0.01
            
            move_values.append((move, q_value + heuristic_bonus))
        
        # Return move with highest value
        best_move = max(move_values, key=lambda x: x[1])[0]
        return best_move
    
    def _evaluate_move(self, state: GameState, state_features: np.ndarray, 
                      move: Move) -> float:
        """Evaluate a specific move using the Q-network."""
        # Get base Q-values for action types
        q_values, _ = self.q_network.forward(state_features)
        
        # Map move to action type
        if isinstance(move, PlayCard):
            base_q = q_values[0]
            # Add card-specific adjustment
            card_bonus = self._card_value_adjustment(state, move.card)
            return base_q + card_bonus
        elif isinstance(move, RollDice):
            return q_values[1]
        elif isinstance(move, Pass):
            return q_values[2]
        return 0.0
    
    def _card_value_adjustment(self, state: GameState, card: Card) -> float:
        """Compute value adjustment for specific card plays."""
        adjustment = 0.0
        
        # Strong preference for 5s early in the round
        if card.rank == 5:
            cards_played = sum(len(state.board.suit_cards[suit]) for suit in Suit)
            if cards_played < 10:
                adjustment += 0.5
        
        # Prefer cards that extend sequences (not 5s)
        if card.rank != 5:
            # Check if it's adjacent (extends a sequence)
            if not state.board.is_empty(card.suit):
                if state.board.is_adjacent(card) and not state.board.has_rank(card.suit, card.rank):
                    adjustment += 0.2
        
        # Slight penalty for high rank cards (harder to extend later)
        if card.rank in [11, 12]:
            adjustment -= 0.1
        
        return adjustment
    
    def _heuristic_move(self, state: GameState, legal_moves: List[Move]) -> Move:
        """Select move using simple heuristics."""
        playable_moves = [m for m in legal_moves if isinstance(m, PlayCard)]
        
        if playable_moves:
            # Prefer 5s
            fives = [m for m in playable_moves if m.card.rank == 5]
            if fives:
                return random.choice(fives)
            
            # Then sequence extensions (cards that are adjacent)
            extensions = []
            for m in playable_moves:
                if m.card.rank != 5:
                    if not state.board.is_empty(m.card.suit):
                        if state.board.is_adjacent(m.card):
                            extensions.append(m)
            
            if extensions:
                return random.choice(extensions)
            
            return random.choice(playable_moves)
        
        return random.choice(legal_moves)
    
    def _heuristic_value(self, state: GameState, move: Move) -> float:
        """Compute heuristic value for a move (0-1)."""
        if isinstance(move, PlayCard):
            value = 0.5
            if move.card.rank == 5:
                value += 0.3
            elif not state.board.is_empty(move.card.suit) and state.board.is_adjacent(move.card):
                value += 0.2
            return value
        elif isinstance(move, RollDice):
            return 0.4
        elif isinstance(move, Pass):
            return 0.1
        return 0.0
    
    def compute_reward(self, state: GameState, action: Move, 
                      next_state: GameState, player_index: int) -> float:
        """
        Compute reward for a state transition with improved shaping.
        """
        reward = 0.0
        
        player = state.players[player_index]
        next_player = next_state.players[player_index]
        
        # Playing a card is good
        if isinstance(action, PlayCard):
            reward += 1.0
            
            # Extra bonus for playing 5s early
            if action.card.rank == 5:
                cards_played = sum(len(state.board.suit_cards[suit]) for suit in Suit)
                if cards_played < 10:
                    reward += 2.0
            
            # Bonus for extending sequences
            if action.card.rank != 5 and not state.board.is_empty(action.card.suit):
                if state.board.is_adjacent(action.card):
                    reward += 0.5
        
        # Hand size reduction is good
        hand_diff = player.hand_size() - next_player.hand_size()
        reward += hand_diff * 2.0
        
        # Round score improvement is good
        round_score_diff = next_player.round_score - player.round_score
        reward += round_score_diff * 1.0
        
        # Penalty for passing
        if isinstance(action, Pass) and action.voluntary:
            reward -= 3.0
        
        # Check if round ended (someone went out)
        if next_state.game_over and not state.game_over:
            # Round just ended
            round_score = next_player.round_score
            reward += round_score * 5.0
            
            # Large bonus for going out first
            if next_player.hand_size() == 0:
                reward += 30.0
            else:
                # Penalty for not going out
                reward -= next_player.hand_size() * 2.0
            
            # Match score matters
            final_score = next_player.match_score
            reward += final_score * 5.0
            
            # Bonus/penalty for placement
            scores = [p.match_score for p in next_state.players]
            max_score = max(scores)
            min_score = min(scores)
            
            if next_player.match_score == max_score:
                reward += 50.0
            elif next_player.match_score == min_score:
                reward -= 30.0
        
        # Clip reward to reasonable range
        return np.clip(reward, -100, 100)
    
    def update_from_experience(self, 
                               state: GameState, 
                               action: Move,
                               reward: float,
                               next_state: GameState,
                               done: bool):
        """Update Q-network from a single experience tuple."""
        if self.q_network is None:
            return
        
        player_index = state.current_player
        
        # Encode states
        state_features = StateEncoder.encode(state, player_index)
        next_state_features = StateEncoder.encode(next_state, player_index)
        
        # Compute target Q-value
        if done:
            target_q = reward
        else:
            # Q-learning: target = r + gamma * max_a' Q(s', a')
            next_q_values, _ = self.q_network.forward(next_state_features)
            target_q = reward + self.discount_factor * np.max(next_q_values)
        
        # Get action type index for update
        if isinstance(action, PlayCard):
            action_idx = 0
        elif isinstance(action, RollDice):
            action_idx = 1
        elif isinstance(action, Pass):
            action_idx = 2
        else:
            action_idx = 0
        
        # Update network
        self.q_network.update(state_features, action_idx, target_q, self.learning_rate)
        self.total_updates += 1
    
    def store_experience(self, 
                        state: GameState,
                        action: Move,
                        reward: float,
                        next_state: GameState,
                        done: bool):
        """Store experience in replay buffer."""
        # Track cumulative reward
        self.current_episode_reward += reward
        self.episode_step_count += 1
        
        # Create lightweight copies of states
        experience = {
            'state': StateEncoder.encode(state, state.current_player),
            'action': action,
            'reward': reward,
            'next_state': StateEncoder.encode(next_state, state.current_player),
            'done': done
        }
        self.replay_buffer.append(experience)
    
    def train_from_replay(self, batch_size: int = 64):
        """Train Q-network using experience replay."""
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample random batch
        batch = random.sample(self.replay_buffer, batch_size)
        
        for experience in batch:
            state_features = experience['state']
            action = experience['action']
            reward = experience['reward']
            next_state_features = experience['next_state']
            done = experience['done']
            
            # Compute target
            if done:
                target_q = reward
            else:
                next_q_values, _ = self.q_network.forward(next_state_features)
                target_q = reward + self.discount_factor * np.max(next_q_values)
            
            # Get action index
            if isinstance(action, PlayCard):
                action_idx = 0
            elif isinstance(action, RollDice):
                action_idx = 1
            elif isinstance(action, Pass):
                action_idx = 2
            else:
                action_idx = 0
            
            # Update network
            self.q_network.update(state_features, action_idx, target_q, self.learning_rate)
            self.total_updates += 1
    
    def decay_epsilon(self, decay_rate: float = 0.995, min_epsilon: float = 0.05):
        """Decay exploration rate."""
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
    
    def end_episode(self):
        """Called at the end of an episode."""
        # Store the cumulative reward for this episode
        avg_reward = self.current_episode_reward / max(self.episode_step_count, 1)
        self.episode_rewards.append(avg_reward)
        
        # Reset counters
        self.current_episode_reward = 0.0
        self.episode_step_count = 0
    
    def save_weights(self, filepath: str):
        """Save network weights to file."""
        if self.q_network is None:
            print("No network to save")
            return
        
        weights_data = {
            'weights': self.q_network.weights,
            'biases': self.q_network.biases,
            'state_dim': self.q_network.state_dim,
            'action_dim': self.q_network.action_dim,
            'hidden_dims': self.q_network.hidden_dims,
            'epsilon': self.epsilon,
            'total_updates': self.total_updates
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(weights_data, f)
        print(f"Saved weights to {filepath}")
    
    def load_weights(self, filepath: str):
        """Load network weights from file."""
        try:
            with open(filepath, 'rb') as f:
                weights_data = pickle.load(f)
            
            # Initialize network if needed
            if self.q_network is None:
                self.q_network = ImprovedQNetwork(
                    weights_data['state_dim'],
                    weights_data['action_dim'],
                    weights_data['hidden_dims']
                )
            
            # Load weights
            self.q_network.weights = weights_data['weights']
            self.q_network.biases = weights_data['biases']
            self.epsilon = weights_data.get('epsilon', self.epsilon)
            self.total_updates = weights_data.get('total_updates', 0)
            
            print(f"Loaded weights from {filepath}")
            print(f"  Epsilon: {self.epsilon:.3f}")
            print(f"  Total updates: {self.total_updates}")
        except FileNotFoundError:
            print(f"No weights file found at {filepath}")
        except Exception as e:
            print(f"Error loading weights: {e}")


# Pre-configured variants
class RLAgentExplore(RLAgent):
    """RL agent with higher exploration rate."""
    
    def __init__(self, name: str = "RL-Explore"):
        super().__init__(
            name=name,
            epsilon=0.3,
            learning_rate=0.0005,
            discount_factor=0.95,
            use_heuristics=True
        )


class RLAgentExploit(RLAgent):
    """RL agent with lower exploration rate (more exploitation)."""
    
    def __init__(self, name: str = "RL-Exploit"):
        super().__init__(
            name=name,
            epsilon=0.05,
            learning_rate=0.0005,
            discount_factor=0.95,
            use_heuristics=True
        )


class RLAgentPure(RLAgent):
    """RL agent without heuristic guidance."""
    
    def __init__(self, name: str = "RL-Pure"):
        super().__init__(
            name=name,
            epsilon=0.15,
            learning_rate=0.0005,
            discount_factor=0.95,
            use_heuristics=False
        )