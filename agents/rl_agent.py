"""
Reinforcement Learning agent for Cinquillo 2.0

Total feature count (4-player game): = 209

"""

import random
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from game.entities import GameState, Suit, Card, Deck
from game.rules import Move, Rules, PlayCard, RollDice, Pass
from agents.base_agents import Agent


# ---------------------------------------------------------------------------
# Prioritized Experience Replay
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """
    Circular replay buffer with proportional prioritization.

    Experiences are sampled with probability proportional to
    |TD-error|^alpha.  Importance-sampling weights w_i = (N * P(i))^{-beta}
    are returned and used to scale the gradient update, correcting for the
    non-uniform distribution.  beta is annealed from beta_start to 1.0 over
    training.
    """

    def __init__(self,
                 capacity: int = 100_000,
                 alpha: float = 0.6,
                 beta_start: float = 0.4,
                 beta_increment: float = 5e-4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = beta_increment

        self._buffer: list = []
        self._priorities: list = []
        self._pos: int = 0
        self._max_priority: float = 1.0

    # ------------------------------------------------------------------
    def add(self, experience: dict) -> None:
        """Insert an experience with maximum current priority."""
        if len(self._buffer) < self.capacity:
            self._buffer.append(experience)
            self._priorities.append(self._max_priority)
        else:
            self._buffer[self._pos] = experience
            self._priorities[self._pos] = self._max_priority
        self._pos = (self._pos + 1) % self.capacity

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> Tuple[list, np.ndarray, np.ndarray]:
        """
        Returns (samples, indices, importance_sampling_weights).
        Anneals beta toward 1.0 on every call.
        """
        n = len(self._buffer)
        batch_size = min(batch_size, n)

        probs = np.array(self._priorities[:n], dtype=np.float64) ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        samples = [self._buffer[i] for i in indices]

        # Importance-sampling weights (normalised so max weight == 1)
        weights = (n * probs[indices]) ** (-self.beta)
        weights = (weights / weights.max()).astype(np.float32)

        self.beta = min(1.0, self.beta + self.beta_increment)
        return samples, indices, weights

    # ------------------------------------------------------------------
    def update_priorities(self, indices: np.ndarray, td_errors: list) -> None:
        """Update stored priorities after a training step."""
        for idx, td_err in zip(indices, td_errors):
            p = float(abs(td_err)) + 1e-6
            self._priorities[idx] = p
            if p > self._max_priority:
                self._max_priority = p

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

TARGET_SYNC_FREQ = 1000   # gradient steps between target-network syncs


class ImprovedQNetwork:
    """
    Three-output Q-network [PlayCard, RollDice, Pass] with
    Double-DQN target network support.

    Architecture: state_dim → 512 → 256 → 128 → 3  (all ReLU hidden).
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int = 42,
                 hidden_dims: List[int] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [256, 128]

        self.weights: List[np.ndarray] = []
        self.biases:  List[np.ndarray] = []

        prev = state_dim
        for h in self.hidden_dims:
            self.weights.append(np.random.randn(prev, h) * np.sqrt(2.0 / prev))
            self.biases.append(np.zeros(h))
            prev = h
        self.weights.append(np.random.randn(prev, action_dim) * np.sqrt(2.0 / prev))
        self.biases.append(np.zeros(action_dim))

        # Target network — initially identical
        self.target_weights: List[np.ndarray] = [w.copy() for w in self.weights]
        self.target_biases:  List[np.ndarray] = [b.copy() for b in self.biases]

    # ------------------------------------------------------------------
    def sync_target(self) -> None:
        """Hard-copy online weights into the target network."""
        self.target_weights = [w.copy() for w in self.weights]
        self.target_biases  = [b.copy() for b in self.biases]

    # ------------------------------------------------------------------
    def forward(self,
                state: np.ndarray,
                use_target: bool = False) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Forward pass; returns (q_values, activations)."""
        ws = self.target_weights if use_target else self.weights
        bs = self.target_biases  if use_target else self.biases

        activations = [state]
        for i in range(len(self.hidden_dims)):
            h = np.dot(activations[-1], ws[i]) + bs[i]
            h = np.maximum(0.0, h)           # ReLU
            activations.append(h)

        q = np.dot(activations[-1], ws[-1]) + bs[-1]
        q = np.clip(q, -100.0, 100.0)
        return q, activations

    # ------------------------------------------------------------------
    def update(self,
               state: np.ndarray,
               action_idx: int,
               target: float,
               lr: float,
               importance_weight: float = 1.0) -> float:
        """
        Single-step gradient-descent update.
        Returns |TD error| for PER priority update.
        """
        target = float(np.clip(target, -100.0, 100.0))
        q_values, activations = self.forward(state, use_target=False)

        q_pred   = q_values[action_idx]
        td_error = target - q_pred          # sign for PER (return |td_error|)
        error    = float(np.clip(q_pred - target, -10.0, 10.0)) * importance_weight

        dq = np.zeros_like(q_values)
        dq[action_idx] = error

        d_act = dq
        grads = []
        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.clip(np.outer(activations[i], d_act), -1.0, 1.0)
            db = np.clip(d_act, -1.0, 1.0)
            grads.append((dW, db))
            if i > 0:
                d_act = np.dot(d_act, self.weights[i].T)
                d_act[activations[i] <= 0] = 0.0
                d_act = np.clip(d_act, -1.0, 1.0)

        grads.reverse()
        for i, (dW, db) in enumerate(grads):
            self.weights[i] -= lr * dW
            self.biases[i]  -= lr * db
            self.weights[i]  = np.clip(self.weights[i], -10.0, 10.0)
            self.biases[i]   = np.clip(self.biases[i],  -10.0, 10.0)

        return abs(td_error)


# ---------------------------------------------------------------------------
# State encoder
# ---------------------------------------------------------------------------

class StateEncoder:
    """
    Encodes game state into a feature vector.
    """

    @staticmethod
    def encode(state: GameState, player_index: int) -> np.ndarray:
        features: List[float] = []
        player = state.players[player_index]

        # ── Core card / board encoding (120 features) ──────────────────
        features.extend(StateEncoder._encode_cards(player.hand))
        features.extend(StateEncoder._encode_board(state.board))
        features.extend(StateEncoder._encode_playable_cards(state, player_index))

        # ── Scalar game-state features ──────────────────────────────────
        features.append(player.hand_size() / 10.0)

        for p in state.players:
            if p.index != player_index:
                features.append(p.hand_size() / 10.0)

        features.append(1.0 if state.current_player == player_index else 0.0)
        features.append(1.0 if state.dice_state.wild_active        else 0.0)
        features.append(1.0 if state.dice_state.double_play_active else 0.0)

        features.append(state.round_number / 10.0)
        features.append(state.turn_number  / 100.0)
        features.append(float(np.clip(player.round_score / 50.0, -1.0, 1.0)))
        features.append(float(np.clip(player.match_score / 50.0, -1.0, 1.0)))

        # Strategic aggregates
        features.append(sum(1 for c in player.hand if c.rank == 5) / 4.0)
        features.append(len(StateEncoder._get_extension_cards(state, player)) / 10.0)

        hand_sizes = [p.hand_size() for p in state.players]
        max_hand   = max(hand_sizes)
        features.append(1.0 - player.hand_size() / max(max_hand, 1))

        # ── Variant config (18 features) ────────────────────────────────
        # Full one-hot encoding for all good and bad effects so the network
        # can learn distinct strategies per variant (previously only WILD and
        # DOUBLE_PLAY were one-hotted; INFO_REVEAL/GIVE_CARD were invisible,
        # and TAKE_CARDS/FORCED_PASS/NEGATIVE_POINTS/REVEAL_HAND were
        # collapsed into one scalar, making them indistinguishable).
        from game.entities import ScoringMode, GoodDiceEffect, BadDiceEffect, MatchEndMode
        variant = state.variant

        features.append(variant.voluntary_pass_penalty / 5.0)            # 1
        features.append(variant.points_per_card        / 3.0)            # 1

        # Scoring mode (2)
        features.append(1.0 if variant.scoring_mode == ScoringMode.WINNER_TAKES_ALL else 0.0)
        features.append(1.0 if variant.scoring_mode == ScoringMode.DOUBLE_PENALTY   else 0.0)

        features.append(variant.dice_good_probability)                    # 1

        # Good effect — full one-hot (4)
        features.append(1.0 if variant.dice_good_effect == GoodDiceEffect.WILD        else 0.0)
        features.append(1.0 if variant.dice_good_effect == GoodDiceEffect.DOUBLE_PLAY else 0.0)
        features.append(1.0 if variant.dice_good_effect == GoodDiceEffect.INFO_REVEAL else 0.0)
        features.append(1.0 if variant.dice_good_effect == GoodDiceEffect.GIVE_CARD   else 0.0)

        # Bad effect — full one-hot (4)
        features.append(1.0 if variant.dice_bad_effect == BadDiceEffect.TAKE_CARDS      else 0.0)
        features.append(1.0 if variant.dice_bad_effect == BadDiceEffect.FORCED_PASS     else 0.0)
        features.append(1.0 if variant.dice_bad_effect == BadDiceEffect.NEGATIVE_POINTS else 0.0)
        features.append(1.0 if variant.dice_bad_effect == BadDiceEffect.REVEAL_HAND     else 0.0)

        # Bad effect severity scalars (2): magnitude of TAKE_CARDS / NEGATIVE_POINTS
        features.append(variant.dice_bad_cards_count    / 5.0)
        features.append(variant.dice_bad_penalty_points / 5.0)

        # Match end mode (2)
        features.append(1.0 if variant.match_end_mode == MatchEndMode.TARGET_SCORE else 0.0)
        features.append(1.0 if variant.match_end_mode == MatchEndMode.FIXED_ROUNDS  else 0.0)

        features.append(variant.fixed_rounds_count / 10.0)                # 1
        # Total variant features: 18  (was 11, +7)

        # ── Info-reveal (41 features, from v5) ─────────────────────────
        revealed_target = state.dice_state.get_revealed_target(player_index)
        if revealed_target is not None:
            features.append(1.0)
            features.extend(StateEncoder._encode_cards(state.players[revealed_target].hand))
        else:
            features.append(0.0)
            features.extend([0.0] * 40)

        # ── NEW v6: Suit progress (4 features) ─────────────────────────
        suit_order = [Suit.OROS, Suit.COPAS, Suit.ESPADAS, Suit.BASTOS]
        for suit in suit_order:
            features.append(len(state.board.suit_cards[suit]) / 10.0)

        # ── NEW v6: Suit edge positions (8 features) ────────────────────
        for suit in suit_order:
            if state.board.is_empty(suit):
                features.append(-1.0)
                features.append(-1.0)
            else:
                min_r, max_r = state.board.get_min_max(suit)
                features.append(Deck.RANK_INDEX[min_r] / 9.0)
                features.append(Deck.RANK_INDEX[max_r] / 9.0)

        # ── NEW v6: Opponent score differences (n-1 features) ───────────
        for p in state.players:
            if p.index != player_index:
                diff = float(np.clip(
                    (player.match_score - p.match_score) / 50.0, -2.0, 2.0
                ))
                features.append(diff)

        # ── NEW v6: Min-hand gap (1 feature) ────────────────────────────
        opp_sizes = [p.hand_size() for p in state.players if p.index != player_index]
        if opp_sizes:
            min_opp = min(opp_sizes)
            features.append(float(np.clip((player.hand_size() - min_opp) / 10.0, -1.0, 1.0)))
        else:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    # ------------------------------------------------------------------
    @staticmethod
    def _encode_cards(cards: List[Card]) -> List[float]:
        enc = [0.0] * 40
        suit_order  = [Suit.OROS, Suit.COPAS, Suit.ESPADAS, Suit.BASTOS]
        rank_to_idx = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 10:7, 11:8, 12:9}
        for card in cards:
            si = suit_order.index(card.suit)
            ri = rank_to_idx[card.rank]
            enc[si * 10 + ri] = 1.0
        return enc

    @staticmethod
    def _encode_board(board) -> List[float]:
        enc = [0.0] * 40
        suit_order  = [Suit.OROS, Suit.COPAS, Suit.ESPADAS, Suit.BASTOS]
        rank_to_idx = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 10:7, 11:8, 12:9}
        for si, suit in enumerate(suit_order):
            for rank in board.suit_cards[suit]:
                enc[si * 10 + rank_to_idx[rank]] = 1.0
        return enc

    @staticmethod
    def _encode_playable_cards(state: GameState, player_index: int) -> List[float]:
        enc = [0.0] * 40
        player      = state.players[player_index]
        suit_order  = [Suit.OROS, Suit.COPAS, Suit.ESPADAS, Suit.BASTOS]
        rank_to_idx = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 10:7, 11:8, 12:9}
        for card in player.hand:
            if state.dice_state.wild_active:
                can = True
            elif state.board.is_empty(card.suit):
                can = (card.rank == 5)
            elif card.rank == 5:
                can = not state.board.has_rank(card.suit, 5)
            else:
                can = (state.board.is_adjacent(card) and
                       not state.board.has_rank(card.suit, card.rank))
            if can:
                si = suit_order.index(card.suit)
                enc[si * 10 + rank_to_idx[card.rank]] = 1.0
        return enc

    @staticmethod
    def _get_extension_cards(state: GameState, player) -> List[Card]:
        return [
            c for c in player.hand
            if c.rank != 5
            and not state.board.is_empty(c.suit)
            and state.board.is_adjacent(c)
            and not state.board.has_rank(c.suit, c.rank)
        ]


# ---------------------------------------------------------------------------
# RL Agent
# ---------------------------------------------------------------------------

class RLAgent(Agent):
    """
    Q-learning agent with Double DQN, Prioritized Experience Replay,
    and domain-specific heuristic guidance.

    Action abstraction
    ------------------
    The Q-network produces three outputs:
      0 → PlayCard   1 → RollDice   2 → Pass

    When multiple PlayCard moves are legal, the network score for action 0
    is the same for all of them.  Card selection within PlayCard is refined
    by _card_value_adjustment() which adds a learned-style bonus for 5s,
    sequence extensions, chains, and blocking — keeping the 3-output design
    simple while still discriminating between specific cards.
    """

    def __init__(self,
                 name: str   = "RL",
                 epsilon: float   = 0.1,
                 learning_rate: float = 5e-4,
                 discount_factor: float = 0.95,
                 use_heuristics: bool = True):
        super().__init__(name)
        self.epsilon         = epsilon
        self.learning_rate   = learning_rate
        self.discount_factor = discount_factor
        self.use_heuristics  = use_heuristics

        self.q_network: Optional[ImprovedQNetwork] = None
        self.replay_buffer = PrioritizedReplayBuffer(capacity=100_000)

        self.total_updates          = 0
        self.episode_rewards: list  = []
        self.current_episode_reward = 0.0
        self.episode_step_count     = 0

        # Prevent degenerate looping on optional non-play actions.
        self.max_consecutive_voluntary_passes = 2
        self.max_consecutive_rolls = 2
        self._consecutive_voluntary_passes = 0
        self._consecutive_rolls = 0
        self._last_seen_turn_number: Optional[int] = None

    # ------------------------------------------------------------------
    # Move selection
    # ------------------------------------------------------------------

    # ── Action index mapping (42 actions) ──────────────────────────────────
    #   0-39 : PlayCard (suit 0-3, rank-slot 0-9)
    #   40   : RollDice
    #   41   : Pass
    _SUIT_ORDER  = [Suit.OROS, Suit.COPAS, Suit.ESPADAS, Suit.BASTOS]
    _RANK_TO_IDX = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 10:7, 11:8, 12:9}
    _ROLL_IDX    = 40
    _PASS_IDX    = 41

    @staticmethod
    def _move_to_idx(move: Move) -> int:
        """Map any legal Move to a 0-41 action index."""
        if isinstance(move, PlayCard):
            si = RLAgent._SUIT_ORDER.index(move.card.suit)
            ri = RLAgent._RANK_TO_IDX[move.card.rank]
            return si * 10 + ri
        if isinstance(move, RollDice):
            return RLAgent._ROLL_IDX
        return RLAgent._PASS_IDX   # Pass

    def choose_move(self, state: GameState, legal_moves: List[Move]) -> Move:
        """Epsilon-greedy with heuristic exploration."""
        if not legal_moves:
            raise ValueError("No legal moves available")

        self._sync_action_streak_tracking(state)
        constrained_moves = self._apply_action_constraints(legal_moves)

        if len(constrained_moves) == 1:
            move = constrained_moves[0]
            self._record_action_streak(move)
            return move

        state_features = StateEncoder.encode(state, state.current_player)

        if self.q_network is None:
            self.q_network = ImprovedQNetwork(len(state_features), 42)

        if random.random() < self.epsilon:
            move = (self._heuristic_move(state, constrained_moves)
                    if self.use_heuristics
                    else random.choice(constrained_moves))
        else:
            move = self._choose_best_move(state, state_features, constrained_moves)

        self._record_action_streak(move)
        return move

    def _sync_action_streak_tracking(self, state: GameState) -> None:
        """Reset per-game action streaks when a new game starts."""
        if self._last_seen_turn_number is None or state.turn_number < self._last_seen_turn_number:
            self._consecutive_voluntary_passes = 0
            self._consecutive_rolls = 0
        self._last_seen_turn_number = state.turn_number

    def _apply_action_constraints(self, legal_moves: List[Move]) -> List[Move]:
        """Remove optional actions that exceed the configured streak cap."""
        constrained = list(legal_moves)

        if self._consecutive_voluntary_passes >= self.max_consecutive_voluntary_passes:
            constrained = [
                m for m in constrained
                if not (isinstance(m, Pass) and getattr(m, 'voluntary', False))
            ]

        if self._consecutive_rolls >= self.max_consecutive_rolls:
            constrained = [m for m in constrained if not isinstance(m, RollDice)]

        return constrained if constrained else legal_moves

    def _record_action_streak(self, move: Move) -> None:
        """Track consecutive optional non-play actions chosen by the RL agent."""
        if isinstance(move, RollDice):
            self._consecutive_rolls += 1
            self._consecutive_voluntary_passes = 0
        elif isinstance(move, Pass) and getattr(move, 'voluntary', False):
            self._consecutive_voluntary_passes += 1
            self._consecutive_rolls = 0
        else:
            self._consecutive_rolls = 0
            self._consecutive_voluntary_passes = 0

    def _choose_best_move(self,
                          state: GameState,
                          state_features: np.ndarray,
                          legal_moves: List[Move]) -> Move:
        """
        Greedy action selection using per-card Q-values.

        Each legal move maps to a dedicated output neuron (0-41), so the
        network learns card-specific values rather than a shared PlayCard
        bucket.  The heuristic tiebreaker (weight 0.10) guides early training
        before Q-values are meaningful without overriding learned policy.
        """
        q_values, _ = self.q_network.forward(state_features)
        best_move, best_val = None, float('-inf')
        for move in legal_moves:
            idx = self._move_to_idx(move)
            val = q_values[idx]
            if self.use_heuristics:
                val += self._heuristic_value(state, move) * 0.10
            if val > best_val:
                best_val  = val
                best_move = move
        return best_move

    # ------------------------------------------------------------------
    # Card-level heuristic adjustments
    # ------------------------------------------------------------------


    def _heuristic_move(self, state: GameState, legal_moves: List[Move]) -> Move:
        """
        Priority ordering:
          1. Play a 5  (opens new suit)
          2. Block revealed opponent's next extension
          3. Chain move  (playing A immediately makes B in same suit playable)
          4. Sequence extension
          5. Any playable card
          6. Roll dice / Pass
        """
        plays = [m for m in legal_moves if isinstance(m, PlayCard)]
        if not plays:
            return random.choice(legal_moves)

        player = state.players[state.current_player]

        # Priority 1: 5s
        fives = [m for m in plays if m.card.rank == 5]
        if fives:
            return random.choice(fives)

        # Priority 2: blocking revealed opponent
        revealed_target = state.dice_state.get_revealed_target(state.current_player)
        if revealed_target is not None:
            rev_hand = state.players[revealed_target].hand
            blocking = []
            for m in plays:
                our_idx = Deck.RANK_INDEX[m.card.rank]
                for opp_c in rev_hand:
                    if opp_c.suit == m.card.suit:
                        if abs(our_idx - Deck.RANK_INDEX[opp_c.rank]) == 1:
                            blocking.append(m)
                            break
            if blocking:
                return random.choice(blocking)

        # Priority 3: chain moves
        chains = []
        for m in plays:
            if m.card.rank == 5:
                continue
            our_idx = Deck.RANK_INDEX[m.card.rank]
            for other in player.hand:
                if other is not m.card and other.suit == m.card.suit:
                    if abs(our_idx - Deck.RANK_INDEX[other.rank]) == 1:
                        chains.append(m)
                        break
        if chains:
            return random.choice(chains)

        # Priority 4: extensions
        ext = [m for m in plays
               if m.card.rank != 5
               and not state.board.is_empty(m.card.suit)
               and state.board.is_adjacent(m.card)]
        if ext:
            return random.choice(ext)

        return random.choice(plays)

    def _heuristic_value(self, state: GameState, move: Move) -> float:
        """Returns a 0–1 scalar used as a tiny tiebreaker alongside Q-values."""
        if isinstance(move, PlayCard):
            v = 0.5
            if move.card.rank == 5:
                v += 0.3
            elif (not state.board.is_empty(move.card.suit)
                  and state.board.is_adjacent(move.card)):
                v += 0.2
            revealed = state.dice_state.get_revealed_target(state.current_player)
            if revealed is not None:
                our_idx = Deck.RANK_INDEX[move.card.rank]
                for opp_c in state.players[revealed].hand:
                    if opp_c.suit == move.card.suit:
                        if abs(our_idx - Deck.RANK_INDEX[opp_c.rank]) == 1:
                            v += 0.15
                            break
            return min(1.0, v)
        if isinstance(move, RollDice):
            return 0.4
        return 0.1   # Pass

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def compute_reward(self,
                       state: GameState,
                       action: Move,
                       next_state: GameState,
                       player_index: int) -> float:
        """
        Action-immediate reward: only inspects state → next_state_after_action,
        which is the result of applying the single action.  This avoids
        penalising RL for opponent-caused hand growth (e.g. TAKE_CARDS between
        RL turns) which was the root cause of V1/V2 underperformance.

        The Q-bootstrap target still uses the next RL-turn state (handled in
        the training loop), so temporal credit assignment remains correct.
        """
        reward = 0.0
        player      = state.players[player_index]
        next_player = next_state.players[player_index]
        ppc         = state.variant.points_per_card

        if isinstance(action, PlayCard):
            # Direct hand reduction (always exactly -1 card)
            reward += 3.0
            # Bonus for 5s (open new suit)
            if action.card.rank == 5:
                reward += 2.0
            # Bonus for sequence extension
            elif (not state.board.is_empty(action.card.suit)
                  and state.board.is_adjacent(action.card)):
                reward += 1.0
            # Chain bonus: does this make another card in hand immediately playable?
            for c in player.hand:
                if c is not action.card and c.suit == action.card.suit:
                    if abs(Deck.RANK_INDEX[c.rank] - Deck.RANK_INDEX[action.card.rank]) == 1:
                        reward += 0.5
                        break

        elif isinstance(action, RollDice):
            # Reward/penalise based on what the dice immediately produced
            hand_delta  = next_player.hand_size() - player.hand_size()
            score_delta = next_player.match_score  - player.match_score
            if next_state.dice_state.wild_active:
                reward += 2.0           # got Wild — very valuable
            elif next_state.dice_state.double_play_active:
                reward += 1.5           # got Double Play
            if hand_delta > 0:
                reward -= hand_delta * 2.0 * ppc   # took cards (bad)
            if score_delta < 0:
                reward += score_delta * 2.0         # lost points

        elif isinstance(action, Pass):
            if action.voluntary:
                reward -= state.variant.voluntary_pass_penalty * 3.0

        # Terminal: computed against the final scored state
        if next_state.game_over:
            if next_player.hand_size() == 0:
                reward += 30.0                      # RL won the round
            else:
                reward -= next_player.hand_size() * 2.0 * ppc  # RL lost

            scores = [p.match_score for p in next_state.players]
            if next_player.match_score == max(scores):
                reward += 20.0
            elif next_player.match_score == min(scores):
                reward -= 10.0

        return float(np.clip(reward, -100.0, 100.0))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def store_experience(self,
                         state: GameState,
                         action: Move,
                         reward: float,
                         next_state: GameState,
                         done: bool) -> None:
        """Encode and buffer an (s, a, r, s', done) transition.

        Also stores next_legal_idx so the bootstrap argmax is masked to
        legal moves only — prevents uninitialised Q-values for cards not
        in hand from polluting Bellman targets.
        """
        self.current_episode_reward += reward
        self.episode_step_count     += 1

        if not done and not next_state.game_over:
            next_legal     = Rules.get_legal_moves(next_state)
            next_legal_idx = [self._move_to_idx(m) for m in next_legal]
        else:
            next_legal_idx = []

        exp = {
            'state':          StateEncoder.encode(state,      state.current_player),
            'action_idx':     self._move_to_idx(action),
            'reward':         reward,
            'next_state':     StateEncoder.encode(next_state, state.current_player),
            'next_legal_idx': next_legal_idx,
            'done':           done,
        }
        self.replay_buffer.add(exp)

    def train_from_replay(self, batch_size: int = 64) -> None:
        """Double DQN update with Prioritized Experience Replay and legal masking."""
        if len(self.replay_buffer) < batch_size:
            return

        samples, indices, is_weights = self.replay_buffer.sample(batch_size)
        td_errors = []

        for exp, iw in zip(samples, is_weights):
            sf    = exp['state']
            r     = exp['reward']
            nsf   = exp['next_state']
            done  = exp['done']

            if done:
                target_q = r
            else:
                legal_idx   = exp['next_legal_idx']
                q_online, _ = self.q_network.forward(nsf, use_target=False)
                if legal_idx:
                    best_a = legal_idx[int(np.argmax([q_online[i] for i in legal_idx]))]
                else:
                    best_a = int(np.argmax(q_online))
                q_target, _ = self.q_network.forward(nsf, use_target=True)
                target_q    = r + self.discount_factor * q_target[best_a]

            td_err = self.q_network.update(
                sf, exp['action_idx'], target_q, self.learning_rate, float(iw)
            )
            td_errors.append(td_err)
            self.total_updates += 1

        self.replay_buffer.update_priorities(indices, td_errors)

        if self.total_updates % TARGET_SYNC_FREQ == 0:
            self.q_network.sync_target()

    def update_from_experience(self,
                               state: GameState,
                               action: Move,
                               reward: float,
                               next_state: GameState,
                               done: bool) -> None:
        """Legacy online update (single transition, no IS weight)."""
        if self.q_network is None:
            return
        pi = state.current_player
        sf  = StateEncoder.encode(state,      pi)
        nsf = StateEncoder.encode(next_state, pi)

        if done:
            target_q = reward
        else:
            q_online, _ = self.q_network.forward(nsf, use_target=False)
            next_legal   = Rules.get_legal_moves(next_state)
            legal_idx    = [self._move_to_idx(m) for m in next_legal]
            if legal_idx:
                best_a = legal_idx[int(np.argmax([q_online[i] for i in legal_idx]))]
            else:
                best_a = int(np.argmax(q_online))
            q_target, _ = self.q_network.forward(nsf, use_target=True)
            target_q    = reward + self.discount_factor * q_target[best_a]

        a_idx = self._move_to_idx(action)
        self.q_network.update(sf, a_idx, target_q, self.learning_rate)
        self.total_updates += 1

        if self.total_updates % TARGET_SYNC_FREQ == 0:
            self.q_network.sync_target()

    def decay_epsilon(self, decay_rate: float = 0.995, min_epsilon: float = 0.05) -> None:
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)

    def end_episode(self) -> None:
        avg = self.current_episode_reward / max(self.episode_step_count, 1)
        self.episode_rewards.append(avg)
        self.current_episode_reward = 0.0
        self.episode_step_count     = 0
        self._consecutive_voluntary_passes = 0
        self._consecutive_rolls = 0
        self._last_seen_turn_number = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_weights(self, filepath: str) -> None:
        if self.q_network is None:
            print("No network to save")
            return
        data = {
            'weights':        self.q_network.weights,
            'biases':         self.q_network.biases,
            'target_weights': self.q_network.target_weights,
            'target_biases':  self.q_network.target_biases,
            'state_dim':      self.q_network.state_dim,
            'action_dim':     self.q_network.action_dim,
            'hidden_dims':    self.q_network.hidden_dims,
            'epsilon':        self.epsilon,
            'total_updates':  self.total_updates,
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved weights → {filepath}  (state_dim={self.q_network.state_dim})")

    def load_weights(self, filepath: str) -> None:
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            saved_dim = data['state_dim']

            if self.q_network is None:
                self.q_network = ImprovedQNetwork(
                    saved_dim,
                    data['action_dim'],
                    data.get('hidden_dims', [256, 128]),
                )

            if self.q_network.state_dim != saved_dim:
                raise ValueError(
                    f"Dimension mismatch: saved={saved_dim}, "
                    f"current encoder={self.q_network.state_dim}. "
                    f"Retrain with train_agent.py."
                )

            self.q_network.weights = data['weights']
            self.q_network.biases  = data['biases']

            # Load target weights if present (v6+); fall back to sync from main
            if 'target_weights' in data:
                self.q_network.target_weights = data['target_weights']
                self.q_network.target_biases  = data['target_biases']
            else:
                self.q_network.sync_target()

            self.epsilon       = data.get('epsilon',       self.epsilon)
            self.total_updates = data.get('total_updates', 0)

            print(f"Loaded weights ← {filepath}")
            print(f"  state_dim={saved_dim}  hidden={self.q_network.hidden_dims}"
                  f"  ε={self.epsilon:.3f}  updates={self.total_updates}")

        except FileNotFoundError:
            print(f"No weights file at {filepath} — starting fresh.")
        except ValueError as e:
            print(f"[RLAgent] {e}")
            print("  → Initialising fresh network.")
            self.q_network = None
        except Exception as e:
            print(f"Error loading weights: {e}")


# ---------------------------------------------------------------------------
# Pre-configured variants
# ---------------------------------------------------------------------------

class RLAgentExplore(RLAgent):
    def __init__(self, name="RL-Explore"):
        super().__init__(name, epsilon=0.3,  learning_rate=5e-4,
                         discount_factor=0.95, use_heuristics=True)

class RLAgentExploit(RLAgent):
    def __init__(self, name="RL-Exploit"):
        super().__init__(name, epsilon=0.05, learning_rate=5e-4,
                         discount_factor=0.95, use_heuristics=True)

class RLAgentPure(RLAgent):
    def __init__(self, name="RL-Pure"):
        super().__init__(name, epsilon=0.15, learning_rate=5e-4,
                         discount_factor=0.95, use_heuristics=False)