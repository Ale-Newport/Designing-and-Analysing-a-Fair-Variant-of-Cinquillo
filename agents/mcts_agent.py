"""
Optimised Monte Carlo Tree Search (MCTS) agent for Cinquillo 2.0.

Information-reveal awareness:
  When GoodDiceEffect.INFO_REVEAL is active for the rolling player, that player
  knows the full hand of the opponent with fewest cards (u).  During MCTS
  rollouts, _fast_card_selection checks dice_state.get_revealed_target() and
  preferentially chooses cards that block u's most urgently playable cards.

  When BadDiceEffect.REVEAL_HAND is active, other players know the current
  player's hand.  The MCTS agent at those opponent positions uses the revealed
  hand to select better blocking moves.

Speed optimisations (unchanged):
  1. Early rollout termination
  2. Cached computations
  3. Reduced state copying
  4. Optimised hot paths
"""
import math
import random
import time
from typing import List, Optional
from dataclasses import dataclass, field

from game.entities import Card, GameState, ScoringMode, Deck
from game.rules import Move, Rules, PlayCard, Pass, RollDice
from agents.base_agents import Agent


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    state: GameState
    parent: Optional['MCTSNode'] = None
    move_to_here: Optional[Move] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0
    untried_moves: List[Move] = field(default_factory=list)
    
    # Cache for performance
    _is_terminal: Optional[bool] = None
    
    def __post_init__(self):
        """Initialise untried moves."""
        if not self.untried_moves:
            self.untried_moves = Rules.get_legal_moves(self.state)
    
    def is_fully_expanded(self) -> bool:
        """Check if all moves have been tried."""
        return len(self.untried_moves) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node (cached)."""
        if self._is_terminal is None:
            self._is_terminal = Rules.is_terminal(self.state)
        return self._is_terminal
    
    def best_child(self, exploration_weight: float = 1.414) -> 'MCTSNode':
        """Select best child using UCT."""
        if not self.children:
            return None
            
        log_parent = math.log(max(1, self.visits))
        best_score = float('-inf')
        best_child = None
        
        for child in self.children:
            if child.visits == 0:
                return child
            
            exploitation = child.total_reward / child.visits
            exploration = exploration_weight * math.sqrt(log_parent / child.visits)
            score = exploitation + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def most_visited_child(self) -> 'MCTSNode':
        """Return child with most visits."""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visits)
    
    def add_child(self, move: Move, state: GameState) -> 'MCTSNode':
        """Add a child node for the given move."""
        child = MCTSNode(
            state=state,
            parent=self,
            move_to_here=move
        )
        self.children.append(child)
        if move in self.untried_moves:
            self.untried_moves.remove(move)
        return child


class MCTSAgent(Agent):
    """
    Optimised Monte Carlo Tree Search agent.

    Information-reveal integration:
      The agent is aware of two dice reveal effects and exploits them during
      card selection inside rollouts:

      1. GoodDiceEffect.INFO_REVEAL — the current player can see the hand of u
         (the opponent with fewest cards).  _fast_card_selection calls
         _find_blocking_moves() to prefer cards that are adjacent (in rank
         index) to cards held by u, thereby blocking u's sequence extensions.

      2. BadDiceEffect.REVEAL_HAND — all opponents know p's hand.  Each opponent
         (when it is their rollout turn) calls _find_blocking_moves() against p's
         revealed hand, so they can block p's most urgent plays.
    """
    
    def __init__(self, 
                 name: str = "MCTS-Improved",
                 num_iterations: int = 300,
                 exploration_weight: float = 1.414,
                 max_rollout_depth: int = 40,
                 time_limit: float = 0.5):
        """
        Args:
            num_iterations:  Max MCTS iterations per move (hard cap).
            max_rollout_depth: Max moves per rollout simulation.
            time_limit: Wall-clock seconds budget per choose_move call.
                        The iteration loop stops as soon as EITHER
                        num_iterations OR time_limit is reached — whichever
                        comes first.  Prevents catastrophically long moves on
                        variants with many rounds or non-advancing dice effects.
        """
        super().__init__(name)
        self.num_iterations = num_iterations
        self.exploration_weight = exploration_weight
        self.max_rollout_depth = max_rollout_depth
        self.time_limit = time_limit

    def _dice_can_advance_turn(self, state: GameState) -> bool:
        """Return whether rolling dice can realistically advance the game this turn."""
        from game.entities import BadDiceEffect, GoodDiceEffect
        variant = state.variant
        return (
            variant.dice_bad_effect == BadDiceEffect.FORCED_PASS
            or variant.dice_good_effect == GoodDiceEffect.WILD
            or variant.dice_good_effect == GoodDiceEffect.DOUBLE_PLAY
        )

    def _root_filter_moves(self, state: GameState, legal_moves: List[Move]) -> List[Move]:
        """Filter root moves to avoid non-progress dice when a card play exists."""
        filtered_moves = [m for m in legal_moves if not (isinstance(m, Pass) and m.voluntary)]
        candidate_moves = filtered_moves if filtered_moves else legal_moves

        has_card_move = any(isinstance(m, PlayCard) for m in candidate_moves)
        if has_card_move and not self._dice_can_advance_turn(state):
            no_dice = [m for m in candidate_moves if not isinstance(m, RollDice)]
            if no_dice:
                return no_dice
        return candidate_moves
    
    def choose_move(self, state: GameState, legal_moves: List[Move]) -> Move:
        """Choose move using MCTS with a wall-clock time budget."""
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        filtered_moves = self._root_filter_moves(state, legal_moves)

        if len(filtered_moves) == 1:
            return filtered_moves[0]
        
        root = MCTSNode(state=state.copy())
        root.untried_moves = filtered_moves
        player_index = state.current_player
        
        deadline = time.time() + self.time_limit
        for _ in range(self.num_iterations):
            # Stop early if wall-clock budget is exhausted
            if time.time() >= deadline:
                break
            self._mcts_iteration(root, state, player_index)
        
        best_child = root.most_visited_child()
        return best_child.move_to_here if best_child else random.choice(filtered_moves)
    
    def _mcts_iteration(self, root: MCTSNode, state: GameState, player_index: int):
        """Single MCTS iteration."""
        node = root
        sim_state = state.copy()
        
        # Selection
        while not node.is_terminal() and node.is_fully_expanded():
            best_child = node.best_child(self.exploration_weight)
            if best_child is None:
                break
            node = best_child
            sim_state = node.state.copy()
        
        # Expansion
        if not node.is_terminal() and not node.is_fully_expanded() and node.untried_moves:
            move = random.choice(node.untried_moves)
            sim_state = Rules.apply_move(sim_state, move)
            node = node.add_child(move, sim_state)
        
        # Simulation
        reward = self._rollout(sim_state, player_index)
        
        # Backpropagation
        self._backpropagate(node, reward)
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate reward up the tree."""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
    
    def _rollout(self, state: GameState, player_index: int) -> float:
        """
        Rollout with early termination and information-reveal awareness.

        Key fix: RollDice moves whose effect does NOT advance the current
        player's turn (INFO_REVEAL and REVEAL_HAND both leave the same player
        active) are excluded from the rollout policy when card plays are
        available.  Without this fix the rollout burns its entire depth budget
        rolling dice on the same player's turn and never makes game progress,
        which is catastrophic for variants like Open Book.
        """
        depth = 0
        
        while depth < self.max_rollout_depth:
            if Rules.is_terminal(state):
                break
            
            if depth > 10 and self._should_terminate_early(state):
                break
            
            legal_moves = Rules.get_legal_moves(state)
            if not legal_moves:
                break
            
            move = self._fast_rollout_policy(state, legal_moves)
            state = Rules.apply_move(state, move)
            depth += 1
        
        return self._evaluate_terminal_state(state, player_index)
    
    def _should_terminate_early(self, state: GameState) -> bool:
        """Check if rollout should terminate early."""
        hand_sizes = [p.hand_size() for p in state.players]
        min_hand = min(hand_sizes)
        max_hand = max(hand_sizes)
        
        if min_hand <= 2 and max_hand >= 8:
            return True
        
        if min_hand == 0:
            return True
        
        return False
    
    def _fast_rollout_policy(self, state: GameState, legal_moves: List[Move]) -> Move:
        """
        Fast rollout policy.  Avoids voluntary passes and uses revealed-hand
        information if available to make informed card selections.

        Bug fix: RollDice is only considered when its effect will actually
        advance the turn (i.e. the bad-effect is FORCED_PASS) OR when there
        are no card moves available.  For INFO_REVEAL and REVEAL_HAND effects,
        rolling dice leaves the same player active — including dice in the
        rollout policy for those variants caused the rollout to spin through
        its entire depth budget rolling dice without ever playing a card.
        """
        card_moves = []
        dice_moves = []
        forced_pass = None
        
        for move in legal_moves:
            if isinstance(move, PlayCard):
                card_moves.append(move)
            elif isinstance(move, RollDice):
                dice_moves.append(move)
            elif isinstance(move, Pass) and not move.voluntary:
                forced_pass = move

        # Determine whether rolling dice on this variant can advance the turn.
        # FORCED_PASS bad-effect advances the turn; INFO_REVEAL / REVEAL_HAND
        # good/bad effects do NOT.  Only include dice in the rollout policy when
        # there is a realistic chance the roll produces turn advancement.
        dice_can_advance = self._dice_can_advance_turn(state)

        if card_moves:
            # Only consider dice when they can meaningfully advance the game
            if dice_moves and dice_can_advance and random.random() < 0.1:
                return random.choice(dice_moves)
            return self._fast_card_selection(state, card_moves)
        
        if dice_moves:
            return random.choice(dice_moves)
        
        if forced_pass:
            return forced_pass
        
        return legal_moves[0]
    
    def _fast_card_selection(self, state: GameState, card_moves: List[PlayCard]) -> PlayCard:
        """
        Fast card selection with simplified heuristics.

        Information-reveal integration:
          If the current player has a revealed opponent hand in dice_state
          (from either INFO_REVEAL good or REVEAL_HAND bad effects), the agent
          uses _find_blocking_moves() to prefer cards that are adjacent in rank
          index to the opponent's held cards in the same suit — preventing them
          from extending their sequences next turn.
        """
        # Priority 1: Play 5s on empty suits (always high value)
        for move in card_moves:
            if move.card.rank == 5 and state.board.is_empty(move.card.suit):
                return move
        
        # Priority 2: Use revealed opponent hand for informed blocking.
        #
        # Case A — INFO_REVEAL good: current player can see the opponent u's hand.
        revealed_target = state.dice_state.get_revealed_target(state.current_player)
        if revealed_target is not None:
            revealed_hand = state.players[revealed_target].hand
            blocking = self._find_blocking_moves(card_moves, revealed_hand)
            if blocking:
                return random.choice(blocking)
        
        # Case B — REVEAL_HAND bad: current player p's hand was revealed, so
        # check if ANY opponent whose turn it is now can see p's hand.
        # (During rollouts, sim_state.current_player iterates through opponents.)
        # We look for any entry in revealed_hands that maps this player to p.
        else:
            for i, p in enumerate(state.players):
                if i != state.current_player:
                    if state.dice_state.get_revealed_target(state.current_player) == i:
                        # This shouldn't happen — just defensive
                        pass
            # Check if the current rollout player can see any opponent's hand
            # via the REVEAL_HAND path (opponents were given view of p's hand)
            cur = state.current_player
            for other_idx, target_idx in state.dice_state.revealed_hands.items():
                if other_idx == cur:
                    # cur can see target_idx's hand
                    known_hand = state.players[target_idx].hand
                    blocking = self._find_blocking_moves(card_moves, known_hand)
                    if blocking:
                        return random.choice(blocking)
                    break
        
        # Priority 3: Extend existing sequences (quick check)
        for move in card_moves:
            if not state.board.is_empty(move.card.suit):
                min_rank, max_rank = state.board.get_min_max(move.card.suit)
                if min_rank and max_rank:
                    if move.card.rank < min_rank or move.card.rank > max_rank:
                        return move
        
        # Priority 4: Play high cards (10, 11, 12 are hard to play)
        high_cards = [m for m in card_moves if m.card.rank >= 10]
        if high_cards:
            return random.choice(high_cards)
        
        # Default: random card
        return random.choice(card_moves)

    def _find_blocking_moves(self, card_moves: List[PlayCard],
                             revealed_hand: List[Card]) -> List[PlayCard]:
        """
        Return a subset of card_moves whose cards are adjacent (in Deck rank
        index) to at least one card in revealed_hand of the same suit.

        Playing such a card blocks the opponent from extending their sequence
        in that suit next turn.

        Args:
            card_moves:    Candidate PlayCard moves to filter.
            revealed_hand: The known cards in the opponent's hand.

        Returns:
            List of blocking PlayCard moves (may be empty).
        """
        blocking = []
        for move in card_moves:
            for opp_card in revealed_hand:
                if opp_card.suit == move.card.suit:
                    our_idx = Deck.RANK_INDEX[move.card.rank]
                    their_idx = Deck.RANK_INDEX[opp_card.rank]
                    if abs(our_idx - their_idx) == 1:
                        blocking.append(move)
                        break  # One match per move is enough
        return blocking
    
    def _evaluate_terminal_state(self, state: GameState, player_index: int) -> float:
        """Evaluate terminal state (optimised)."""
        if not Rules.is_terminal(state):
            return self._fast_non_terminal_eval(state, player_index)
        
        sim_state = state.copy()
        Rules.compute_round_scores(sim_state)
        
        round_scores = [p.round_score for p in sim_state.players]
        player_round_score = round_scores[player_index]
        
        if state.variant.scoring_mode == ScoringMode.WINNER_TAKES_ALL:
            if sim_state.winner == player_index:
                return 1.0
            
            max_score = max(round_scores)
            min_score = min(round_scores)
            
            if max_score == min_score:
                return 0.0
            
            return min(0.4, (player_round_score - min_score) / (max_score - min_score))
        
        # DOUBLE_PENALTY mode
        max_score = max(round_scores)
        min_score = min(round_scores)
        
        if max_score == min_score:
            return 0.5
        
        return (player_round_score - min_score) / (max_score - min_score)
    
    def _fast_non_terminal_eval(self, state: GameState, player_index: int) -> float:
        """Fast heuristic evaluation with minimal computation."""
        player = state.players[player_index]
        player_hand_size = player.hand_size()
        
        if player_hand_size == 0:
            return 1.0
        
        opponent_sizes = [p.hand_size() for p in state.players if p.index != player_index]
        min_opp = min(opponent_sizes)
        
        if min_opp == 0:
            return 0.0
        
        avg_opp = sum(opponent_sizes) / len(opponent_sizes)
        hand_ratio = player_hand_size / avg_opp
        score = max(0.0, 1.0 - (hand_ratio * 0.5))
        
        if player.round_score < 0:
            score *= 0.9
        
        return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Optimised variants
# ---------------------------------------------------------------------------

class MCTSAgentSuperFast(MCTSAgent):
    """Super fast MCTS with minimal iterations."""
    
    def __init__(self, name: str = "MCTS-SuperFast"):
        super().__init__(
            name=name,
            num_iterations=100,
            exploration_weight=1.414,
            max_rollout_depth=50,
            time_limit=0.1,
        )


class MCTSAgentFast(MCTSAgent):
    """Fast MCTS with fewer iterations."""
    
    def __init__(self, name: str = "MCTS-Fast"):
        super().__init__(
            name=name,
            num_iterations=500,
            exploration_weight=1.414,
            max_rollout_depth=50,
            time_limit=0.2,
        )


class MCTSAgentStandard(MCTSAgent):
    """Standard MCTS configuration."""
    
    def __init__(self, name: str = "MCTS"):
        super().__init__(
            name=name,
            num_iterations=1000,
            exploration_weight=1.414,
            max_rollout_depth=100,
            time_limit=0.8,
        )


class MCTSAgentDeep(MCTSAgent):
    """Stronger MCTS with more iterations."""
    
    def __init__(self, name: str = "MCTS-Deep"):
        super().__init__(
            name=name,
            num_iterations=2000,
            exploration_weight=1.414,
            max_rollout_depth=150,
            time_limit=2.0,
        )