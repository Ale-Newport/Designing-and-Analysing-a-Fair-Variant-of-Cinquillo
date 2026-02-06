"""
Optimized Monte Carlo Tree Search (MCTS) agent for Cinquillo 2.0.
Speed optimizations:
1. Early rollout termination
2. Cached computations
3. Reduced state copying
4. Optimized hot paths
"""
import math
import random
from typing import List, Optional
from dataclasses import dataclass, field

from game.entities import Card, GameState, ScoringMode
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
        """Initialize untried moves."""
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
        """
        Select best child using UCT (Upper Confidence Bound for Trees).
        Optimized with inline calculation.
        """
        if not self.children:
            return None
            
        log_parent = math.log(max(1, self.visits))  # Avoid log(0)
        best_score = float('-inf')
        best_child = None
        
        for child in self.children:
            if child.visits == 0:
                return child  # Immediate return for unvisited
            
            # Inline UCT calculation (faster than function call)
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
        # Safe removal
        if move in self.untried_moves:
            self.untried_moves.remove(move)
        return child


class MCTSAgent(Agent):
    """
    Optimized Monte Carlo Tree Search agent.
    Speed improvements without sacrificing quality.
    """
    
    def __init__(self, 
                 name: str = "MCTS-Improved",
                 num_iterations: int = 1000,
                 exploration_weight: float = 1.414,
                 max_rollout_depth: int = 100):
        """
        Initialize optimized MCTS agent.
        
        Args:
            name: Agent name
            num_iterations: Number of MCTS iterations per move
            exploration_weight: UCT exploration parameter
            max_rollout_depth: Maximum depth for rollout simulations
        """
        super().__init__(name)
        self.num_iterations = num_iterations
        self.exploration_weight = exploration_weight
        self.max_rollout_depth = max_rollout_depth
    
    def choose_move(self, state: GameState, legal_moves: List[Move]) -> Move:
        """Choose move using MCTS."""
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # Filter out voluntary passes
        filtered_moves = [
            m for m in legal_moves 
            if not (isinstance(m, Pass) and m.voluntary)
        ]
        
        if not filtered_moves:
            filtered_moves = legal_moves
        
        if len(filtered_moves) == 1:
            return filtered_moves[0]
        
        root = MCTSNode(state=state.copy())
        root.untried_moves = filtered_moves
        player_index = state.current_player
        
        # Run MCTS iterations
        for _ in range(self.num_iterations):
            self._mcts_iteration(root, state, player_index)
        
        # Return move with most visits
        best_child = root.most_visited_child()
        return best_child.move_to_here if best_child else random.choice(filtered_moves)
    
    def _mcts_iteration(self, root: MCTSNode, state: GameState, player_index: int):
        """Single MCTS iteration."""
        node = root
        sim_state = state.copy()
        
        # Selection: traverse tree using UCT
        while not node.is_terminal() and node.is_fully_expanded():
            best_child = node.best_child(self.exploration_weight)
            if best_child is None:
                break
            node = best_child
            sim_state = node.state.copy()
        
        # Expansion: add a new child
        if not node.is_terminal() and not node.is_fully_expanded() and node.untried_moves:
            move = random.choice(node.untried_moves)
            sim_state = Rules.apply_move(sim_state, move)
            node = node.add_child(move, sim_state)
        
        # Simulation: rollout to terminal state
        reward = self._rollout(sim_state, player_index)
        
        # Backpropagation: update statistics
        self._backpropagate(node, reward)
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate reward up the tree."""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
    
    def _rollout(self, state: GameState, player_index: int) -> float:
        """
        Optimized rollout with early termination.
        
        Key optimizations:
        1. Early termination when outcome is clear
        2. Lightweight state tracking
        3. Fast move selection
        """
        depth = 0
        
        while depth < self.max_rollout_depth:
            # Early termination check
            if Rules.is_terminal(state):
                break
            
            # Check if outcome is clear (someone nearly done)
            if depth > 10 and self._should_terminate_early(state):
                break
            
            legal_moves = Rules.get_legal_moves(state)
            if not legal_moves:
                break
            
            # Fast rollout policy
            move = self._fast_rollout_policy(state, legal_moves)
            state = Rules.apply_move(state, move)
            depth += 1
        
        # Evaluate final state
        return self._evaluate_terminal_state(state, player_index)
    
    def _should_terminate_early(self, state: GameState) -> bool:
        """
        Check if rollout should terminate early.
        Returns True if outcome is highly predictable.
        """
        hand_sizes = [p.hand_size() for p in state.players]
        min_hand = min(hand_sizes)
        max_hand = max(hand_sizes)
        
        # If someone has ≤2 cards and others have ≥8, outcome is very likely
        if min_hand <= 2 and max_hand >= 8:
            return True
        
        # If someone has 0 cards, they've won
        if min_hand == 0:
            return True
        
        return False
    
    def _fast_rollout_policy(self, state: GameState, legal_moves: List[Move]) -> Move:
        """
        Fast rollout policy with minimal computation.
        
        Key optimizations:
        1. Avoid voluntary passes
        2. Simple card selection without heavy computation
        3. Minimal move filtering
        """
        # Quick categorization
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
        
        # Play cards if available (90% of the time)
        if card_moves:
            if dice_moves and random.random() < 0.1:
                return random.choice(dice_moves)
            return self._fast_card_selection(state, card_moves)
        
        # Roll dice if available
        if dice_moves:
            return random.choice(dice_moves)
        
        # Forced pass
        if forced_pass:
            return forced_pass
        
        return legal_moves[0]
    
    def _fast_card_selection(self, state: GameState, card_moves: List[PlayCard]) -> PlayCard:
        """
        Fast card selection with simplified heuristics.
        
        Optimizations:
        1. Quick priority checks without full scoring
        2. Early returns for obvious choices
        3. Minimal state inspection
        """
        # Priority 1: Play 5s on empty suits (quick check)
        for move in card_moves:
            if move.card.rank == 5 and state.board.is_empty(move.card.suit):
                return move
        
        # Priority 2: Extend sequences (simple check)
        for move in card_moves:
            if not state.board.is_empty(move.card.suit):
                min_rank, max_rank = state.board.get_min_max(move.card.suit)
                if min_rank and max_rank:
                    # Prefer edges
                    if move.card.rank < min_rank or move.card.rank > max_rank:
                        return move
        
        # Priority 3: Play high cards
        high_cards = [m for m in card_moves if m.card.rank >= 10]
        if high_cards:
            return random.choice(high_cards)
        
        # Default: random card
        return random.choice(card_moves)
    
    def _evaluate_terminal_state(self, state: GameState, player_index: int) -> float:
        """
        Evaluate terminal state (optimized).
        """
        if not Rules.is_terminal(state):
            return self._fast_non_terminal_eval(state, player_index)
        
        # Simulate scoring
        sim_state = state.copy()
        Rules.compute_round_scores(sim_state)
        
        round_scores = [p.round_score for p in sim_state.players]
        player_round_score = round_scores[player_index]
        
        # Fast evaluation based on scoring mode
        if state.variant.scoring_mode == ScoringMode.WINNER_TAKES_ALL:
            if sim_state.winner == player_index:
                return 1.0
            
            # Quick normalization
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
        """
        Fast heuristic evaluation with minimal computation.
        """
        player = state.players[player_index]
        player_hand_size = player.hand_size()
        
        if player_hand_size == 0:
            return 1.0
        
        # Quick opponent assessment
        opponent_sizes = [p.hand_size() for p in state.players if p.index != player_index]
        min_opp = min(opponent_sizes)
        
        if min_opp == 0:
            return 0.0
        
        # Simplified evaluation (fewer components for speed)
        avg_opp = sum(opponent_sizes) / len(opponent_sizes)
        hand_ratio = player_hand_size / avg_opp
        
        # Single metric: relative hand size
        score = max(0.0, 1.0 - (hand_ratio * 0.5))
        
        # Quick penalty check
        if player.round_score < 0:
            score *= 0.9  # 10% penalty for negative round score
        
        return max(0.0, min(1.0, score))


# Optimized variants
class MCTSAgentSuperFast(MCTSAgent):
    """Super fast MCTS with minimal iterations."""
    
    def __init__(self, name: str = "MCTS-SuperFast"):
        super().__init__(
            name=name,
            num_iterations=100,
            exploration_weight=1.414,
            max_rollout_depth=50,
        )


class MCTSAgentFast(MCTSAgent):
    """Fast MCTS with fewer iterations."""
    
    def __init__(self, name: str = "MCTS-Fast"):
        super().__init__(
            name=name,
            num_iterations=500,
            exploration_weight=1.414,
            max_rollout_depth=50,
        )


class MCTSAgentStandard(MCTSAgent):
    """Standard MCTS configuration."""
    
    def __init__(self, name: str = "MCTS"):
        super().__init__(
            name=name,
            num_iterations=1000,
            exploration_weight=1.414,
            max_rollout_depth=100,
        )


class MCTSAgentDeep(MCTSAgent):
    """Stronger MCTS with more iterations."""
    
    def __init__(self, name: str = "MCTS-Deep"):
        super().__init__(
            name=name,
            num_iterations=2000,
            exploration_weight=1.414,
            max_rollout_depth=150,
        )