"""
Optimised Monte Carlo Tree Search (MCTS) agent for Cinquillo 2.0.
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
    """Node in the MCTS tree"""
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
        if not self.untried_moves:
            self.untried_moves = Rules.get_legal_moves(self.state)
    
    def is_fully_expanded(self) -> bool:
        """check if all moves have been tried"""
        return len(self.untried_moves) == 0
    
    def is_terminal(self) -> bool:
        """check if this is a terminal node (cached)"""
        if self._is_terminal is None:
            self._is_terminal = Rules.is_terminal(self.state)
        return self._is_terminal
    
    def best_child(self, exploration_weight: float = 1.414) -> 'MCTSNode':
        """select best child using UCT"""
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
        """return child with most visits"""
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visits)
    
    def add_child(self, move: Move, state: GameState) -> 'MCTSNode':
        """add a child node for the given move"""
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
    Optimised Monte Carlo Tree Search agent
    """
    
    def __init__(self, 
                 name: str = "MCTS-Improved",
                 num_iterations: int = 300,
                 exploration_weight: float = 1.414,
                 max_rollout_depth: int = 40,
                 time_limit: float = 0.5):
        """
        Args:
            num_iterations:  Max MCTS iterations per move
            max_rollout_depth: Max moves per rollout simulation
            time_limit: Wall-clock seconds budget per choose_move call
        """
        super().__init__(name)
        self.num_iterations = num_iterations
        self.exploration_weight = exploration_weight
        self.max_rollout_depth = max_rollout_depth
        self.time_limit = time_limit

    def _dice_can_advance_turn(self, state: GameState) -> bool:
        """return whether rolling dice can realistically advance the game this turn"""
        from game.entities import BadDiceEffect, GoodDiceEffect
        variant = state.variant
        return (
            variant.dice_bad_effect == BadDiceEffect.FORCED_PASS
            or variant.dice_good_effect == GoodDiceEffect.WILD
            or variant.dice_good_effect == GoodDiceEffect.DOUBLE_PLAY
        )

    def _root_filter_moves(self, state: GameState, legal_moves: List[Move]) -> List[Move]:
        """filter root moves to avoid non-progress dice when a card play exists"""
        filtered_moves = [m for m in legal_moves if not (isinstance(m, Pass) and m.voluntary)]
        candidate_moves = filtered_moves if filtered_moves else legal_moves

        has_card_move = any(isinstance(m, PlayCard) for m in candidate_moves)
        if has_card_move and not self._dice_can_advance_turn(state):
            no_dice = [m for m in candidate_moves if not isinstance(m, RollDice)]
            if no_dice:
                return no_dice
        return candidate_moves
    
    def choose_move(self, state: GameState, legal_moves: List[Move]) -> Move:
        """choose move using MCTS with a wall-clock time budget"""
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
        """single MCTS iteration"""
        node = root
        sim_state = state.copy()
        
        # selection
        while not node.is_terminal() and node.is_fully_expanded():
            best_child = node.best_child(self.exploration_weight)
            if best_child is None:
                break
            node = best_child
            sim_state = node.state.copy()
        
        # expansion
        if not node.is_terminal() and not node.is_fully_expanded() and node.untried_moves:
            move = random.choice(node.untried_moves)
            sim_state = Rules.apply_move(sim_state, move)
            node = node.add_child(move, sim_state)
        
        # simulation
        reward = self._rollout(sim_state, player_index)
        
        # backpropagation
        self._backpropagate(node, reward)
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """backpropagate reward up the tree"""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
    
    def _rollout(self, state: GameState, player_index: int) -> float:
        """
        rollout with early termination and information-reveal awareness
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
        """check if rollout should terminate early"""
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
        fast rollout policy avoids voluntary passes and uses revealed-hand information if available to make informed card selections
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

        dice_can_advance = self._dice_can_advance_turn(state)

        if card_moves:
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
        fast card selection with simplified heuristics
        """
        
        for move in card_moves:
            if move.card.rank == 5 and state.board.is_empty(move.card.suit):
                return move
        
        revealed_target = state.dice_state.get_revealed_target(state.current_player)
        if revealed_target is not None:
            revealed_hand = state.players[revealed_target].hand
            blocking = self._find_blocking_moves(card_moves, revealed_hand)
            if blocking:
                return random.choice(blocking)
        
        else:
            for i, p in enumerate(state.players):
                if i != state.current_player:
                    if state.dice_state.get_revealed_target(state.current_player) == i:
                        pass

            cur = state.current_player
            for other_idx, target_idx in state.dice_state.revealed_hands.items():
                if other_idx == cur:
                    known_hand = state.players[target_idx].hand
                    blocking = self._find_blocking_moves(card_moves, known_hand)
                    if blocking:
                        return random.choice(blocking)
                    break
        
        for move in card_moves:
            if not state.board.is_empty(move.card.suit):
                min_rank, max_rank = state.board.get_min_max(move.card.suit)
                if min_rank and max_rank:
                    if move.card.rank < min_rank or move.card.rank > max_rank:
                        return move
        
        high_cards = [m for m in card_moves if m.card.rank >= 10]
        if high_cards:
            return random.choice(high_cards)
        

        return random.choice(card_moves)

    def _find_blocking_moves(self, card_moves: List[PlayCard], revealed_hand: List[Card]) -> List[PlayCard]:
        """
        args:
            card_moves: candidate PlayCard moves to filter
            revealed_hand: the known cards in the opponent's hand

        returns:
            List of blocking PlayCard moves
        """
        blocking = []
        for move in card_moves:
            for opp_card in revealed_hand:
                if opp_card.suit == move.card.suit:
                    our_idx = Deck.RANK_INDEX[move.card.rank]
                    their_idx = Deck.RANK_INDEX[opp_card.rank]
                    if abs(our_idx - their_idx) == 1:
                        blocking.append(move)
                        break
        return blocking
    
    def _evaluate_terminal_state(self, state: GameState, player_index: int) -> float:
        """evaluate terminal state"""
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
        
        max_score = max(round_scores)
        min_score = min(round_scores)
        
        if max_score == min_score:
            return 0.5
        
        return (player_round_score - min_score) / (max_score - min_score)
    
    def _fast_non_terminal_eval(self, state: GameState, player_index: int) -> float:
        """fast heuristic evaluation with minimal computation"""
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
    """super fast MCTS with minimal iterations"""
    
    def __init__(self, name: str = "MCTS-SuperFast"):
        super().__init__(
            name=name,
            num_iterations=100,
            exploration_weight=1.414,
            max_rollout_depth=50,
            time_limit=0.1,
        )


class MCTSAgentFast(MCTSAgent):
    """fast MCTS with fewer iterations"""
    
    def __init__(self, name: str = "MCTS-Fast"):
        super().__init__(
            name=name,
            num_iterations=500,
            exploration_weight=1.414,
            max_rollout_depth=50,
            time_limit=0.2,
        )


class MCTSAgentStandard(MCTSAgent):
    """standard MCTS configuration"""
    
    def __init__(self, name: str = "MCTS"):
        super().__init__(
            name=name,
            num_iterations=1000,
            exploration_weight=1.414,
            max_rollout_depth=100,
            time_limit=0.8,
        )


class MCTSAgentDeep(MCTSAgent):
    """deep MCTS with more iterations"""
    
    def __init__(self, name: str = "MCTS-Deep"):
        super().__init__(
            name=name,
            num_iterations=2000,
            exploration_weight=1.414,
            max_rollout_depth=150,
            time_limit=2.0,
        )