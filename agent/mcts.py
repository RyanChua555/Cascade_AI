from __future__ import annotations


import math
import random
from . import constants # type: ignore
from dataclasses import dataclass, field
from referee.game import GamePhase
from referee.game.actions import CascadeAction, EatAction, MoveAction, PlaceAction
from referee.game.board import Board
from referee.game.coord import CARDINAL_DIRECTIONS, Coord
from referee.game.player import PlayerColor
from referee.game import Direction
from .program import evaluate
from .program import GameState
from referee.game import Action


MAX_DIST = 14          # max Manhattan distance on 8×8 board

BOARD_N = 8


@dataclass
class MCTSNode:

    state: GameState
    parent: MCTSNode | None = None
    action: Action | None = None
    visits: int = 0
    value: float = 0.0
    children: list[MCTSNode] = field(default_factory=list)
    untried_actions: list[Action] = field(default_factory=list)

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def add_child(self, action: Action, child_state: GameState) -> MCTSNode:
        child = MCTSNode(state=child_state, parent=self, action=action)
        self.children.append(child)
        return child

    def update(self, reward: float) -> None:
        self.visits += 1
        self.value += reward

    def uct_score(self, exploration_constant: float = constants.MCTS_EXPLORATION_CONSTANT) -> float:
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else self.visits
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    def best_child(self, exploration_constant: float = constants.MCTS_EXPLORATION_CONSTANT) -> MCTSNode | None:
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.uct_score(exploration_constant))


class MCTSTree:

    def __init__(self, root_state: GameState):
        self.root = MCTSNode(state=root_state)

    def clone_root_state(self) -> GameState:
        return self.root.state.clone()

    def selection(self, node: MCTSNode) -> MCTSNode:
        while node.is_fully_expanded and not node.state.game_over:
            best = node.best_child()
            if best is None:
                break
            node = best
        return node

    def expand(self, node: MCTSNode, action: Action, next_state: GameState) -> MCTSNode:
        return node.add_child(action, next_state)
    
    def rollout_policy(self, state: GameState, agent_color: PlayerColor) -> Action:
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            raise ValueError("No legal actions available")
 
        board  = state._board
        bstate = board._state
        color  = board.turn_color
 
        if state.phase == GamePhase.PLACEMENT:
            return self.placement_policy(state, color)
 
        best_eat: EatAction | None = None
        best_eat_val = -1
        for a in legal_actions:
            if isinstance(a, EatAction):
                dest = Coord(a.coord.r + a.direction.r, a.coord.c + a.direction.c)
                val = bstate[dest].height
                if val > best_eat_val:
                    best_eat_val = val
                    best_eat = a
        if best_eat is not None:
            return best_eat
        
        return random.choice(legal_actions)
 
    def placement_policy(self, state: GameState, agent_color: PlayerColor) -> PlaceAction:
        legal_places = state.get_legal_actions()
        legal_places = [a for a in legal_places if isinstance(a, PlaceAction)]

        def placement_score(action: PlaceAction) -> float:
            coord = action.coord
            score = 0.0
            score -= abs(3.5 - coord.r) * 0.05  # prefer center rows
            score -= abs(3.5 - coord.c) * 0.05  # prefer center columns
            return score + random.random() * 0.1  # small random tie-breaker
 
        return max(legal_places, key=placement_score)
    



    def simulate(self, state: GameState, agent_color: PlayerColor) -> float:
        """Rollout from state for up to MAX_ROLLOUT_DEPTH plies, then evaluate."""
        sim_state = state.clone()

        for _ in range(int(constants.MAX_ROLLOUT_DEPTH)):
            if sim_state.game_over:
                break
            try:
                action = self.rollout_policy(sim_state, agent_color)
            except ValueError:
                break
            sim_state.apply_action(action)

        return evaluate(sim_state._board, agent_color)



    def total_stack_height_count(self, board: Board, color: PlayerColor) -> int:
        return sum(cell.height for cell in board._state.values() if cell.color == color)
    

    def backpropagate(self, node: MCTSNode | None, reward: float, agent_color: PlayerColor) -> None:
        while node is not None:
            if node.state.turn_color == agent_color:
                node.update(1.0 - reward)       
            else:
                node.update(reward)  
            node = node.parent


    def best_action(self) -> Action | None:
        if not self.root.children:
            return None
        return max(self.root.children, key=lambda child: child.visits).action