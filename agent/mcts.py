from __future__ import annotations

import math
from dataclasses import dataclass, field
from referee.game import GamePhase
from referee.game.actions import CascadeAction, EatAction, MoveAction, PlaceAction
from referee.game.board import Board
from referee.game.coord import CARDINAL_DIRECTIONS, Coord
from referee.game.player import PlayerColor
from referee.game import Direction

from .program import GameState
from referee.game import Action

MAX_DIST = 14 # max orthogonal distance on an 8x8 board

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

    def uct_score(self, exploration_constant: float = math.sqrt(2.0)) -> float:
        if self.visits == 0:
            return float("inf")

        parent_visits = self.parent.visits if self.parent else self.visits
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    def best_child(self, exploration_constant: float = math.sqrt(2.0)) -> MCTSNode | None:
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
        import random

        board = state._board
        color = board.turn_color

        legal_actions = state.get_legal_actions()

        if not legal_actions:
            raise ValueError("No legal actions available")

        def action_score(action: Action) -> float:
            next_state = state.clone()
            next_state.apply_action(action)
            return self.rollout_score(next_state, agent_color)

        # Small epsilon-greedy random chance to avoid deterministic rollouts
        if random.random() < 0.1:
            return random.choice(legal_actions)

        return max(legal_actions, key=action_score)
    

    def rollout_score(self, state: GameState, agent_color: PlayerColor) -> float:
        enemy_color = agent_color.opponent
        board = state._board
        agent_stack_height = self.total_stack_height_count(board, agent_color)
        enemy_stack_height = self.total_stack_height_count(board, enemy_color)

        if state.game_over:
            winner = board.winner_color
            if winner == agent_color:
                return 1.0
            elif winner == enemy_color:
                return 0.0
            return 0.5

        # --- Attack distance: how close are we to eating an enemy ---
        min_attack_dist = MAX_DIST
        for coord, cell in board._state.items():
            if cell.color != agent_color:
                continue
            for enemy_coord, enemy_cell in board._state.items():
                if enemy_cell.color != enemy_color:
                    continue
                min_orth, max_orth = self.orthogonal(coord, enemy_coord)
                dist = min_orth + max(max_orth - cell.height, 0)
                if dist < min_attack_dist:
                    min_attack_dist = dist

        # --- Threat distance: how close is an enemy to eating us ---
        min_threat_dist = MAX_DIST
        for coord, cell in board._state.items():
            if cell.color != enemy_color:
                continue
            for friendly_coord, friendly_cell in board._state.items():
                if friendly_cell.color != agent_color:
                    continue
                # Only a real threat if enemy is tall enough to eat us
                if cell.height < friendly_cell.height:
                    continue
                min_orth, max_orth = self.orthogonal(coord, friendly_coord)
                dist = min_orth + max(max_orth - cell.height, 0)
                if dist < min_threat_dist:
                    min_threat_dist = dist

        # --- Scores ---
        token_score = agent_stack_height / (agent_stack_height + enemy_stack_height)
        dist_score = 1.0 - (min_attack_dist / MAX_DIST)
        threat_score = min_threat_dist / MAX_DIST

        # --- Dynamic weighting based on game phase ---
        progress = min(board.play_phase_turn_count / 300, 1.0)
        w_token = 0.4 + 0.4 * progress
        w_dist = (1.0 - w_token) * 0.6
        w_threat = (1.0 - w_token) * 0.4

        return w_token * token_score + w_dist * dist_score + w_threat * threat_score

    def total_stack_height_count(self, board: Board, color: PlayerColor) -> int:
        return sum(
            cell.height for cell in board._state.values()
            if cell.color == color
        )

    '''Helper function for calculating orthogonal distance between two coordinates. Returns a tuple of (min_orth, max_orth)'''
    def orthogonal(self, a: Coord, b: Coord) -> tuple[int, int]:
        dx = abs(a.c - b.c)
        dy = abs(a.r - b.r)
        return (dx, dy) if dx <= dy else (dy, dx)
    
    def simulate(self, state: GameState) -> float:
        raise NotImplementedError("Not implemented yet")

    def backpropagate(self, node: MCTSNode | None, reward: float) -> None:
        while node is not None:
            node.update(reward)
            node = node.parent

    def best_action(self) -> Action | None:
        if not self.root.children:
            return None
        return max(self.root.children, key=lambda child: child.visits).action