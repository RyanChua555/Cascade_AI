from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from referee.game import GamePhase
from referee.game.actions import CascadeAction, EatAction, MoveAction, PlaceAction
from referee.game.board import Board
from referee.game.coord import CARDINAL_DIRECTIONS, Coord
from referee.game.player import PlayerColor
from referee.game import Direction

from .program import GameState
from referee.game import Action

MAX_DIST = 14          # max Manhattan distance on 8×8 board
MAX_ROLLOUT_DEPTH = 20 # simulation depth limit
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
    
 
    def rollout_score(self, state: GameState, agent_color: PlayerColor) -> float:
        """
        Static board evaluation from agent_color's perspective. Returns [0, 1].
        Used both as alpha-beta leaf evaluation and MCTS rollout terminal score.

        Features:
          1. token_score         — raw material ratio
          2. tall_score          — fraction of tokens in h≥3 stacks (combat-ready)
          3. consolidation_score — average stack height vs enemy (fewer taller = better)
          4. eat_threat_score    — immediate captures minus immediate vulnerabilities
          5. attack_dist_score   — proximity of our best attacker to any enemy
          draw_penalty           — penalise nearing turn limit when materially ahead
        """
        enemy_color = agent_color.opponent
        board  = state._board
        bstate = board._state

        if state.game_over:
            winner = board.winner_color
            if winner == agent_color:   return 1.0
            elif winner == enemy_color: return 0.0
            return 0.45  # draw

        agent_stacks: list[tuple[Coord, int]] = []
        enemy_stacks: list[tuple[Coord, int]] = []
        for coord, cell in bstate.items():
            if cell.color == agent_color:
                agent_stacks.append((coord, cell.height))
            elif cell.color == enemy_color:
                enemy_stacks.append((coord, cell.height))

        agent_tokens = sum(h for _, h in agent_stacks)
        enemy_tokens = sum(h for _, h in enemy_stacks)
        total_tokens = agent_tokens + enemy_tokens
        if total_tokens == 0:
            return 0.5

        # 1. Token ratio
        token_score = agent_tokens / total_tokens

        # 2. Tall-stack concentration (h≥3 can eat most things and cascade far)
        agent_tall = sum(h for _, h in agent_stacks if h >= 3)
        enemy_tall = sum(h for _, h in enemy_stacks if h >= 3)
        tall_total = agent_tall + enemy_tall
        tall_score = (agent_tall / tall_total) if tall_total > 0 else 0.5

        # 3. Consolidation: higher average stack height = more concentrated power
        agent_avg_h = agent_tokens / len(agent_stacks) if agent_stacks else 0.0
        enemy_avg_h = enemy_tokens / len(enemy_stacks) if enemy_stacks else 0.0
        max_avg = max(agent_avg_h, enemy_avg_h, 1.0)
        consolidation_score = (agent_avg_h - enemy_avg_h) / (2.0 * max_avg) + 0.5

        # 4. Immediate eat threats vs vulnerabilities
        eat_threats    = 0
        vulnerabilities = 0
        for (ac, ah) in agent_stacks:
            for d in CARDINAL_DIRECTIONS:
                nr, nc = ac.r + d.r, ac.c + d.c
                if 0 <= nr < BOARD_N and 0 <= nc < BOARD_N:
                    nbr = bstate[Coord(nr, nc)]
                    if nbr.color == enemy_color and ah >= nbr.height:
                        eat_threats += nbr.height
        for (ec, eh) in enemy_stacks:
            for d in CARDINAL_DIRECTIONS:
                nr, nc = ec.r + d.r, ec.c + d.c
                if 0 <= nr < BOARD_N and 0 <= nc < BOARD_N:
                    nbr = bstate[Coord(nr, nc)]
                    if nbr.color == agent_color and eh >= nbr.height:
                        vulnerabilities += nbr.height
        max_threat = max(agent_tokens, enemy_tokens, 1)
        eat_threat_score = 0.5 + (eat_threats - vulnerabilities) / (2.0 * max_threat)
        eat_threat_score = max(0.0, min(1.0, eat_threat_score))

        # 5. Attack distance: closest h≥2 agent stack to any enemy
        min_attack_dist = float(MAX_DIST)
        for (ac, ah) in agent_stacks:
            if ah < 2:
                continue
            for (ec, _) in enemy_stacks:
                d = abs(ac.r - ec.r) + abs(ac.c - ec.c)
                if d < min_attack_dist:
                    min_attack_dist = d
        if min_attack_dist == float(MAX_DIST):
            for (ac, _) in agent_stacks:
                for (ec, _) in enemy_stacks:
                    d = abs(ac.r - ec.r) + abs(ac.c - ec.c)
                    if d < min_attack_dist:
                        min_attack_dist = d
        attack_dist_score = 1.0 - (min_attack_dist / MAX_DIST)

        # Draw penalty
        turns_left   = max(0, 300 - board.play_phase_turn_count)
        draw_risk    = 1.0 - (turns_left / 300.0)
        lead         = (agent_tokens - enemy_tokens) / total_tokens
        draw_penalty = draw_risk * max(0.0, lead) * 0.15

        # Dynamic weights
        progress = min(board.play_phase_turn_count / 200.0, 1.0)
        w_token = 0.25 + 0.30 * progress
        w_tall  = 0.20 - 0.05 * progress
        w_cons  = 0.10 + 0.10 * progress
        w_eat   = 0.20
        w_dist  = 1.0 - w_token - w_tall - w_cons - w_eat

        score = (
            w_token * token_score
            + w_tall  * tall_score
            + w_cons  * consolidation_score
            + w_eat   * eat_threat_score
            + w_dist  * attack_dist_score
            - draw_penalty
        )
        return max(0.0, min(1.0, score))



    def simulate(self, state: GameState, agent_color: PlayerColor) -> float:
        """Rollout from state for up to MAX_ROLLOUT_DEPTH plies, then evaluate."""
        sim_state = state.clone()

        for _ in range(MAX_ROLLOUT_DEPTH):
            if sim_state.game_over:
                break
            try:
                action = self.rollout_policy(sim_state, agent_color)
            except ValueError:
                break
            sim_state.apply_action(action)

        return self.rollout_score(sim_state, agent_color)



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