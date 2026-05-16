from __future__ import annotations

import math
from dataclasses import dataclass, field
import random
from referee.game import GamePhase
from referee.game.actions import CascadeAction, EatAction, MoveAction, PlaceAction
from referee.game.board import Board
from referee.game.coord import CARDINAL_DIRECTIONS, Coord
from referee.game.player import PlayerColor
from referee.game import Direction

from .program import GameState
from referee.game import Action

MAX_DIST = 14 # max orthogonal distance on an 8x8 board

MAX_ROLLOUT_DEPTH = 30

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
        """
        Select an action for a simulation step without cloning the board for
        every candidate.  Priority order:
          1. Immediate EatActions (free captures — always worth taking).
          2. CascadeActions that push an enemy off the board or pile tokens
             (cheap approximation: prefer cascades aimed at enemy stacks).
          3. Everything else, scored with a lightweight static heuristic and a
             small random tie-breaker so rollouts aren't deterministic.
        """
        legal_actions = state.get_legal_actions()

        if not legal_actions:
            raise ValueError("No legal actions available")

        return random.choice(legal_actions) 
    
    def rollout_score(self, state: GameState, agent_color: PlayerColor) -> float:
        """
        Static evaluation of a board position from agent_color's perspective.
        Returns a value in [0, 1] where 1 = certain win, 0 = certain loss.
 
        Five features, all normalised to [0, 1] before weighting:
 
        1. token_score       raw token count ratio.  Fundamental material
                              balance; weighted more heavily late game.
 
        2. tall_score        ratio of "tall" tokens (in stacks h≥3) to total.
                              Tall stacks are the primary offensive weapon: they
                              can EAT anything of equal or lesser height and
                              CASCADE further.  Dispersed h=1 tokens (the main
                              failure mode we observed) score near 0 here.
 
        3. eat_threat_score  immediate EAT opportunities minus immediate
                              vulnerabilities.  Directly rewards positions where
                              we can capture on the next move.
 
        4. attack_dist_score how close our tallest stack is to any enemy.
                              Encourages convergence rather than aimless drift.
 
        5. draw_penalty      penalises positions close to the turn limit or
                              with low token counts on both sides (draw-prone
                              endgames).  Draws score 0.5 which is neutral, but
                              we want to actively avoid them when winning.
        """
        enemy_color = agent_color.opponent
        board = state._board
        bstate = board._state
 
        # Terminal positions
        if state.game_over:
            winner = board.winner_color
            if winner == agent_color:
                return 1.0
            elif winner == enemy_color:
                return 0.0
            return 0.45  # draw is slightly bad — prefer decisive wins
 
        # --- Gather per-color stack lists once ---
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
        token_score = agent_tokens / total_tokens  # [0, 1]
 
        # 2. Tall-stack concentration
        # Tokens sitting in stacks of height >= 3 are "combat-ready".
        # h=1 tokens scattered after a cascade are nearly useless alone.
        agent_tall = sum(h for _, h in agent_stacks if h >= 3)
        enemy_tall = sum(h for _, h in enemy_stacks if h >= 3)
        tall_total = agent_tall + enemy_tall
        tall_score = (agent_tall / tall_total) if tall_total > 0 else 0.5
 
        # 3. Immediate eat threats vs vulnerabilities
        eat_threats    = 0
        vulnerabilities = 0
        for (ac, ah) in agent_stacks:
            for d in CARDINAL_DIRECTIONS:
                nr, nc = ac.r + d.r, ac.c + d.c
                if 0 <= nr < 8 and 0 <= nc < 8:
                    nbr = bstate[Coord(nr, nc)]
                    if nbr.color == enemy_color and ah >= nbr.height:
                        eat_threats += nbr.height  # weight by tokens captured
        for (ec, eh) in enemy_stacks:
            for d in CARDINAL_DIRECTIONS:
                nr, nc = ec.r + d.r, ec.c + d.c
                if 0 <= nr < 8 and 0 <= nc < 8:
                    nbr = bstate[Coord(nr, nc)]
                    if nbr.color == agent_color and eh >= nbr.height:
                        vulnerabilities += nbr.height
        max_threat = max(agent_tokens, enemy_tokens, 1)
        eat_threat_score = 0.5 + (eat_threats - vulnerabilities) / (2.0 * max_threat)
        eat_threat_score = max(0.0, min(1.0, eat_threat_score))
 
        # 4. Attack distance — how close is our tallest stack to any enemy?
        # Only consider stacks tall enough to actually threaten (h >= 2).
        min_attack_dist = float(MAX_DIST)
        for (ac, ah) in agent_stacks:
            if ah < 2:
                continue
            for (ec, _) in enemy_stacks:
                d = abs(ac.r - ec.r) + abs(ac.c - ec.c)
                if d < min_attack_dist:
                    min_attack_dist = d
        if min_attack_dist == float(MAX_DIST):
            # No tall stacks — fall back to any stack's distance
            for (ac, _) in agent_stacks:
                for (ec, _) in enemy_stacks:
                    d = abs(ac.r - ec.r) + abs(ac.c - ec.c)
                    if d < min_attack_dist:
                        min_attack_dist = d
        attack_dist_score = 1.0 - (min_attack_dist / MAX_DIST)  # [0, 1]
 
        # 5. Draw penalty — if we're ahead on tokens, penalise being near the
        # turn limit (a draw would waste our advantage).
        turns_left = max(0, 300 - board.play_phase_turn_count)
        draw_risk = 1.0 - (turns_left / 300.0)  # 0 early, 1 at limit
        # Only penalise when we're actually ahead; penalty scales with lead
        lead = (agent_tokens - enemy_tokens) / total_tokens  # [-1, 1]
        draw_penalty = draw_risk * max(0.0, lead) * 0.15
 
        # --- Dynamic weighting by game phase ---
        progress = min(board.play_phase_turn_count / 200.0, 1.0)
        w_token  = 0.25 + 0.35 * progress   # 0.25 early → 0.60 late
        w_tall   = 0.30 - 0.15 * progress   # 0.30 early → 0.15 late
        w_eat    = 0.25
        w_dist   = 1.0 - w_token - w_tall - w_eat  # remainder
 
        score = (
            w_token * token_score
            + w_tall  * tall_score
            + w_eat   * eat_threat_score
            + w_dist  * attack_dist_score
            - draw_penalty
        )
        return max(0.0, min(1.0, score))
 
    def total_stack_height_count(self, board: Board, color: PlayerColor) -> int:
        return sum(
            cell.height for cell in board._state.values()
            if cell.color == color
        )
    

    def placement_policy(self, state: GameState, agent_color: PlayerColor) -> PlaceAction:
        legal_actions = [a for a in state.get_legal_actions() if isinstance(a, PlaceAction)]
        if not legal_actions:
            raise ValueError("No legal placement actions available")
        return random.choice(legal_actions)
    
    
    '''Helper function for calculating orthogonal distance between two coordinates. Returns a tuple of (min_orth, max_orth)'''
    def orthogonal(self, a: Coord, b: Coord) -> tuple[int, int]:
        dx = abs(a.c - b.c)
        dy = abs(a.r - b.r)
        return (dx, dy) if dx <= dy else (dy, dx)
    
    def simulate(self, state: GameState) -> float:
        # The perspective we always score from is the agent that owns the tree.
        agent_color: PlayerColor = self.root.state.turn_color
 
        # Work on a cheap clone so we never mutate the node's state.
        sim_state = state.clone()
 
        for _ in range(MAX_ROLLOUT_DEPTH):
            if sim_state.game_over:
                break
 
            try:
                action = self.rollout_policy(sim_state, agent_color)
            except ValueError:
                # No legal actions — game is effectively over.
                break
 
            sim_state.apply_action(action)
 
        return self.rollout_score(sim_state, agent_color)

    def backpropagate(self, node: MCTSNode | None, reward: float) -> None:
        while node is not None:
            node.update(reward)
            node = node.parent

    def best_action(self) -> Action | None:
        if not self.root.children:
            return None
        return max(self.root.children, key=lambda child: child.visits).action

