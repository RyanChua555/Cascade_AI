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
MAX_ROLLOUT_DEPTH = 30 # simulation depth limit
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cascade_value(
        self,
        action: CascadeAction,
        bstate: dict,
        opponent: PlayerColor,
    ) -> float:
        """
        Deterministic cascade score — call ONCE and store the result.
        Returns:
          -1.0  tokens fall off the board AND no enemy is hit (self-elimination)
           0.0  no enemies hit and no tokens lost
          >0.0  proportional to enemy tokens hit
        Never includes randomness — add jitter at the call site if needed.
        """
        r, c = action.coord.r, action.coord.c
        height = bstate[action.coord].height
        enemy_hits = 0
        tokens_lost = 0
        for step in range(1, height + 1):
            nr = r + action.direction.r * step
            nc = c + action.direction.c * step
            if 0 <= nr < BOARD_N and 0 <= nc < BOARD_N:
                cell = bstate[Coord(nr, nc)]
                if cell.color == opponent:
                    enemy_hits += cell.height + 1
            else:
                tokens_lost += 1
        if tokens_lost > 0 and enemy_hits == 0:
            return -1.0
        return float(enemy_hits)

    # ------------------------------------------------------------------
    # Rollout policy
    # ------------------------------------------------------------------

    def rollout_policy(self, state: GameState, agent_color: PlayerColor) -> Action:
        """
        Fast action selection for rollout simulations. No board cloning.

        Priority order:
          0. Placement phase  → placement_policy
          1. EatAction        → always take the tallest-target eat
          2. CascadeAction    → only if it hits an enemy AND doesn't
                                self-eliminate (lose tokens off-board for free)
          3. MoveAction       → merges first, then approach nearest enemy
          4. Fallback         → random legal action
        """
        legal_actions = state.get_legal_actions()
        if not legal_actions:
            raise ValueError("No legal actions available")

        board = state._board
        bstate = board._state
        color = board.turn_color
        opponent = color.opponent

        # ---- 0. Placement phase ----------------------------------------
        if state.phase == GamePhase.PLACEMENT:
            return self.placement_policy(state, color)

        # ---- 1. Eat: take best available eat ---------------------------
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

        # ---- 2. Cascade: only enemy-hitting, non-self-eliminating ------
        # Score is computed once per action and stored — never re-evaluated
        # with different random values, which was the bug causing fatal
        # self-cascades (turn 173 in the observed game).
        best_cascade: CascadeAction | None = None
        best_cascade_val = 0.0  # must strictly exceed 0 to be accepted
        for a in legal_actions:
            if isinstance(a, CascadeAction):
                base_val = self._cascade_value(a, bstate, opponent)
                if base_val > 0:
                    # Add jitter once here, after the veto check
                    val = base_val + random.random() * 0.1
                    if val > best_cascade_val:
                        best_cascade_val = val
                        best_cascade = a
        if best_cascade is not None:
            return best_cascade

        # ---- 3. Move: merges first, then approach ----------------------
        # Pre-compute nearest enemy to any friendly — one pass, no early break.
        nearest_enemy: Coord | None = None
        min_dist = float(MAX_DIST) + 1.0
        for coord, cell in bstate.items():
            if cell.color == opponent:
                for fc, fc_cell in bstate.items():
                    if fc_cell.color == color:
                        d = abs(coord.r - fc.r) + abs(coord.c - fc.c)
                        if d < min_dist:
                            min_dist = d
                            nearest_enemy = coord

        best_move: Action | None = None
        best_move_val = -1.0
        for a in legal_actions:
            if not isinstance(a, MoveAction):
                continue
            dest = Coord(a.coord.r + a.direction.r, a.coord.c + a.direction.c)
            dest_cell = bstate[dest]

            if dest_cell.color == color:
                # Merge: always beats approach — score above 1.0 ceiling of approach
                merged_h = bstate[a.coord].height + dest_cell.height
                val = 1.1 + merged_h * 0.05 + random.random() * 0.02
            elif nearest_enemy is not None:
                # Approach nearest enemy
                dist_after = abs(dest.r - nearest_enemy.r) + abs(dest.c - nearest_enemy.c)
                val = 1.0 - (dist_after / MAX_DIST) + random.random() * 0.05
            else:
                val = random.random() * 0.05

            if val > best_move_val:
                best_move_val = val
                best_move = a

        if best_move is not None:
            return best_move

        # ---- 4. Fallback (only cascades with no-enemy path remain) ----
        return random.choice(legal_actions)

    # ------------------------------------------------------------------
    # Placement policy
    # ------------------------------------------------------------------

    def placement_policy(self, state: GameState, agent_color: PlayerColor) -> PlaceAction:
        """
        Score placement cells on four features:
          1. Centre proximity  — closer to centre = better
          2. Enemy distance    — further from enemies = safer (capped at 5)
          3. Friendly distance — ideal gap ≈ 2 cells for future merges
          4. Edge penalty      — row/col 0 or 7 are cascade-vulnerable
        """
        legal_places: list[PlaceAction] = [
            a for a in state.get_legal_actions() if isinstance(a, PlaceAction)
        ]
        if not legal_places:
            raise ValueError("No legal placement actions available")

        bstate   = state._board._state
        opponent = agent_color.opponent

        enemy_coords:    list[Coord] = []
        friendly_coords: list[Coord] = []
        for coord, cell in bstate.items():
            if cell.color == opponent:
                enemy_coords.append(coord)
            elif cell.color == agent_color:
                friendly_coords.append(coord)

        CENTRE = 3.5

        def score(action: PlaceAction) -> float:
            c = action.coord
            dist_centre = abs(c.r - CENTRE) + abs(c.c - CENTRE)
            centre_score = 1.0 - (dist_centre / 14.0)

            if enemy_coords:
                min_enemy = min(abs(c.r - e.r) + abs(c.c - e.c) for e in enemy_coords)
                enemy_score = min(min_enemy, 5) / 5.0
            else:
                enemy_score = 1.0

            if friendly_coords:
                min_friendly = min(abs(c.r - f.r) + abs(c.c - f.c) for f in friendly_coords)
                friendly_score = max(0.0, 1.0 - abs(min_friendly - 2) / 4.0)
            else:
                friendly_score = 0.5

            on_edge = (c.r == 0 or c.r == 7 or c.c == 0 or c.c == 7)
            edge_penalty = 0.25 if on_edge else 0.0

            return (
                0.40 * centre_score
                + 0.25 * enemy_score
                + 0.25 * friendly_score
                - edge_penalty
                + random.random() * 0.02
            )

        return max(legal_places, key=score)

    # ------------------------------------------------------------------
    # Rollout score (static board evaluation)
    # ------------------------------------------------------------------

    def rollout_score(self, state: GameState, agent_color: PlayerColor) -> float:
        """
        Evaluate board from agent_color's perspective. Returns [0, 1].

        Features:
          1. token_score         raw material ratio
          2. tall_score          fraction of tokens in h≥3 stacks
          3. consolidation_score average stack height vs enemy
          4. eat_threat_score    immediate captures minus vulnerabilities
          5. attack_dist_score   proximity of our best stack to an enemy
          draw_penalty           penalise approaching turn limit when ahead
        """
        enemy_color = agent_color.opponent
        board = state._board
        bstate = board._state

        if state.game_over:
            winner = board.winner_color
            if winner == agent_color:
                return 1.0
            elif winner == enemy_color:
                return 0.0
            return 0.45  # draw slightly below neutral

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

        # 2. Tall-stack concentration (h≥3 = combat-ready)
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
        eat_threats = 0
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

        # Draw penalty: penalise approaching turn limit when materially ahead
        turns_left = max(0, 300 - board.play_phase_turn_count)
        draw_risk = 1.0 - (turns_left / 300.0)
        lead = (agent_tokens - enemy_tokens) / total_tokens
        draw_penalty = draw_risk * max(0.0, lead) * 0.15

        # Dynamic weights
        progress = min(board.play_phase_turn_count / 200.0, 1.0)
        w_token = 0.25 + 0.30 * progress   # 0.25 → 0.55
        w_tall  = 0.20 - 0.05 * progress   # 0.20 → 0.15
        w_cons  = 0.10 + 0.10 * progress   # 0.10 → 0.20
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

    # ------------------------------------------------------------------
    # Simulate
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

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