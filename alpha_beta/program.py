# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

import copy
import time
import random

from referee.game import (
    PlayerColor, Coord, Direction,
    Action, PlaceAction, MoveAction, EatAction, CascadeAction,
    Board, GamePhase, BOARD_N, CARDINAL_DIRECTIONS,
)


TOTAL_TIME_BUDGET = 180.0
TIME_BUFFER       = 6.0
TIME_FRACTION     = 0.10      # fraction of remaining time per turn
MAX_TIME_PER_TURN = 6.0
MIN_TIME_PER_TURN = 0.05


INF = float("inf")
AB_DEPTH_INITIAL = 3          # starting iterative-deepening depth
AB_DEPTH_MAX     = 6          # never search deeper than this

BOARD_CENTRE = 3.5



class GameState:
    """Thin wrapper around Board that adds legal-action generation."""

    def __init__(
        self,
        initial_player: PlayerColor = PlayerColor.RED,
        board: Board | None = None,
    ):
        self._board = board if board is not None else Board(initial_player=initial_player)

    @property
    def phase(self) -> GamePhase:
        return self._board.phase

    @property
    def turn_color(self) -> PlayerColor:
        return self._board.turn_color

    @property
    def turn_count(self) -> int:
        return self._board.turn_count

    @property
    def play_phase_turn_count(self) -> int:
        return self._board.play_phase_turn_count

    @property
    def game_over(self) -> bool:
        return self._board.game_over

    def clone(self) -> "GameState":
        return GameState(board=copy.deepcopy(self._board))

    def apply_action(self, action: Action) -> None:
        self._board.apply_action(action)

    def render(self, use_color: bool = False, use_unicode: bool = False) -> str:
        return self._board.render(use_color=use_color, use_unicode=use_unicode)


    def get_legal_actions(self) -> list[Action]:
        actions: list[Action] = []
        current_player = self._board.turn_color
        opponent = current_player.opponent
        state = self._board._state

        if self.phase == GamePhase.PLACEMENT:
            for r in range(BOARD_N):
                for c in range(BOARD_N):
                    coord = Coord(r, c)
                    if not state[coord].is_empty:
                        continue
                    adj_opp = False
                    for d in CARDINAL_DIRECTIONS:
                        nr, nc = r + d.r, c + d.c
                        if 0 <= nr < BOARD_N and 0 <= nc < BOARD_N:
                            if state[Coord(nr, nc)].color == opponent:
                                adj_opp = True
                                break
                    if not adj_opp:
                        actions.append(PlaceAction(coord))
            return actions

        # Play phase
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                coord = Coord(r, c)
                cell = state[coord]
                if cell.color != current_player:
                    continue
                for d in CARDINAL_DIRECTIONS:
                    dr, dc = r + d.r, c + d.c
                    if not (0 <= dr < BOARD_N and 0 <= dc < BOARD_N):
                        continue
                    dest = Coord(dr, dc)
                    dest_cell = state[dest]
                    if dest_cell.is_empty or dest_cell.color == current_player:
                        actions.append(MoveAction(coord, d))
                    if dest_cell.color == opponent and cell.height >= dest_cell.height:
                        actions.append(EatAction(coord, d))
                if cell.height >= 2:
                    for d in CARDINAL_DIRECTIONS:
                        actions.append(CascadeAction(coord, d))
        return actions


def evaluate(board: Board, agent_color: PlayerColor) -> float:
    """
    Static board evaluation from agent_color's perspective.
    Returns a value roughly in [-1, 1].

    Features (all normalised):
      1. Token ratio                  - raw material balance
      2. Eat-threat count             - immediate capture opportunities
      3. Vulnerability count          - stacks we're about to lose
      4. Stack-height concentration   - tall stacks are more powerful
      5. Mobility ratio               - number of legal actions available
      6. Centre control               - stacks near board centre
    """
    enemy_color = agent_color.opponent
    state = board._state

    agent_tokens  = 0
    enemy_tokens  = 0
    agent_stacks  = []   # (coord, height)
    enemy_stacks  = []

    for coord, cell in state.items():
        if cell.color == agent_color:
            agent_tokens += cell.height
            agent_stacks.append((coord, cell.height))
        elif cell.color == enemy_color:
            enemy_tokens += cell.height
            enemy_stacks.append((coord, cell.height))

    total_tokens = agent_tokens + enemy_tokens
    if total_tokens == 0:
        return 0.0

    # 1. Token ratio [-1, 1]
    token_score = (agent_tokens - enemy_tokens) / total_tokens

    # 2 & 3. Eat threats and vulnerability
    eat_threats    = 0  # how many enemy stacks we can eat right now
    vulnerabilities = 0  # how many of our stacks an enemy can eat right now

    for (ac, ah) in agent_stacks:
        for d in CARDINAL_DIRECTIONS:
            nr, nc = ac.r + d.r, ac.c + d.c
            if 0 <= nr < BOARD_N and 0 <= nc < BOARD_N:
                nbr = state[Coord(nr, nc)]
                if nbr.color == enemy_color and ah >= nbr.height:
                    eat_threats += 1

    for (ec, eh) in enemy_stacks:
        for d in CARDINAL_DIRECTIONS:
            nr, nc = ec.r + d.r, ec.c + d.c
            if 0 <= nr < BOARD_N and 0 <= nc < BOARD_N:
                nbr = state[Coord(nr, nc)]
                if nbr.color == agent_color and eh >= nbr.height:
                    vulnerabilities += 1

    threat_score = (eat_threats - vulnerabilities) / max(
        len(agent_stacks) + len(enemy_stacks), 1
    )

    # 4. Tallest-stack advantage (tall stacks can eat more things)
    agent_max_h = max((h for _, h in agent_stacks), default=0)
    enemy_max_h = max((h for _, h in enemy_stacks), default=0)
    height_score = (agent_max_h - enemy_max_h) / max(agent_max_h + enemy_max_h, 1)

    # 5. Centre control
    agent_centre = sum(
        1.0 - (abs(c.r - BOARD_CENTRE) + abs(c.c - BOARD_CENTRE)) / 7.0
        for c, _ in agent_stacks
    ) / max(len(agent_stacks), 1)
    enemy_centre = sum(
        1.0 - (abs(c.r - BOARD_CENTRE) + abs(c.c - BOARD_CENTRE)) / 7.0
        for c, _ in enemy_stacks
    ) / max(len(enemy_stacks), 1)
    centre_score = agent_centre - enemy_centre  # in [-1, 1]

    # Dynamic weighting based on game progress
    progress = min(board.play_phase_turn_count / 150.0, 1.0)
    w_token   = 0.35 + 0.35 * progress
    w_threat  = 0.30
    w_height  = 0.15
    w_centre  = (1.0 - w_token - w_threat - w_height)

    score = (
        w_token  * token_score
        + w_threat * threat_score
        + w_height * height_score
        + w_centre * centre_score
    )
    return max(-1.0, min(1.0, score))



def order_actions(actions: list[Action], board: Board, color: PlayerColor) -> list[Action]:
    """
    Cheap heuristic ordering for alpha-beta move ordering.
    Priority (descending):
      1. EatAction (immediate capture)
      2. CascadeAction that hits an enemy
      3. MoveAction that merges with a friendly tall stack
      4. Everything else
    """
    state = board._state
    opponent = color.opponent

    def score(action: Action) -> float:
        if isinstance(action, EatAction):
            dest = Coord(action.coord.r + action.direction.r,
                         action.coord.c + action.direction.c)
            # Higher enemy height = more valuable capture
            return 1000.0 + state[dest].height

        if isinstance(action, CascadeAction):
            # Count enemy tokens in path
            r, c = action.coord.r, action.coord.c
            h = state[action.coord].height
            hits = 0
            for i in range(1, h + 1):
                nr = r + action.direction.r * i
                nc = c + action.direction.c * i
                if 0 <= nr < BOARD_N and 0 <= nc < BOARD_N:
                    cell = state[Coord(nr, nc)]
                    if cell.color == opponent:
                        hits += cell.height
            return 500.0 + hits

        if isinstance(action, MoveAction):
            dest = Coord(action.coord.r + action.direction.r,
                         action.coord.c + action.direction.c)
            dest_cell = state[dest]
            if dest_cell.color == color:
                return 200.0 + dest_cell.height  # merging
            return 0.0

        return 0.0

    return sorted(actions, key=score, reverse=True)



class AlphaBeta:

    def __init__(self, agent_color: PlayerColor, deadline: float):
        self.agent_color = agent_color
        self.deadline    = deadline
        self.nodes       = 0
        self._timed_out  = False

    def timed_out(self) -> bool:
        if self._timed_out:
            return True
        if time.time() >= self.deadline:
            self._timed_out = True
        return self._timed_out

    def search(self, state: GameState) -> Action | None:
        """
        Iterative-deepening alpha-beta.  Returns the best action found before
        the deadline, falling back to a random legal action if nothing completes.
        """
        legal = state.get_legal_actions()
        if not legal:
            return None
        if len(legal) == 1:
            return legal[0]

        best_action = random.choice(legal)  # safe fallback

        for depth in range(1, AB_DEPTH_MAX + 1):
            if self.timed_out():
                break
            self._timed_out = False  # reset per iteration
            action, _ = self._root_search(state, depth)
            if not self._timed_out and action is not None:
                best_action = action
            if self.timed_out():
                break

        return best_action

    def _root_search(
        self, state: GameState, depth: int
    ) -> tuple[Action | None, float]:
        alpha, beta = -INF, INF
        best_action = None
        best_value  = -INF

        legal = order_actions(
            state.get_legal_actions(), state._board, self.agent_color
        )

        for action in legal:
            if self.timed_out():
                break
            child = state.clone()
            child.apply_action(action)
            value = self._minimax(child, depth - 1, alpha, beta, False)
            if value > best_value:
                best_value  = value
                best_action = action
            alpha = max(alpha, best_value)

        return best_action, best_value

    def _minimax(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        maximising: bool,
    ) -> float:
        self.nodes += 1

        if self.timed_out():
            return 0.0

        if state.game_over:
            winner = state._board.winner_color
            if winner == self.agent_color:
                return 1.0
            elif winner is None:
                return 0.0
            else:
                return -1.0

        if depth == 0:
            return evaluate(state._board, self.agent_color)

        legal = order_actions(
            state.get_legal_actions(), state._board, state.turn_color
        )
        if not legal:
            return evaluate(state._board, self.agent_color)

        if maximising:
            value = -INF
            for action in legal:
                if self.timed_out():
                    break
                child = state.clone()
                child.apply_action(action)
                value = max(value, self._minimax(child, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if value >= beta:
                    break  # beta cut-off
            return value
        else:
            value = INF
            for action in legal:
                if self.timed_out():
                    break
                child = state.clone()
                child.apply_action(action)
                value = min(value, self._minimax(child, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if value <= alpha:
                    break  # alpha cut-off
            return value



def placement_score(coord: Coord, board: Board, color: PlayerColor) -> float:
    """
    Score a placement coordinate during the placement phase.

    Prefers:
      - Cells near the board centre
      - Cells far from opponent stacks
      - Cells with many empty neighbours (mobility)
    """
    opponent = color.opponent
    state    = board._state

    # 1. Centre proximity
    dist_centre = abs(coord.r - BOARD_CENTRE) + abs(coord.c - BOARD_CENTRE)
    centre_score = 1.0 - (dist_centre / 7.0)

    # 2. Distance from nearest enemy stack
    min_opp_dist = BOARD_N * 2
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            cell = state[Coord(r, c)]
            if cell.color == opponent:
                d = abs(coord.r - r) + abs(coord.c - c)
                if d < min_opp_dist:
                    min_opp_dist = d
    opp_score = min(min_opp_dist / 7.0, 1.0)

    # 3. Friendly clustering (prefer being near own stacks for merge potential)
    min_friendly_dist = BOARD_N * 2
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            cell = state[Coord(r, c)]
            if cell.color == color:
                d = abs(coord.r - r) + abs(coord.c - c)
                if d < min_friendly_dist:
                    min_friendly_dist = d
    # Want to be close to friendly (2-3 away is ideal for future merges)
    friendly_score = 1.0 - abs(min_friendly_dist - 2) / 6.0

    # 4. Empty neighbours (mobility)
    empty_nbrs = sum(
        1
        for d in CARDINAL_DIRECTIONS
        if (0 <= coord.r + d.r < BOARD_N and 0 <= coord.c + d.c < BOARD_N)
        and state[Coord(coord.r + d.r, coord.c + d.c)].is_empty
    )
    mobility_score = empty_nbrs / 4.0

    return (
        0.35 * centre_score
        + 0.30 * opp_score
        + 0.20 * friendly_score
        + 0.15 * mobility_score
    )


def choose_placement(state: GameState, color: PlayerColor) -> Action:
    legal = state.get_legal_actions()
    return max(legal, key=lambda a: placement_score(a.coord, state._board, color))



class Agent:
    """
    Alpha-beta minimax agent with iterative deepening.
    Falls back to a greedy one-ply search if the deadline is very tight.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        self._color      = color
        self._state      = GameState(initial_player=PlayerColor.RED)
        self._start_time = time.time()
        print(f"Agent initialised as {color}")

    def action(self, **referee: dict) -> Action:
        """
        Called by the referee at the start of our turn. Returns the chosen
        action.
        """
        time_remaining: float | None = referee.get("time_remaining", None)  # type: ignore

        # Placement phase
        if self._state.phase == GamePhase.PLACEMENT:
            chosen = choose_placement(self._state, self._color)
            print(f"Agent ({self._color}): PLACE {chosen.coord}")
            return chosen

        # Play phase
        budget   = self._turn_budget(time_remaining)
        deadline = time.time() + budget

        ab     = AlphaBeta(self._color, deadline)
        action = ab.search(self._state)

        print(
            f"Agent ({self._color}): {ab.nodes} nodes, "
            f"{budget:.2f}s budget → {action}"
        )

        if action is None:
            legal  = self._state.get_legal_actions()
            action = random.choice(legal) if legal else MoveAction(Coord(0, 0), Direction.Down)

        return action

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        self._state.apply_action(action)

    def _turn_budget(self, time_remaining: float | None) -> float:
        if time_remaining is not None:
            available = max(time_remaining - TIME_BUFFER, 0.0)
        else:
            elapsed   = time.time() - self._start_time
            available = max(TOTAL_TIME_BUDGET - elapsed - TIME_BUFFER, 0.0)

        budget = available * TIME_FRACTION
        return max(MIN_TIME_PER_TURN, min(MAX_TIME_PER_TURN, budget))