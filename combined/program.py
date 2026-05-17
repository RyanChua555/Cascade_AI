# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

import copy
import time
import random

from referee.game import PlayerColor, Coord, Direction, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction, Board, GamePhase, BOARD_N, CARDINAL_DIRECTIONS


# Total CPU seconds available per game (referee hard limit is 180s).
TOTAL_TIME_BUDGET = 180.0

# Reserve this many seconds as a safety buffer so we never hit the hard limit.
TIME_BUFFER = 5.0

# Fraction of remaining time to spend on each individual alpha-beta call.
TIME_FRACTION_PER_TURN = 0.05

# Hard cap per turn.
MAX_TIME_PER_TURN = 5.0

# Minimum time per turn.
MIN_TIME_PER_TURN = 0.1

# Alpha-beta iterative deepening limits.
AB_MAX_DEPTH = 6

INF = float("inf")

BOARD_CENTRE = (BOARD_N - 1) / 2.0

class GameState:
    """
    Internal representation of the current game state from the agent's view.
    """

    def __init__(self, initial_player: PlayerColor = PlayerColor.RED, board: Board | None = None):
        self._board = board if board is not None else Board(initial_player=initial_player)

    @property
    def phase(self) -> GamePhase:
        return self._board.phase

    def clone(self) -> "GameState":
        return GameState(board=copy.deepcopy(self._board))

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

    def apply_action(self, action: Action) -> None:
        self._board.apply_action(action)

    def render(self, use_color: bool = False, use_unicode: bool = False) -> str:
        return self._board.render(use_color=use_color, use_unicode=use_unicode)

    def get_legal_actions(self) -> list[Action]:
        """
        Generate all legal actions available to the current player.
        """
        actions: list[Action] = []
        current_player = self._board.turn_color
        opponent = current_player.opponent

        if self.phase == GamePhase.PLACEMENT:
            for r in range(BOARD_N):
                for c in range(BOARD_N):
                    coord = Coord(r, c)
                    if not self._board[coord].is_empty:
                        continue

                    adjacent_to_opponent = False
                    for direction in CARDINAL_DIRECTIONS:
                        adj_r = coord.r + direction.r
                        adj_c = coord.c + direction.c
                        if 0 <= adj_r < BOARD_N and 0 <= adj_c < BOARD_N:
                            adj_cell = self._board[Coord(adj_r, adj_c)]
                            if adj_cell.color == opponent:
                                adjacent_to_opponent = True
                                break

                    if not adjacent_to_opponent:
                        actions.append(PlaceAction(coord))
            return actions

        for r in range(BOARD_N):
            for c in range(BOARD_N):
                coord = Coord(r, c)
                cell = self._board[coord]

                if cell.color != current_player:
                    continue

                for direction in CARDINAL_DIRECTIONS:
                    dest_r = coord.r + direction.r
                    dest_c = coord.c + direction.c

                    if not (0 <= dest_r < BOARD_N and 0 <= dest_c < BOARD_N):
                        continue

                    dest_coord = Coord(dest_r, dest_c)
                    dest_cell = self._board[dest_coord]

                    if dest_cell.is_empty or dest_cell.color == current_player:
                        actions.append(MoveAction(coord, direction))

                    if dest_cell.color == opponent and cell.height >= dest_cell.height:
                        actions.append(EatAction(coord, direction))

                if cell.height >= 2:
                    for direction in CARDINAL_DIRECTIONS:
                        actions.append(CascadeAction(coord, direction))

        return actions




# ---------------------------------------------------------------------------
# Move ordering for alpha-beta (critical for pruning efficiency)
# ---------------------------------------------------------------------------

def order_actions(actions: list[Action], board: Board, color: PlayerColor) -> list[Action]:
    """
    Order actions so alpha-beta sees the best moves first, maximising cutoffs.

    Priority (descending):
      1. EatAction   — immediate capture; always the most forcing move
      2. CascadeAction that hits enemy tokens along its path
      3. MoveAction  that merges with a friendly stack
      4. Everything else
    """
    bstate   = board._state
    opponent = color.opponent

    def score(action: Action) -> float:
        if isinstance(action, EatAction):
            dest = Coord(action.coord.r + action.direction.r,
                         action.coord.c + action.direction.c)
            return 1000.0 + bstate[dest].height

        if isinstance(action, CascadeAction):
            r, c   = action.coord.r, action.coord.c
            height = bstate[action.coord].height
            hits   = 0
            for i in range(1, height + 1):
                nr = r + action.direction.r * i
                nc = c + action.direction.c * i
                if 0 <= nr < BOARD_N and 0 <= nc < BOARD_N:
                    cell = bstate[Coord(nr, nc)]
                    if cell.color == opponent:
                        hits += cell.height
            return 500.0 + hits

        if isinstance(action, MoveAction):
            dest = Coord(action.coord.r + action.direction.r,
                         action.coord.c + action.direction.c)
            dest_cell = bstate[dest]
            if dest_cell.color == color:
                return 200.0 + dest_cell.height  # merge
            return 0.0

        return 0.0

    return sorted(actions, key=score, reverse=True)


# ---------------------------------------------------------------------------
# Iterative-deepening alpha-beta minimax
# ---------------------------------------------------------------------------

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

        for depth in range(1, AB_MAX_DEPTH + 1):
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
                    break  # β cut-off
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
                    break  # α cut-off
            return value

def evaluate(board: Board, agent_color: PlayerColor) -> float:
    """
    Static board evaluation from agent_color's perspective.
    Returns a value roughly in [-1, 1].

    Features (all normalised):
      1. Token ratio                  – raw material balance
      2. Eat-threat count             – immediate capture opportunities
      3. Vulnerability count          – stacks we're about to lose
      4. Stack-height concentration   – tall stacks are powerful
      5. Mobility ratio               – number of legal actions available
      6. Centre control               – stacks near board centre
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
    # Early game: position matters more; late game: material dominates
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
class Agent:
    """
    Game-playing agent using iterative-deepening alpha-beta minimax as the
    primary search, with MCTS available as a fallback.

    Alpha-beta searches deeply (depth 3-6 depending on branching factor) and
    uses the richer rollout_score evaluation function from mcts.py at leaf
    nodes.  Move ordering (captures first) maximises pruning efficiency.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        self._color      = color
        self._game_state = GameState(initial_player=PlayerColor.RED)
        self._start_time = time.time()
        self._tree       = None  # kept for MCTS fallback / tree reuse

        print(f"Agent initialised as {color}")

        
    def action(self, **referee: dict) -> Action:
        time_remaining: float | None = referee.get("time_remaining", None)  # type: ignore
        budget = self.turn_budget(time_remaining)
        deadline = time.time() + budget

        # ------------------------------------------------------------------
        # Placement phase
        # ------------------------------------------------------------------
        if self._game_state.phase == GamePhase.PLACEMENT:
            chosen = self._placement_action()
            print(f"Agent ({self._color}): PLACE {chosen.coord}")
            return chosen

        legal_actions = self._game_state.get_legal_actions()

        if not legal_actions:
            return MoveAction(Coord(0, 0), Direction.Down)

        if len(legal_actions) == 1:
            return legal_actions[0]

        # ------------------------------------------------------------------
        # Hybrid search policy
        # ------------------------------------------------------------------
        #
        # Early / mid game:
        #   Use alpha-beta because tactical accuracy matters most.
        #
        # Late game:
        #   Use MCTS because branching factor shrinks and long-term
        #   rollout evaluation becomes stronger.
        #
        # Timeout safety:
        #   If alpha-beta times out too early or fails to return a move,
        #   immediately fall back to MCTS.
        #
        # ------------------------------------------------------------------

        play_turns = self._game_state.play_phase_turn_count

        # Rough branching estimate
        branching = len(legal_actions)

        # Endgame heuristic
        endgame = (
            play_turns > 120
            or branching < 12
        )

        # ------------------------------------------------------------------
        # PRIMARY: Alpha-beta
        # ------------------------------------------------------------------

        if not endgame:    
            # --- Play phase: iterative-deepening alpha-beta ---
            budget   = self._turn_budget(time_remaining)
            deadline = time.time() + budget

            ab     = AlphaBeta(self._color, deadline)
            action = ab.search(self._game_state)

            print(
                f"Agent ({self._color}): {ab.nodes} nodes, "
                f"{budget:.2f}s budget → {action}"
            )

            if action is None:
                action = random.choice(legal_actions) if legal_actions else MoveAction(Coord(0, 0), Direction.Down)

            return action


        # ------------------------------------------------------------------
        # FALLBACK / ENDGAME: MCTS
        # ------------------------------------------------------------------

        remaining_budget = max(0.05, deadline - time.time())

        mcts_action = self.run_mcts(remaining_budget)

        if mcts_action is not None:
            print(
                f"Agent ({self._color}): "
                f"MCTS chose {mcts_action}"
            )
            return mcts_action

        # ------------------------------------------------------------------
        # Final emergency fallback
        # ------------------------------------------------------------------

        fallback = random.choice(legal_actions)

        print(
            f"Agent ({self._color}): "
            f"random fallback {fallback}"
        )

        return fallback

    def _placement_action(self) -> PlaceAction:
        """Heuristic placement: prefer centre, avoid edges, cluster with own stacks."""
        legal_places: list[PlaceAction] = [
            a for a in self._game_state.get_legal_actions() if isinstance(a, PlaceAction)
        ]
        if not legal_places:
            raise ValueError("No legal placement actions available")
        
        board    = self._game_state._board
        bstate   = board._state
        opponent = self._color.opponent
        CENTRE   = 3.5

        enemy_coords    = [c for c, cell in bstate.items() if cell.color == opponent]
        friendly_coords = [c for c, cell in bstate.items() if cell.color == self._color]

        def score(a: PlaceAction) -> float:
            c = a.coord
            centre_score = 1.0 - (abs(c.r - CENTRE) + abs(c.c - CENTRE)) / 14.0
            enemy_score  = (min(abs(c.r-e.r)+abs(c.c-e.c) for e in enemy_coords) / 5.0
                           if enemy_coords else 1.0)
            enemy_score  = min(enemy_score, 1.0)
            if friendly_coords:
                min_f = min(abs(c.r-f.r)+abs(c.c-f.c) for f in friendly_coords)
                friendly_score = max(0.0, 1.0 - abs(min_f - 2) / 4.0)
            else:
                friendly_score = 0.5
            edge_penalty = 0.25 if (c.r in (0,7) or c.c in (0,7)) else 0.0
            return (0.40*centre_score + 0.25*enemy_score + 0.25*friendly_score
                    - edge_penalty + random.random()*0.02)

        return max(legal_places, key=score)

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        self._advance_tree(action)
        self._game_state.apply_action(action)

    def _advance_tree(self, action: Action) -> None:
        if self._tree is None:
            return
        matched_child = None
        for child in self._tree.root.children:
            if child.action == action:
                matched_child = child
                break
        if matched_child is None:
            self._tree = None
            return
        matched_child.parent = None
        self._tree.root = matched_child

    def turn_budget(self, time_remaining: float | None) -> float:
        if time_remaining is not None:
            available = max(time_remaining - TIME_BUFFER, 0.0)
        else:
            elapsed   = time.time() - self._start_time
            available = max(TOTAL_TIME_BUDGET - elapsed - TIME_BUFFER, 0.0)
        budget = available * TIME_FRACTION_PER_TURN
        return max(MIN_TIME_PER_TURN, min(MAX_TIME_PER_TURN, budget))

    def run_mcts(self, time_budget: float) -> Action | None:
        """
        MCTS fallback — kept intact for completeness and potential future use.
        Not called by action() in the current implementation.
        """
        from .mcts import MCTSTree

        if self._tree is None:
            root_state = self._game_state.clone()
            self._tree = MCTSTree(root_state)
            self._tree.root.untried_actions = root_state.get_legal_actions()
        else:
            if self._tree.root.state.turn_count != self._game_state.turn_count:
                root_state = self._game_state.clone()
                self._tree = MCTSTree(root_state)
                self._tree.root.untried_actions = root_state.get_legal_actions()

        tree = self._tree

        if len(tree.root.untried_actions) == 1 and not tree.root.children:
            return tree.root.untried_actions[0]

        deadline = time.time() + time_budget
        iterations = 0

        def _expansion_priority(a: Action) -> int:
            if isinstance(a, EatAction):     return 2
            if isinstance(a, CascadeAction): return 1
            return 0

        while time.time() < deadline:
            node = tree.selection(tree.root)

            if not node.state.game_over and node.untried_actions:
                node.untried_actions.sort(key=_expansion_priority)
                action = node.untried_actions.pop()
                next_state = node.state.clone()
                next_state.apply_action(action)
                child = tree.expand(node, action, next_state)
                child.untried_actions = next_state.get_legal_actions()
                node = child

            reward = tree.simulate(node.state, self._color)
            tree.backpropagate(node, reward, self._color)
            iterations += 1

        print(f"Agent ({self._color}): {iterations} MCTS iterations in {time_budget:.2f}s")
        return tree.best_action()
    
    
    def _turn_budget(self, time_remaining: float | None) -> float:
        if time_remaining is not None:
            available = max(time_remaining - TIME_BUFFER, 0.0)
        else:
            elapsed   = time.time() - self._start_time
            available = max(TOTAL_TIME_BUDGET - elapsed - TIME_BUFFER, 0.0)

        budget = available * TIME_FRACTION_PER_TURN
        return max(MIN_TIME_PER_TURN, min(MAX_TIME_PER_TURN, budget))