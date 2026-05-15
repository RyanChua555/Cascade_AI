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

# Fraction of remaining time to spend on each individual MCTS call.
TIME_FRACTION_PER_TURN = 0.05

# Hard cap per turn so early turns don't burn too much time on their own.
MAX_TIME_PER_TURN = 5.0

# Minimum time to give MCTS even in very late games.
MIN_TIME_PER_TURN = 0.1


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
            # Placement phase: can place on empty cells not adjacent to
            # opponent stacks.
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

        # Play phase: generate MOVE, EAT, and CASCADE actions.
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

class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Cascade game events.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        self._color = color
        self._game_state = GameState(initial_player=PlayerColor.RED)

        self._start_time = time.time()
        self._time_used = 0.0

        print(f"MCTS agent initialised as {color}")


    def action(self, **referee: dict) -> Action:
        """
        Called by the referee at the start of our turn.  Returns the chosen
        action.
        """
        time_remaining: float | None = referee.get("time_remaining", None)  # type: ignore

        # --- Play phase: run MCTS ---
        budget = self.turn_budget(time_remaining)
        action = self.run_mcts(budget)

        if action is None:
            # Fallback: pick a random legal action so we never return none
            legal = self._game_state.get_legal_actions()
            action = random.choice(legal) if legal else MoveAction(Coord(0, 0), Direction.Down)
            print(f"MCTS agent ({self._color}): fallback random action {action}")
        else:
            print(f"MCTS agent ({self._color}): MCTS chose {action}")

        return action

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Called by the referee after every turn (ours and the opponent's).
        Keep our internal game state in sync.
        """
        self._game_state.apply_action(action)


    def turn_budget(self, time_remaining: float | None) -> float:
        """
        Compute how many seconds to give to MCTS this turn.

        Uses the time_remaining hint from the referee when available.
        Falls back to a conservative estimate based on elapsed wall time.
        """
        if time_remaining is not None:
            available = max(time_remaining - TIME_BUFFER, 0.0)
        else:
            elapsed = time.time() - self._start_time
            available = max(TOTAL_TIME_BUDGET - elapsed - TIME_BUFFER, 0.0)

        budget = available * TIME_FRACTION_PER_TURN
        budget = max(MIN_TIME_PER_TURN, min(MAX_TIME_PER_TURN, budget))
        return budget

    def run_mcts(self, time_budget: float) -> Action | None:
        """
        Run the four-phase MCTS loop for `time_budget` seconds and return
        the best action found.

        """
        # Import here to avoid a circular import at module level.
        from .mcts import MCTSTree

        root_state = self._game_state.clone()
        tree = MCTSTree(root_state)

        # Initialise the root's untried actions.
        tree.root.untried_actions = root_state.get_legal_actions()

        # If there is only one legal action, skip MCTS entirely.
        if len(tree.root.untried_actions) == 1:
            return tree.root.untried_actions[0]

        deadline = time.time() + time_budget
        iterations = 0

        while time.time() < deadline:
            # ---- 1. Selection ----
            node = tree.selection(tree.root)

            # ---- 2. Expansion ----
            # If the node is not terminal and has untried actions, expand one.
            if not node.state.game_over and node.untried_actions:
                action = node.untried_actions.pop()
                next_state = node.state.clone()
                next_state.apply_action(action)

                child = tree.expand(node, action, next_state)

                # Initialise the child's untried actions for future expansion.
                child.untried_actions = next_state.get_legal_actions()

                node = child


            # ---- 3. Simulation ----
            reward = tree.simulate(node.state)

            # ---- 4. Backpropagation ----
            tree.backpropagate(node, reward)

            iterations += 1

        print(f"MCTS agent ({self._color}): {iterations} iterations in {time_budget:.2f}s")

        return tree.best_action()