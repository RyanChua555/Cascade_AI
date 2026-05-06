# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

import copy

from referee.game import PlayerColor, Coord, Direction, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction, Board, GamePhase, BOARD_N, CARDINAL_DIRECTIONS


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
            # Placement phase: can place on empty cells not adjacent to opponent stacks
            for r in range(BOARD_N):
                for c in range(BOARD_N):
                    coord = Coord(r, c)
                    if not self._board[coord].is_empty:
                        continue

                    # Check if adjacent to opponent stack
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

        # Play phase: generate MOVE, EAT, and CASCADE actions
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                coord = Coord(r, c)
                cell = self._board[coord]

                if cell.color != current_player:
                    continue

                # Try each cardinal direction for MOVE and EAT
                for direction in CARDINAL_DIRECTIONS:
                    dest_r = coord.r + direction.r
                    dest_c = coord.c + direction.c

                    # Check bounds
                    if not (0 <= dest_r < BOARD_N and 0 <= dest_c < BOARD_N):
                        continue

                    dest_coord = Coord(dest_r, dest_c)
                    dest_cell = self._board[dest_coord]

                    # MOVE: destination is empty or friendly stack
                    if dest_cell.is_empty or dest_cell.color == current_player:
                        actions.append(MoveAction(coord, direction))

                    # EAT: destination is enemy with height <= ours
                    if dest_cell.color == opponent and cell.height >= dest_cell.height:
                        actions.append(EatAction(coord, direction))

                # CASCADE: stack height >= 2, always valid in some direction
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
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        self._color = color
        self._turn_count = 0
        self._game_state = GameState()

        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as RED (first player)")
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")

    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object.
        """

        # Below we have hardcoded actions to be played depending on whether
        # the agent is playing as BLUE or RED. Obviously this won't work beyond
        # the initial moves of the game, so you should use some game playing
        # technique(s) to determine the best action to take.

        # During placement phase (first 8 turns total, 4 per player)
        if self._turn_count < 4:
            match self._color:
                case PlayerColor.RED:
                    print("Testing: RED is playing a PLACE action")
                    return PlaceAction(Coord(0, self._turn_count))
                case PlayerColor.BLUE:
                    print("Testing: BLUE is playing a PLACE action")
                    return PlaceAction(Coord(7, self._turn_count))

        # During play phase
        match self._color:
            case PlayerColor.RED:
                print("Testing: RED is playing a MOVE action")
                return MoveAction(Coord(0, 0), Direction.Down)
            case PlayerColor.BLUE:
                print("Testing: BLUE is playing a MOVE action")
                return MoveAction(Coord(7, 0), Direction.Up)

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state.
        """
        self._game_state.apply_action(action)
        if color == self._color:
            self._turn_count += 1

        # There are four possible action types: PLACE, MOVE, EAT, and CASCADE.
        # Below we check which type of action was played and print out the
        # details of the action for demonstration purposes. You should replace
        # this with your own logic to update your agent's internal game state.
        match action:
            case PlaceAction(coord):
                print(f"Testing: {color} played PLACE action at {coord}")
            case MoveAction(coord, direction):
                print(f"Testing: {color} played MOVE action:")
                print(f"  Coord: {coord}")
                print(f"  Direction: {direction}")
            case EatAction(coord, direction):
                print(f"Testing: {color} played EAT action:")
                print(f"  Coord: {coord}")
                print(f"  Direction: {direction}")
            case CascadeAction(coord, direction):
                print(f"Testing: {color} played CASCADE action:")
                print(f"  Coord: {coord}")
                print(f"  Direction: {direction}")
            case _:
                raise ValueError(f"Unknown action type: {action}")
