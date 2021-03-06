import warnings
import copy
import numpy as np
from bughouse.BugHouseBoard import BughouseBoards
import bughouse.constants as constants


class BughouseState(object):
    def __init__(self, matrices, time, team, board, _boards_fen):
        # old state
        # self.player = player # Player that is allowed to move next
        # self.board = board
        # self.pockets = pockets
        # self.time_remaining = time_remaining
        # variables for bord init
        # State matrices
        self.matrice_stack = matrices
        self.time_remaining = time
        self.team = team
        self.board = board
        self._fen = _boards_fen
        #self.communation = []# ToDo


class BughouseEnv():
    Color = bool
    COLORS = [WHITE, BLACK] = [True, False]
    COLOR_NAMES = ["black", "white"]

    PieceType = int
    PIECE_TYPES = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] = range(1, 7)
    PIECE_SYMBOLS = [None, "p", "n", "b", "r", "q", "k"]
    PIECE_NAMES = [None, "pawn", "knight", "bishop", "rook", "queen", "king"]

    TEAMS = [BOTTOM, TOP] = [TEAM_A, TEAM_B] = [0, 1]
    BOARDS = [LEFT, RIGHT] = [BOARD_A, BOARD_B] = [0, 1]
    MAXIMIUM_TIME_INIT = 300.0 # In Seconds


    def __init__(self, team=0, board=0, max_time = MAXIMIUM_TIME_INIT):
        self.max_time = max_time
        self.legal_moves = dict.fromkeys(constants.LABELS, 0)
        self.team = team
        self.board = board
        self.last_board_moved = board
        self.color = (team == board)  # Bottom left (00) is White(1) same as Top Right (11)
        self.boards = BughouseBoards()
        self.time_remaining = np.full((2, 2), self.max_time)  # (Teams, Boards)

    def __call__(self, action, build_matrices = True) -> BughouseState:
        return self.propapagete_board(action, build_matrices)

    def load_state(self, state : BughouseState):
        self.boards.reset_boards()
        self.time_remaining = state.time_remaining
        self.boards.set_fen(state._fen)

    def propapagete_board(self, action, build_matrices = True) -> BughouseState:
        self.push(action, self.team, self.board)  # Execute the action
        board = self.board
        #time = self.flip_time(board)
        time = self.time_remaining
        player_color = self.boards.boards[self.board].turn
        team = self.get_team(player_color, board)
        _fen = self.boards.fen()
        if build_matrices:
            return BughouseState(self.boards.to_numpy(self.board), time, team, board, _fen)
        else:
            return BughouseState(None, time, team, board, _fen)

    def get_state(self, board = None, team = None, build_matrices = True) -> BughouseState:
        _fen = self.boards.fen()
        time = self.time_remaining
        if not ((board is not None) and (team is not None)):
            board = self.last_board_moved
            player_color = self.boards.boards[board].turn
            team = self.get_team(player_color, board)
        if build_matrices:
            return BughouseState(self.boards.to_numpy(board), time, team, board, _fen)
        else:
            return BughouseState(None, time, team, board, _fen)

    def get_board_state(self, build_matrices=True, player_view=False) -> BughouseState:
        """
        Shows the state of the specified board independent on who moved last
        :return: BugHouseState object
        """
        _fen = self.boards.fen()
        player_color = self.boards.boards[self.board].turn
        time = self.time_remaining
        team = self.get_team(player_color, self.board)
        view_color = int(self.color) if player_view else None
        if build_matrices:
            return BughouseState(self.boards.to_numpy(self.board, view_color), time, team, self.board, _fen)
        else:
            return BughouseState(None, time, team, self.board, _fen)

    def game_finished(self, board = None):
        """
        Check if the game has finished
        :return: Bool
        """

        return self.boards.is_game_over(board)

    def get_score(self, board = None):
        """
        Checks the result of a game
        Score: 1 White win, -1 Black Win, 0 Draw
        :return: Array [Left Board result, Right Board result, game finished]
        """
        return self.boards.result(board)

    def set_time_remaining(self, time, team=None, board=0):
        """
        Update the time for the last player that made a move on a specified board

        :param time:
        :param team:
        :param board:
        :return:
        """
        if team is None:
            player_color = self.boards.boards[board].turn
            team = self.get_team(player_color, board)
            self.time_remaining[team, board] = time
        else:
            self.time_remaining[team, board] = time

    def push(self, uci_move: str, team, board):
        """
        Make a move for any player

        :param uci_move:
        :param team:
        :param board:
        :return:
        """
        self.last_board_moved = board
        self.boards.boards[board].push_uci(uci_move)

    def push_san(self, san_move: str, team, board, to_uci: bool = False):
        """
        Make a move for any player, input is short algebraic notation (san)

        :param san_move:
        :param team:
        :param board:
        :param to_uci: return the converted san move as uci
        :return:
        """
        self.last_board_moved = board
        move = self.boards.boards[board].parse_san(san_move)
        self.boards.boards[board].push(move)
        if to_uci:
            return move.uci()

    def san_to_uci(self, san_move: str, team, board):
        move = self.boards.boards[board].parse_san(san_move)
        return move.uci()


    def push_action(self, action: int, board):
        self.push(constants.LABELS[action], 0, board)

    def get_legal_moves_dict(self, side=None):
        """
        Returns a dictionary of legal moves
        :return:
        """
        board = side
        if board is None:
            board = self.board
        legal_moves = copy.deepcopy(self.legal_moves)
        for lm in self.boards.generate_legal_moves(board):
            legal_moves[lm.uci()] = 1
        return legal_moves

    def get_legal_moves_vector(self, side=None):
        """
        Returns a value vector of legal moves wihtout the keys
        :return:
        """
        board = side
        if board is None:
            board = self.board
        legal_moves = copy.deepcopy(self.legal_moves)
        for lm in self.boards.generate_legal_moves(board):
            legal_moves[lm.uci()] = 1
        return legal_moves.values()

    def reset(self):
        self.boards.reset_boards() #Reset Board
        self.time_remaining = np.full((2, 2), self.max_time) #  Reset Time

    def get_player_color(self,player, board):
        return int(player == board)

    def get_team(self,player_color, board):
        return int(player_color == board)

    def flip_time(self, board):
        #  Truns the time representation so that the active player is at the bottom left
        time_remaining = np.zeros((2, 2)) # (Teams, Boards)
        player_color = self.boards.boards[self.board].turn
        team = self.get_team(player_color, board)
        # My board
        time_remaining[0, 0] = time_remaining[int(team), int(board)]
        time_remaining[1, 0] = time_remaining[int(not team), int(board)]
        # Partner board
        time_remaining[0, 1] = time_remaining[int(team), int(not board)]
        time_remaining[1, 1] = time_remaining[int(not team), int(not board)]



