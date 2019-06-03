import warnings
import copy
import numpy as np
from bughouse.BugHouseBoard import BughouseBoards
import bughouse.constants as constants


class BughouseState(object):
    def __init__(self, matrices, time, _pockets_left, _pockets_right, _boards_fen):
        # old state
        # self.player = player # Player that is allowed to move next
        # self.board = board
        # self.pockets = pockets
        # self.time_remaining = time_remaining
        # variables for bord init
        # State matrices
        self.piece_mat, self.pocket_mat, self.player_mat, self.movecount_mat, self.promoted_mat,\
        self.castling_mat = matrices
        self.time_remaining = time
        self._pockets_left =_pockets_left
        self._pockets_right = _pockets_right
        self._boards_fen = _boards_fen
        #self.communation = []# ToDo

    def getStackedNumpyArray(self):
        castling_mat = np.concatenate(self.castling_mat,axis=0)
        castling_mat = np.concatenate(castling_mat,axis=0)
        piece_mat = np.concatenate(self.piece_mat,axis=0)
        piece_mat = np.concatenate(piece_mat,axis=0)
        player_mat = np.concatenate(self.player_mat,axis=0)
        pocket_mat = np.concatenate(self.pocket_mat,axis=0)
        pocket_mat = np.concatenate(pocket_mat,axis=0)
        promoted_mat = np.concatenate(self.promoted_mat,axis=0)
        wholeStackedState = np.concatenate([piece_mat,pocket_mat,player_mat,self.movecount_mat,promoted_mat,castling_mat])
        return wholeStackedState

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
    MAX_TIME = 600 # In Seconds ToDo get real time

    def __init__(self, team=0, board=0):
        self.legal_moves = dict.fromkeys(constants.LABELS, 0)
        self.team = team
        self.board = board
        self.color = (team == board)  # Bottom left (00) is White(1) same as Top Right (11)
        self.boards = BughouseBoards()
        self.time_remaining = np.full((2, 2), self.MAX_TIME) # (Teams, Boards)

    def __call__(self, action) -> BughouseState:
        return self.propapagete_board(action)

    def load_state(self, state : BughouseState):
        self.time_remaining = state.time_remaining
        self.boards.set_fen(state._boards_fen)
        self.boards.boards[self.board].turn = state.player
        self.boards.set_pockets(state._pockets_left,state._pockets_right)

    def propapagete_board(self, action) -> BughouseState:
        self.push(action, self.team, self.board)  # Execute the action
        time = self.time_remaining
        _boards_fen = self.boards.board_fen()
        _pockets_left, _pockets_right = self.boards.get_pockets()
        return BughouseState(self.boards.to_numpy(flip=False), time, _pockets_left, _pockets_right, _boards_fen)

    def get_state(self):
        time = self.time_remaining
        _boards_fen = self.boards.board_fen()
        _pockets_left, _pockets_right = self.boards.get_pockets()
        return BughouseState(self.boards.to_numpy(flip=False), time, _pockets_left, _pockets_right, _boards_fen)

    def game_finished(self):
        """
        Check if the game has finished
        :return: Bool
        """

        return self.boards.is_game_over()

    def get_score(self):
        """
        Checks the result of a game
        Score: 1 White win, -1 Black Win, 0 Draw
        :return: Array [Left Board result, Right Board result, game finished]
        """
        return self.boards.result()

    def set_time_remaining(self, time, team, board):
        """
        Used to set the remaining time

        :param time:
        :param team:
        :param board:
        :return:
        """
        self.time_remaining[team, board] = time

    def push(self, uci_move: str, team, board):
        """
        Make a move for any player

        :param uci_move:
        :param team:
        :param board:
        :return:
        """
        self.boards.boards[board].push_uci(uci_move)
        # color = self.boards.boards[board].turn
        # if color == (team == board):
        #     self.boards.boards[board].push_uci(uci_move)
        # else:
        #     warnings.warn("Warning")

    def push_san(self, san_move: str, team, board, to_uci: bool = False):
        """
        Make a move for any player, input is short algebraic notation (san)

        :param san_move:
        :param team:
        :param board:
        :param to_uci: return the converted san move as uci
        :return:
        """
        move = self.boards.boards[board].parse_san(san_move)
        self.boards.boards[board].push(move)
        if to_uci:
            return move.uci()

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
        self.time_remaining = np.full((2, 2), self.MAX_TIME) #  Reset Time

