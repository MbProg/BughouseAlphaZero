import chess.variant
import copy
import numpy as np

class _BughouseBoardState:
    def __init__(self, board):
        self.board_state = chess._BoardState(board)
        self.pockets = board._bughouse_boards.boards[board.other_board_id].pockets.copy()
        self.pockets_self = board.pockets.copy()

    def restore(self, board):
        self.board_state.restore(board)
        board._bughouse_boards.boards[board.other_board_id].pockets = self.pockets.copy()
        board.pockets = self.pockets_self.copy()


class _BughouseBoard(chess.variant.CrazyhouseBoard):
    def __init__(self, bughouse_boards: "BughouseBoards", other_board_id, fen) -> None:
        self.other_board_id = other_board_id
        self._bughouse_boards = bughouse_boards
        self.score = 0  # 1 For white winning, -1 for Black winning
        self.finished = False
        super().__init__(fen)

    def _end_game(self, score = None):
        # ToDo incoperate in is_variant_end function?
        if score is not None:
            self.score = score
        self.finished = True

    def _board_state(self):
        return _BughouseBoardState(self)

    def _push_capture(self, move, capture_square, piece_type, was_promoted):
        pocket = not self.turn
        if was_promoted:
            self._bughouse_boards.boards[self.other_board_id].pockets[pocket].add(chess.PAWN)
        else:
            self._bughouse_boards.boards[self.other_board_id].pockets[pocket].add(piece_type)

    def set_fen(self, fen):
        super(chess.variant.CrazyhouseBoard, self).set_fen(fen)

    def to_numpy(self, flip: bool):
        #order of figurs
        # Black = 0, White = 1
        # PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING

        # Piece postions on the board and promoted pieces
        piece_mat = np.zeros((2, 6, 8, 8))
        promoted_mat = np.zeros((2, 8, 8))  # First White then Black
        rank = 0
        file = 0
        # check each tile for figures and promotion
        for square in chess.SQUARES_180:
            mask = chess.BB_SQUARES[square]
            if self.occupied & mask:
                if self.pawns & mask:
                    color = int(bool(self.occupied_co[chess.WHITE] & mask))
                    fig = chess.PAWN - 1
                    piece_mat[color, fig, rank, file] = 1
                elif self.knights & mask:
                    color = int(bool(self.occupied_co[chess.WHITE] & mask))
                    fig = chess.KNIGHT - 1
                    piece_mat[color, fig, rank, file] = 1
                elif self.bishops & mask:
                    color = int(bool(self.occupied_co[chess.WHITE] & mask))
                    fig = chess.BISHOP - 1
                    piece_mat[color, fig, rank, file] = 1
                elif self.rooks & mask:
                    color = int(bool(self.occupied_co[chess.WHITE] & mask))
                    fig = chess.ROOK - 1
                    piece_mat[color, fig, rank, file] = 1
                elif self.queens & mask:
                    color = int(bool(self.occupied_co[chess.WHITE] & mask))
                    fig = chess.QUEEN - 1
                    piece_mat[color, fig, rank, file] = 1
                elif self.kings & mask:
                    color = int(bool(self.occupied_co[chess.WHITE] & mask))
                    fig = chess.KING - 1
                    piece_mat[color, fig, rank, file] = 1
                # check for promotion
                if self.promoted & mask:
                    color = int(bool(self.occupied_co[chess.WHITE] & mask))
                    promoted_mat[color, rank, file]

            file  += 1
            if file >= 8:
                file = 0
                rank += 1

        # Get Pockets
        normalization = 8.0  # max number of pawns
        pocket_mat = self.get_pockets_numpy(normalization)

        # check which player is allowed to move
        player_mat = np.zeros((2, 8, 8))
        player_mat[int(self.turn), :, :] = 1 #Set Matrix to 1 for player that is allowed to move

        # check the move count of the players
        movecount_mat = np.full((8, 8), self.fullmove_number)

        #check castling rights
        castling_mat = np.zeros((2, 2, 8, 8))
        # WHITE
        # check for King Side Castling
        if bool(self.castling_rights & chess.BB_H1) is True:
            # White can castle with the h1 rook
            castling_mat[chess.WHITE, 0, :, :] = 1
        # check for Queen Side Castling
        if bool(self.castling_rights & chess.BB_A1) is True:
            castling_mat[chess.WHITE, 1, :, :] = 1

        # BLACK
        # check for King Side Castling
        if bool(self.castling_rights & chess.BB_H8) is True:
            # White can castle with the h1 rook
            castling_mat[chess.BLACK, 0, :, :] = 1
        # check for Queen Side Castling
        if bool(self.castling_rights & chess.BB_A8) is True:
            castling_mat[chess.BLACK, 1, :, :] = 1

        # ToDo implment flippping

        return piece_mat, pocket_mat, player_mat, movecount_mat, promoted_mat, castling_mat

    def get_pockets_numpy(self, normalization = 8):
        # oder of Pocket LEFT, RIGHT
        # order of colors WHITE = 1, BLACK = 0
        # White PAWN, KNIGHT, BISHOP, ROOK, QUEEN then Black PAWN, KNIGHT, BISHOP, ROOK, QUEEN
        # 8x8 is network input size
        ret_pockets = np.zeros((2, 5, 8, 8))
        for pt, count in self.pockets[chess.BLACK].pieces.items():
            ret_pockets[int(chess.BLACK), pt - 1, :, :] = float(count) / normalization
        for pt, count in self.pockets[chess.WHITE].pieces.items():
            ret_pockets[int(chess.WHITE), pt - 1, :, :] = float(count) / normalization
        return ret_pockets

    def to_numpy_simplified(self, flip: bool):
        board_state_game = np.zeros((8, 8))
        rank = 0
        file = 0
        for square in chess.SQUARES_180:
            fig = 0
            mask = chess.BB_SQUARES[square]
            if not self.occupied & mask:
                fig = 0
            elif self.pawns & mask:
                color = bool(self.occupied_co[chess.WHITE] & mask)
                fig = chess.PAWN + ((not color)*6)
            elif self.knights & mask:
                color = bool(self.occupied_co[chess.WHITE] & mask)
                fig = chess.KNIGHT + ((not color)*6)
            elif self.bishops & mask:
                color = bool(self.occupied_co[chess.WHITE] & mask)
                fig = chess.BISHOP + ((not color)*6)
            elif self.rooks & mask:
                color = bool(self.occupied_co[chess.WHITE] & mask)
                fig = chess.ROOK + ((not color)*6)
            elif self.queens & mask:
                color = bool(self.occupied_co[chess.WHITE] & mask)
                fig = chess.QUEEN + ((not color)*6)
            elif self.kings & mask:
                color = bool(self.occupied_co[chess.WHITE] & mask)
                fig = chess.KING + ((not color)*6)
            board_state_game[rank, file] = fig
            file  += 1
            if file >= 8:
                file = 0
                rank += 1
        if flip:
            return np.flip(np.flip(board_state_game, 0), 1)
        return board_state_game


class BughouseBoards:
    aliases = ["Bughouse", "Bug House", "BH"]
    uci_variant = "bughouse"
    xboard_variant = "bughouse"
    starting_fen = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1", #8/3K4/8/5k2/8/8/8/8 w KQkq - 0 1
                    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"]

    tbw_suffix = tbz_suffix = None
    tbw_magic = tbz_magic = None
    TEAMS = [BOTTOM, TOP] = [TEAM_A, TEAM_B] = [0, 1]
    BOARDS = [LEFT, RIGHT] = [BOARD_A, BOARD_B] = [0, 1]

    def __init__(self, fen=starting_fen, chess960=False):
        self.boards = [_BughouseBoard(self, 1, fen[0]),_BughouseBoard(self, 0, fen[1])]
        self.fen = fen

    def reset_boards(self):
        for board in self.boards:
            board.reset_board()

    def reset_board(self, board_id: int):
        self.boards[board_id].reset_board()

    def clear_boards(self):
        for board in self.boards:
            board.clear_board()

    def clear_boards(self, board_id: int):
        self.boards[board_id].clear_board()

    def push(self, move, board):
        """
        Makes a move on the selected board
        :param move:
        :return: The move
        """
        return self.boards[board].push(move)

    def push_san(self, san, board):
        return self.boards[board].push_san(san)

    def push_uci(self, uci, board):
        return self.boards[board].push_uci(uci)

    def can_claim_fifty_moves(self):
        return False

    def is_seventyfive_moves(self):
        return False

    def set_fen(self, fen):
        self.boards[0].set_fen(fen[0])
        self.boards[1].set_fen(fen[1])

    def set_pockets(self, pockets_left, pockets_right):
        self.boards[self.LEFT].pockets = copy.deepcopy(pockets_left)
        self.boards[self.RIGHT].pockets = copy.deepcopy(pockets_right)

    def get_pockets(self):
        return self.boards[self.LEFT].pockets.copy(), self.boards[self.RIGHT].pockets.copy()

    def board_fen(self, promoted=None):
        fen = [self.boards[0].board_fen(), self.boards[1].board_fen()]
        return fen

    def is_checkmate(self) -> bool:
        return self.boards[self.LEFT].is_checkmate() or self.boards[self.RIGHT].is_checkmate()

    def is_game_over(self) -> bool:
        return not (any(True for _ in self.boards[self.LEFT].legal_moves) or
                    any(True for _ in self.boards[self.RIGHT].legal_moves))

    def is_threefold_repetition(self):
        return self.boards[self.LEFT].is_repetition(3) and self.boards[self.RIGHT].is_repetition(3)

    def result(self):
        # ToDo check win conditions
        # the returned arrays constist of [Left Board, Right Board, Game has ended]
        # score of 1 means White wins, -1 means Black wins and 0 is a draw
        # Check external influnce like time or withdraw
        if self.boards[self.LEFT].finished or self.boards[self.RIGHT].finished:
            return np.asarray([self.boards[self.LEFT].score, self.boards[self.RIGHT].score, 1])
        # Checkmate
        if self.is_checkmate():
            if self.boards[self.LEFT].is_checkmate():
                return np.asarray([1, 0, 1]) if self.boards[self.LEFT].turn == chess.WHITE \
                    else np.asarray([-1, 0, 1])
            else:
                return np.asarray([1, 0, 1]) if self.boards[self.RIGHT].turn == chess.WHITE \
                    else np.asarray([-1, 0, 1])
        # Check for a draw
        if self.is_threefold_repetition():
            return np.asarray([0, 0, 1])
        # Still ongoing
        return np.asarray([0, 0, 0])

    def set_result(self, board, score):
        self.boards[board]._end_game(score)

    def generate_legal_moves(self, board):
        return self.boards[board].generate_legal_moves()

    def to_numpy(self, flip: bool):
        # Flip to get both player on the same side
        piece_mat_l, pocket_mat_l, player_mat_l, movecount_mat_l, promoted_mat_l, castling_mat_l =\
            self.boards[0].to_numpy(False)
        piece_mat_r, pocket_mat_r, player_mat_r, movecount_mat_r, promoted_mat_r, castling_mat_r = \
            self.boards[0].to_numpy(False)
        return np.array([piece_mat_l, piece_mat_r+1]), np.array([pocket_mat_l, pocket_mat_r+1]), \
               np.array([player_mat_l, player_mat_r+1]), np.array([movecount_mat_l, movecount_mat_r+1]), \
               np.array([promoted_mat_l, promoted_mat_r+1]),np.array([castling_mat_l, castling_mat_r+1])

    def to_numpy_simplified(self, flip: bool):
        return np.concatenate((self.boards[0].to_numpy_simplified(False), self.boards[1].to_numpy_simplified(flip)),
                              axis=1)

    # def get_pockets_numpy(self, create_mat:bool = True):
    #     # oder of Pocket LEFT, RIGHT
    #     # order of colors WHITE = 1, BLACK = 0
    #     # White PAWN, KNIGHT, BISHOP, ROOK, QUEEN then Black PAWN, KNIGHT, BISHOP, ROOK, QUEEN
    #     # 8x8 is network input size
    #     normalization = 8.0 # max number of pawns
    #     if create_mat:
    #         ret_pockets = np.zeros((2, 2, 5, 8, 8))
    #         for pt, count in self.boards[self.LEFT].pockets[chess.WHITE].pieces.items():
    #             ret_pockets[self.LEFT, chess.WHITE, pt-1, :, :] = float(count) / normalization
    #         for pt, count in self.boards[self.LEFT].pockets[chess.BLACK].pieces.items():
    #             ret_pockets[self.LEFT, chess.BLACK, pt-1, :, :] = float(count) / normalization
    #         for pt, count in self.boards[self.RIGHT].pockets[chess.WHITE].pieces.items():
    #             ret_pockets[self.RIGHT, chess.WHITE, pt-1, :, :] = float(count) / normalization
    #         for pt, count in self.boards[self.RIGHT].pockets[chess.BLACK].pieces.items():
    #             ret_pockets[self.RIGHT, chess.BLACK, pt-1, :, :] = float(count) / normalization
    #     else:
    #         ret_pockets = np.zeros((2, 2, 5))
    #         for pt, count in self.boards[self.LEFT].pockets[chess.WHITE].pieces.items():
    #             ret_pockets[self.LEFT, chess.WHITE, pt-1] = float(count) / normalization
    #         for pt, count in self.boards[self.LEFT].pockets[chess.BLACK].pieces.items():
    #             ret_pockets[self.LEFT, chess.BLACK, pt-1] = float(count) / normalization
    #         for pt, count in self.boards[self.RIGHT].pockets[chess.WHITE].pieces.items():
    #             ret_pockets[self.RIGHT, chess.WHITE, pt-1] = float(count) / normalization
    #         for pt, count in self.boards[self.RIGHT].pockets[chess.BLACK].pieces.items():
    #             ret_pockets[self.RIGHT, chess.BLACK, pt-1] = float(count) / normalization
    #
    #     return ret_pockets