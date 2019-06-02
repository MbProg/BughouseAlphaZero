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
        board._bughouse_boards.boards[board.other_board_id].pockets = copy.deepcopy(self.pockets)
        board.pockets = copy.deepcopy(self.pockets_self)


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
        # White PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING then Black PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING
        board_state = np.zeros((12, 8, 8))
        rank = 0
        file = 0
        for square in chess.SQUARES_180:
            mask = chess.BB_SQUARES[square]
            if not self.occupied & mask:
                pass
            elif self.pawns & mask:
                color = bool(self.occupied_co[chess.WHITE] & mask)
                fig = chess.PAWN + ((not color)*6) - 1
                board_state[fig, rank, file] = 1
            elif self.knights & mask:
                color = bool(self.occupied_co[chess.WHITE] & mask)
                fig = chess.KNIGHT + ((not color)*6) - 1
                board_state[fig, rank, file] = 1
            elif self.bishops & mask:
                color = bool(self.occupied_co[chess.WHITE] & mask)
                fig = chess.BISHOP + ((not color)*6) - 1
                board_state[fig, rank, file] = 1
            elif self.rooks & mask:
                color = bool(self.occupied_co[chess.WHITE] & mask)
                fig = chess.ROOK + ((not color)*6) - 1
                board_state[fig, rank, file] = 1
            elif self.queens & mask:
                color = bool(self.occupied_co[chess.WHITE] & mask)
                fig = chess.QUEEN + ((not color)*6) - 1
                board_state[fig, rank, file] = 1
            elif self.kings & mask:
                color = bool(self.occupied_co[chess.WHITE] & mask)
                fig = chess.KING + ((not color)*6) - 1
                board_state[fig, rank, file] = 1
            file  += 1
            if file >= 8:
                file = 0
                rank += 1
        if flip:
            return np.flip(np.flip(board_state, 1), 2)
        return board_state

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

    def reset_boards(self, board_id: int):
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
        return copy.deepcopy(self.boards[self.LEFT].pockets), copy.deepcopy(self.boards[self.RIGHT].pockets)

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
        return np.concatenate((self.boards[0].to_numpy(False), self.boards[1].to_numpy(flip)), axis=1)

    def to_numpy_simplified(self, flip: bool):
        return np.concatenate((self.boards[0].to_numpy_simplified(False), self.boards[1].to_numpy_simplified(flip)),
                              axis=1)

    def get_pockets_numpy(self):
        # oder of Pocket LEFT, RIGHT
        # order of figurs
        # White PAWN, KNIGHT, BISHOP, ROOK, QUEEN then Black PAWN, KNIGHT, BISHOP, ROOK, QUEEN
        ret_pockets = np.zeros((2, 10))

        for pt, count in self.boards[self.LEFT].pockets[chess.WHITE].pieces.items():
            ret_pockets[self.LEFT, pt-1] = count

        for pt, count in self.boards[self.LEFT].pockets[chess.BLACK].pieces.items():
            ret_pockets[self.LEFT, pt+4] = count

        for pt, count in self.boards[self.RIGHT].pockets[chess.WHITE].pieces.items():
            ret_pockets[self.LEFT, pt - 1] = count

        for pt, count in self.boards[self.RIGHT].pockets[chess.BLACK].pieces.items():
            ret_pockets[self.LEFT, pt + 4] = count

        return ret_pockets