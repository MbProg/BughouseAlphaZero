import chess.variant
import copy
import numpy as np
import copy

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
        super().set_fen(fen)

    def to_numpy_single(self, player_color):
        #order of figurs
        # Black = 0, White = 1
        # PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING

        # Piece postions on the board and promoted pieces
        piece_mat = np.zeros((2, 6, 8, 8))
        promoted_mat = np.zeros((2, 8, 8))  # First White then Black

        rank = 0
        file = 0
        # check each tile for figures and promotion
        # correct postionion so that current player get pos 0 and opponent pos 1
        #  Get pieces and promoted
        for square in chess.SQUARES_180:
            mask = chess.BB_SQUARES[square]
            if self.occupied & mask:
                if self.pawns & mask:
                    color = int(bool(self.occupied_co[chess.WHITE] & mask) != player_color)
                    fig = chess.PAWN - 1
                    piece_mat[color, fig, rank, file] = 1
                elif self.knights & mask:
                    color = int(bool(self.occupied_co[chess.WHITE] & mask) != player_color)
                    fig = chess.KNIGHT - 1
                    piece_mat[color, fig, rank, file] = 1
                elif self.bishops & mask:
                    color = int(bool(self.occupied_co[chess.WHITE] & mask) != player_color)
                    fig = chess.BISHOP - 1
                    piece_mat[color, fig, rank, file] = 1
                elif self.rooks & mask:
                    color = int(bool(self.occupied_co[chess.WHITE] & mask) != player_color)
                    fig = chess.ROOK - 1
                    piece_mat[color, fig, rank, file] = 1
                elif self.queens & mask:
                    color = int(bool(self.occupied_co[chess.WHITE] & mask) != player_color)
                    fig = chess.QUEEN - 1
                    piece_mat[color, fig, rank, file] = 1
                elif self.kings & mask:
                    color = int(bool(self.occupied_co[chess.WHITE] & mask) != player_color)
                    fig = chess.KING - 1
                    piece_mat[color, fig, rank, file] = 1
            file += 1
            if file >= 8:
                file = 0
                rank += 1

        # Fill promotion Mat
        index_toggle = True # Controls the order of the matrices, (False if player is White)
        if player_color == chess.WHITE:
            index_toggle = not index_toggle

        for square in chess.SquareSet(self.promoted):
            row = square // 8
            col = square % 8
            row = 7 - row
            if self.piece_at(square).color == chess.WHITE:
                promoted_mat[int(index_toggle), row, col] = 1
            else:
                promoted_mat[int(not index_toggle), row, col] = 1
        if player_color == chess.BLACK:
            piece_mat = np.flip(np.flip(piece_mat, 2), 3)
            promoted_mat = np.flip(np.flip(promoted_mat, 1), 2)
            # en_passent_mat = np.flip(np.flip(en_passent_mat, 1), 2)

        piece_mat = np.concatenate(piece_mat, axis=0)
        # Get Pockets
        pocket_mat = self.get_pockets_numpy(player_color) # Flip if current is WHITE

        # Team mat
        color_mat = np.full((8, 8), int(self.turn))

        # check the move count of the players
        max_moves = 100.0
        norm_moves = 1.0 if self.fullmove_number > max_moves else self.fullmove_number / max_moves
        movecount_mat = np.full((8, 8), norm_moves)

        #check castling rights
        castling_mat = self.get_castling_mat(player_color)
        return piece_mat, pocket_mat, color_mat, movecount_mat, promoted_mat, castling_mat

    def get_pockets_numpy(self, player):
        # oder of Pocket LEFT, RIGHT
        # order of colors WHITE = 1, BLACK = 0
        # White PAWN, KNIGHT, BISHOP, ROOK, QUEEN then Black PAWN, KNIGHT, BISHOP, ROOK, QUEEN
        # 8x8 is network input size
        ret_pockets = np.zeros((2, 5, 8, 8))
        index_toggle = True # Controls the order of the matrices, (False if player is White)
        if player == chess.WHITE:
            index_toggle = not index_toggle

        for pt, count in self.pockets[chess.WHITE].pieces.items():
            if pt ==chess.PAWN:
                ret_pockets[int(index_toggle), pt - 1, :, :] = float(count) / 8.0
            if pt == chess.KNIGHT:
                ret_pockets[int(index_toggle), pt - 1, :, :] = float(count) / 2.0
            if pt == chess.BISHOP:
                ret_pockets[int(index_toggle), pt - 1, :, :] = float(count) / 2.0
            if pt == chess.ROOK:
                ret_pockets[int(index_toggle), pt - 1, :, :] = float(count) / 2.0
            if pt == chess.QUEEN:
                ret_pockets[int(index_toggle), pt - 1, :, :] = float(count)
        for pt, count in self.pockets[chess.BLACK].pieces.items():
            if pt ==chess.PAWN:
                ret_pockets[int(not index_toggle), pt - 1, :, :] = float(count) / 8.0
            if pt == chess.KNIGHT:
                ret_pockets[int(not index_toggle), pt - 1, :, :] = float(count) / 2.0
            if pt == chess.BISHOP:
                ret_pockets[int(not index_toggle), pt - 1, :, :] = float(count) / 2.0
            if pt == chess.ROOK:
                ret_pockets[int(not index_toggle), pt - 1, :, :] = float(count) / 2.0
            if pt == chess.QUEEN:
                ret_pockets[int(not index_toggle), pt - 1, :, :] = float(count)
        return np.concatenate(ret_pockets, axis = 0)

    def get_castling_mat(self, player: bool):
        #check castling rights
        castling_mat = np.zeros((2, 2, 8, 8))
        # WHITE
        if player: #check if player is WHITE
            if bool(self.castling_rights & chess.BB_H1) is True:
            # White can castle with the h1 rook
                castling_mat[0, 0, :, :] = 1
            # check for Queen Side Castling
            if bool(self.castling_rights & chess.BB_A1) is True:
                castling_mat[0, 1, :, :] = 1
            # BLACK
            # check for King Side Castling
            if bool(self.castling_rights & chess.BB_H8) is True:
                # White can castle with the h1 rook
                castling_mat[1, 0, :, :] = 1
            # check for Queen Side Castling
            if bool(self.castling_rights & chess.BB_A8) is True:
                castling_mat[1, 1, :, :] = 1
        else:
            if bool(self.castling_rights & chess.BB_H1) is True:
                castling_mat[1, 0, :, :] = 1
            if bool(self.castling_rights & chess.BB_A1) is True:
                castling_mat[1, 1, :, :] = 1
            if bool(self.castling_rights & chess.BB_H8) is True:
                castling_mat[0, 0, :, :] = 1
            if bool(self.castling_rights & chess.BB_A8) is True:
                castling_mat[0, 1, :, :] = 1
        return np.concatenate(castling_mat, axis = 0)

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

    def print_board(self, flip: bool):
        print_board = np.full((8,8),'♙')
        rank = 0
        file = 0
        for square in chess.SQUARES_180:
            fig = '_'
            mask = chess.BB_SQUARES[square]
            if not self.occupied & mask:
                fig = '_'
            elif self.pawns & mask:
                color = bool(self.occupied_co[chess.WHITE] & mask)
                if color:
                    fig = '♙'
                else:
                    fig = '♟'
            elif self.knights & mask:
                color = bool(self.occupied_co[chess.WHITE] & mask)
                if color:
                    fig = '♘'
                else:
                    fig = '♞'
            elif self.bishops & mask:
                color = bool(self.occupied_co[chess.WHITE] & mask)
                if color:
                    fig = '♗'
                else:
                    fig = '♝'
            elif self.rooks & mask:
                color = bool(self.occupied_co[chess.WHITE] & mask)
                if color:
                    fig = '♖'
                else:
                    fig = '♜'
            elif self.queens & mask:
                color = bool(self.occupied_co[chess.WHITE] & mask)
                if color:
                    fig = '♕'
                else:
                    fig = '♛'
            elif self.kings & mask:
                color = bool(self.occupied_co[chess.WHITE] & mask)
                if color:
                    fig = '♔'
                else:
                    fig = '♚'
            print_board[rank, file] = fig
            file  += 1
            if file >= 8:
                file = 0
                rank += 1
        if flip:
            return np.flip(np.flip(print_board, 0), 1)
        return print_board

    def fifty_moves_rule(self):
        return True if self.halfmove_clock >= 50 else False


class BughouseBoards:
    aliases = ["Bughouse", "Bug House", "BH"]
    uci_variant = "bughouse"
    xboard_variant = "bughouse"
    starting_fen = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1",
                    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"]

    tbw_suffix = tbz_suffix = None
    tbw_magic = tbz_magic = None
    TEAMS = [BOTTOM, TOP] = [TEAM_A, TEAM_B] = [0, 1]
    BOARDS = [LEFT, RIGHT] = [BOARD_A, BOARD_B] = [0, 1]

    def __init__(self, fen=starting_fen, chess960=False):
        self.boards = [_BughouseBoard(self, 1, fen[0]),_BughouseBoard(self, 0, fen[1])]

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

    def fen(self, promoted:bool =None):
        f = self.boards[0].fen()
        fen = [self.boards[0].fen(), self.boards[1].fen()]
        #ToDo add board_fen = [fen[0].split(" ", 1)[0], fen[1].split(" ", 1)[0]] for history
        return fen

    def board_fen(self, promoted=None):
        board_fen = [self.boards[0].board_fen(), self.boards[1].board_fen()]
        return board_fen

    def is_checkmate(self, board = None) -> bool:
        return self.boards[self.LEFT].is_checkmate() or self.boards[self.RIGHT].is_checkmate()

    def fifty_moves_rule(self, board = None):
        if board is None:
            return self.boards[self.LEFT].fifty_moves_rule() or self.boards[self.RIGHT].fifty_moves_rule()
        else:
            return self.boards[board].fifty_moves_rule()


    def is_threefold_repetition(self, board = None):
        if board is None:
            return self.boards[self.LEFT].is_repetition(3) or self.boards[self.RIGHT].is_repetition(3)
        else:
            self.boards[board].is_repetition(3)

    def is_game_over(self, board = None) -> bool:
        if board is None:
            return (self.is_checkmate() or self.possible_stalemate() or self.fifty_moves_rule())
        else:
            return (self.is_checkmate(board) or self.possible_stalemate(board) or self.fifty_moves_rule(board))

    def possible_stalemate(self, board = None) -> bool:
        if board is None:
            return not (any(self.boards[self.LEFT].generate_legal_moves())
                                           or any(self.boards[self.RIGHT].generate_legal_moves()))
        else:
            return not any(self.boards[board].generate_legal_moves())

    def result(self, board = None):
        # the returned arrays constist of [Left Board, Right Board, Game has ended]
        # score of 1 means White wins, -1 means Black wins and 0 is a draw
        # Check external influnce like time or withdraw

        # Checkmate
        if board is None:
            if self.is_checkmate():
                if self.boards[self.LEFT].is_checkmate():
                    return np.asarray([-1, 0, 1]) if self.boards[self.LEFT].turn == chess.WHITE \
                        else np.asarray([1, 0, 1])
                else:
                    return np.asarray([-1, 0, 1]) if self.boards[self.RIGHT].turn == chess.WHITE \
                        else np.asarray([1, 0, 1])
            # Check for a draw
            if (self.is_threefold_repetition() or self.fifty_moves_rule() or  self.possible_stalemate()):
                return np.asarray([0, 0, 1])
            # Still ongoing
            return np.asarray([0, 0, 0])
        else:
            if self.is_checkmate(board):
                return -1 if self.boards[board].turn == chess.WHITE else 1
            # Check for a draw
            if (self.is_threefold_repetition(board) or self.fifty_moves_rule(board) or  self.possible_stalemate(board)):
                return 0
    def set_result(self, board, score):
        self.boards[board]._end_game(score)

    def generate_legal_moves(self, board):
        return self.boards[board].generate_legal_moves()

    def to_numpy(self, board: int, player_color: int = None):
        # Flip to get both player on the same side
        # main board
        other_board = 1 if board == 0 else 0
        my_color = self.boards[board].turn
        if player_color is not None:
            my_color = player_color
        partner_color = not my_color
        piece_mat_m, pocket_mat_m, color_mat_m, movecount_mat_m, promoted_mat_m, castling_mat_m =\
            self.boards[board].to_numpy_single(my_color)
        piece_mat_p, pocket_mat_p, color_mat_p, movecount_mat_p, promoted_mat_p, castling_mat_p =\
            self.boards[other_board].to_numpy_single(partner_color)
        ret_mat = np.concatenate([piece_mat_m,piece_mat_p, pocket_mat_m, pocket_mat_p, np.stack([color_mat_m, color_mat_p]), np.stack([movecount_mat_m, movecount_mat_p]), promoted_mat_m, promoted_mat_p, castling_mat_m, castling_mat_p])
        return ret_mat

    def to_numpy_simplified(self, flip: bool):
        return np.concatenate((self.boards[0].to_numpy_simplified(False), self.boards[1].to_numpy_simplified(flip)),
                              axis=1)
    def __copy__(self):
        newone = type(self)()
        newone.boards = self.boards.copy()
        return newone

    def __deepcopy__(self, memodict={}):
        newone = type(self)()
        for i in range(2):
            newone.boards[i].pawns = self.boards[i].pawns
            newone.boards[i].knights = self.boards[i].knights
            newone.boards[i].bishops = self.boards[i].bishops
            newone.boards[i].rooks = self.boards[i].rooks
            newone.boards[i].queens = self.boards[i].queens
            newone.boards[i].kings = self.boards[i].kings

            newone.boards[i].occupied_co[chess.WHITE] = self.boards[i].occupied_co[chess.WHITE]
            newone.boards[i].occupied_co[chess.BLACK] = self.boards[i].occupied_co[chess.BLACK]
            newone.boards[i].occupied = self.boards[i].occupied
            newone.boards[i].promoted = self.boards[i].promoted
            newone.boards[i].pockets[chess.WHITE] = self.boards[i].pockets[chess.WHITE].copy()
            newone.boards[i].pockets[chess.BLACK] = self.boards[i].pockets[chess.BLACK].copy()
        return newone

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