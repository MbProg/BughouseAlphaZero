from collections import namedtuple
import numpy as np
import chess.variant


class _BughouseBoard(chess.variant.CrazyhouseBoard):
    def __init__(self, bughouse_boards: "BughouseBoards",other_board_id, fen) -> None:
        self.other_board_id = other_board_id
        self._bughouse_boards = bughouse_boards
        super().__init__(fen)




class Board():
    

    def __init__(self, fen=starting_fen, chess960=False):
        pass

