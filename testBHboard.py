import numpy as np
import chess
import chess.variant
from BughouseEnv import BughouseEnv

# Mini Example for Agent
b = chess.Board()
fen= b.fen()

agent = BughouseEnv(0, 0)
state = agent('a2a3')
agent2 = BughouseEnv(0, 0)
agent2.load_state(state)
moves = agent2.get_legal_moves_dict()
for key, value in moves.items():
    if value == 1:
        print(key)

agent = BughouseEnv(0, 0)
moves = agent.get_legal_moves_dict()
start = agent.get_state() # genrate a state
for key, value in moves.items():
    if value == 1:
        print(key)
state2 = agent("a2a3")
state = agent.get_state()
agent2 = BughouseEnv(0, 0)
agent2.load_state(state)
print("______")
moves = agent2.get_legal_moves_dict()
for key, value in moves.items():
    if value == 1:
        print(key)
game_over = agent.game_finished()
score = agent.get_score()
print(agent.boards.boards[0])
board = chess.variant.CrazyhouseBoard()
agent.load_state(start)
print(agent.boards.boards[0])
state = agent("a2a3")
moves = agent.get_legal_moves_dict()
for key, value in moves.items():
    if value == 1:
        print(key)
print()

#
# from BugHouseBoard import BughouseBoards
#
# chess.BB_ALL
#
# legal_moves = {}
# for o in chess.SQUARE_NAMES:
#     for t in chess.SQUARE_NAMES:
#         if o != t:
#             legal_moves[o+t] = 0
# #promotion moves
# #placement moves
#
#
# bh_boards = BughouseBoards()
# for lm in bh_boards.generate_legal_moves(BughouseBoards.LEFT):
#     legal_moves[lm.uci()] = 1
#     print(lm.uci())
# print(bh_boards.generate_legal_moves(BughouseBoards.LEFT))
# print(bh_boards.generate_legal_moves(BughouseBoards.RIGHT))
# bh_boards.boards[0].pockets[chess.WHITE].add(chess.QUEEN)
# #bh_boards.boards[0].pockets[chess.WHITE].add(chess.PAWN)
# #bh_boards.boards[0].pockets[chess.WHITE].add(chess.PAWN)
# print(bh_boards.generate_legal_moves(BughouseBoards.LEFT))
# print(bh_boards.generate_legal_moves(BughouseBoards.RIGHT))
# print(bh_boards.get_win_status())
# for lm in bh_boards.generate_legal_moves(BughouseBoards.LEFT):
#     legal_moves[lm.uci()] = 1
#     print(lm.uci())
# bh_boards.push_uci("Q@h6",BughouseBoards.LEFT)
# print(bh_boards.boards[0])
# fen = bh_boards.board_fen()
# bh_boards2 = BughouseBoards()
# print(bh_boards2.boards[0])
# bh_boards2.set_fen(fen)
# print(bh_boards2.boards[0])
# print(bh_boards.boards[0].legal_moves)
# bh_boards2.boards[0].turn = bh_boards.boards[0].turn
# print(bh_boards2.boards[0].legal_moves)
# bs = bh_boards.to_numpy_simplified(True)
# bh_boards.boards[0].pockets[chess.WHITE].add(chess.PAWN)
# bh_boards.boards[0].pockets[chess.WHITE].add(chess.KNIGHT)
# bh_boards.boards[0].pockets[chess.WHITE].add(chess.BISHOP)
# bh_boards.boards[0].pockets[chess.WHITE].add(chess.ROOK)
# bh_boards.boards[0].pockets[chess.WHITE].add(chess.QUEEN)
# ret = bh_boards.get_pockets_numpy()
# bh_boards.boards[0].is_checkmate()
# print()


