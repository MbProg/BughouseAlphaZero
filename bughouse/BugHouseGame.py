import sys
import numpy as np

sys.path.append('..')
from Game import Game
import bughouse.constants as constants
from bughouse.BughouseEnv import BughouseEnv


class BugHouseGame(Game):
    # we although in the base class everything works with board and we inherit the names
    # but we are using state instead of board
    def __init__(self, board=constants.BOTTOM, team=constants.LEFT, max_time=300):
        Game.__init__(self)
        self.environment = BughouseEnv(board, team, max_time)

    def getInitBoard(self, build_matrices=True):
        return self.environment.get_board_state(build_matrices=build_matrices)

    def getBoardSize(self):
        return (12, 16, 8)  #

    def getActionSize(self):
        return len(constants.LABELS)

    def getActionString(self, action):
        return constants.LABELS[action]

    def getActionNumber(self, actionString):
        for counter, elem in enumerate(constants.LABELS):
            if elem == actionString:
                return counter

    def getNextState(self, action, player=0, state=None, time=None, play_other_board = False, build_matrices=True,
                     boardView=False, player_view=False):
        if player_view:
            boardView = True
        state = None
        if state is not None:
            self.environment.load_state(state)
        if play_other_board is False:
            if time is not None:
                self.environment.set_time_remaining(time,None, self.environment.board)
            state = self.environment(constants.LABELS[action])
            return state, -player
        else:
            if time is not None:
                self.environment.set_time_remaining(time,None, int(not self.environment.board))
            self.environment.push(constants.LABELS[action], 0, int(not self.environment.board))
            if boardView:
                state = self.environment.get_board_state(build_matrices=build_matrices, player_view=player_view)
            else:
                state = self.environment.get_state(build_matrices=build_matrices)
            return state, player

    def getCurrentBoardState(self, build_matrices=True):
        return self.environment.get_board_state(build_matrices=build_matrices)
        if state is not None:
            self.environment.load_state(state)


    def setState(self, state):
        self.environment.load_state(state)

    def getValidMoves(self, state, player):

        "Any zero value in top row in a valid move"
        return np.array(list(self.environment.get_legal_moves_dict().values()))

    def getGameEnded(self, state, player):
        finished = self.environment.game_finished(self.environment.board)
        if finished:
            score = self.environment.get_score(self.environment.board)
            if score == 0:
                # draw has very little value.
                return 1e-4
            elif score == player:
                return +1
            elif score == -player:
                return -1
            else:
                raise ValueError('Unexpected winstate found: ', winstate)
        return 0

    def getCanonicalForm(self, state, player):
        return state

    def getSymmetries(self, state, pi):
        """Board is left/right board symmetric"""
        return [(state.board, pi)]

    def stringRepresentation(self, state):
        return str(state._fen[self.environment.board])


def printBughouse(agent):
    builder = []
    firstBoardRows = agent.boards.boards[0].__str__().split('\n')
    secondBoardRows = agent.boards.boards[1].__str__().split('\n')
    for counter, row in enumerate(firstBoardRows):
        builder.append(firstBoardRows[counter] + ' | ' + secondBoardRows[(len(secondBoardRows) - counter) - 1] + '\n')
    return ''.join(builder)


def display(state):
    agent.load_state(state)

    print(" -----------------------")
    print(_strPrintBughouse(agent))
    print(" -----------------------")
