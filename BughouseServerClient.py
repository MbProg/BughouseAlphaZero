import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time
from Arena import Arena
from MCTS import MCTS
from websocket import create_connection
import re


class BughouseServerClient():

    def __init__(self, game, nnet, args, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.args = args
        self.game = game
        self.display = display
        self.nnet = nnet
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.game = game
        self.display = display
        self.inputBox = []
        self.myTurn = False
        self.curPlayer = 1

    ########################################################
    # i) two words 'move e2e4' -> extract only the second word
    def extractFromMoveFEN(self, moveString):
        if moveString.startswith('move'):
            return moveString.split(' ')[1]
        else:
            return ''


    def makeMove(self, ws, moveAsFEN):
        ws.send('move ' + moveAsFEN)

    def ownColor(self, partnerNumberString):
        if (partnerNumberString == 'partner 0') or (partnerNumberString == 'partner 2'):
            return 'black'
        else:
            return 'white'

    def setOwnColor(self, partnerNumberString):
        if (partnerNumberString == 'partner 0') or (partnerNumberString == 'partner 2'):
            self.color = 'black'
        else:
            self.color =  'white'

    def beginGame(self, ownColor):
        if ownColor == 'white':
            return True
        else:
            return False

    def isMyTurn(self, msg):
        if 'move' in msg.split(' '):
            return True
        else:
            return False


    def playAgainstServer(self):

        ws = create_connection("ws://127.0.0.1/websocketclient")
        print("Receiving...")

        state = self.game.getInitBoard()

        while True:
            result = ws.recv()
            if result == 'protover 4':
                ws.send('feature san=1, time=1, variants=bughouse, otherboard=1')

            if result == "go":
                self.myTurn = True
                curPlayer = 1
                break
            if result == 'playother':
                self.myTurn = False
                curPlayer = 0
                break
        self.curPlayer = curPlayer

        while True:



            if self.myTurn:
                action=''  # run action
                self.myTurn = False
                state, curPlayer, action = self.performActionAgainsNetworkPlayer(state, curPlayer)
                dataList = []
                self.makeMove(ws, action)
                action = ''



            msg = ws.recv()
            if msg.startswith('pmove'):
                continue

            action = self.extractFromMoveFEN(msg)

            if action != '':
                state, curPlayer,action = self.performActionAgainsNetworkPlayer(state,curPlayer,action)
                action = ''
                self.myTurn = True



        ws.close()



    def performActionAgainsNetworkPlayer(self, state,curPlayer, actionString=''):
        if actionString=='':
            actionNumber = np.argmax(self.mcts.getActionProb(state, temp=0))
            # action = players[curPlayer+1](self.game.getCanonicalForm(state, curPlayer))
            actionString = self.game.getActionString(actionNumber)
        else:
            actionNumber = self.game.getActionNumber(actionString)
        valids = self.game.getValidMoves(self.game.getCanonicalForm(state, curPlayer), 1)

        if valids[actionNumber] == 0:
            print(actionNumber)
            assert valids[actionNumber] > 0
        state, curPlayer = self.game.getNextState(state, curPlayer, actionNumber)

        return state, curPlayer,actionString