import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time
from Arena import Arena
from MCTS import MCTS
from websocket import create_connection


class BugHouseArena(Arena):
    """
    """
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

        # self.player1 = lambda x: np.argmax(nmcts.getActionProb(x, temp=0))
        # self.player2 = lambda x: np.argmax(nmcts.getActionProb(x, temp=0))
        self.game = game
        self.display = display

    def playAgainstServer(self):

        ws = create_connection("ws://127.0.0.1/websocketclient")
        print("Sending 'Hello, World'...")
        ws.send("Hello, World")
        print("Sent")
        print("Receiving...")
        dataList = []
        state = self.game.getInitBoard()

        sent = True
        insertDataList = True
        gameStart = False
        action = ''
        InitialMove = False
        curPlayer = 1

        while True:
            result = ws.recv()
            if gameStart:
                action = result

            dataList.append(result)

            if result == "go":
                gameStart = True


            # dataList.append("datalist:"+result)

            if result == 'protover 4':
                ws.send("feature")
                if result == 'accepted':
                    print("Yeahh, wait for game")

                    # print(dataList)

            # if I have partner 1 or partner 2 as team member -> I am the beginner
            if gameStart:
                if ("partner 1" in dataList or "partner2" in dataList):
                    InitialMove = True

                if InitialMove:
                    action  # run action
                    InitialMove = False
                    state, curPlayer,action = self.performActionAgainsNetworkPlayer(state,curPlayer)
                    dataList = []
                    ws.send(action)
                    action = ''
                    continue
                if action != '':
                    state, curPlayer,action = self.performActionAgainsNetworkPlayer(state,curPlayer,action)
                    ws.send(action)
                    action = ''

                else:
                    state, curPlayer,action = self.performActionAgainsNetworkPlayer(state,curPlayer)
                    ws.send(action)
                    action = ''

                dataList = []
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

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        # players = [self.player2, None, self.player1]
        curPlayer = 1
        state = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(state, curPlayer)==0:
            it+=1
            if verbose:
                assert(self.display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(state)
            action = np.argmax(self.mcts.getActionProb(state,temp=0))
            # action = players[curPlayer+1](self.game.getCanonicalForm(state, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(state, curPlayer),1)

            if valids[action]==0:
                print(action)
                assert valids[action] >0
            state, curPlayer = self.game.getNextState(state, curPlayer, action)
            print(state._boards_fen[0])
        if verbose:
            assert(self.display)
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(state, 1)))
            self.display(state)
        return self.game.getGameEnded(state, 1)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        num = int(num/2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult==1:
                oneWon+=1
            elif gameResult==-1:
                twoWon+=1
            else:
                draws+=1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=maxeps, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()

        self.player1, self.player2 = self.player2, self.player1
        
        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult==-1:
                oneWon+=1                
            elif gameResult==1:
                twoWon+=1
            else:
                draws+=1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=num, et=eps_time.avg,
                                                                                                       total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()
            
        bar.finish()

        return oneWon, twoWon, draws
