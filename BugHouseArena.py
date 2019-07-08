import numpy as np
from pytorch_classification.utils import Bar, AverageMeter
import time
from Arena import Arena
from MCTS import MCTS
from numpy.random import choice
import bughouse.constants as constants
from bughouse.BugHouseGame import BugHouseGame
from WebSocketGameClient import WebSocketGameClient
import threading
from bughouse.BughouseEnv import BughouseEnv
from MCTS import MCTS

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

    def playAgainstServer(self, random = False):

        wsgc = WebSocketGameClient()
        curPlayer = 1
        ifmillis = 100.0 # use if gametime is given in milliseconds
        delay = 60.0/ifmillis
        connection_thread = threading.Thread(target=wsgc.connect)
        connection_thread.daemon = True
        connection_thread.start()
        # Waiting for Connection
        while wsgc.connected == False:
            pass
        # Waiting for game to start
        while wsgc.game_started == False:
            pass
        start_time = time.time()
        self.time_remaining = np.full((2, 2), wsgc.max_time / ifmillis)
        self.game = BugHouseGame(wsgc.my_team, wsgc.my_board)
        self.mcts = MCTS(self.game, self.nnet, self.args)
        state = self.game.getInitBoard()
        while wsgc.game_started == True:
            if not wsgc.my_turn and wsgc.check_my_stack():
                time_remaining = start_time - time.time() - delay
                action = self.game.getActionNumber(wsgc.pop_my_stack())
                state, curPlayer = self.game.getNextState(curPlayer, action)

            if wsgc.check_partner_stack():
                time_remaining = start_time - time.time() - delay
                action = self.game.getActionNumber(wsgc.pop_partner_stack())
                state, curPlayer = self.game.getNextState(curPlayer, action, play_other_board=True)
                if self.mcts.is_running():
                    print("eval_new_State")
                    self.mcts.eval_new_state(state)
                    time.sleep(0.05)

            if wsgc.my_turn and not self.mcts.is_running():
                if self.game.getValidMoves(self.game.getCanonicalForm(state, curPlayer), curPlayer).sum() >= 1:
                    self.mcts.startMCTS(state, depth=0)
                # time.sleep(0.2)

            if wsgc.my_turn and self.mcts.has_finished() and not wsgc.check_partner_stack():
                actions = self.mcts.stopMCTS(temp=0.2)
                # action = np.argmax(actions)
                # ToDo add drawing and change temp for more random
                if random:
                    actions = self.game.getValidMoves(self.game.getCanonicalForm(state, curPlayer), curPlayer)
                    action = choice(np.arange(len(actions)), 1, p=(actions/actions.sum()))[0]
                valids = self.game.getValidMoves(self.game.getCanonicalForm(state, curPlayer), curPlayer)
                if valids[action] == 0:
                    i = 0
                    for v in valids:
                        if v > 0:
                            print(constants.LABELS[i])
                        i += 1
                    assert valids[action] > 0
                state, curPlayer = self.game.getNextState(curPlayer, action, state)
                wsgc.send_action(self.game.getActionString(action))
                time_remaining = start_time - time.time() - delay

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
        while self.game.getGameEnded(state, curPlayer) == 0:
            it += 1
            if verbose:
                assert (self.display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(state)
            self.mcts.startMCTS(state, depth=0)
            while self.mcts.has_finished():
                time.sleep(0.03)
                pass
            actions = self.mcts.stopMCTS(temp=0.75)
            action = np.argmax(actions)

            draw = choice(np.arange(len(actions)), 1, p=actions)
            # if action != draw:
            #   print("RANDOM SHIT")
            #   print(actions[int(draw)])
            #    print(actions[action])
            # else:
            # print("SAME")
            # print(actions[int(draw)])
            # print(actions[action])
            # action = np.argmax(self.mcts.getActionProb_seq(state, temp=0))
            # action = players[curPlayer+1](self.game.getCanonicalForm(state, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(state, curPlayer), 1)

            if valids[action] == 0:
                print(action)
                assert valids[action] > 0
            state, curPlayer = self.game.getNextState(curPlayer, action, state)
            print(state._fen[0])
        if verbose:
            assert (self.display)
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

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps + 1,
                                                                                                       maxeps=maxeps,
                                                                                                       et=eps_time.avg,
                                                                                                       total=bar.elapsed_td,
                                                                                                       eta=bar.eta_td)
            bar.next()

        self.player1, self.player2 = self.player2, self.player1

        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps + 1,
                                                                                                       maxeps=num,
                                                                                                       et=eps_time.avg,
                                                                                                       total=bar.elapsed_td,
                                                                                                       eta=bar.eta_td)
            bar.next()

        bar.finish()

        return oneWon, twoWon, draws
