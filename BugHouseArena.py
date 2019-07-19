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
        self.tick_time = args.tick_time

        # self.player1 = lambda x: np.argmax(nmcts.getActionProb(x, temp=0))
        # self.player2 = lambda x: np.argmax(nmcts.getActionProb(x, temp=0))
        self.game = game
        self.display = display

    def playAgainstServer(self, random = False):

        wsgc = WebSocketGameClient(args=self.args)
        delay = 0.000
        timefactor = 100.0
        connection_thread = threading.Thread(target=wsgc.connect)
        connection_thread.daemon = True
        connection_thread.start()
        # Waiting for Connection

        while wsgc.connected == False:
            pass
        # Main Loop  for playing Games
        while True:
            # Waiting for game to start
            while wsgc.game_started == False:
                pass

            curPlayer = 1 # Parameter without use in this function but necessary for game
            half_turn = 0 # count the turns
            start_time = time.time() # start tiem of the game

            max_time = (wsgc.max_time/timefactor) # maximum time for each player
            # Count the remainign time of each player in these variables
            my_time_remaining = np.full((2,), max_time) # I am pos 0, oponnent is pos 1
            other_time_remaining = np.full((2,), max_time) # Postiions doesnt matter only time counter
            # create a game object with my position and the max time
            self.game = BugHouseGame(wsgc.my_team, wsgc.my_board, max_time)
            # create an mcts object to search for actions
            self.mcts = MCTS(self.game, self.nnet, self.args)
            state = self.game.getInitBoard() # Inital state
            # parameter that toggles when a play is made on the other board, value is not important
            other_board_toggle = not wsgc.my_turn
            wsgc.ready_check() # Tell the game client I am ready to send my moves
            print("Player Ready")
            # Main rouine for playing a single game
            while wsgc.game_started == True and wsgc.player_ready:
                if wsgc.my_turn and self.mcts.has_finished() and not wsgc.check_my_stack():
                    print(time.time(), "B4 STOP MCTS >>" ,state._fen[0], state._fen[1])
                    action = None
                    if random:
                        actions = np.array(self.mcts.stopMCTS(temp=0))
                        actions = self.game.getValidMoves(self.game.getCanonicalForm(state, curPlayer), curPlayer)
                        action = choice(np.arange(len(actions)), 1, p=(actions/actions.sum()))[0]
                    else:
                        if self.args.mctsTmpDepth < (half_turn / 2.0):
                            actions = np.asarray(self.mcts.stopMCTS(temp=self.args.mctsTmp))
                            action = choice(np.arange(len(actions)), 1, p=(actions / actions.sum()))[0]
                        else:
                            actions = self.mcts.stopMCTS(temp=0)
                            action = np.argmax(actions)
                    # check if move is valid
                    valids = self.game.getValidMoves(self.game.getCanonicalForm(state, curPlayer), curPlayer)
                    if valids[action] == 0:
                        print(action)
                        i = 0
                        for v in valids:
                            if v > 0:
                                print(constants.LABELS[i])
                            i += 1
                        print("my_board", wsgc.my_board)
                        print(state._fen[0], state._fen[1])
                        print(self.mcts._mcts_eval_state._fen[0], self.mcts._mcts_eval_state._fen[1])
                        assert valids[action] > 0
                    my_time_remaining[0] = max_time - (time.time() - start_time) - delay + (max_time - my_time_remaining[1])
                    state, curPlayer = self.game.getNextState(action, time=my_time_remaining[0], build_matrices=False)
                    print(time.time(), "STOP MCTS >>", state._fen[0], state._fen[1])
                    wsgc.send_action(self.game.getActionString(action))
                    half_turn += 1

                if wsgc.my_turn and not self.mcts.is_running():
                    # check if I am in stalemate (no valid moves)
                    if self.game.getValidMoves(self.game.getCanonicalForm(state, curPlayer), curPlayer).sum() >= 1:
                        self.mcts.startMCTS(state)

                # Look if we got a new move from our opponent or the partner board
                if wsgc.check_stack_id():
                    # check which board made a move, keeping the order consistent
                    board_id = wsgc.pop_stack_id()
                    # play on my board
                    if board_id == 0 and wsgc.check_my_stack():
                        # calculate the time
                        my_time_remaining[1] = max_time - (time.time() - start_time) - delay + (
                                    max_time - my_time_remaining[0])
                        # play the action on our simulated game
                        action = self.game.getActionNumber(wsgc.pop_my_stack())
                        state, curPlayer = self.game.getNextState(action, time=my_time_remaining[1],
                                                                  build_matrices=False, player_view=True)
                        half_turn += 1

                    # play on the other board
                    elif board_id == 1 and wsgc.check_partner_stack():
                        actionString = wsgc.pop_partner_stack()
                        # print(actionString)
                        # print(time.time(), "PLAY OTHER >>", state._fen[0], state._fen[1])
                        # calculate the time
                        other_time_remaining[int(other_board_toggle)] = max_time - (time.time() - start_time) - delay + \
                                                                        (max_time - other_time_remaining[
                                                                            int(not other_board_toggle)])
                        # play the action on our simulated game
                        action = self.game.getActionNumber(actionString)
                        state, curPlayer = self.game.getNextState(action,time=other_time_remaining[int(other_board_toggle)],
                                                                  play_other_board=True, build_matrices=False,
                                                                  player_view=False)
                        # toggle the other board so we know a turn has been played
                        other_board_toggle = not other_board_toggle

                        # check if we need to reevalute a running mcts task
                        if self.mcts.is_running():
                            my_time_remaining[0] = max_time - (time.time() - start_time) - delay + (
                                    max_time - my_time_remaining[1])
                            self.mcts.eval_new_state(state, my_time_remaining[0])

            time.sleep(self.tick_time)



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
            self.mcts.startMCTS(state)

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
