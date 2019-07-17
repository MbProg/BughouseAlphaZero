import math
import numpy as np
import threading
import time
import copy

EPS = 1e-8

class MCTSData():
    """
    This class handles data acces of the MCTS tree.
    """

    def __init__(self, args):
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s
        self.lock = threading.Lock()

class MCTS():
    AVAILABLE_CORES =  1

    def __init__(self, game, nnet, args):
        self.args = args
        self.game = copy.deepcopy(game)
        self.nnet = nnet
        self.data = MCTSData(args)
        self._mcts_thread = [None]*self.AVAILABLE_CORES
        self._run_mcts = False # Check to allow the thread to start
        self._mcts_finished = [True]*self.AVAILABLE_CORES # Switches to false while thread is running
        self._mcts_probs = None # reulting probabilities
        self._mcts_eval_state = None
        self._mcts_start_time = time.time()
        self._mcts_delta_time = -1.
        self.lock = threading.Lock()

    def startMCTS(self, canonicalBoard, new_time_remaining = None):
        self._mcts_eval_state = copy.deepcopy(canonicalBoard)
        self.game.setState(self._mcts_eval_state)
        self._mcts_start_time = time.time()
        if self.args.network_only:
            self._mcts_delta_time = 0.001
            self._run_mcts = True
            return True

        self._mcts_delta_time = self.eval_time(self._mcts_eval_state, new_time_remaining)
        self._run_mcts = True
        for i in range(self.AVAILABLE_CORES):
            self._mcts_thread[i] = threading.Thread(target=self._thread_getActionProb_seq,
                                                    kwargs={'thread_id': i, 'canonicalBoard': self._mcts_eval_state})
            self._mcts_thread[i].daemon = True
            self._mcts_thread[i].start()
        return True

    def stopMCTS(self, temp=0):
        if not self._run_mcts:
            return False
        self._run_mcts = False
        if self.args.network_only:
            valids = self.game.getValidMoves(self._mcts_eval_state, 1)
            matrices = self._mcts_eval_state.matrice_stack
            self.data.lock.acquire()
            Ps, v = self.nnet.predict(matrices)
            self.data.lock.release()
            Ps = Ps * valids
            self._mcts_probs = Ps
            self._mcts_finished = [True] * self.AVAILABLE_CORES
            return self._mcts_probs
        if not any(self._mcts_finished):
            for i in range(self.AVAILABLE_CORES):
                self._mcts_thread[i].join()
        s = self.game.stringRepresentation(self._mcts_eval_state)
        counts = [self.data.Nsa[(s, a)] if (s, a) in self.data.Nsa else 0 for a in range(self.game.getActionSize())]
        if float(sum(counts)) == 0:
            counts = self.data.Ps[s]
        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            self._mcts_probs = probs
            self._mcts_finished = [True]*self.AVAILABLE_CORES
            return self._mcts_probs

        counts = [x ** (1. / temp) for x in counts]
        probs = [x / float(sum(counts)) for x in counts]
        self._mcts_probs = probs
        self._mcts_finished = [True] * self.AVAILABLE_CORES
        return self._mcts_probs

    def eval_new_state(self, canonicalBoard, new_time_remaining):
        if self._mcts_eval_state is not None:
            valids = self.game.getValidMoves(self._mcts_eval_state, 1)
            self.game.setState(canonicalBoard)
            valids_new = self.game.getValidMoves(canonicalBoard, 1)
            # check if we can make new moves
            if (valids != valids_new).sum() == 0:
                #  We still have the same move we can make
                #  We may want to reevaluate the time
                return
            else:
                if (time.time() - self._mcts_start_time) < (self._mcts_delta_time*self.args.restart_cutoff):
                    self.stopMCTS()
                    self.startMCTS(canonicalBoard, new_time_remaining)
        else:
            self.startMCTS(canonicalBoard)

    def eval_time(self, canonicalBoard, new_time_remaining = None):
        MAX_TIME = self.game.environment.max_time
        my_fen = self.game.stringRepresentation(canonicalBoard)
        f_moves = int(my_fen.rsplit(" ", 1)[-1])
        board = self.game.environment.board
        team = self.game.environment.team
        my_time = None
        if new_time_remaining is not None:
            my_time = new_time_remaining
        else:
            my_time = canonicalBoard.time_remaining[team, board]
        # Opening
        if f_moves <= 4:
            return 1
        # Midgame to the predicted end
        if f_moves <= 30:
            return (MAX_TIME-44)/26.0

        # Overtime for ong games
        ret_time = my_time-canonicalBoard.time_remaining[int(not team), board]
        if ret_time < 0.25:
            return 0.25
        return ret_time


    def has_finished(self):
        if self._run_mcts:
            return time.time() >= (self._mcts_start_time + self._mcts_delta_time)
        else:
            return False

    def is_running(self):
        return self._run_mcts


    def _thread_getActionProb_seq(self, thread_id, canonicalBoard):
        self.lock.acquire()
        self._mcts_finished[thread_id] = False
        self.lock.release()
        counter = 0
        game_copy = copy.deepcopy(self.game)
        while self._run_mcts:
            game_copy.setState(canonicalBoard)
            search_mcts(canonicalBoard, self.data, game_copy, self.nnet)
            counter += 1
        self.lock.acquire()
        self._mcts_finished[thread_id] = True
        self.lock.release()
        return True

def search_mcts(canonicalBoard, data: MCTSData, game, nnet, player=1):
    """
    This function performs one iteration of MCTS. It is recursively called
    till a leaf node is found. The action chosen at each node is one that
    has the maximum upper confidence bound as in the paper.

    Once a leaf node is found, the neural network is called to return an
    initial policy P and a value v for the state. This value is propogated
    up the search path. In case the leaf node is a terminal state, the
    outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
    updated.

    NOTE: the return values are the negative of the value of the current
    state. This is done since v is in [-1,1] and if v is the value of a
    state for the current player, then its value is -v for the other player.

    Returns:
        v: the negative of the value of the current canonicalBoard
    """
    s = game.stringRepresentation(canonicalBoard)
    if s not in data.Es:
        data.lock.acquire()
        data.Es[s] = game.getGameEnded(canonicalBoard, 1)
        data.lock.release()
    if data.Es[s]!=0:
        # terminal node
        # ToDo check if returns are correct
        return (player*data.Es[s])

    # Expand a Node
    if s not in data.Ps:
        # leaf node
        valids = game.getValidMoves(canonicalBoard, 1)
        matrices = game.getCurrentBoardState(True).matrice_stack
        data.lock.acquire()
        data.Ps[s], v = nnet.predict(matrices)
        data.Ps[s] = data.Ps[s]*valids      # masking invalid moves
        sum_Ps_s = np.sum(data.Ps[s])
        if sum_Ps_s > 0:
            data.Ps[s] /= sum_Ps_s    # renormalize
        else:
            # if all valid moves were masked make all valid moves equally probable

            # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
            # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
            print("All valid moves were masked, do workaround.")
            data.Ps[s] = data.Ps[s] + valids
            data.Ps[s] /= np.sum(data.Ps[s])

        data.Vs[s] = valids
        data.Ns[s] = 0
        data.lock.release()
        return (player*v)

    data.lock.acquire()
    valids = data.Vs[s]
    data.lock.release()
    cur_best = -float('inf')
    best_act = -1

    # pick the action with the highest upper confidence bound
    for a in range(game.getActionSize()):
        if valids[a]:
            if (s,a) in data.Qsa:
                u = data.Qsa[(s,a)] + data.args.cpuct*data.Ps[s][a]*math.sqrt(data.Ns[s])/(1+data.Nsa[(s,a)])
            else:
                u = data.args.mctsValueInit + data.args.cpuct*data.Ps[s][a]*math.sqrt(data.Ns[s] + EPS)     # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

    a = best_act
    # ToDo boardview?
    next_s, next_player = game.getNextState(player, a, build_matrices=False)
    # recursion
    v = search_mcts(next_s, data, game, nnet, next_player)
    data.lock.acquire()
    if (s,a) in data.Qsa:
        data.Qsa[(s,a)] = (data.Nsa[(s,a)]*data.Qsa[(s,a)] + v)/(data.Nsa[(s,a)]+1)
        data.Nsa[(s,a)] += 1

    else:
        data.Qsa[(s,a)] = player*v
        data.Nsa[(s,a)] = 1

    data.Ns[s] += 1
    data.lock.release()
    return (player*v)

def prune_mcts_data(data: MCTSData, max_depth = 4):
    key_s = []
    key_sa = []
    for key,value in data.Ns.items():
        if value >= 1 and (int(key.split(" ", 5)[-1]) < max_depth):
            for tuple_key,_ in data.Nsa.items():
                if key == tuple_key[0]:
                    key_sa.append(tuple_key)
            key_s.append(key)

    for key in key_s:
        tmp_dict = {k: data.Ps[k] for k in key_s}
        data.Ps = tmp_dict
        tmp_dict = {k: data.Vs[k] for k in key_s}
        data.Vs = tmp_dict
        tmp_dict = {k: data.Es[k] for k in key_s}
        data.Es = tmp_dict
        tmp_dict = {k: data.Ns[k] for k in key_s}
        data.Ns = tmp_dict

    for tuple_key in key_sa:
        tmp_dict = {k: data.Nsa[k] for k in key_sa}
        data.Nsa = tmp_dict
        tmp_dict = {k: data.Qsa[k] for k in key_sa}
        data.Qsa = tmp_dict

def simplified_value_fct(state):
    value = 0
    value += state.matrice_stack[0].sum()    # My pawns
    value += state.matrice_stack[1].sum()*3  # My knights
    value += state.matrice_stack[2].sum()*3  # My bishop
    value += state.matrice_stack[3].sum()*5    # My rook
    value += state.matrice_stack[4].sum()*9  # My queen

    value -= state.matrice_stack[6].sum()    # Enemy pawns
    value -= state.matrice_stack[7].sum()*3  # Enemy knights
    value -= state.matrice_stack[8].sum()*3  # Enemy bishop
    value -= state.matrice_stack[9].sum()*5    # Enemy rook
    value -= state.matrice_stack[10].sum()*9  # Enemy queen
    return value / 39