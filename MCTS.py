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
    AVAILABLE_CORES = 2

    def __init__(self, game, nnet, args):
        self.args = args
        self.game = game
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

    def getActionProb(self, canonicalBoard, temp=1, n_tasks = 8, depth = 8):
        start = time.time()
        game_copys = []
        nnet_copys = []
        thread_list = []
        for i in range(n_tasks):
            game_copys.append(copy.deepcopy(self.game))
            # nnet_copys.append(copy.deepcopy(self.nnet))
        for i in range(self.args.numMCTSSims):
            for game in game_copys:
                game.setState(canonicalBoard)
            for j in range(n_tasks):
                t = threading.Thread(target=search_mcts, kwargs={'canonicalBoard': canonicalBoard,
                                                                   'data': self.data, 'game': game_copys[j],
                                                                   'nnet': self.nnet, 'depth': depth})
                t.daemon = True
                thread_list.append(t)
                thread_list[j].start()

            for thread in thread_list:
                thread.join()
            thread_list.clear()
        print("end", time.time()-start)
        self.game.setState(canonicalBoard)
        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.data.Nsa[(s,a)] if (s,a) in self.data.Nsa else 0 for a in range(self.game.getActionSize())]
        print(sum(counts))
        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs

    def getActionProb_seq(self, canonicalBoard, temp=1, depth = 0):
        start = time.time()
        for i in range(self.args.numMCTSSims):
            self.game.setState(canonicalBoard)
            search_mcts_seq(canonicalBoard, self.data, self.game, self.nnet, depth)
        print("end", time.time()-start)
        self.game.setState(canonicalBoard)
        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.data.Nsa[(s,a)] if (s,a) in self.data.Nsa else 0 for a in range(self.game.getActionSize())]

        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs

    def startMCTS(self, canonicalBoard, depth=0):
        self._mcts_eval_state = copy.deepcopy(canonicalBoard)
        self.game.setState(self._mcts_eval_state)
        self._mcts_start_time = time.time()
        self._mcts_delta_time = self.eval_time(self._mcts_eval_state)
        self._run_mcts = True
        for i in range(self.AVAILABLE_CORES):
            self._mcts_thread[i] = threading.Thread(target=self.__thread_getActionProb_seq,
                                             kwargs={'thread_id': i, 'canonicalBoard': self._mcts_eval_state, 'depth': depth})
            self._mcts_thread[i].daemon = True
            self._mcts_thread[i].start()
        return True

    def stopMCTS(self, temp=1):
        if not self._run_mcts:
            return False
        self._run_mcts = False
        if not any(self._mcts_finished):
            for i in range(self.AVAILABLE_CORES):
                self._mcts_thread[i].join()
        s = self.game.stringRepresentation(self._mcts_eval_state)
        counts = [self.data.Nsa[(s, a)] if (s, a) in self.data.Nsa else 0 for a in range(self.game.getActionSize())]
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

    def eval_new_state(self, canonicalBoard):
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
                self.stopMCTS()
                self.startMCTS(canonicalBoard)
        else:
            self.startMCTS(canonicalBoard)

    def eval_time(self, canonicalBoard):
        time = canonicalBoard.time_remaining
        self.data.lock.acquire()
        moves, value = self.nnet.predict(canonicalBoard)
        self.data.lock.release()
        # ToDo a good formula for time
        time = 10
        return time

    def has_finished(self):
        return time.time() <= (self._mcts_start_time + self._mcts_delta_time)


    def __thread_getActionProb_seq(self,thread_id , canonicalBoard, depth=0):
        self.lock.acquire()
        self._mcts_finished[thread_id] = False
        self.lock.release()
        counter = 0
        game_copy = copy.deepcopy(self.game)
        while self._run_mcts:
            game_copy.setState(canonicalBoard)
            search_mcts(canonicalBoard, self.data, game_copy, self.nnet, depth)
            counter += 1
        print(counter)
        self.lock.acquire()
        self._mcts_finished[thread_id] = True
        self.lock.release()
        return True

def search_mcts(canonicalBoard, data: MCTSData, game, nnet, depth = 2):
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
        return -data.Es[s]

    if s not in data.Ps:
        # leaf node
        valids = game.getValidMoves(canonicalBoard, 1)
        data.lock.acquire()
        data.Ps[s], v = nnet.predict(canonicalBoard)
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
        if depth <= 0:
            return -v
        else:
            depth = depth - 1

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
                u = data.args.cpuct*data.Ps[s][a]*math.sqrt(data.Ns[s] + EPS)     # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

    a = best_act
    next_s, next_player = game.getNextState(1, a)
    next_s = game.getCanonicalForm(next_s, next_player)
    v = search_mcts(next_s, data, game, nnet, depth)
    data.lock.acquire()
    if (s,a) in data.Qsa:
        data.Qsa[(s,a)] = (data.Nsa[(s,a)]*data.Qsa[(s,a)] + v)/(data.Nsa[(s,a)]+1)
        data.Nsa[(s,a)] += 1

    else:
        data.Qsa[(s,a)] = v
        data.Nsa[(s,a)] = 1

    data.Ns[s] += 1
    data.lock.release()
    return -v

def search_mcts_seq(canonicalBoard, data: MCTSData, game, nnet, depth = 2):
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
        data.Es[s] = game.getGameEnded(canonicalBoard, 1)
    if data.Es[s]!=0:
        # terminal node
        return -data.Es[s]

    if s not in data.Ps:
        # leaf node
        valids = game.getValidMoves(canonicalBoard, 1)
        data.Ps[s], v = nnet.predict(canonicalBoard)
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
        if depth <= 0:
            return -v
        else:
            depth = depth - 1

    valids = data.Vs[s]
    cur_best = -float('inf')
    best_act = -1

    # pick the action with the highest upper confidence bound
    for a in range(game.getActionSize()):
        if valids[a]:
            if (s,a) in data.Qsa:
                u = data.Qsa[(s,a)] + data.args.cpuct*data.Ps[s][a]*math.sqrt(data.Ns[s])/(1+data.Nsa[(s,a)])
            else:
                u = data.args.cpuct*data.Ps[s][a]*math.sqrt(data.Ns[s] + EPS)     # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

    a = best_act
    next_s, next_player = game.getNextState(1, a)
    next_s = game.getCanonicalForm(next_s, next_player)
    v = search_mcts(next_s, data, game, nnet, depth)
    if (s,a) in data.Qsa:
        data.Qsa[(s,a)] = (data.Nsa[(s,a)]*data.Qsa[(s,a)] + v)/(data.Nsa[(s,a)]+1)
        data.Nsa[(s,a)] += 1

    else:
        data.Qsa[(s,a)] = v
        data.Nsa[(s,a)] = 1

    data.Ns[s] += 1
    return -v