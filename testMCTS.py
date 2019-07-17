from bughouse.BughouseEnv import BughouseState
from bughouse.BugHouseGame import BugHouseGame
from MCTS import MCTS
from bughouse.keras.NNet import NNetWrapper as nn
from utils import *
import numpy as np
import bughouse.constants as constants

args = dotdict({
    'tick_time': 0.05,

    'cpuct': 1,
    'mctsTmp' : 0.5,
    'mctsTmpDepth' : 4,
    'mctsValueInit' : -0.1,
    'restart_cutoff' : 0.75,
    'network_only' : False,


    'numIters': 1,
    'numEps': 10,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'arenaCompare': 40,


    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

_fen =  ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[Q] w KQkq - 0 1","1nb1k2q/Pp4p1/2p1r3/p2p1n1p/4pp1P/P2Q1N2/P2PPPPR/RNB1KB2[] w Q - 0 1"]
state = BughouseState(None, np.full((2, 2), 300), 0,0, _fen )

g = BugHouseGame()
g.setState(state)
s = g.getCurrentBoardState(True)
valids = g.getValidMoves(None, None)
i= 0
for v in valids:
    if v > 0:
        print(constants.LABELS[i])
    i += 1
nnet = nn(g, b_randomNet=True)
predict=nnet.predict(s.matrice_stack)
mcts = MCTS(g,nnet,args)
import time
mcts.startMCTS(state)
while not mcts.has_finished():
    time.sleep(0.3)
time.sleep(0)
actions = mcts.stopMCTS(temp=1)
i= 0
for a in actions:
    if a > 0:
        print(constants.LABELS[i], a)
    i += 1
time.sleep(3)
