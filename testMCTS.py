from bughouse.BughouseEnv import BughouseState
from bughouse.BugHouseGame import BugHouseGame
from MCTS import MCTS
from bughouse.keras.NNet import NNetWrapper as nn
from utils import *
import numpy as np
import bughouse.constants as constants
import matplotlib.pyplot as plt

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

def sortByProb(normed_policies):
    tuples = []
    for index,prob in enumerate(normed_policies):
        if prob>0:
            tuples.append((constants.LABELS[index],prob))
    
    tuples.sort(key=lambda tup: tup[1],reverse=True) 
    actions = []
    values = []
    for action,prob in tuples[:10]:
        actions.append(action)
        values.append(prob)
    plot_bar_x(actions,values)



def plot_bar_x(label,values):
    # this is for plotting purpose
    index = np.arange(len(label))
    plt.bar(index, values)
    plt.xlabel('Moves')
    plt.ylabel('Probablity')
    plt.xticks(index, label)
    plt.show()


import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

_fen =  ["2bqkbn1/2pppp2/np2N3/r3P1p1/p2N2B1/5Q2/PPPPKPP1/RNB2r2 w KQkq - 0 1","rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
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
nnet = nn(g,modelweights_path = 'models/model-04.hdf5', b_randomNet=False)
p,v =nnet.predict(s.matrice_stack)
p_valid = p*valids
p_norm = p_valid/np.sum(p_valid)
sortByProb(p_norm)
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
