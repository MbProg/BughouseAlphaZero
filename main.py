import sys
from bughouse.BugHouseGame import BugHouseGame as Game
from bughouse.keras.NNet import NNetWrapper as nn
from BugHouseArena import BugHouseArena
from utils import *
from bughouse.BugHouseGame import display as display


args = dotdict({
    'tick_time' : 0.05,
    'url' : "ws://127.0.0.1/websocketclient",
    'fix_action_input': False,
    'cpuct': 1,
    'mctsTmp' : 0.5,
    'mctsTmpDepth' : 4,
    'mctsValueInit' : -0.3,
    'mctsOpeningTime': 1.0,
    'mctsMidgameDepth': 30,
    'mctsTimeBuffer': 30,
    'restart_cutoff' : 0.75,
    'network_only' : True,
    'random_play' : True,


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

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
if __name__=="__main__":
    # Quick and dirty start parameter ToDo use argparser object
    if len(sys.argv) > 1:
        args.url = sys.argv[1]
    g = Game()
    nnet = nn(g,modelweights_path='finalModel/FinalModelJustForAlex.hdf5', b_randomNet=args.random_play)
    b = BugHouseArena(g, nnet, args, display)
    b.playAgainstServer(random=args.random_play)

