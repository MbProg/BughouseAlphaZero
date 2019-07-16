# from Coach import Coach
# # from othello.OthelloGame import OthelloGame as Game
# from connect4.Connect4Game import Connect4Game as Game
# from connect4.tensorflow.NNet import NNetWrapper as nn
# # from othello.pytorch.NNet import NNetWrapper as nn
# from utils import *

# args = dotdict({
#     'numIters': 1,
#     'numEps': 1,
#     'tempThreshold': 15,
#     'updateThreshold': 0.6,
#     'maxlenOfQueue': 200000,
#     'numMCTSSims': 25,
#     'arenaCompare': 40,
#     'cpuct': 1,

#     'checkpoint': './temp/',
#     'load_model': False,
#     'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
#     'numItersForTrainExamplesHistory': 20,

# })

# if __name__=="__main__":
#     g = Game(6)
#     nnet = nn(g)

#     if args.load_model:
#         nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

#     c = Coach(g, nnet, args)
#     if args.load_model:
#         print("Load trainExamples from file")
#         c.loadTrainExamples()
#     c.learn()



#------------------------------------------------------
from bughouse.BugHouseGame import BugHouseGame as Game
from bughouse.keras.NNet import NNetWrapper as nn
from BugHouseArena import BugHouseArena
from utils import *
from bughouse.BugHouseGame import display as display
args = dotdict({
    'cpuct': 1,
    'mctsTmp' : 0,
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

if __name__=="__main__":
    # g = Game(6)
    # nnet = nn(g)
    #
    # g = Game()
    # nnet = nn(g, b_randomNet=False)
    # b = BugHouseArena(g,nnet,args,display)
    # b.playAgainstServer(random=False)

    g = Game()
    nnet = nn(g, b_randomNet=True)
    b = BugHouseArena(g,nnet,args,display)
    b.playAgainstServer(random=True)

    # g = Game()
    # nnet = nn(g, b_randomNet=True)
    # b = BugHouseArena(g,nnet,args,display)
    # b.playAgainstServer(random=True)

    #b = BugHouseArena(g,nnet,args,display)
    # b.playGame()
    # if args.load_model:
    #     nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    # c = Coach(g, nnet, args)
    # if args.load_model:
    #     print("Load trainExamples from file")
    #     c.loadTrainExamples()
# c.learn()