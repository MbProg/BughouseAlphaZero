import os
import shutil
import time
import random
import numpy as np
import math
import sys
sys.path.append('../../')
from utils import *
from pytorch_classification.utils import Bar, AverageMeter
from NeuralNet import NeuralNet
from .BugHouseNet import BugHouseNet as bughouseNet
import matplotlib.pyplot as plt
args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 32,
    'num_channels': 512,
})

class NNetWrapper(NeuralNet):
    def __init__(self, game, modelweights_path = 'models/model-05.hdf5'):
        self.nnet = bughouseNet(game, args, modelweights_path)
        self.depth, self.height, self.width = game.getBoardSize()
        self.action_size = game.getActionSize()


    def train(self, examples):
        """
        This function trains the neural network with examples obtained from
        self-play.

        Input:
            examples: a list of training examples, where each example is of form
                      (board, pi, v). pi is the MCTS informed policy vector for
                      the given board, and v is its value. The examples has
                      board in its canonical form.
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        print('Start BughouseNN fit...')
        self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = args.batch_size, epochs = args.epochs)

    def train(self, trainset, valset, filename = 'VGG16'):
        input_boards, target_pis, target_vs = list(zip(*trainset))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        val_boards, val_pis, val_vs = list(zip(*valset))
        val_boards = np.asarray(val_boards)
        val_pis = np.asarray(val_pis)
        val_vs = np.asarray(val_vs)
        print('Start BughouseNN fit...')
        history = self.nnet.model.fit(x = input_boards, y = [target_pis, target_vs],validation_data = (val_boards,[val_pis,val_vs]),
         batch_size = args.batch_size, epochs = args.epochs)
        self.nnet.model.save('models\\'+filename+'.h5')
        # print(history.history.keys())
        # keys = history.history.keys()
        # val_loss = history.history['val_loss']
        # val_policy_loss = history.history['val_policy_loss']
        # val_value_loss = history.history['val_value_loss']
        # loss = history.history['loss']
        # policy_loss = history.history['policy_loss']
        # value_loss = history.history['value_loss']
        
        # epochs = range(1,len(loss) + 1)
        # self.plotHistory(history)
        # plt.plot(epochs,loss,'bo',label='loss')
        # plt.plot(epochs,val_loss,'b',label='val_loss')
        # plt.title = 'Training and validation loss'
        # plt.legend()
        # plt.show()

    def plotHistory(self, history):

        val_loss = history.history['val_loss']
        val_policy_loss = history.history['val_policy_loss']
        val_value_loss = history.history['val_value_loss']
        loss = history.history['loss']
        policy_loss = history.history['policy_loss']
        value_loss = history.history['value_loss']
        
        epochs = range(1,len(loss) + 1)

        # fig, ax = plt.subplots(nrows=2, ncols=2)
        fig = plt.figure()

        plt.subplot(2, 2, 1)
        plt.plot(epochs,loss,'bo',label='loss')
        plt.plot(epochs,val_loss,'b',label='val_loss')
        plt.title = 'Training and validation loss'
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(epochs,policy_loss,'bo',label='policy_loss')
        plt.plot(epochs,val_policy_loss,'b',label='val_policy_loss')
        plt.title = 'Training and validation policy loss'
        plt.legend()        

        plt.subplot(2, 2, 3)
        plt.plot(epochs,value_loss,'bo',label='value_loss')
        plt.plot(epochs,val_value_loss,'b',label='val_value_loss')
        plt.title = 'Training and validation value loss'
        plt.legend()


        plt.show()
    def predict(self, board):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
        # timing
        start = time.time()

        # preparing input
        board = board[np.newaxis, :, :]

        # run
        pi, v = self.nnet.model.predict(board)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]
            
    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        self.nnet.model.load_weights(filepath)