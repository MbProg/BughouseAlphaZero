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
from .RandomNet import RandomNet as randomNet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 32,
    'num_channels': 512,
})


class NNetWrapper(NeuralNet):
    def __init__(self, game, modelweights_path='models/model-05.hdf5', b_randomNet=False):
        if b_randomNet:
            self.nnet = randomNet(args)
        else:
            self.nnet = bughouseNet(args, modelweights_path)
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
        pass

    def train(self, trainset, valset, filename='VGG16'):
        pass



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

        # predict
        pi, v = self.nnet.predict(board)
        

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi, v

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
            raise ("No model in path {}".format(filepath))
        self.nnet.model.load_weights(filepath)