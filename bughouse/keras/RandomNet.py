import sys

sys.path.append('..')
from utils import *
from constants import NB_LABELS

import numpy as np

class RandomNet():
    def __init__(self, args, modelweights_path=''):
        self.action_size = NB_LABELS

    def predict(self, board):
        """
        Input:
            board: current board in its canonical form.

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
        prob = np.ones(self.action_size)/self.action_size
        v = np.random.uniform(low=-1, high=1)
        return prob,v