
import sys
sys.path.append('..')
from utils import *
from constants import NB_LABELS
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation
from keras import optimizers

class BugHouseNet():
    def __init__(self, game, args):
        self.CLASSES_LEN = NB_LABELS
        self.channel_pos = 'channels_last'
        self.inp_shape = (60,8,8) # TODO: this should be read from the environment
        inp = Input(self.inp_shape)
        # Block 1
        x = self.__VGG_Conv2DBlock(64, (3,3), 'relu', 'same',self.channel_pos, inp, 2, inp_shape=self.inp_shape)
        # Block 2 
        x = self.__VGG_Conv2DBlock(128, (3,3), 'relu', 'same',self.channel_pos, x, 2)
        # Block 3
        x = self.__VGG_Conv2DBlock(256, (3,3), 'relu', 'same',self.channel_pos, x)
        # Block 4
        x = self.__VGG_Conv2DBlock(512, (3,3), 'relu', 'same',self.channel_pos, x)
        # Block 5 
        x = self.__VGG_Conv2DBlock(512, (3,3), 'relu', 'same',self.channel_pos, x)

        x = Flatten()(x)
        dense_1 = Dense(128, activation='relu')(x)

        value = Dense(1, activation='tanh', name='value')(dense_1)
        policy = Dense(self.CLASSES_LEN, activation='softmax', name='policy')(dense_1)

        self.model = Model(inp, [policy,value])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=optimizers.RMSprop(args.lr))
        self.model.summary()

    def __VGG_Conv2DBlock(self, depth, kernelshape, activation, padding,channel_pos, x, conv_amount = 3, inp_shape=None):
        ''' 
        channel_pos must be 3, because keras has a problem with channel_first in BatchNormalization which is not fixed yet
        thus use: A = np.moveaxis(A,0,-1) to move the channel axis to the last index

        '''

        bn_axis = 3
            
        if inp_shape==None:
            x = Conv2D(depth, kernel_size=kernelshape, padding = padding,data_format=channel_pos)(x)
        else:
            x = Conv2D(depth, kernel_size=kernelshape, padding = padding, input_shape=inp_shape,data_format=channel_pos)(x)
        x = BatchNormalization(axis=bn_axis)(x)
        x = Activation(activation)(x)
        x = Conv2D(depth, kernel_size=kernelshape, padding = padding,data_format=channel_pos)(x)
        x = BatchNormalization(axis=bn_axis)(x)
        x = Activation(activation)(x)
        if conv_amount == 3:
            x = Conv2D(depth, kernel_size=kernelshape, padding = padding,data_format=channel_pos)(x)
            x = BatchNormalization(axis=bn_axis)(x)
            x = Activation(activation)(x)
            
        return x
