import sys

sys.path.append('..')
from utils import *
from constants import NB_LABELS
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import *


class BugHouseNet():
    def __init__(self, game, args, modelweights_path='models/model-05.hdf5'):
        self.CLASSES_LEN = NB_LABELS
        self.channel_pos = 'channels_last'
        self.inp_shape = (60, 8, 8)  # TODO: this should be read from the environment
        self.model = self.__getResidualNetwork(self.inp_shape, output_policy=self.CLASSES_LEN)
        sgd = optimizers.SGD(lr=0.000, momentum=0.9, decay=0.0, nesterov=False)
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=sgd,
                           metrics=['accuracy'], loss_weights=[0.999, 0.001])
        self.model.load_weights(modelweights_path)

    def __VGG_Conv2DBlock(self, depth, kernelshape, activation, padding, channel_pos, x, conv_amount=3, inp_shape=None):
        ''' 
        channel_pos must be 3, because keras has a problem with channel_first in BatchNormalization which is not fixed yet
        thus use: A = np.moveaxis(A,0,-1) to move the channel axis to the last index

        '''

        bn_axis = 3

        if inp_shape == None:
            x = Conv2D(depth, kernel_size=kernelshape, padding=padding, data_format=channel_pos)(x)
        else:
            x = Conv2D(depth, kernel_size=kernelshape, padding=padding, input_shape=inp_shape, data_format=channel_pos)(
                x)
        x = BatchNormalization(axis=bn_axis)(x)
        x = Activation(activation)(x)
        x = Conv2D(depth, kernel_size=kernelshape, padding=padding, data_format=channel_pos)(x)
        x = BatchNormalization(axis=bn_axis)(x)
        x = Activation(activation)(x)
        if conv_amount == 3:
            x = Conv2D(depth, kernel_size=kernelshape, padding=padding, data_format=channel_pos)(x)
            x = BatchNormalization(axis=bn_axis)(x)
            x = Activation(activation)(x)

        return x

    def __getResidualNetwork(input_shape, output_value=1, output_policy=2272):

        channel_pos = 'channels_first'
        inp_shape = Input(input_shape, name='input1')
        x = Conv2D(256, kernel_size=(3, 3), padding='same', input_shape=input_shape, data_format=channel_pos,
                   name='conv2d_Prep')(inp_shape)
        x = BatchNormalization(axis=1, name='batch_normalization_prep')(x)
        x_a1 = Activation('relu', name='activation_prep')(x)
        activated_x = x_a1

        #     activated_x, x
        def createResidualBlock(block_nr, activated_x):
            nr = block_nr * 2 - 1
            x = Conv2D(256, kernel_size=(3, 3), name='conv2d_' + str(nr), padding='same', data_format=channel_pos)(
                activated_x)
            x = BatchNormalization(axis=1, name='batch_normalization_' + str(nr))(x)
            x = Activation('relu', name='activation_' + str(nr))(x)
            x = Conv2D(256, kernel_size=(3, 3), name='conv2d_' + str(nr + 1), padding='same', data_format=channel_pos)(
                x)
            x = BatchNormalization(axis=1, name='batch_normalization_' + str(nr + 1))(x)
            x = keras.layers.add([x, activated_x], name='add_' + str(block_nr))
            activated_x = Activation('relu', name='activation_' + str(nr + 1))(x)
            return activated_x

        # build eight residual blocks
        for i in range(1, 8):
            activated_x = createResidualBlock(i, activated_x)

        # Value header
        x = Conv2D(1, kernel_size=(1, 1), name='value_conv2d', padding='same', data_format=channel_pos)(activated_x)
        xb = BatchNormalization(axis=1, name='value_batch_normalization')(x)
        xA = Activation('relu', name='value_activation')(xb)
        xF = Flatten(name='value_flatten')(xA)
        dense_1 = Dense(256, activation='relu', name='value_dense')(xF)
        value = Dense(output_value, activation='tanh', name='value')(dense_1)

        # Policy Header
        xConv = Conv2D(8, kernel_size=(7, 7), padding='same', name='policy_conv2d', data_format=channel_pos)(
            activated_x)
        xb = BatchNormalization(axis=1, name='policy_batch_normalization')(xConv)
        xA = Activation('relu', name='policy_activation')(xb)
        xF = Flatten(name='policy_flatten')(xA)
        policy = Dense(output_policy, activation='softmax', name='policy')(xF)

        from keras.models import Model
        model = Model(inp_shape, [policy, value])

        model.summary()
        return model
import sys
sys.path.append('..')
from utils import *
from constants import NB_LABELS
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import *

class BugHouseNet():
    def __init__(self, game, args, modelweights_path = 'models/model-05.hdf5'):
        self.CLASSES_LEN = NB_LABELS
        self.channel_pos = 'channels_last'
        self.inp_shape = (60,8,8) # TODO: this should be read from the environment
        self.model = self.__getResidualNetwork(self.inp_shape, output_policy=self.CLASSES_LEN)
        sgd = optimizers.SGD(lr=0.000, momentum=0.9, decay=0.0, nesterov=False)
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=sgd,
                    metrics=['accuracy'], loss_weights=[0.999,0.001])
        self.model.load_weights(modelweights_path)


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

    def __getResidualNetwork(input_shape, output_value=1, output_policy=2272):

        channel_pos = 'channels_first'
        inp_shape = Input(input_shape,name='input1')
        x = Conv2D(256, kernel_size=(3,3), padding = 'same', input_shape=input_shape,data_format=channel_pos,name='conv2d_Prep')(inp_shape)
        x = BatchNormalization(axis=1,name='batch_normalization_prep')(x)
        x_a1 = Activation('relu',name='activation_prep')(x)
        activated_x = x_a1

    #     activated_x, x
        def createResidualBlock(block_nr, activated_x):
            nr = block_nr *2 -1
            x = Conv2D(256, kernel_size=(3,3), name = 'conv2d_'+str(nr), padding='same',data_format=channel_pos)(activated_x)
            x = BatchNormalization(axis=1, name = 'batch_normalization_'+str(nr))(x)
            x = Activation('relu',name = 'activation_'+str(nr))(x)
            x = Conv2D(256, kernel_size=(3,3), name = 'conv2d_'+str(nr+1),padding = 'same',data_format=channel_pos)(x)
            x = BatchNormalization(axis=1, name = 'batch_normalization_'+str(nr+1))(x)
            x = keras.layers.add([x,activated_x],name='add_' + str(block_nr))
            activated_x = Activation('relu',name='activation_'+str(nr+1))(x)
            return activated_x

        # build eight residual blocks
        for i in range (1,8):
            activated_x = createResidualBlock(i, activated_x)



        # Value header
        x = Conv2D(1, kernel_size=(1,1),name='value_conv2d', padding = 'same',data_format=channel_pos)(activated_x)
        xb = BatchNormalization(axis=1,name='value_batch_normalization')(x)
        xA = Activation('relu',name='value_activation')(xb)
        xF = Flatten(name='value_flatten')(xA)
        dense_1 = Dense(256, activation='relu',name='value_dense')(xF)
        value = Dense(output_value, activation='tanh', name='value')(dense_1)

        # Policy Header
        xConv = Conv2D(8, kernel_size=(7,7), padding = 'same',name='policy_conv2d',data_format=channel_pos)(activated_x)
        xb = BatchNormalization(axis=1,name='policy_batch_normalization')(xConv)
        xA = Activation('relu',name='policy_activation')(xb)
        xF = Flatten(name='policy_flatten')(xA)
        policy = Dense(output_policy, activation='softmax', name='policy')(xF)


        from keras.models import Model
        model = Model(inp_shape, [policy,value])

        model.summary()
        return model
import sys
sys.path.append('..')
from utils import *
from constants import NB_LABELS
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import *

class BugHouseNet():
    def __init__(self, game, args, modelweights_path = 'models/model-05.hdf5'):
        self.CLASSES_LEN = NB_LABELS
        self.channel_pos = 'channels_last'
        self.inp_shape = (60,8,8) # TODO: this should be read from the environment
        self.model = self.__getResidualNetwork(self.inp_shape, output_policy=self.CLASSES_LEN)
        sgd = optimizers.SGD(lr=0.000, momentum=0.9, decay=0.0, nesterov=False)
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=sgd,
                    metrics=['accuracy'], loss_weights=[0.999,0.001])
        self.model.load_weights(modelweights_path)


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

    def __getResidualNetwork(input_shape, output_value=1, output_policy=2272):

        channel_pos = 'channels_first'
        inp_shape = Input(input_shape,name='input1')
        x = Conv2D(256, kernel_size=(3,3), padding = 'same', input_shape=input_shape,data_format=channel_pos,name='conv2d_Prep')(inp_shape)
        x = BatchNormalization(axis=1,name='batch_normalization_prep')(x)
        x_a1 = Activation('relu',name='activation_prep')(x)
        activated_x = x_a1

    #     activated_x, x
        def createResidualBlock(block_nr, activated_x):
            nr = block_nr *2 -1
            x = Conv2D(256, kernel_size=(3,3), name = 'conv2d_'+str(nr), padding='same',data_format=channel_pos)(activated_x)
            x = BatchNormalization(axis=1, name = 'batch_normalization_'+str(nr))(x)
            x = Activation('relu',name = 'activation_'+str(nr))(x)
            x = Conv2D(256, kernel_size=(3,3), name = 'conv2d_'+str(nr+1),padding = 'same',data_format=channel_pos)(x)
            x = BatchNormalization(axis=1, name = 'batch_normalization_'+str(nr+1))(x)
            x = keras.layers.add([x,activated_x],name='add_' + str(block_nr))
            activated_x = Activation('relu',name='activation_'+str(nr+1))(x)
            return activated_x

        # build eight residual blocks
        for i in range (1,8):
            activated_x = createResidualBlock(i, activated_x)



        # Value header
        x = Conv2D(1, kernel_size=(1,1),name='value_conv2d', padding = 'same',data_format=channel_pos)(activated_x)
        xb = BatchNormalization(axis=1,name='value_batch_normalization')(x)
        xA = Activation('relu',name='value_activation')(xb)
        xF = Flatten(name='value_flatten')(xA)
        dense_1 = Dense(256, activation='relu',name='value_dense')(xF)
        value = Dense(output_value, activation='tanh', name='value')(dense_1)

        # Policy Header
        xConv = Conv2D(8, kernel_size=(7,7), padding = 'same',name='policy_conv2d',data_format=channel_pos)(activated_x)
        xb = BatchNormalization(axis=1,name='policy_batch_normalization')(xConv)
        xA = Activation('relu',name='policy_activation')(xb)
        xF = Flatten(name='policy_flatten')(xA)
        policy = Dense(output_policy, activation='softmax', name='policy')(xF)


        from keras.models import Model
        model = Model(inp_shape, [policy,value])

        model.summary()
        return model