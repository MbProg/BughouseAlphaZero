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
import keras
import tensorflow as tf

def acc_sign(y_true, y_pred):
    return K.mean(K.equal(K.sign(y_pred), K.sign(y_true)), axis=-1)


def acc_round(y_true, y_pred):
    # each interval for class -1, 0, 1 must have a width of 2/3 to be equally distributed
    # therefore you must convert (1/3) to (1/2) and only then round to the closest integer
    # 1/3 * x = 1/2
    return K.mean(K.equal(K.round(y_pred * 1.5), K.round(y_true)), axis=-1)

def acc_reg(y_true,y_pred):
    return K.constant(1) - K.square(K.mean((y_pred-y_true), axis=1))

def acc_round_unequal(y_true,y_pred):
    return K.mean(K.equal(K.round(y_true),K.round(y_pred)), axis=-1)
graph = None

class BugHouseNet():
    def __init__(self, args, modelweights_path='models/model-05.hdf5'):

        self.CLASSES_LEN = NB_LABELS
        self.channel_pos = 'channels_last'
        self.inp_shape = (60, 8, 8)  # TODO: this should be read from the environment

        # self.model = self.__getResidualNetwork(self.inp_shape, output_policy=self.CLASSES_LEN)
        # sgd = optimizers.SGD(lr=0.000, momentum=0.9, decay=0.0, nesterov=False)

        # self.model.compile(loss={'policy':'categorical_crossentropy',
        #                     'value':'mean_squared_error'}, optimizer=sgd,
        #             metrics={'policy':'accuracy', 'value':[acc_round, acc_sign]}, loss_weights=[0.25,0.75])

        self.model = keras.models.load_model('finalModel/FinalModelNewData.h5',custom_objects={'acc_round': acc_round,'acc_sign':acc_sign})
        # self.model.load_weights('finalModel/model-05.hdf5')
        # self._evaluate([300],'datasetSmallPart/')
        global graph
        graph = tf.get_default_graph()

        # a prediction because first is always very slow
        self.predict(np.random.rand(1, self.inp_shape[0],self.inp_shape[1],self.inp_shape[2]))
        # dataTest = np.load('data.npy')
        # p,v = self.model.predict(dataTest)
        # print(p,v)
        # dataset_path = 'dataset/'                     # relative directory to the dataset

        # x_val, policies_val, values_val = read_files_data([6001],path=dataset_path)
        # self.model.evaluate(x_val,[policies_val,values_val],verbose=True)

        # print('success')
        # self.model = self.__getResidualNetwork(self.inp_shape, output_policy=self.CLASSES_LEN)
        # sgd = optimizers.SGD(lr=0.000, momentum=0.9, decay=0.0, nesterov=False)
        # self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=sgd,
        #                    metrics=['accuracy'], loss_weights=[0.999, 0.001])
        # self.model.load_weights(modelweights_path)

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

    def _evaluate(self, file_ids, dataset_path='dataset/'):
        x_val, policies_val, values_val = self.__read_files_data(file_ids,path=dataset_path)
        print(self.model.metrics_names)
        print(self.model.evaluate(x_val,[policies_val,values_val]))

    def __read_file_data(self,ID, path='dataset/'):
        import zarr
        dataset = zarr.group(store=zarr.ZipStore(path + str(ID) +'.zip', mode="r"))
        X = np.array(dataset['states'])
        policies = np.array(dataset['policies'])
        values = np.array(dataset['values'])
        return X,policies,values

    def __read_files_data(self,IDs, path='dataset/'):
        X = np.array([])
        for ID in IDs:
            if X.size==0:
                X, policies, values = self.__read_file_data(ID, path)
                values = np.array(values)
            else:
                X_ID, policies_ID, values_ID = self.__read_file_data(ID, path)
                X = np.concatenate((X,X_ID))
                policies = np.concatenate((policies, policies_ID))
                values = np.concatenate((values,np.array(values_ID)))
        return X,policies,values
        
    def __getResidualNetwork(self, input_shape, output_value=1, output_policy=2272):
        
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
        x = Conv2D(4, kernel_size=(1,1),name='value_conv2d', padding = 'same',data_format=channel_pos)(activated_x)
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


    def predict(self, data):
        """
        Input: numpy array with shape (1, shape of state) e.g. (1, 60, 8 , 8)
                should be channel_first
        """
        # data = data[0][np.newaxis, :, :]
        # random_data = np.random.rand(1,60,8,8)
        # data = np.float32(data)
        with graph.as_default():
            pi, v = self.model.predict(data)      
        # pi, v = self.model.predict(data)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]