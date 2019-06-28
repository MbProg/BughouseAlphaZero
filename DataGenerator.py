import numpy as np
import keras
import pickle
import zarr

class RL_Datapoint():
    def __init__(self, state, policy, values):
        self.state = state
        self.policy = policy
        self.values = values


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, file_IDs, zip_length , batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True, path='dataset/'):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.file_IDs = file_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.zip_length = zip_length
        self.fileID = 0
        self.offset = 0
        self.path = path
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.file_IDs)*self.zip_length / self.batch_size))

    def __get_fileID(self,index_start):
        file_id = int(index_start/self.zip_length)
        return file_id
    
    def __read_file_data(self, ID):
        dataset = zarr.group(store=zarr.ZipStore(self.path + str(ID) +'.zip', mode="r"))
        self.X = np.array(dataset['states'])
        self.policies = np.array(dataset['policies'])
        self.values = np.array(dataset['values'])

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if ((index+1)*self.batch_size) > (self.fileID+1)*self.zip_length:
            self.fileID = self.__get_fileID((index+1)*self.batch_size)
            if self.shuffle == True: np.random.shuffle(self.indexes)
            self.__read_file_data(self.fileID)
            self.offset = index
            
        indexes = self.indexes[(index-self.offset)*self.batch_size:((index-self.offset)+1)*self.batch_size]

        # Find list of IDs
        # list_IDs_temp = self.list_IDs[indexes] 

        # Generate data
        X,y = self.__data_generation_zip(indexes)
        # X, y = self.__data_generation(list_IDs_temp)

        return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.zip_length)
        # self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation_zip(self,indexes):

        X = self.X[indexes]
        y1 = self.policies[indexes]
        y2 = self.values[indexes]
        return X,[y1,y2]

        
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y1 = np.empty((self.batch_size, 2272), dtype=dict)
        y2 = np.empty((self.batch_size), dtype=dict)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            with open(self.path + str(ID) + '.pkl','rb') as pkl_file:
                try:
                    rl_datapoint = pickle.load(pkl_file)
                except EOFError:
                    break

            # Store sample
            X[i,] = rl_datapoint.state

            # Store class
            y1[i] = rl_datapoint.policy
            y2[i] = rl_datapoint.values # [rl_datapoint.policy,rl_datapoint.values]
        return X, [y1,y2]

def __VGG_Conv2DBlock( depth, kernelshape, activation, padding,channel_pos, x, conv_amount = 3, inp_shape=None):
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
def plotHistory( history):
    import matplotlib.pyplot as plt

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

def __read_file_data(ID, path='dataset/'):
    dataset = zarr.group(store=zarr.ZipStore(path + str(ID) +'.zip', mode="r"))
    X = np.array(dataset['states'])
    policies = np.array(dataset['policies'])
    values = np.array(dataset['values'])
    return X,policies,values

def generator(batch_size, datasetFileLength, path='dataset/'):
    while True:
        files_sequence = list(range(datasetFileLength))
        np.random.shuffle(files_sequence)

        for file_step, file_id in enumerate(files_sequence): 
            X, policies, values = __read_file_data(file_id,path)
            rand_indices = np.arange(len(X))
            np.random.shuffle(rand_indices)

            for i in range(int(len(X)/batch_size)):
                batch_indices = rand_indices[i*batch_size: (i+1)*batch_size]
                yield X[batch_indices],[policies[batch_indices],values[batch_indices]]

import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.layers import *

def getResidualNetwork(input_shape):
    
    channel_pos = 'channels_first'
    inp_shape = Input(input_shape,name='input1')
    x = Conv2D(256, kernel_size=(3,3), padding = 'same', input_shape=input_shape,data_format=channel_pos,name='conv2d_1')(inp_shape)
    x = BatchNormalization(axis=1,name='batch_normalization_1')(x)
    x_a1 = Activation('relu',name='activation_1')(x)

    x = Conv2D(256, kernel_size=(3,3),name ='conv2d_2' ,padding = 'same',data_format=channel_pos)(x_a1)
    x = BatchNormalization(axis=1, name = 'batch_normalization_2')(x)
    x = Activation('relu',name = 'activation_2')(x)
    x = Conv2D(256, kernel_size=(3,3), name = 'conv2d_3',padding = 'same',data_format=channel_pos)(x)
    x = BatchNormalization(axis=1, name = 'batch_normalization_3')(x)

    x = keras.layers.add([x,x_a1],name='add1')
    x_a2 = Activation('relu',name='activation_3')(x)
    x = Conv2D(256, kernel_size=(3,3),name = 'conv2d_4', padding = 'same',data_format=channel_pos)(x_a2)
    x = BatchNormalization(axis=1,name = 'batch_normalization_4')(x)
    x = Activation('relu',name='activation_4')(x)
    x = Conv2D(256, kernel_size=(3,3), name='conv2d_5',padding = 'same',data_format=channel_pos)(x)
    x = BatchNormalization(axis=1,name='batch_normalization_5')(x)

    x = keras.layers.add([x,x_a2],name='add_2')
    x_a3 = Activation('relu',name='activation_5')(x)
    x = Conv2D(256, kernel_size=(3,3),name='conv2d_6', padding = 'same',data_format=channel_pos)(x_a3)
    x = BatchNormalization(axis=1,name='batch_normalization_6')(x)
    x = Activation('relu',name='activation_6')(x)
    x = Conv2D(256, kernel_size=(3,3), name='conv2d_7',padding = 'same',data_format=channel_pos)(x)
    x = BatchNormalization(axis=1,name='batch_normalization_7')(x)

    x = keras.layers.add([x,x_a3],name='add_3')
    x_a4 = Activation('relu',name='activation_7')(x)
    x = Conv2D(256, kernel_size=(3,3), name = 'conv2d_8',padding = 'same',data_format=channel_pos)(x_a4)
    x = BatchNormalization(axis=1,name='batch_normalization_8')(x)
    x = Activation('relu',name='activation_8')(x)
    x = Conv2D(256, kernel_size=(3,3), name='conv2d_9',padding = 'same',data_format=channel_pos)(x)
    x = BatchNormalization(axis=1,name='batch_normalization_9')(x)

    x = keras.layers.add([x,x_a4],name='add4')
    x_a5 = Activation('relu',name='activation_9')(x)
    x = Conv2D(256, kernel_size=(3,3),name='conv2d_10', padding = 'same',data_format=channel_pos)(x_a5)
    x = BatchNormalization(axis=1,name='batch_normalization_10')(x)
    x = Activation('relu',name='activation_10')(x)
    x = Conv2D(256, kernel_size=(3,3),name='conv2d_11', padding = 'same',data_format=channel_pos)(x)
    x = BatchNormalization(axis=1,name='batch_normalization_11')(x)

    x = keras.layers.add([x,x_a5],name='add_5')
    x_a6 = Activation('relu',name='activation_11')(x)
    x = Conv2D(256, kernel_size=(3,3), name='conv2d_12',padding = 'same',data_format=channel_pos)(x_a6)
    x = BatchNormalization(axis=1,name='batch_normalization_12')(x)
    x = Activation('relu',name='activation_12')(x)
    x = Conv2D(256, kernel_size=(3,3), name='conv2d_13',padding = 'same',data_format=channel_pos)(x)
    x = BatchNormalization(axis=1,name='batch_normalization_13')(x)

    x = keras.layers.add([x,x_a6],name='add6')
    x_a7 = Activation('relu',name='activation_13')(x)
    x = Conv2D(256, kernel_size=(3,3), name='conv2d_14',padding = 'same',data_format=channel_pos)(x_a7)
    x = BatchNormalization(axis=1,name='batch_normalization_14')(x)
    x = Activation('relu',name='activation_14')(x)
    x = Conv2D(256, kernel_size=(3,3), name='conv2d_15',padding = 'same',data_format=channel_pos)(x)
    x = BatchNormalization(axis=1,name='batch_normalization_15')(x)

    x = keras.layers.add([x,x_a7],name='add_7')
    x_a8 = Activation('relu',name='activation_15')(x)
    x = Conv2D(1, kernel_size=(1,1),name='conv2d_17', padding = 'same',data_format=channel_pos)(x_a8)
    xb = BatchNormalization(axis=1,name='batch_normalization_17')(x)
    xConv = Conv2D(2, kernel_size=(1,1), padding = 'same',name='conv2d_16',data_format=channel_pos)(x_a8)
    xA = Activation('relu',name='activation_17')(xb)
    xb = BatchNormalization(axis=1,name='batch_normalization_16')(xConv)
    xF = Flatten(name='flatten_2')(xA)
    xA = Activation('relu',name='activation_16')(xb)

    dense_1 = Dense(256, activation='relu',name='dense_1')(xF)
    xF = Flatten(name='flatten_1')(xA)

    value = Dense(1, activation='tanh', name='value')(dense_1)
    policy = Dense(CLASSES_LEN, activation='softmax', name='policy')(xF)

    from keras.models import Model
    model = Model(inp_shape, [policy,value])

    model.summary()
    return model

# ----------- parameters ----------------
files_len = 10
file_ids = np.arange(files_len)
# np.random.shuffle(file_ids)
train_ids = file_ids[:int(len(file_ids)*0.9)]
val_ids = file_ids[int(len(file_ids)*0.9):]
# training_generator = DataGenerator(train_ids,1000, 128,(60,8,8),path='dataset')
# val_generator = DataGenerator(val_ids,1000, 128,(60,8,8))

filepath="models/model-{epoch:02d}.hdf5"
zip_length = 10000
data_len = files_len * zip_length
batch_size = 256
steps_per_epoch = int((len(train_ids)*zip_length)/batch_size)
CLASSES_LEN = 2272
channel_pos = 'channels_last'
dataset_path = 'dataset/'
inp_shape = (60,8,8) # TODO: this should be read from the environment

# ----------- NN Architecture ----------------

# inp = Input(inp_shape)
# # Block 1
# x = __VGG_Conv2DBlock(64, (3,3), 'relu', 'same',channel_pos, inp, 2, inp_shape=inp_shape)
# # Block 2 
# x = __VGG_Conv2DBlock(96, (3,3), 'relu', 'same',channel_pos, x, 2)
# # Block 3
# x = __VGG_Conv2DBlock(128, (3,3), 'relu', 'same',channel_pos, x)
# # Block 4
# x = __VGG_Conv2DBlock(160, (3,3), 'relu', 'same',channel_pos, x)
# # Block 5 
# x = __VGG_Conv2DBlock(190, (3,3), 'relu', 'same',channel_pos, x)

# x = Flatten()(x)
# dense_1 = Dense(128, activation='relu')(x)

# value = Dense(1, activation='tanh', name='value')(dense_1)
# policy = Dense(CLASSES_LEN, activation='softmax', name='policy')(dense_1)

# model = Model(inp, [policy,value])
model = getResidualNetwork(inp_shape)
sgd = optimizers.SGD(lr=0.000, momentum=0.9, decay=0.1/5, nesterov=False)

def acc_reg(y_true,y_pred):
    return K.constant(1) - K.square(K.mean((y_pred-y_true), axis=1))

model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=sgd,
              metrics=['accuracy', acc_reg], loss_weights=[0.999,0.001])
              
# model.summary()



#
val_id = files_len -1 
x_val, policies_val, values_val = __read_file_data(val_id,path=dataset_path)
    
# callbacks 
from datetime import datetime
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
logdir="logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, update_freq='batch')

# learning rate
from  LearningRateScheduler import *
epochs = 20
batch_len = epochs * int(data_len/ (batch_size))
max_lr = 0.001*8
total_it = batch_len
min_lr = 0.0001
print('BatchLen: ', batch_len, ' - DataLen: ', data_len)
lr_schedule = OneCycleSchedule(start_lr=max_lr/8, max_lr=max_lr, cycle_length=total_it*.4, cooldown_length=total_it*.6, finish_lr=min_lr)
scheduler = LinearWarmUp(lr_schedule, start_lr=min_lr, length=total_it/30)
bt = BatchLearningRateScheduler(scheduler)

# losshistory = LossHistory()
callbacks_list = [checkpoint,tensorboard_callback,bt]
# callbacks_list = [checkpoint]
model.load_weights('models/model-05.hdf5')
# history = model.fit_generator(generator=training_generator,validation_data=val_generator,epochs=5,callbacks=callbacks_list)
history = model.fit_generator(generator(batch_size,len(train_ids),path=dataset_path), steps_per_epoch=int((len(train_ids)*zip_length)/batch_size), callbacks=callbacks_list,
                    epochs=epochs, validation_data=(x_val, [policies_val,values_val]))
model.save('models\\BughouseNet220620190437.h5')
plotHistory(history)

# with open('dataset/' + str(0) + '.pkl','rb') as pkl_file:
#     rl_datapoint = pickle.load(pkl_file)
#     print(rl_datapoint.policy.shape)
#     print(rl_datapoint.values.shape)
