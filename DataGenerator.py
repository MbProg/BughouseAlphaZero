import numpy as np
import keras
import pickle


class RL_Datapoint():
    def __init__(self, state, policy, values):
        self.state = state
        self.policy = policy
        self.values = values


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = self.list_IDs[indexes] 

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y1 = np.empty((self.batch_size, 2272), dtype=dict)
        y2 = np.empty((self.batch_size), dtype=dict)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            with open('dataset/' + str(ID) + '.pkl','rb') as pkl_file:
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
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation
from keras import optimizers
from keras.callbacks import ModelCheckpoint

CLASSES_LEN = 2272
channel_pos = 'channels_last'
inp_shape = (60,8,8) # TODO: this should be read from the environment
inp = Input(inp_shape)
# Block 1
x = __VGG_Conv2DBlock(64, (3,3), 'relu', 'same',channel_pos, inp, 2, inp_shape=inp_shape)
# Block 2 
x = __VGG_Conv2DBlock(96, (3,3), 'relu', 'same',channel_pos, x, 2)
# Block 3
x = __VGG_Conv2DBlock(128, (3,3), 'relu', 'same',channel_pos, x)
# Block 4
x = __VGG_Conv2DBlock(160, (3,3), 'relu', 'same',channel_pos, x)
# Block 5 
x = __VGG_Conv2DBlock(190, (3,3), 'relu', 'same',channel_pos, x)

x = Flatten()(x)
dense_1 = Dense(128, activation='relu')(x)

value = Dense(1, activation='tanh', name='value')(dense_1)
policy = Dense(CLASSES_LEN, activation='softmax', name='policy')(dense_1)

model = Model(inp, [policy,value])
sgd = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.1/5, nesterov=False)

model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=sgd,
              metrics=['accuracy'])
              
model.summary()

whole_ids = np.arange(1133611)
np.random.shuffle(whole_ids)
train_ids = whole_ids[:int(len(whole_ids)*0.9)]
val_ids = whole_ids[int(len(whole_ids)*0.9):]
training_generator = DataGenerator(train_ids, 256,(60,8,8))
val_generator = DataGenerator(val_ids, 256,(60,8,8))
filepath="model-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
callbacks_list = [checkpoint]

history = model.fit_generator(generator=training_generator,validation_data=val_generator,epochs=5,callbacks=callbacks_list)
model.save('models\\BughouseNet220620190437.h5')
plotHistory(history)

# with open('dataset/' + str(0) + '.pkl','rb') as pkl_file:
#     rl_datapoint = pickle.load(pkl_file)
#     print(rl_datapoint.policy.shape)
#     print(rl_datapoint.values.shape)
