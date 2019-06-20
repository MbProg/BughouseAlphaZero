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
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
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
        y = np.empty((self.batch_size), dtype=dict)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            with open('dataset/' + str(ID) + '.pkl','rb') as pkl_file:
                try:
                    rl_datapoint = pickle.load(pkl_file)
                except EOFError:
                    break

            # Store sample
            X[i,] = rl_datapoint.state.matrice_stack

            # Store class
            y[i] = {'policy': rl_datapoint.policy, 'value': rl_datapoint.values} # [rl_datapoint.policy,rl_datapoint.values]

        return X, y

dg = DataGenerator(np.arange(4997), None, dim=(60,8,8))
dg.on_epoch_end()
dg.__getitem__(0)