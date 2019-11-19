import  tensorflow.keras as keras
import numpy as np

class DataLoader(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle()
        
    def shuffle(self):
        indeces = np.random.permutation(self.x.shape[0])
        return self.x[indeces], self.y[indeces]
        
    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            np.array(np.load("../data/processed/" + str(file_name)+ ".npy")) for file_name in batch_x]), np.array(batch_y)
