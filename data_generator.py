import numpy as np
import h5py

from utilities import calculate_scalar, scale


class DataGenerator(object):
    
    def __init__(self, hdf5_path, batch_size, fold_for_validation, scale, seed=1234):
        
        self.random_state = np.random.RandomState(seed)

        # Load data
        hf = h5py.File(hdf5_path, 'r')
        self.x = hf['feature'][:]
        self.y = hf['hasbird'][:][:, np.newaxis]
        self.itemids = [e.decode() for e in hf['itemid'][:]]
        self.datasetids = [e.decode() for e in hf['datasetid'][:]]
        self.folds = hf['fold'][:]        
        hf.close()
        
        self.batch_size = batch_size
        self.fold_for_validation = fold_for_validation
        self.scale = scale
        
        (self.mean, self.std) = calculate_scalar(self.x)
        
        self.tr_samples = np.sum(self.folds != fold_for_validation)
        self.va_samples = np.sum(self.folds == fold_for_validation)
        
        (self.tr_indexes, self.va_indexes) = self.get_init_indexes()
        
    def get_init_indexes(self):
        """Get indexes of training and validation data. 
        """
        
        samples = len(self.y)
        tr_indexes = []
        va_indexes = []
        
        for n in range(samples):
            if self.folds[n] == self.fold_for_validation:
                va_indexes.append(n)
            else:
                tr_indexes.append(n)

        tr_indexes = np.array(tr_indexes)
        va_indexes = np.array(va_indexes)
        
        return tr_indexes, va_indexes
        
    def generate_train(self):
        
        batch_size = self.batch_size
        indexes = self.tr_indexes
        samples = len(indexes)
        
        self.random_state.shuffle(indexes)
        
        iteration = 0
        pointer = 0
        
        while True:
            
            # Reset pointer
            if pointer >= samples:
                pointer = 0
                self.random_state.shuffle(indexes)
            
            # Get batch indexes
            batch_indexes = indexes[pointer : pointer + batch_size]
            pointer += batch_size
            
            iteration += 1
            
            if self.scale:
                batch_x = scale(self.x[batch_indexes], self.mean, self.std)
            
            batch_y = self.y[batch_indexes]
            
            yield batch_x, batch_y
        
    def generate_validate(self, data_type, max_iteration):
    
        batch_size = self.batch_size
        
        if data_type == 'train':
            indexes = self.tr_indexes
            
        elif data_type == 'validate':
            indexes = self.va_indexes
            
        else:
            raise Exception("Invalid data_type!")
            
        samples = len(indexes)
        
        iteration = 0
        pointer = 0
        
        while True:
            
            if iteration == max_iteration:
                break
            
            # Reset pointer
            if pointer >= samples:
                break
            
            # Get batch indexes
            batch_indexes = indexes[pointer : pointer + batch_size]
            pointer += batch_size
            
            iteration += 1
            
            if self.scale:
                batch_x = scale(self.x[batch_indexes], self.mean, self.std)
            
            batch_y = self.y[batch_indexes]
            
            yield batch_x, batch_y