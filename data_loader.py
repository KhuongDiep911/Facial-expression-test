from parameters import DATASET, NETWORK
import numpy as np


def load_data(validation=False, test=False):
    
    training_dict = dict()
    validation_dict = dict()
    test_dict = dict()

    if DATASET.name == "Fer2013":

        # load training dataset
        training_dict['X'] = np.load(DATASET.train_folder + "/images.npy")
        training_dict['X'] = training_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
        if NETWORK.use_landmarks:
            training_dict['X2'] = np.load(DATASET.train_folder + "/landmarks.npy")

        # load validation dataset
        if validation:
            validation_dict['X'] = np.load(DATASET.train_folder + "/images.npy")
            validation_dict['X'] = validation_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
            if NETWORK.use_landmarks:
                validation_dict['X2'] = np.load(DATASET.validation_folder + "/landmarks.npy")
        
        # load test dataset
        if test:
            test_dict['X'] = np.load(DATASET.train_folder + "/images.npy")
            test_dict['X'] = test_dict['X'].reshape([-1, NETWORK.input_size, NETWORK.input_size, 1])
            if NETWORK.use_landmarks:
                test_dict['X2'] = np.load(DATASET.test_folder + "/landmarks.npy")
        
        if not validation and not test:
            return training_dict
        elif not test:
            return training_dict, validation_dict
        else: 
            return training_dict, validation_dict, test_dict
    else:
        print( "Unknown dataset")
        exit()