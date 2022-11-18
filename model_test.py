import tensorflow as tf 
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.merge_ops import merge_outputs, merge
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.estimator import regression 
from tflearn.optimizers import Momentum, Adam, SGD


from parameters import NETWORK, HYPERPARAMS

def build_model(optimizer=HYPERPARAMS.optimizer, optimizer_param=HYPERPARAMS.optimizer_param, 
    learning_rate=HYPERPARAMS.learning_rate, keep_prob=HYPERPARAMS.keep_prob,
    learning_rate_decay=HYPERPARAMS.learning_rate_decay, decay_step=HYPERPARAMS.decay_step):

    if NETWORK.model == 'alexNet':
        return alexNet(optimizer, optimizer_param, learning_rate, keep_prob, learning_rate_decay, decay_step)
    if NETWORK.model == 'resNet':
        return resNet(optimizer, optimizer_param, learning_rate, keep_prob, learning_rate_decay, decay_step)
    if NETWORK.model == 'mlp':
        return MLP(optimizer, optimizer_param, learning_rate, keep_prob, learning_rate_decay, decay_step)
    else:
        print( "ERROR: no model " + str(NETWORK.model))
        exit()

def alexNet(optimizer=HYPERPARAMS.optimizer, optimizer_param=HYPERPARAMS.optimizer_param, 
            learning_rate=HYPERPARAMS.learning_rate, keep_prob=HYPERPARAMS.keep_prob,
            learning_rate_decay=HYPERPARAMS.learning_rate_decay, decay_step=HYPERPARAMS.decay_step):
            
    
    # print('Felix')
    # network = conv_1d(network, 96, 11, strides=4, activation=NETWORK.activation)
    # network = max_pool_1d(network, 3, strides=2)
    
    # network = batch_normalization(network)
    # print('WangJa')
    # network = conv_1d(network, 256, 5, activation=NETWORK.activation)
    # network = max_pool_1d(network, 3, strides=2)
    
    # network = batch_normalization(network)
    # network = conv_1d(network, 384, 3, activation=NETWORK.activation)
    # network = conv_1d(network, 384, 3, activation=NETWORK.activation)
    # network = conv_1d(network, 256, 3, activation=NETWORK.activation)
    # network = max_pool_1d(network, 3, strides=2)
    # network = batch_normalization(network)
    # network = fully_connected(network, 4096, activation='tanh')
    # network = dropout(network, 0.2)
    # network = fully_connected(network, 4096, activation='tanh')
    # network = dropout(network, 0.2)
    # network = fully_connected(network,  NETWORK.output_size, activation='softmax')
    if NETWORK.use_landmarks:
        landmarks_network = input_data(shape=[None, 68, 2], name='input')
        landmarks_network = fully_connected(landmarks_network, 1024, activation=NETWORK.activation)
        landmarks_network = batch_normalization(landmarks_network)
        landmarks_network = fully_connected(landmarks_network, 128, activation=NETWORK.activation)
        landmarks_network = batch_normalization(landmarks_network)
        landmarks_network = fully_connected(landmarks_network, NETWORK.output_size, activation='softmax')
    

    if optimizer == 'momentum':
        optimizer = Momentum(learning_rate=learning_rate, momentum=optimizer_param, 
                    lr_decay=learning_rate_decay, decay_step=decay_step)
    elif optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate, beta1=optimizer_param, beta2=learning_rate_decay)
    else:
        print( "Unknown optimizer: {}".format(optimizer))
    network = regression(network, optimizer=optimizer, loss=NETWORK.loss, learning_rate=learning_rate, name='output')

    return network

def MLP(optimizer=HYPERPARAMS.optimizer, optimizer_param=HYPERPARAMS.optimizer_param, 
            learning_rate=HYPERPARAMS.learning_rate, keep_prob=HYPERPARAMS.keep_prob,
            learning_rate_decay=HYPERPARAMS.learning_rate_decay, decay_step=HYPERPARAMS.decay_step):
            
    network = input_data(shape=[None, 68, 2], name='input')
    print('Felix')
    network = fully_connected(network, 64, activation=NETWORK.activation,
                                 regularizer='L2', weight_decay=0.001)
    network = batch_normalization(network)
    network = fully_connected(network, 64, activation=NETWORK.activation,
                                    regularizer='L2', weight_decay=0.001)
    network = batch_normalization(network)
    network = dropout(network, 0.8)
    network = fully_connected(network, 64, activation='tanh',
                                    regularizer='L2', weight_decay=0.001)
    network = batch_normalization(network)
    # network = dropout(network, 0.8)
    network = fully_connected(network, NETWORK.output_size, activation='softmax')


    

    if optimizer == 'momentum':
        optimizer = Momentum(learning_rate=learning_rate, momentum=optimizer_param, 
                    lr_decay=learning_rate_decay, decay_step=decay_step)
    elif optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate, beta1=optimizer_param, beta2=learning_rate_decay)
    else:
        print( "Unknown optimizer: {}".format(optimizer))
    network = regression(network, optimizer=optimizer, loss=NETWORK.loss, learning_rate=learning_rate, name='output')

    return network