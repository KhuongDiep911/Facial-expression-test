import os


class Dataset:
    name = 'Fer2013'
    train_folder = 'fer2013-dataset/Training'
    validation_folder = 'fer2013-dataset/PublicTest'
    test_folder = 'fer2013-dataset/PrivateTest'
    shape_predictor_path='shape_predictor_68_face_landmarks.dat'
    trunc_trainset_to = -1  # put the number of train images to use (-1 = all images of the train set)
    trunc_validationset_to = -1
    trunc_testset_to = -1

class Network:
    model = 'mlp' # {'alexNet', 'mlp'}
    input_size = 48
    output_size = 7
    activation = 'relu'
    loss = 'categorical_crossentropy'
    use_landmarks = True
    # use_hog_and_landmarks = True
    # use_hog_sliding_window_and_landmarks = True
    # use_batchnorm_after_conv_layers = True
    # use_batchnorm_after_fully_connected_layers = False

class Hyperparams:
    keep_prob = 0.956   # dropout = 1 - keep_prob
    learning_rate = 0.016
    learning_rate_decay = 0.864
    decay_step = 50
    optimizer = 'adam'  # {'momentum', 'adam', 'rmsprop', 'adagrad', 'adadelta'}
    optimizer_param = 0.95   # momentum value for Momentum optimizer, or beta1 value for Adam

class Training:
    batch_size = 64
    epochs = 100
    snapshot_step = 500
    vizualize = True
    logs_dir = "logs"
    checkpoint_dir = "checkpoints/chk"
    best_checkpoint_path = "checkpoints/best/"
    max_checkpoints = 1
    checkpoint_frequency = 1.0 # in hours
    save_model = True
    save_model_path = "best_model/saved_model.bin"

def make_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

DATASET = Dataset()
NETWORK = Network()
TRAINING = Training()
HYPERPARAMS = Hyperparams()
# VIDEO_PREDICTOR = VideoPredictor()
# OPTIMIZER = OptimizerSearchSpace()

make_dir(TRAINING.logs_dir)
make_dir(TRAINING.checkpoint_dir)