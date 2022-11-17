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
    input_size = 48
    output_size = 72
    activation = 'relu'
    loss = 'categorical_crossentropy'
    use_landmarks = True
    # use_hog_and_landmarks = True
    # use_hog_sliding_window_and_landmarks = True
    # use_batchnorm_after_conv_layers = True
    # use_batchnorm_after_fully_connected_layers = False