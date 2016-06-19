#coding: utf-8
import sys
import os
import time

import numpy as np
np.random.seed(1234)

import theano
import theano.tensor as T

import lasagne
# from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Conv2DLayer, MaxPool2DLayer, InputLayer
from lasagne.layers import DenseLayer, ElemwiseMergeLayer, FlattenLayer
from lasagne.layers import ConcatLayer, ReshapeLayer, get_output_shape
from lasagne.layers import Conv1DLayer, DimshuffleLayer, LSTMLayer, SliceLayer
from lasagne.layers import MaxPool1DLayer, dropout, get_output, get_all_params
from lasagne.updates import adagrad
from lasagne.nonlinearities import softmax, sigmoid, rectify, tanh
from lasagne.objectives import binary_accuracy

def load_dataset(file_path):
    X = []
    y = []
    values = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            phrases = line.split()
            features = [float(datum) for datum in phrases[1:-2]]
            features.append(float(phrases[-2].split(';')[0]))
            X.append([features])
            value = float(phrases[-1])
            values.append(value)
            if value <= 0:
                y.append([0])
            else:
                y.append([1])
    # We reserve the last 10000 training examples for validation.
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.int32)
    values = np.array(values).astype(np.float32)
    train_size = int(0.9 * len(y))
    X_train, X_val  = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    values_train, values_val = values[:train_size], values[train_size:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X, y, X_train, y_train, X_val, y_val, values_train, values_val

def build_convpool_conv1d(input_var, nb_classes, input_size=20, n_chanels=1, activity=softmax):
    """
    Builds the complete network with 1D-conv layer to integrate time from sequences of EEG images.

    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :return: a pointer to the output of last layer
    """
    # Input layer
    network = InputLayer(shape=(None, n_chanels, input_size), input_var=input_var)

    network = Conv1DLayer(network, 1024, 5)

    network = MaxPool1DLayer(network, 2)

    network = Conv1DLayer(network, 512, 5)

    network = MaxPool1DLayer(network, 2)

    network = Conv1DLayer(network, 256, 2)

    network = FlattenLayer(network)

    # A fully-connected layer of 512 units with 50% dropout on its inputs:
    network = DenseLayer(dropout(network, p=.5),
            num_units=64, nonlinearity=rectify)

    # And, finally, the output layer with 50% dropout on its inputs:
    network = DenseLayer(dropout(network, p=.5),
            num_units=nb_classes, nonlinearity=activity)

    return network

def build_convpool_dnn(input_var, nb_classes, input_size=20, n_chanels=1, activity=softmax):
    """
    Builds the complete network with 1D-conv layer to integrate time from sequences of EEG images.

    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :return: a pointer to the output of last layer
    """
    # Input layer
    network = InputLayer(shape=(None, n_chanels, input_size), input_var=input_var)

    network = DenseLayer(dropout(network, p=.2),
            num_units=11, nonlinearity=rectify)

    network = DenseLayer(dropout(network, p=.2),
            num_units=5, nonlinearity=rectify)

    network = DenseLayer(dropout(network, p=.2),
            num_units=nb_classes, nonlinearity=activity)

    return network

def build_convpool_mix(input_var, nb_classes, input_size=20, n_chanels=1, activity=softmax):
    """
    Builds the complete network with 1D-conv layer to integrate time from sequences of EEG images.

    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :return: a pointer to the output of last layer
    """
    # Input layer
    input = InputLayer(shape=(None, n_chanels, input_size), input_var=input_var)

    conv1 = Conv1DLayer(input, 1024, 5)

    pool1 = MaxPool1DLayer(conv1, 2)

    conv2 = Conv1DLayer(pool1, 512, 5)

    pool2 = MaxPool1DLayer(conv2, 2)

    conv3 = Conv1DLayer(pool2, 256, 2)

    conv_layer = FlattenLayer(conv3)

    rnn_input = DimshuffleLayer(conv2, (0, 2, 1))

    rnnpool = LSTMLayer(rnn_input, num_units=256, nonlinearity=tanh)

    rnn_layer = SliceLayer(rnnpool, -1, 1)

    network = ConcatLayer([conv_layer, rnn_layer])

    # A fully-connected layer of 512 units with 50% dropout on its inputs:
    network = DenseLayer(dropout(network, p=.5),
            num_units=64, nonlinearity=rectify)

    # And, finally, the output layer with 50% dropout on its inputs:
    network = DenseLayer(dropout(network, p=.5),
            num_units=nb_classes, nonlinearity=activity)

    return network

def build_convpool_cascade(input_var, nb_classes, input_size=20, n_chanels=1, activity=softmax):
    """
    Builds the complete network with 1D-conv layer to integrate time from sequences of EEG images.

    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :return: a pointer to the output of last layer
    """
    # Input layer
    network = InputLayer(shape=(None, n_chanels, input_size), input_var=input_var)

    network = Conv1DLayer(network, 1024, 5)

    network = MaxPool1DLayer(network, 2)

    network = DimshuffleLayer(network, (0, 2, 1))

    network = LSTMLayer(network, num_units=256, grad_clipping=100, nonlinearity=tanh)

    network = SliceLayer(network, -1, 1)

    # A fully-connected layer of 512 units with 50% dropout on its inputs:
    network = DenseLayer(dropout(network, p=.5),
            num_units=64, nonlinearity=rectify)

    # And, finally, the output layer with 50% dropout on its inputs:
    network = DenseLayer(dropout(network, p=.5),
            num_units=nb_classes, nonlinearity=activity)

    return network

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################# Self define loss ###############################
def self_binary_crossentropy(output, target, possitive_punishment=1):
    """
    更大的正例惩罚力度
    Compute the crossentropy of binary random variables.
    Output and target are each expectations of binary random
    variables; target may be exactly 0 or 1 but output must
    lie strictly between 0 and 1.
    Notes
    -----
    We could use the x log y op to support output=0 and output=1.
    The gradient would still be undefined though.
    We do not sum, crossentropy is computed by component.
    TODO : Rewrite as a scalar, and then broadcast to tensor.
    """
    return -(possitive_punishment * target * T.log(output) + (1.0 - target) * T.log(1.0 - output))


def run_dnn(learning_rate=0.001, dnn_strategy='mix', possitive_punishment=1):
    #input_var = T.TensorType('float32', ((False,) * 3))()        # Notice the () at the end
    input_var = T.ftensor3('X')
    target_var = T.imatrix('y')
    network = build_convpool_mix(input_var, 1, activity=sigmoid)
    if dnn_strategy == 'dnn':
        build_convpool_dnn(input_var, 1, activity=sigmoid)
    elif dnn_strategy == 'conv1d':
        build_convpool_conv1d(input_var, 1, activity=sigmoid)
    elif dnn_strategy == 'cascade':
        build_convpool_cascade(input_var, 1, activity=sigmoid)
    elif dnn_strategy == 'mix':
        pass
    else:
        raise AttributeError("This dnn_strategy is not supported!")

    l_output = get_output(network)
    loss = self_binary_crossentropy(l_output, target_var, possitive_punishment=possitive_punishment).mean()

    all_params = get_all_params(network, trainable=True)
    updates = adagrad(loss, all_params, learning_rate=learning_rate)
    train = theano.function([input_var, target_var], loss, updates=updates)

    test_prediction = get_output(network, deterministic=True)
    test_loss = self_binary_crossentropy(test_prediction, target_var, possitive_punishment=possitive_punishment).mean()
    #test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
    test_acc = binary_accuracy(test_prediction, target_var).mean()

    #calculate win rate
    win_rate_result1 = []
    win_rate_result2 = []
    for win_rate_threhold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        tmp1 = T.sum(T.switch(T.and_(T.gt(test_prediction, win_rate_threhold), T.eq(target_var, 1)), 1, 0), dtype=theano.config.floatX)
        tmp2 = T.sum(T.switch(T.gt(test_prediction, win_rate_threhold), 1, 0), dtype=theano.config.floatX)
        test_win_rate = (tmp1 + 0.00001) / (tmp2 + 0.00001)
        win_rate_result1.append(test_win_rate)
        win_rate_result2.append(tmp1)

    val = theano.function([input_var, target_var], [test_prediction, test_loss, test_acc, T.as_tensor_variable(win_rate_result1), T.as_tensor_variable(win_rate_result2)])

    _, _, X_train, y_train, X_val, y_val, _, _ = load_dataset('../../data/data.txt')
    test_data, test_label, _, _, _, _, _, _ = load_dataset('../../data/test_data.txt')

    num_epochs = 200
    batch_size = 128
    for epoch in xrange(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()

        #train
        for batch in iterate_minibatches(X_train, y_train, batch_size):
            inputs, targets = batch
            err= train(inputs, targets)
            train_err += err
            train_batches += 1

        #validate
        _, val_err, val_acc, _, _ = val(X_val, y_val)

        #predict
        predict_result, loss, acc, win_rate_result1, win_rate_result2 = val(test_data, test_label)

        # Then we print the results for this epoch:
        sys.stdout.write("Epoch {} of {} took {:.3f}s\n".format(
            epoch + 1, num_epochs, time.time() - start_time))
        sys.stdout.write("  training loss:\t\t{:.6f}\n".format(train_err / train_batches))
        sys.stdout.write("  validation loss:\t\t{}\n".format(val_err))
        sys.stdout.write("  validation accuracy:\t\t{:.2f} %\n".format(val_acc * 100))
        sys.stdout.write('loss is {} and acc is {}\n'.format(loss, acc))
        for ix in xrange(len(win_rate_result1)):
            sys.stdout.write(
                'win_rate is {} and the possitive num is {}\n'.format(win_rate_result1[ix], win_rate_result2[ix]))
        sys.stdout.flush()



    '''
    with open('128_predict_result', 'w') as f:
        for ix in xrange(len(test_label)):
            f.write('{} {}\n'.format(test_label[ix], predict_result[ix]))
    sys.stdout.flush()

    print 'Done!'
    '''

if __name__ == '__main__':
    learning_rate_list = [0.001]
    dnn_strategy_list = ['mix']
    possitive_punishment_list = [1]

    for dnn_strategy in dnn_strategy_list:
        for learning_rate in learning_rate_list:
            for possitive_punishment in possitive_punishment_list:
                sys.stdout.write('learning_rate is {}, dnn_strategy is {}, possitive_punishment is {}\n'.format(learning_rate, dnn_strategy, possitive_punishment))
                sys.stdout.flush()
                run_dnn(learning_rate, dnn_strategy, possitive_punishment)
