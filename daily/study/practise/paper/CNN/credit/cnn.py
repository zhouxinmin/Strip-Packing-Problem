# ! /usr/bin/env python
# -*- coding: utf-8 --
# ---------------------------
# @Time    : 2017/5/2 18:32
# @Site    : 
# @File    : cnn.py
# @Author  : zhouxinmin
# @Email   : 1141210649@qq.com
# @Software: PyCharm


import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
from sklearn import preprocessing
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params


def pickle_output(vars, file_name):
    import cPickle
    f = open(file_name, 'wb')
    cPickle.dump(vars, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def load_data_set(save_data=False):
    # from sklearn.utils import shuffle
    credit_card = np.loadtxt(open("credit_card_clients_import.csv", "rb"), delimiter=",", skiprows=0)
    data_set_x = credit_card[:, :-1]
    data_set_y = credit_card[:, -1]
    min_max_scaler = preprocessing.MinMaxScaler()
    data_set_x_minmax = min_max_scaler.fit_transform(data_set_x)
    new_data_set_x = []
    for x in data_set_x_minmax:
        a = x.reshape(4, 6)
        b = a.T
        new_data_set_x.append([b])
    # print("shuffle data...")
    # data_set_x, data_set_y = shuffle(data_set_x, data_set_y, random_state=0)
    print("Import was successful !")
    X_train, X_val, X_test = np.array(new_data_set_x[:19200]), np.array(new_data_set_x[19200:24000]), np.array(new_data_set_x[24000:])
    y_train, y_val, y_test = np.array(map(int, data_set_y[:19200])), \
                             np.array(map(int, data_set_y[19200:24000])), \
                             np.array(map(int, data_set_y[24000:]))
    print len(new_data_set_x), len(X_train), len(y_train), len(X_val), len(y_val), len(X_test), len(y_test)
    if save_data:
        pickle_output((X_train, y_train), "train")
        pickle_output((X_val, y_val), "val")
        pickle_output((X_test, y_test), "test")

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 6, 4),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 20 kernels of size 2x3. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=20, filter_size=(6, 1),
        nonlinearity=lasagne.nonlinearities.sigmoid,
        W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    # network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Global pooling layer
    # This layer pools globally across all trailing dimensions beyond the 2nd.
    network = lasagne.layers.GlobalPoolLayer(network)

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=0),
        num_units=100,
        nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=0),
        num_units=2,
        nonlinearity=lasagne.nonlinearities.softmax)

    return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    # print len(inputs), len(targets)
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


def main(model='cnn', num_epochs=5):
    # Load the dataset
    print "Loading data..."
    X_train, y_train, X_val, y_val, X_test, y_test = load_data_set()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs', dtype=theano.config.floatX)
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print "Building model and compiling functions..."

    if model == 'cnn':
        network = build_cnn(input_var)
    else:
        print "Unrecognized model type %r.", model
        return

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    # loss = lasagne.objectives.binary_crossentropy(prediction,target_var)
    loss = loss.mean()
    l2_penalty = regularize_layer_params(network, l2) * 1e-1
    loss += l2_penalty
    acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
                 dtype=theano.config.floatX)

    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.001, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    # test_loss = lasagne.objectives.binary_crossentropy(test_prediction,target_var)

    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], [loss, acc], updates=updates, allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc], allow_input_downcast=True)

    test_pre = theano.function([input_var, target_var], [prediction], on_unused_input='ignore')

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    train_out = open("result/train_loss.txt", 'w')
    val_out = open("result/val_loss.txt", 'w')
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_acc = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            # print type(targets), targets
            err, acc = train_fn(inputs, targets)
            train_err += err
            train_acc += acc
            # train_err += train_fn(inputs, targets)
            train_batches += 1
        train_out.write(str(train_err)+"\r\n")

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
        val_out.write(str(val_err)+"\r\n")

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  training accuracy:\t\t{:.2f} %".format(
            train_acc / train_batches * 100))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    train_out.close()
    val_out.close()
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 5, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
        pre = test_pre(inputs, targets)
        print "预测概率：", pre
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

if __name__ == '__main__':
    main()
