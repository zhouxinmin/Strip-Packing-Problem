from __future__ import print_function

import sys
import os
import time

import theano
import theano.tensor as T
# from sklearn import preprocessing

import lasagne
import numpy as np
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params


def cPickle_output(vars, file_name):
    import cPickle
    f = open(file_name, 'wb')
    cPickle.dump(vars, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def load_dataset(savedata = False):
    import scipy.io as sio
    # from sklearn.utils import shuffle
    matfn = 'left_right.mat'
    data = sio.loadmat(matfn)
    datas = []
    labels = []
    left1 = np.array(data['left1'])
    left1 = left1.transpose(2, 0, 1)
    left1 = left1.reshape(60,3*750)
    left2 = np.array(data['left2'])
    left2 = left2.transpose(2, 0, 1)
    left2 = left2.reshape(60, 3 * 750)
    left3 = np.array(data['left3'])
    left3 = left3.transpose(2, 0, 1)
    left3 = left3.reshape(80, 3 * 750)

    right1 = np.array(data['right1'])
    right1 = right1.transpose(2, 0, 1)
    right1 = right1.reshape(60,3*750)
    right2 = np.array(data['right2'])
    right2 = right2.transpose(2, 0, 1)
    right2 = right2.reshape(60, 3 * 750)
    right3 = np.array(data['right3'])
    right3 = right3.transpose(2, 0, 1)
    right3 = right3.reshape(80, 3 * 750)
    lable0 = np.zeros(200)
    lable1 = np.ones(200)
    datas.append(left1)
    datas.append(left2)
    datas.append(left3)
    datas.append(right1)
    datas.append(right2)
    datas.append(right3)
    labels.append(lable0)
    labels.append(lable1)
    data_set = np.vstack(datas)
    label_set = np.hstack(labels)
    print("shuffle data...")
    data_set_x = np.array(data_set)
    data_set_y = np.array(label_set)
    # data_set_x, data_set_y = shuffle(data_set_x, data_set_y, random_state=0)
    print("shuffle data done")
    print (np.shape(data_set))
    print(np.shape(label_set))
    # data_set_x = data_set_x.reshape([200, 47200])
    data_set_x = data_set_x.reshape([400, 3*750])
    # data_set_x = (data_set_x*0.1)**2
    data_set_x = (data_set_x) ** 2
    # data_mean = np.mean(data_set_x, 3)
    data_tmp = data_set_x.reshape(400 * 3, 750)
    data = []
    for j in range(50):
        data.append(np.mean(data_tmp[:, j * 15:(j + 1) * 15], 1))
    data = np.array(data)
    data = data.transpose(1, 0)
    data_set_x = data.reshape(400, 3, 50)
    print (np.shape(data_set_x))
    # min_max_scaler = preprocessing.MinMaxScaler()
    # data_set_x = min_max_scaler.fit_transform(data_set_x)
    data_set_x = data_set_x.reshape([400, 1, 3, 50])


    # data_mean = np.mean(data_set_x,3)
    # data_mean = np.repeat(data_mean,2000,2)
    # data_mean = data_mean.reshape(96,1,2,2000)
    # data_std = np.std(data_set_x,3)
    # data_std = np.repeat(data_std,2000,2)
    # data_std = data_std.reshape(96,1,2,2000)
    # data_set_x = (data_set_x - data_mean)/data_std

    X_train, X_val = data_set_x[:-50], data_set_x[-50:]
    y_train, y_val = data_set_y[:-50], data_set_y[-50:]
    if(savedata):
        cPickle_output((X_train, y_train), "train")
        cPickle_output((X_val, y_val), "val")

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val,X_val,y_val


def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 3, 50),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=2, filter_size=(3, 1),
            nonlinearity=lasagne.nonlinearities.sigmoid,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    # network = lasagne.layers.Conv2DLayer(
    #         network, num_filters=50, filter_size=(1,1),
    #         nonlinearity=lasagne.nonlinearities.sigmoid,
    #         W=lasagne.init.GlorotUniform())
    # network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1,10))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    # network = lasagne.layers.Conv2DLayer(
    #         network, num_filters=5, filter_size=(1, 5),
    #         nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1, 5))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    # network = lasagne.layers.DenseLayer(
    #         lasagne.layers.dropout(network, p=0),
    #         num_units=10,
    #         nonlinearity=lasagne.nonlinearities.sigmoid,
            # W=lasagne.init.GlorotUniform())
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=0),
        num_units=100,
        # nonlinearity=lasagne.nonlinearities.sigmoid,
        W=lasagne.init.GlorotUniform())
    # network = lasagne.layers.DenseLayer(
    #     lasagne.layers.dropout(network, p=0),
    #     num_units=100,
    #     # nonlinearity=lasagne.nonlinearities.sigmoid,
    #     W=lasagne.init.GlorotUniform())
    # network = lasagne.layers.DenseLayer(
    #     lasagne.layers.dropout(network, p=0),
    #     num_units=100,
    #     # nonlinearity=lasagne.nonlinearities.sigmoid,
    #     W=lasagne.init.GlorotUniform())

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

def main(model='cnn', num_epochs=100000):
    # Load the dataset
    # print("Loading data...")
    # X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs', dtype=theano.config.floatX)
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")

    if model == 'cnn':
        network = build_cnn(input_var)
    else:
        print("Unrecognized model type %r." % model)
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
    train_fn = theano.function([(input_var), target_var], [loss,acc], updates=updates,allow_input_downcast=True)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([(input_var), target_var], [test_loss, test_acc],allow_input_downcast=True)

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
        for batch in iterate_minibatches(X_train, y_train, 50, shuffle=True):
            inputs, targets = batch
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
        for batch in iterate_minibatches(X_val, y_val, 5, shuffle=False):
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
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

if __name__ == '__main__':
    main()
