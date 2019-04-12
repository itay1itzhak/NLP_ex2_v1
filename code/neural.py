#!/usr/bin/env python

import numpy as np
import random

from softmax import softmax
from sigmoid import sigmoid, sigmoid_grad
from gradcheck import gradcheck_naive

def forward(data, label, params, dimensions):
    """
    runs a forward pass and returns the probability of the correct word for eval.
    label here is an integer for the index of the label.
    This function is used for model evaluation.
    """
    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    params[ofs:ofs+ Dx * H]
    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Compute the probability
    ### YOUR CODE HERE: forward propagation
    z1 = np.dot(data,W1) + b1
    h = sigmoid(z1)
    z2 = np.dot(h,W2) + b2
    y_hat = softmax(z2)
    J = -np.sum(label*np.log(y_hat))
    return {"z1": z1,
            "h": h,
            "z2": z2,
            "y_hat": y_hat,
            "J": J}
    ### END YOUR CODE

def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    forward_results = forward(data, labels, params, dimensions)
    z1 = forward_results['z1']
    h = forward_results['h']
    z2 = forward_results['z2']
    y_hat = forward_results['y_hat']
    cost = forward_results['J']
    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    grad_softmax = y_hat - labels
    grad_hidden = np.dot(grad_softmax,W2.T) * sigmoid_grad(h)

    gradW1 = np.dot(data.T,grad_hidden)
    gradb1 = np.sum(grad_hidden,axis=0)
    gradW2 = np.dot(h.T,grad_softmax)
    gradb2 = np.sum(grad_softmax,axis=0)

    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    #raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
