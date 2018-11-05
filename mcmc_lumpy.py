#!/usr/bin/env python3
# pylint: disable=R0914,C0103,R0915,C0301
"""
    mcmc lumpy
"""
import numpy as np
import numpy.random as npr
#import matplotlib.pyplot as plt

class markov_chain:
    """
        The markov chain for storing lumpy background representations
    """
    def __init__(self):
        self._chain = [] # Create a list to store elements to chain

    def add(self, element):
        """ adds element to the chain """
        self._chain.append(element)

    def get(self):
        """ returns the entire chain """
        return self._chain

class phi_matrix:
    """
        a representation of theta
    """
    def __init__(self, N):
        # create phi matrix
        self._phi = np.zeros(N, 3)


def create_lumpy_background(dim, Nbar, DC, magnitude, stddev):
    """
        Creates a lumpy background
    """

    # initialize image
    b = DC*np.ones((dim, dim))

    # N is the number of lumps
    N = npr.poisson(Nbar)

    # create list to store pos
    pos = []

    # Create lumpy background image
    for _ in range(N):
        pos.append(npr.rand(2, 1)*dim)
        X, Y = np.meshgrid(
            [i - pos[-1][0] for i in range(dim)],
            [i - pos[-1][1] for i in range(dim)])
        lmp = magnitude*np.exp(-0.5*(X**2+Y**2)/stddev**2)
        b = b + lmp
    return b, N, pos

def proposal_density():
    """
        Returns a proposal from proposal_density
    """

    return

def calculate_posterior():
    """
        Calculates the posterior probability
    """

    return

def acceptance():
    """
        Returns the theta using the acceptance/rejection decision
    """

    return
