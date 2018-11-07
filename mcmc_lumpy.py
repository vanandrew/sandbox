#!/usr/bin/env python3
# pylint: disable=R0914,C0103,R0915,C0301,W0611,W0102,R0913
"""
    mcmc lumpy
"""
from copy import deepcopy
import numpy as np
import numpy.random as npr
import scipy.stats as ss
import matplotlib.pyplot as plt

class phi_matrix:
    """
        a representation of theta
    """
    def __init__(self, centers, stddev=10, Nstar=10, dim=64):
        # create list to store theta
        self._theta = []

        # set the N'
        self._Nprime = Nstar*100

        # set flip probability
        self._eta = 0.04/self._Nprime

        # set gaussian shift density (1/10 of lump function width)
        self._gsd = stddev/10

        # create phi matrix
        self._phi = np.zeros((self._Nprime, 3))

        # assign centers to phi matrix
        for n, pos in enumerate(centers):
            self._phi[n, 1:3] = np.squeeze(pos)
            self._phi[n, 0] = 1

        # randomly assign centers in the phi matrix
        for n, _ in enumerate(self._phi):
            # check if already assigned
            if self._phi[n, 0] != 1:
                # assign a random center
                self._phi[n, 1:3] = npr.rand(2)*dim

        # Save the current phi_matrix in theta
        self._theta.append(deepcopy(self._phi))

    def shift_centers(self):
        """
            shifts center of a single alpha element = 1
        """
        # get only realized lumps
        pos = [center[1:3] for center in self._phi if center[0] == 1]

        # get the number of lumps set
        N = len(pos)

        # randomly choose a lump to shift
        choice = npr.choice(N)
        print(choice)

        # shift the correct lump
        i = 0
        for n, _ in enumerate(self._phi):
            if self._phi[n, 0] == 1:
                if i == choice:
                    self._phi[n, 1] = self._phi[n, 1] + npr.randn()*self._gsd
                    self._phi[n, 2] = self._phi[n, 2] + npr.randn()*self._gsd
                    break
                else:
                    i = i + 1

    def flip_alpha(self):
        """
            flips an alpha with probability eta
        """
        for n, _ in enumerate(self._phi):
            if npr.rand() < self._eta:
                self._phi[n, 0] = int(not self._phi[n, 0])

    def flip_and_shift(self):
        """
            flips and shifts centers
        """
        self.flip_alpha()
        self.shift_centers()

    def grab_chain(self):
        """
            returns the markov chain
        """
        return self._theta

    def acceptance(self):
        """
            Updates the markov chain using the acceptance/rejection decision
        """
        # calculate posterior ratio and calculate probability acceptance
        p1a, p2a = self._calculate_posterior_components(theta=self._phi)
        p1b, p2b = self._calculate_posterior_components(theta=self._theta[-1])
        posterior_ratio = np.prod(p1a/p1b)*(p2a/p2b)
        print(posterior_ratio)
        prob_accept = np.minimum(1, posterior_ratio)
        print(prob_accept)

        # add new if within probability
        if npr.rand() < prob_accept:
            self._add_new()
        # add previous if not within probability
        else:
            self._add_last()
            # reset phi to last theta
            self._phi = self._theta[-1]

    @staticmethod
    def _calculate_posterior_components(theta, var_noise=0.1, dim=64, Nbar=10):
        """
            Calculates the posterior probability
        """
        # generate the mumpy background using theta
        b, N, pos = create_lumpy_background(pos=theta)
        print(pos)

        # add gaussian noise
        gH0 = b + npr.normal(0, var_noise**0.5, (dim, dim))

        # calculate the probabiltiy g given b
        prgbH0 = ss.norm(0, var_noise**0.5).pdf(gH0 - b).ravel()

        # calculate pr(N)*pr({c_n})
        prNprcn = np.exp(-Nbar)*(Nbar/(dim*dim))**N

        # return components
        return prgbH0, prNprcn

    def _add_new(self):
        """
            adds the proposed phi to the markov chain
        """
        self._theta.append(deepcopy(self._phi))

    def _add_last(self):
        """
            adds the last phi to the chain
        """
        self._theta.append(self._theta[-1])

def create_lumpy_background(Nbar=10, DC=20, magnitude=1, stddev=10, dim=64, pos=[]):
    """
        Creates a lumpy background
    """

    # initialize image
    b = DC*np.ones((dim, dim))

    # if pos empty generate a new background
    if isinstance(pos, list):
        # N is the number of lumps
        N = npr.poisson(Nbar)

        # Create lumpy background image
        for _ in range(N):
            pos.append(npr.rand(2, 1)*dim)
            X, Y = np.meshgrid(
                [i - pos[-1][0] for i in range(dim)],
                [i - pos[-1][1] for i in range(dim)])
            lmp = magnitude*np.exp(-0.5*(X**2+Y**2)/stddev**2)
            b = b + lmp
    else: # use the provided pos to generate background using settings
        # get only realized lumps
        pos = [center[1:3] for center in pos if center[0] == 1]

        # get the number of lumps set
        N = len(pos)

        # Loop over positions to generate background
        for center in pos:
            X, Y = np.meshgrid(
                [i - center[0] for i in range(dim)],
                [i - center[1] for i in range(dim)])
            lmp = magnitude*np.exp(-0.5*(X**2+Y**2)/stddev**2)
            b = b + lmp

    # return backgroun, number of lumps, lump positions
    return b, N, pos

def main():
    """
        Main
    """
    _, _, pos = create_lumpy_background()
    phi = phi_matrix(centers=pos)
    for i in range(5000):
        print("")
        print(i)
        phi.flip_and_shift()
        phi.acceptance()
    #print(phi.grab_chain())



if __name__ == "__main__":
    main()
