#!/usr/bin/env python3
# pylint: disable=R0914,C0103,R0915,C0301,W0611,W0102,R0913,R0902
"""
    mcmc lumpy
"""
from concurrent.futures import ProcessPoolExecutor
import pickle
import numpy as np
import numpy.random as npr
import scipy.stats as ss
import scipy.ndimage as snd
from matplotlib import pyplot as plt

class phi_matrix:
    """
        a representation of theta
    """
    def __init__(self, centers, g, n, h, stddev=10, var_noise=0.1, dim=64, Nbar=10):
        # Save dim
        self._dim = dim

        # Save var noise
        self._var_noise = var_noise

        # Save Nbar
        self._Nbar = Nbar

        # create list to store theta
        self._theta = []

        # Set Nstar
        Nstar = len(centers)

        # set the N'
        self._Nprime = Nstar*100

        # set flip probability
        self._eta = 0.04/self._Nprime

        # set gaussian shift density (1/10 of lump function width)
        self._gsd = stddev/10

        # create phi matrix
        self._phi = np.zeros((self._Nprime, 3))

        # create gaussian noise
        self._noise = n

        # assign centers to phi matrix
        for i, pos in enumerate(centers):
            self._phi[i, 1:3] = np.squeeze(pos)
            self._phi[i, 0] = 1

        # randomly assign centers in the phi matrix
        for i, _ in enumerate(self._phi):
            # check if already assigned
            if self._phi[i, 0] != 1:
                # assign a random center
                self._phi[i, 1:3] = npr.rand(2)*dim

        # Save the current phi_matrix in theta
        self._theta.append(np.copy(self._phi))

        # Save g into a variable
        self._g = g

        # save system psf
        self._h = h

    def shift_centers(self):
        """
            shifts center of a single alpha element = 1
        """
        # get realized lumps from previous realization
        pos = [center[1:3] for center in self._theta[-1] if center[0] == 1]

        # get the number of lumps set
        N = len(pos)

        # randomly choose a lump to shift
        choice = npr.choice(N)

        # shift the correct lump
        i = 0
        for n, _ in enumerate(self._phi):
            if self._phi[n, 0] == 1 and self._theta[-1][n, 0] == 1:
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

    def grab_chain(self, real=True):
        """
            returns the markov chain
        """
        if real: # only return entries where alpha = 1
            return [np.array([row for row in theta if row[0] == 1]) for theta in self._theta]
        # return everything
        return self._theta

    def grab_g(self):
        """
            returns image g
        """
        return self._g

    def acceptance(self):
        """
            Updates the markov chain using the acceptance/rejection decision
        """
        # calculate posterior ratio and calculate probability acceptance
        p1a, p2a = self._calculate_posterior_components(theta=self._phi)
        p1b, p2b = self._calculate_posterior_components(theta=self._theta[-1])
        posterior_ratio = np.prod(p1a/p1b)*(p2a/p2b)
        prob_accept = np.minimum(1, posterior_ratio)

        # add new if within probability
        if npr.rand() < prob_accept:
            # print('added new')
            self._add_new()
        # add previous if not within probability
        else:
            # print('added last')
            self._add_last()
            # reset phi to last theta
            self._phi = np.copy(self._theta[-1])

    def _calculate_posterior_components(self, theta):
        """
            Calculates the posterior probability
        """
        # Get number of lumps
        b, N, _ = create_lumpy_background(pos=theta)

        # create signal absent image
        s = snd.filters.gaussian_filter(b, self._h) + self._noise

        prgbH0 = ss.norm(s, self._var_noise**0.5).pdf(self._g).ravel()

        # calculate pr(N)*pr({c_n})
        prNprcn = np.exp(-self._Nbar)*(self._Nbar/(self._dim*self._dim))**N

        # return components
        return prgbH0, prNprcn

    def _add_new(self):
        """
            adds the proposed phi to the markov chain
        """
        self._theta.append(np.copy(self._phi))

    def _add_last(self):
        """
            adds the last phi to the chain
        """
        self._theta.append(self._theta[-1])

def calculate_BKE(g, b, s, var):
    """
        calculate the BKE likelihood ratio
    """
    # unpack arrays to 1 dimension
    g1 = g.ravel()
    b1 = b.ravel()
    s1 = s.ravel()
    K_inv = np.eye(g1.shape[0])*(1/var)
    breakpoint()

    # return the likelihood ratio
    return np.exp(np.dot((g1-b1-s1/2), np.matmul(K_inv, s1)))

def create_lumpy_background(Nbar=10, DC=20, magnitude=1, stddev=10, dim=64, pos=[]):
    """
        Creates a lumpy background
    """

    # initialize image
    b = DC*np.ones((dim, dim))

    # if pos empty generate a new background
    if isinstance(pos, list):
        # create realized pos list
        real_pos = []

        # N is the number of lumps
        N = npr.poisson(Nbar)

        # Create lumpy background image
        for _ in range(N):
            real_pos.append(npr.rand(2, 1)*dim)
            X, Y = np.meshgrid(
                [i - real_pos[-1][0] for i in range(dim)],
                [i - real_pos[-1][1] for i in range(dim)])
            lmp = magnitude*np.exp(-0.5*(X**2+Y**2)/stddev**2)
            b = b + lmp
    else: # use the provided pos to generate background using settings
        # get only realized lumps
        real_pos = [center[1:3] for center in pos if center[0] == 1]

        # get the number of lumps set
        N = len(real_pos)

        # Loop over positions to generate background
        for center in real_pos:
            X, Y = np.meshgrid(
                [i - center[0] for i in range(dim)],
                [i - center[1] for i in range(dim)])
            lmp = magnitude*np.exp(-0.5*(X**2+Y**2)/stddev**2)
            b = b + lmp

    # return backgroun, number of lumps, lump positions
    return b, N, real_pos

def run_mcmc(phi, signal, var_noise, h, skip_iterations, iterations):
    """
        Runs the mcmc for one image
    """

    # generate markow chain
    for _ in range(iterations):
        phi.flip_and_shift()
        phi.acceptance()

    # Get g
    g = phi.grab_g()
    cum_ratio = 0
    for i in range(skip_iterations, iterations):
        pos = phi.grab_chain(real=False)[i]
        b, _, _ = create_lumpy_background(pos=pos)
        ratio = calculate_BKE(g, snd.filters.gaussian_filter(b, h), signal, var_noise)
        cum_ratio += ratio
    lr = cum_ratio/(iterations-skip_iterations)

    # return ratio
    return lr

def main():
    """
        Main
    """
    # Variables
    signal_intensity = 0.1
    var_noise = 0.1
    dim = 64
    gaussian_sigma = 2
    obj_dim1 = [28, 33]
    obj_dim2 = [29, 32]
    num_examples = 100 # for each set
    skip_iterations = 500
    iterations = 150000

    # Create signal
    signal = np.zeros((64, 64))
    signal[obj_dim1[0]:obj_dim1[1], obj_dim2[0]:obj_dim2[1]] = signal_intensity
    signal[obj_dim2[0]:obj_dim2[1], obj_dim1[0]:obj_dim1[1]] = signal_intensity
    signal = snd.filters.gaussian_filter(signal, gaussian_sigma)

    # make images
    phi_set = []
    for k in range(num_examples):
        print(k)
        # signal present
        b, _, pos = create_lumpy_background()
        noise = npr.normal(0, var_noise**0.5, (dim, dim))
        g = signal + snd.filters.gaussian_filter(b, gaussian_sigma) + noise
        phi_set.append(phi_matrix(centers=pos, g=g, n=noise, h=gaussian_sigma))

        # signal absent images
        b, _, pos = create_lumpy_background()
        noise = npr.normal(0, var_noise**0.5, (dim, dim))
        g = snd.filters.gaussian_filter(b, gaussian_sigma) + noise
        phi_set.append(phi_matrix(centers=pos, g=g, n=noise, h=gaussian_sigma))

    # calculate lambdas
    lr = []

    # single process
    # for k, phi in enumerate(phi_set):
    #     print(k)
    #     lr.append(run_mcmc(phi, signal, var_noise, gaussian_sigma, skip_iterations, iterations))

    # multiprocess
    job = []
    with ProcessPoolExecutor(max_workers=10) as e:
        for k, phi in enumerate(phi_set):
            print(k)
            job.append(e.submit(run_mcmc, phi, signal, var_noise, gaussian_sigma, skip_iterations, iterations))
        for j in job:
            lr.append(j.result())

    with open('ratios.pkl', 'wb') as f:
        pickle.dump(lr, f)

if __name__ == "__main__":
    main()
