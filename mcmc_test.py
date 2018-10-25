#!/usr/bin/env python3
# pylint: disable=R0914,C0103,R0915,C0301
"""
    mcmc test
"""

import numpy.random as nprand
import scipy.stats as ss
from matplotlib import pyplot as plt

# set target value
target_value = 90
target_std = 25

# set initial value
current_value = 80
value_chain = []
value_chain.append(current_value)

# create proposal distribution N(0,25)
noise = lambda: 5*nprand.randn()

# Generate proposed value
for i in range(1000):
    proposed_value = current_value + noise()

    # Find the height of target at proposed and current value
    proposed_height = ss.norm(proposed_value, target_std).pdf(target_value)
    current_height = ss.norm(current_value, target_std).pdf(target_value)

    # accept if proposed_height > current_height
    if proposed_height > current_height:
        current_value = proposed_value
        value_chain.append(current_value)
    else: # only accept by probability if proposed height < current_height
        accept_prob = proposed_height/current_height
        if nprand.rand() < accept_prob:
            current_value = proposed_value
            value_chain.append(current_value)
        else:
            value_chain.append(current_value)

# generate histogram
plt.hist(value_chain, 100)
plt.show()
