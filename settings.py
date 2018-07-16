"""
Global settings
"""
import task
import os
import json
import numpy as np
import random
import time
import method
import getopt
import sys
import argparse



"""
Settings.py -- a file with all of the project settings.
This includes the rehearsal algorithm used, number of neurons, etc.

These settings are implemented in `main.py` and `rehearsal.py`

N.B. Settings can be overwritten by main.py if command line options are given
"""

save = True # whether to save the data

"""
The rehearsal algorithm to use.
Options are:
    - method.catastrophicForgetting (No rehearsal)
    - method.random (Random rehearsal)
    - method.recency (Recency rehearsal)
    - method.sweep (Sweep rehearsal)
    - method.pseudo (Random pseudorehearsal)
    - method.pseudoSweep (Sweep pseudorehearsal)
"""
mymethod = method.pseudoSweep

numPseudoItems = 128 # How many pseudoitems to generate
bufferRefreshRate = 5 # How many epochs to train a buffer (sweep pseudorehearsal)

inputNodes = 32         # Number of input neurons
hiddenNodes = 16       # Number of hidden neurons
outputNodes = 32         # Number of output neurons
numLayers = 1 # Number of hidden layers of the network
numInterventions = 10    # Number of intervening trials
numPatterns = 20+numInterventions  # Total number of patterns to learn

populationSize = numPatterns-numInterventions
repeats = 10# Number of times to repeat the experiment completely on a new population
             # including new intervening trials and a new network (1 means no repeats)

auto = False     # Whether the learning is autoassociative (e.g. [1,0] -> [1,0])
                 # or heteroassociative (e.g. [1,0] -> [0,0])
                 # Input patterns are uniquely generated

printRate = 1000

learningConstant = 0.3
momentumConstant = 0.9
errorCriterion = 0.90 # Minimum Goodness
maxIterations = 10000
maxInitialIterations = 20000


batch_size = 1
outputFile = None
