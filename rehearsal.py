import task
import os
import json
import numpy as np
import random as rand
import settings
from Instance import Instance
import tools

"""
Method to train a buffer of items e.g. [Instance([0,1], [1,0]), Instance([1,0], [1,0])]
to a given criterion.
"""
def trainBuffer(network, instances, maxIterations=0, quiet=False):
    if maxIterations == 0:
        maxIterations = settings.maxIterations

    input = [instances[i].input for i in range(len(instances))]
    teacher = [instances[i].teacher for i in range(len(instances))]

    # Train the network using backpropagation
    tools.train(
        model=network,
        input=input,
        teacher=teacher,
        maxIterations=maxIterations
        )

"""
Catastrophic Forgetting
"""
def catastrophicForgetting(network, intervention):
    trainBuffer(network, [intervention], settings.maxIterations)
"""
Recency rehearsal (or random if random=True)
"""
def recency(network, intervention, learned, random=False):
    # newInstance = intervention
    #
    # if random:          # Random rehearsal
    #     rand.shuffle(learned)
    #
    # buffer2=learned[0]
    # buffer3=learned[0]
    # buffer4=learned[len(learned)-1]
    # if len(learned) >= 3:
    #     buffer2=learned[-1]
    #     buffer3=learned[-2]
    #     buffer4=learned[-3]
    #
    #
    # interveningDataset = [newInstance, buffer2, buffer3, buffer4]
    # trainBuffer(network, interveningDataset)
    pass
"""
Random rehearsal
"""
def random(network, intervention, learned):
    # recency(network, intervention, learned, random=True)
    pass

"""
Random pseudorehearsal
"""
def pseudo(network, intervention, numPseudoItems):
    # # Pseudoitems for pseudorehearsal
    # pseudoItems = generatePseudoPairs(network, numPseudoItems)
    # rand.shuffle(pseudoItems)
    # buffer = [intervention, pseudoItems[-1], pseudoItems[-2], pseudoItems[-3]]
    # trainBuffer(network, buffer)
    pass

"""
Generates pseudoitem pairs (input and output) for pseudorehearsal
"""
def generatePseudoPairs(network, numItems):
    mytask = task.Task(inputNodes=settings.inputNodes, hiddenNodes=settings.hiddenNodes,
                outputNodes=settings.outputNodes, populationSize=numItems, auto=False).task
    pseudoInputs = mytask['inputPatterns']
    pseudoItems = [Instance(a, network.predict(np.array([a]))[0]) for a in pseudoInputs]
    return pseudoItems


"""
Sweep rehearsal
"""
def sweep(network, intervention, learned):
    iterations = 0

    currentError = tools.getGoodness(model=network,
                                     input=intervention.input,
                                     teacher=intervention.teacher)
    while currentError < settings.errorCriterion and iterations < settings.maxIterations:
        iterations+=1

        if iterations % settings.printRate == 0:
            print("{} times, goodness: {}".format(iterations, currentError))
        rand.shuffle(learned)
        if len(learned) >= 3:
            buffer = [intervention, learned[-1], learned[-2], learned[-3]]
        elif len(learned) == 2:
            buffer = [intervention, learned[-1], learned[-2], learned[-1]]
        elif len(learned) == 1:
            buffer = [intervention, learned[-1], learned[-1], learned[-1]]
        # rand.shuffle(buffer)
        trainBuffer(network,
                    instances=buffer,
                    maxIterations=settings.bufferRefreshRate,
                    quiet=True)

        # update current error
        currentError = tools.getGoodness(model=network,
                                         input=intervention.input,
                                         teacher=intervention.teacher)

def pseudoSweep(network, intervention, numPseudoItems):
    iterations = 0
    currentError = tools.getGoodness(model=network,
                                     input=intervention.input,
                                     teacher=intervention.teacher)

    pseudoItems = generatePseudoPairs(network, numPseudoItems)
    while currentError < settings.errorCriterion and iterations < settings.maxIterations:
        iterations+=1
        if iterations % settings.printRate == 0:
            print("{} times, goodness: {}".format(iterations, currentError))

        rand.shuffle(pseudoItems)
        buffer = [intervention, pseudoItems[-1], pseudoItems[-2], pseudoItems[-3]]
        rand.shuffle(buffer)

        trainBuffer(network,
                    instances=buffer,
                    maxIterations=settings.bufferRefreshRate,
                    quiet=True)

        # update current error
        currentError = tools.getGoodness(model=network,
                                         input=intervention.input,
                                         teacher=intervention.teacher)
