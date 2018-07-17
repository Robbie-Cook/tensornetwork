import tensorflow as tf
from tensorflow import keras
import numpy as np
import task
import settings
import copy
import rehearsal
from Instance import Instance
import method
import tools
import os
import argparse

"""
Argument parser for changing settings
"""

parser = argparse.ArgumentParser()
parser.add_argument("--numLayers")
parser.add_argument("--numRepeats")
parser.add_argument("--bufferRefreshRate")

args = parser.parse_args()

if args.numLayers != None:
    settings.numLayers = int(args.numLayers)

if args.numRepeats != None:
    settings.numRepeats = int(args.numRepeats)

if args.bufferRefreshRate != None:
    settings.bufferRefreshRate = int(args.bufferRefreshRate)

"""
Open files for information
"""
if settings.save:
    i = 0
    while "output{}.txt".format(i) in os.listdir("data"):
        i+=1
    settings.outputFile = open("data/output{}.txt".format(i), 'w')
    settings.infoFile = open("info/info{}.txt".format(i), 'w')

    settings.infoFile.write(
        "Learning constant: {}\n \
        Momentum constant: {}\n \
        Number of layers: {}\n \
        Algorithm: {} \n \
        Pseudo refresh rate: {}\n \
        Number of repeats {}".format(
            settings.learningConstant,
            settings.momentumConstant,
            settings.numLayers,
            settings.mymethod,
            settings.bufferRefreshRate,
            settings.repeats)
    )
    settings.infoFile.flush()

"""
Algorithm start
"""
goodnesses = [0 for i in range(settings.numInterventions+1)]

for x in range(settings.repeats):
    print("Repeat: {}\n".format(x+1))

    model = keras.Sequential()
    model.add(keras.layers.Dense(settings.inputNodes, activation='sigmoid'))
    for i in range(settings.numLayers):
        model.add(keras.layers.Dense(settings.hiddenNodes, activation='sigmoid'))
    model.add(keras.layers.Dense(settings.outputNodes, activation='sigmoid'))

    # Configure a model for error regression.
    model.compile(
        optimizer=tf.train.MomentumOptimizer(
            learning_rate = settings.learningConstant,
            momentum = settings.momentumConstant
            ),
        loss='mse',
        metrics=['mse'])

    mytask = task.Task(
            inputNodes=settings.inputNodes,
            hiddenNodes=settings.hiddenNodes,
            outputNodes=settings.outputNodes,
            populationSize=settings.numPatterns,
            auto=settings.auto,
        )

    interventions = [mytask.popTask() for a in range(0, settings.numInterventions)]
    # print("Interventions input: {}".format(interventions))

    data = np.array(mytask.task['inputPatterns'])
    teacher = np.array(mytask.task['teacher'])

    # print("Data: {}, Teacher: {}".format(data, teacher))

    tools.train(model, data, teacher)

    goodness = tools.getGoodness(model=model, input=data, teacher=teacher)

    print("Goodness", goodness)
    goodnesses[0] += (goodness)

    learnedInput = copy.deepcopy(data.tolist())
    learnedOutput = copy.deepcopy(teacher.tolist())
    learned = []
    for i in range(len(learnedInput)):
        learned.append(Instance(learnedInput[i], learnedOutput[i]))

    for j in range(0, len(interventions)):
        interventionInput = interventions[j]['inputPatterns'][0]
        interventionOutput = interventions[j]['teacher'][0]
        intervention = Instance(interventionInput, interventionOutput)

        meth = settings.mymethod
        if meth == method.catastrophicForgetting:
            rehearsal.catastrophicForgetting(
                network=model,
                intervention=intervention
                )
        elif meth == method.recency:
            rehearsal.recency(
                network=model,
                intervention=intervention,
                learned=learned,
                random=False
            )
        elif meth == method.random:
            rehearsal.recency(
                network=model,
                intervention=intervention,
                learned=learned,
                random=True
            )
        elif meth == method.pseudo:
            rehearsal.pseudo(
                network=model,
                intervention=intervention,
                numPseudoItems=settings.numPseudoItems
            )
        elif meth == method.sweep:
            rehearsal.sweep(
                network=model,
                intervention=intervention,
                learned=learned
            )

        elif meth == method.pseudoSweep:
            rehearsal.pseudoSweep(
                network=model,
                intervention=intervention,
                numPseudoItems=settings.numPseudoItems
            )
        else:
            print("Method not valid")
            exit(0)
        goodness = tools.getGoodness(model=model, input=data, teacher=teacher)
        goodnesses[j+1] += goodness
        print("({}) Goodness: {}".format(j+2, goodness))
        learned.append(intervention)

finalGoodnesses = [goodness/settings.repeats for goodness in goodnesses]
print(finalGoodnesses)
for g in finalGoodnesses:
    settings.outputFile.write(str(g) + "\n")
