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
                learnt=learnt,
                random=False
            )
        elif meth == method.random:
            rehearsal.recency(
                network=model,
                intervention=intervention,
                learnt=learnt,
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