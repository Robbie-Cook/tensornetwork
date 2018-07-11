import settings
import copy
import numpy as np

def getGoodness(model, input, teacher):
    inputArray = np.array(copy.deepcopy(input))
    teacherArray = np.array(copy.deepcopy(teacher))

    if inputArray.ndim == 1 and teacherArray.ndim == 1:
        inputArray = np.array([inputArray.tolist()])
        teacherArray = np.array([teacherArray.tolist()])

    outputArray = model.predict(inputArray)
    totalGoodness = 0

    for i in range(len(inputArray)):
        for j in range(len(inputArray[i])):
            teacherArray[i][j] = teacherArray[i][j]*2-1
            outputArray[i][j] = outputArray[i][j]*2-1
        totalGoodness += np.dot(teacherArray[i], outputArray[i])/len(inputArray[i])
    totalGoodness = totalGoodness/len(inputArray)

    return totalGoodness

def train(model, input, teacher, maxIterations=0):
    input = np.array(input)
    teacher = np.array(teacher)
    iterations = 0
    if maxIterations == 0:
        maxIterations = settings.maxInitialIterations

    while (
        getGoodness(model, input, teacher) < settings.errorCriterion and
        iterations < maxIterations
        ) :
        model.train_on_batch(input, teacher)
        iterations += 1
        if iterations % 1000 == 0:
            print("Trained for {} iterations, goodness {}".format(iterations, getGoodness(model, input, teacher)))
