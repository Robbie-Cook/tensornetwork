import numpy as np

class Instance:
    # This is a simple encapsulation of a `input signal : output signal`
    # pair in our training set.
    def __init__(self, features, teacher = list() ):
        self.input = np.array(features)
        if len(teacher) > 0:
            self.teacher  = np.array(teacher)
        else:
            self.teacher  = None

    def __str__(self):
        if len(self.teacher) == 0:
            return "({})".format(self.input)
        else:
            return "({}, {})".format(self.input, self.teacher)
#endclass Instance
