import numpy as np
# In theory sigmoid function is 1 / 1 + exp (-x) and the output value will be in the range of -1 to 1

def sigmoid(inputs):
    scores = [1 / float(1 + np.exp(-x)) for x in inputs]
    return scores


inputs = [1,2,3,4,5]
print ("Sigmoid Function Output :: {}".format(sigmoid(inputs)))
