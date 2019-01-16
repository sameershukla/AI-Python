import numpy as np
# In theory sigmoid function is 1 / 1 + exp (-x) and the output value will be in the range of -1 to 1

def sigmoid(inputs):
    return np.exp(inputs) / float(sum(np.exp(inputs)))


inputs = [1,2,3,4,5]
print ("Sigmoid Function Output :: {}".format(sigmoid(inputs)))
