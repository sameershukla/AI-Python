class Perceptron:

    def __init__(self, no_of_epochs, bias):
        self.no_of_epochs = no_of_epochs
        self.bias = bias

    def predict(self, input, weight):
        # bias
        activation = self.bias
        for i in range(len(input) - 1):
            # Wx + b
            activation = weight[i] * input[i] + activation

        return 1.0 if activation >= 0.0 else 0.0

    def perceptron(self, inputs, lr):
        self.weights = [0.0 for i in range(len(inputs[0]))]
        self.errors = []
        for epoch in range(self.no_of_epochs):
            error = 0.0
            # Array of Input Arrays
            for i in inputs:
                # label - error
                prediction = self.predict(i, weights)
                error = i[-1] * prediction
                self.errors.append(error)
                self.weights[0] = self.weights[0] + error * lr
                # Single Array
                for j in range(len(i) - 1):
                    self.weights[j] = self.weights[j] * lr * i[j] * error

        return self


# And Gate
print('--------- And Gate ----------')
X = [[0,0,0],
[1,1,1],
[1,0,0],
[0,1,0]]

weights = [0.5,0.5]
bias = -1
p = Perceptron(5, bias)
for x in X:
        prediction = p.predict(x, weights)
        print("Expected=%d, Predicted=%d" % (x[-1], prediction))

print(' ----------- Or Gate -----------')
# Or Gate
X = [[0,0,0],
    [1,1,1],
    [1,0,1],
    [0,1,1]]

weights = [1.5,1.5]
bias = -1
p = Perceptron(5, bias)
for x in X:
        prediction = p.predict(x, weights)
        print("Expected=%d, Predicted=%d" % (x[-1], prediction))
