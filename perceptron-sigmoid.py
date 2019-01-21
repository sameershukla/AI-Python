class Perceptron(object):

    def __init__(self, lr=0.01, n=10):
        self.lr = lr
        self.n = n

    def fit(self, X, y):
        self.weight = np.zeros(1 + X.shape[1])
        self.errors = []  # Number of misclassifications
        self.predictions = []

        for i in range(self.n):
            err = 0
            prediction = 0
            for xi, target in zip(X, y):
                prediction = target - self.predict(xi)
                self.predictions.append(prediction)
                wt = self.lr * (prediction)
                self.weight[1:] += wt * xi
                self.weight[0] += wt
                err += int(wt != 0.0)
                self.errors.append(err)
        return self

    def predict(self, X):
        wxplusb = np.dot(X, self.weight[1:]) + self.weight[0]
        # if(wxplusb >= 0.0):
        # return 1
        # else:
        # return -1
        # Let's use sigmoid
        return self.sigmoid(wxplusb)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# ------- Training ---------------

# Let's train our model for prediction
# color:     R-1   B-0     R-1    B-0   R-1     B-0     R-1     B-0   ???
# length:    3      2      4      3     3.5     2      5.5      1   4.5
# width:    1.5     1      1.5    1     0.5     0.5    1        1   1

X = np.array([[3,1.5],[2,1],[4,1.5],[3,1],[3.5,0.5],[2,0.5],[5.5,1],[1,1],[4.5,1]])
y = [1,0,1,0,1,0,1,0,1]

p = Perceptron(0.1,1)
p.fit(X,y)
p.predictions