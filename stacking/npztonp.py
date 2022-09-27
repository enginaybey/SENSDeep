import numpy as np
def npztonp(X):
    return np.asarray([X[key] for key in X][0])
