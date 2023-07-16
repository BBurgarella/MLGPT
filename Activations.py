import numpy as np

def ReLU(numpyobject, derivative=False):
    """
    ReLU = 0 if x < 0 or x elsewhat
    this is a very common activation function
    
    """

    if derivative:
        return np.where(numpyobject <= 0, 0, 1)
    else:
        return np.maximum(0, numpyobject)