import numpy as np #

def Euclidean(vector1, vector2):

    return np.sum(np.sqrt((vector1-vector2)**2))

def Manhattan(vector1, vector2):

    return np.sum(np.abs(vector1-vector2))

def Cosine(vector1, vector2):
    pass


