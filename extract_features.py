import numpy as np
from scipy.io import loadmat

def ExtractFeatures(x, NumofPrincComp):
    """
    Extracts specified number of principal components of x.

    Parameters:
        x (numpy.array): Vector of raw data.
        coeff (numpy.array): Basis transformation matrix.
        NumofPrincComp (int): Number of largest principal components to extract.

    Returns:
        numpy.array: Specified largest principal components of x.
    """
    coeff = loadmat("coeff.mat")["coeff"]
    # Perform the matrix multiplication using the transpose of `coeff`
    x_ = coeff.T @ x

    # Select the first NumofPrincComp components
    x_ = x_[:NumofPrincComp]

    return x_