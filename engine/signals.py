import numpy as np


def orNone(other):
    def check(x):
        if x is None:
            return True

        return other(x)

    return check


def rgbTuple():
    def checkRGB(x):
        assert type(x) is tuple, "Signal must be a tuple but is " + str(type(x))
        assert len(x) == 3, (
            "Signal must be a 3-tuple but has " + str(len(x)) + " entries"
        )

    return checkRGB


def npTensor(shape):
    def checkTensor(x):
        assert type(x) is np.ndarray, "Signal must be a numpy array but is " + str(
            type(x)
        )
        for index, dim in enumerate(shape):
            if dim > 0:
                assert x.shape[index] == dim, (
                    "Tensor shape "
                    + str(x.shape)
                    + " does not match definition "
                    + str(shape)
                )

    return checkTensor


def rgbImage(width, height):
    return npTensor((height, width, 3))


def lst():
    def checkLst(x):
        assert type(x) is list, "Signal must be a list but is " + str(type(x))

    return checkLst
