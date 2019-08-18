__all__ = [
    'ConstantLR',
    'InverseScalingLR',
    'OptimalLR'
]


cdef class LRScheduler:

    cpdef double get(self, long t):
        """Returns the learning rate at a given iteration."""
        raise NotImplementedError


cdef class ConstantLR(LRScheduler):
    """Always uses the same learning rate."""

    cdef readonly double learning_rate

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    cpdef double get(self, long t):
        return self.learning_rate


cdef class InverseScalingLR(LRScheduler):
    """Reduces the learning rate using a power schedule.

    Assuming an iteration counter $t$ starting from 0, the learning rate will be:

    .. math:: \\frac{1}{(t+1)^p}

    where $p$ is a user-defined parameter.

    """

    cdef readonly double learning_rate
    cdef readonly double power

    def __init__(self, learning_rate, power=0.5):
        self.learning_rate = learning_rate
        self.power = power

    cpdef double get(self, long t):
        return self.learning_rate / (t + 1) ** self.power


cdef class OptimalLR(LRScheduler):
    """Optimal learning schedule as proposed by Léon Bottou.

    References:
        1. `Stochastic Gradient Descent <https://leon.bottou.org/projects/sgd>`_

    """

    cdef readonly long t0
    cdef readonly double alpha

    def __init__(self, t0=4e3, alpha=1e-4):
        self.t0 = t0
        self.alpha = alpha

    cpdef double get(self, long t):
        return 1. / (self.alpha * (self.t0 + t))
