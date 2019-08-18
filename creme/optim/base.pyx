cimport lr_schedule

import numbers
import typing

from . import lr_schedule


cdef class Optimizer:

    def __init__(self, lr: typing.Union[lr_schedule.LRScheduler, numbers.Number]):
        self.lr = lr_schedule.ConstantLR(lr) if isinstance(lr, numbers.Number) else lr
        self.n_iterations = 0

    @property
    def learning_rate(self) -> float:
        return self.lr.get(self.n_iterations)

    cpdef dict update_before_pred(self, dict w, dict x):
        return w

    cdef dict _update_after_pred(self, dict w, dict g):
        raise NotImplementedError

    cpdef dict update_after_pred(self, dict w, dict g):

        # Update the weights
        w = self._update_after_pred(w, g)

        # Update the iteration counter
        self.n_iterations += 1

        return w
