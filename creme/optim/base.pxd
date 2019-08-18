cimport lr_schedule


cdef class Optimizer:

    cdef readonly lr_schedule.LRScheduler lr
    cdef readonly long n_iterations

    cpdef dict update_before_pred(self, dict w, dict x)

    cdef dict _update_after_pred(self, dict w, dict g)

    cpdef dict update_after_pred(self, dict w, dict g)
