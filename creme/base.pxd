cdef class Estimator:
    pass

cdef class Transformer(Estimator):
    cpdef Transformer fit_one(self, dict x, object y=*)
    cpdef dict transform_one(self, dict x)
