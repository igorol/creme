cimport base
cimport mean


cdef class Var(base.Univariate):
    cdef readonly long ddof
    cdef readonly mean.Mean mean
    cdef readonly double sos
