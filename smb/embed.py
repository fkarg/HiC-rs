import ctypes
import os
import sys
import numpy as np
from scipy.sparse import csr_matrix

EXT = sys.platform == "darwin" and ".dylib" or ".so"


# constructors for used FFI types
# - List with 32bit int (unsigned)
# - List with 64bit float (double)


class FFIArray(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("len", ctypes.c_size_t)]

    # Allow implicit conversions from a sequence of 32-bit unsigned
    # integers.
    @classmethod
    def from_param(cls, seq):
        return cls(seq)

    # Wrap sequence of values. You can specify another type besides a
    # 32-bit unsigned integer.
    def __init__(self, seq, data_type=ctypes.c_uint32):
        array_type = data_type * len(seq)
        raw_seq = array_type(*seq)
        self.data = ctypes.cast(raw_seq, ctypes.c_void_p)
        self.len = len(seq)

    def __len__(self):
        return self.len


class FFIFloatArray(FFIArray):
    # Wrap sequence of values. You can specify another type besides a
    # 64-bit float.
    def __init__(self, seq, data_type=ctypes.c_double):
        array_type = data_type * len(seq)
        raw_seq = array_type(*seq)
        self.data = ctypes.cast(raw_seq, ctypes.c_void_p)
        self.len = len(seq)


# A conversion function that cleans up the result value to make it
# nicer to consume.
def void_array_to_double_list(array, _func=None, _args=None):
    l = array.len
    a = ctypes.cast(array.data, ctypes.POINTER(ctypes.c_double))
    return [a[i] for i in range(l)]


# necessary for relative finding of the actualy dynamic library
my_path = os.path.abspath(os.path.dirname(__file__))

# here should it be:
path = os.path.join(my_path, "target/%s/libhicrs" + EXT)


try:
    lib = ctypes.cdll.LoadLibrary(path % "release")
    # print("using release rs")
except Exception as e:
    try:
        lib = ctypes.cdll.LoadLibrary(path % "debug")
        # print("using debug rs")
    except:
        lib = None
        pass


assert lib, "[HiCIterativeCorrection.smb]: Could not load Rust library. Is it compiled?"


# now, since the lib is loaded, let us add our one function correctly.

# fn csrtest(indptr: Array, indices: Array, data: Array, threads: i32) -> Array {
lib.wrapper_iterative_correct.argtypes = (
    FFIArray,
    FFIArray,
    FFIFloatArray,
    ctypes.c_uint32,
)
lib.wrapper_iterative_correct.restype = FFIFloatArray  # warning: test !!
lib.wrapper_iterative_correct.errcheck = void_array_to_double_list


def iterative_correct(matrix: csr_matrix, iternum=50) -> list:
    return lib.wrapper_iterative_correct(
        matrix.indptr, matrix.indices, matrix.data, iternum
    )


# Further code, unrelated to smb directly but maybe relevant for understanding
# the # interaction better:


class FFITuple(ctypes.Structure):
    _fields_ = [("a", ctypes.c_uint32), ("b", ctypes.c_uint32)]

    def __repr__(self):
        return "({}, {})".format(self.a, self.b)


def void_array_to_tuple_list(array, _func, _args):
    tuple_array = ctypes.cast(array.data, ctypes.POINTER(FFITuple))
    return [tuple_array[i] for i in range(0, array.len)]


def void_array_to_int_list(array, _func=None, _args=None):
    l = array.len
    a = ctypes.cast(array.data, ctypes.POINTER(ctypes.c_int))
    return [a[i] for i in range(l)]


lib.listtest.argtypes = (FFIArray,)
# = (ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t)
lib.listtest.restype = FFIArray
# = (ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t)
lib.listtest.errcheck = void_array_to_int_list
listtest = lib.listtest

# list_to_sum = [1, 2, 3, 4]


def cast_c(l):
    # small helper function. was useful at some point.
    return (ctypes.c_int32 * len(l))(*l)


# l = lib.listtest(list_to_sum)


# row = np.array([0, 0, 1, 2, 2, 2])
# col = np.array([0, 2, 2, 0, 1, 2])
# data = np.array([1, 2, 3, 4, 5, 6])
#
#
# matrix = csr_matrix((data, (row, col)), shape=(3, 3))
# matrix2 = csr_matrix(
#     ([1, 1, 2, 1, 3, 1, 4], ([0, 0, 1, 1, 2, 2, 3], [0, 1, 1, 2, 2, 3, 3])),
#     shape=(4, 4),
# )

# assert list(matrix.indptr) == list(np.array([0, 2, 3, 6]))
# assert list(matrix.indices) == list(np.array([0, 2, 2, 0, 1, 2]))


# e = iterative_correct(matrix, 1)
# print(e)


# testing if calling multiple processes from rust works:
# lib.process()
