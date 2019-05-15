import ctypes
import os
import sys
import numpy as np
from scipy.sparse import csr_matrix

EXT = sys.platform == "darwin" and ".dylib" or ".so"

# lib.process()


class FFITuple(ctypes.Structure):
    _fields_ = [("a", ctypes.c_uint32), ("b", ctypes.c_uint32)]

    def __repr__(self):
        return "({}, {})".format(self.a, self.b)


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
    # 32-bit unsigned integer.
    def __init__(self, seq, data_type=ctypes.c_double):
        array_type = data_type * len(seq)
        raw_seq = array_type(*seq)
        self.data = ctypes.cast(raw_seq, ctypes.c_void_p)
        self.len = len(seq)

# A conversion function that cleans up the result value to make it
# nicer to consume.
def void_array_to_tuple_list(array, _func, _args):
    tuple_array = ctypes.cast(array.data, ctypes.POINTER(FFITuple))
    return [tuple_array[i] for i in range(0, array.len)]


def void_array_to_list(array, _func=None, _args=None):
    l = array.len
    a = ctypes.cast(array.data, ctypes.POINTER(ctypes.c_int))
    return [a[i] for i in range(l)]


my_path = os.path.abspath(os.path.dirname(__file__))

path = os.path.join(my_path, "target/%s/libhicrs" + EXT)


try:
    lib = ctypes.cdll.LoadLibrary(path % "release")
    print("using release rs")
except Exception as e:
    lib = ctypes.cdll.LoadLibrary(path % "debug")
    print("using debug rs")


# lib.listtest.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t)
lib.listtest.argtypes = (FFIArray,)
lib.listtest.restype = FFIArray
lib.listtest.errcheck = void_array_to_list
# lib.listtest.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t)
# lib.listtest.restype = (ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t)
# lib.listtest.errcheck = void_array_to_tuple_list
listtest = lib.listtest

list_to_sum = [1, 2, 3, 4]


def cast_c(l):
    return (ctypes.c_int32 * len(l))(*l)


# l = lib.listtest(cast_c(list_to_sum), len(list_to_sum))
l = lib.listtest(list_to_sum)


row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])


matrix = csr_matrix((data, (row, col)), shape=(3, 3))
matrix2 = csr_matrix((
    [ 1, 1, 2, 1, 3, 1, 4],
    ([0, 0, 1, 1, 2, 2, 3]
    ,[0, 1, 1, 2, 2, 3, 3])),
    shape=(4,4))

assert list(matrix.indptr) == list(np.array([0, 2, 3, 6]))
assert list(matrix.indices) == list(np.array([0, 2, 2, 0, 1, 2]))


lib.csrtest.argtypes = FFIArray, FFIArray, FFIFloatArray
lib.csrtest.restype = FFIArray
lib.csrtest.errcheck = void_array_to_list

def iterative_correct(matrix: csr_matrix) -> list:
    return lib.csrtest(matrix.indptr, matrix.indices, matrix.data)

# e = iterative_correct(matrix)



