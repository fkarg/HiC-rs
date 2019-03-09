import ctypes

# lib.process()



class FFITuple(ctypes.Structure):
    _fields_ = [("a", ctypes.c_uint32),
                ("b", ctypes.c_uint32)]

class FFIArray(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p),
                ("len", ctypes.c_size_t)]

    # Allow implicit conversions from a sequence of 32-bit unsigned
    # integers.
    @classmethod
    def from_param(cls, seq):
        return cls(seq)

    # Wrap sequence of values. You can specify another type besides a
    # 32-bit unsigned integer.
    def __init__(self, seq, data_type = ctypes.c_uint32):
        array_type = data_type * len(seq)
        raw_seq = array_type(*seq)
        self.data = ctypes.cast(raw_seq, ctypes.c_void_p)
        self.len = len(seq)

# A conversion function that cleans up the result value to make it
# nicer to consume.
def void_array_to_tuple_list(array, _func, _args):
    tuple_array = ctypes.cast(array.data, ctypes.POINTER(FFITuple))
    return [tuple_array[i] for i in range(0, array.len)]

def void_array_to_list(array, _func, _args):
    l = array.len
    a = ctypes.cast(array.data, ctypes.POINTER(ctypes.c_int))
    return [a[i] for i in range(l)]

try:
    lib = ctypes.cdll.LoadLibrary("./target/release/libhicrs.so")
    print("using release rs")
except Exception as e:
    lib = ctypes.cdll.LoadLibrary("./target/debug/libhicrs.so")
    print("using debug rs")

print("done!")



# lib.listtest.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t)
lib.listtest.argtypes = FFIArray,
lib.listtest.restype = FFIArray
lib.listtest.errcheck = void_array_to_list
# lib.listtest.argtypes = (ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t)
# lib.listtest.restype = (ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t)
# lib.listtest.errcheck = void_array_to_tuple_list

list_to_sum = [1,2,3,4]
c_array = (ctypes.c_int32 * len(list_to_sum))(*list_to_sum)
l = lib.listtest(c_array, len(list_to_sum))
print(l)
