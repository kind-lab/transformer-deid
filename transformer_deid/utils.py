import numpy as np


def convert_numpy_to_native_type(obj):
    """Convert a numpy type to a native data type."""
    if isinstance(obj, dict):
        # recursion to iterate through sub-dict
        return convert_dict_to_native_types(obj)
    elif isinstance(
        obj, (
            np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64
        )
    ):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
        return {'real': obj.real, 'imag': obj.imag}
    elif isinstance(obj, (np.ndarray, )):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.void)):
        return None


def convert_dict_to_native_types(d):
    """Iterate through a dictionary and convert to native types."""
    for k, v in d.items():
        d[k] = convert_numpy_to_native_type(v)

    return d