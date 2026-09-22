
import warnings
from copy import copy
from functools import lru_cache

from specula import cp, np, array_types
from specula.base_time_obj import BaseTimeObj


# We use lru_cache() instead of cache() for python 3.8 compatibility
@lru_cache(maxsize=None)
def get_properties(cls):
    result = []
    classlist = cls.__mro__
    for cc in classlist:
        result.extend([attr for attr, value in vars(cc).items() if isinstance(value, property) ])
    return result


class BaseDataObj(BaseTimeObj):
    def __init__(self, target_device_idx: int=None, precision: int=None):
        """
        Initialize the base data object.

        Parameters:
        target_device_idx: int, optional
            device to be targeted for data storage. Set to -1 for CPU,
            to 0 for the first GPU device, 1 for the second GPU device, etc.
        precision: int, optional
            if None will use the global_precision, otherwise set to 0 for double, 1 for single
        """
        super().__init__(target_device_idx, precision)
        self.generation_time = -1
        self.tag = ''

    def transferDataTo(self, destobj, force_reallocation=False):
        '''
        Copy CPU/GPU arrays into an existing data object:
        iterate over all self attributes and, if a CPU or GPU array
        is detected, copy data into *destobj* without reallocating.

        Destination (CPU or GPU device) is inferred by *destobj.target_device_idx*,
        which must be set correctly before calling this method.
        '''
        # Get a list of all attributes, but skip properties
        pp = get_properties(type(self))
        attr_list = [attr for attr in dir(self) if attr not in pp]

        for attr in attr_list:
            self_attr = getattr(self, attr)
            self_type = type(self_attr)
            if self_type not in array_types:
                # A genuine numpy/cupy *scalar* (e.g. a BaseValue
                # constructed with an initial scalar value: __init__
                # builds it via self.dtype(value), which for a scalar
                # input produces e.g. numpy.float32 rather than an
                # ndarray). array_types only lists true ndarray types, so
                # without this such a value would be silently skipped
                # below and never updated on the destination object
                # across a device boundary. Scalars are immutable, so
                # this always replaces the attribute rather than copying
                # in-place. (A *0-d array* scalar -- e.g. BaseValue.value
                # built via set_value() on a BaseValue that started with
                # value=None -- is a different case, handled in the DtD/
                # HtH branches below: it IS in array_types, but a 0-d
                # array can't be sliced with `[:]`.)
                if isinstance(self_attr, np.generic) or (cp is not None and isinstance(self_attr, cp.generic)):
                    setattr(destobj, attr, self_attr)
                continue

            dest_attr = getattr(destobj, attr)
            dest_type = type(dest_attr)

            if dest_type not in array_types:
                self.logger.warning(f'destination attribute is not a cupy/numpy array, forcing reallocation ({destobj}.{attr})')
                force_reallocation = True

            # Destination array had the correct type: perform in-place data copy
            if not force_reallocation:
                # Detect whether the array types are correct for all four cases:
                # Device to CPU, CPU to device, device-to-device, and CPU-CPU. Also check whether
                # the target_device_idx is set correctly for the destination object.
                DtD = cp is not None and (self_type == cp.ndarray) and (dest_type == cp.ndarray) and destobj.target_device_idx >= 0
                DtH = cp is not None and (self_type == cp.ndarray) and (dest_type == np.ndarray) and destobj.target_device_idx == -1
                HtD = cp is not None and (self_type == np.ndarray) and (dest_type == cp.ndarray) and destobj.target_device_idx >= 0
                HtH = (self_type == np.ndarray) and (dest_type == np.ndarray) and destobj.target_device_idx == -1
                if DtD:
                    # Performance warnings here are expected, because we might
                    # trigger a peer-to-peer transfer between devices
                    with warnings.catch_warnings():
                        if self.PerformanceWarning:
                            warnings.simplefilter("ignore", category=self.PerformanceWarning)
                        try:
                            dest_attr[:] = self_attr
                        except Exception:
                            # dest_attr[:] = ... raises IndexError for 0-d
                            # arrays ("too many indices"), e.g. a scalar
                            # BaseValue.value built via to_xp() the first
                            # time set_value() runs. The previous fallback
                            # here ("dest_attr = self_attr") only rebound
                            # the local variable, never actually updating
                            # destobj -- so destobj.value silently stayed
                            # frozen at whatever it was on the very first
                            # transfer, forever, with no error anywhere.
                            setattr(destobj, attr, self_attr)
                elif DtH:
                    # Do not set blocking=True for cupy 12.x compatibility.
                    # Blocking is True by default in later versions anyway
                    self_attr.get(out=dest_attr)
                elif HtD:
                    dest_attr.set(self_attr)
                elif HtH:
                    try:
                        dest_attr[:] = self_attr
                    except Exception:
                        # Same 0-d-array case as DtD above.
                        setattr(destobj, attr, self_attr)
                else:
                    self.logger.warning(f'mismatch between target_device_idx and array allocation, forcing reallocation ({destobj}.{attr})')
                    force_reallocation = True

            # Otherwise, reallocate
            if force_reallocation:
                DtD = cp is not None and (self_type == cp.ndarray) and destobj.target_device_idx >= 0
                DtH = cp is not None and (self_type == cp.ndarray) and destobj.target_device_idx == -1
                HtD = (self_type == np.ndarray) and destobj.target_device_idx >= 0
                HtH = (self_type == np.ndarray) and destobj.target_device_idx == -1

                if DtD:
                    # Performance warnings here are expected, because we might
                    # trigger a peer-to-peer transfer between devices
                    with warnings.catch_warnings():
                        if self.PerformanceWarning:
                            warnings.simplefilter("ignore", category=self.PerformanceWarning)
                        setattr(destobj, attr, cp.asarray(self_attr))
                if DtH:
                    # Do not set blocking=True for cupy 12.x compatibility.
                    # Blocking is True by default in later versions anyway
                    setattr(destobj, attr, self_attr.get())
                if HtD:
                    setattr(destobj, attr, cp.asarray(self_attr))
                if HtH:
                    setattr(destobj, attr, np.asarray(self_attr))

        destobj.generation_time = self.generation_time

    def copyTo(self, target_device_idx):
        '''
        Duplicate a data object on another device,
        alllocating all CPU/GPU arrays on the new device.
        '''
        if target_device_idx == self.target_device_idx:
            return self
        else:
            cloned = copy(self)

            if target_device_idx >= 0:
                cloned.xp = cp
            else:
                cloned.xp = np
            cloned.target_device_idx = target_device_idx

            self.transferDataTo(cloned, force_reallocation=True)
            return cloned
