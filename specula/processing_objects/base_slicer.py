from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue

class BaseSlicer(BaseProcessingObj):
    """
    Extracts a subset of values from a BaseValue based on an index, list of indices, or a slice.
    """
    def __init__(self,
                 indices=None,   # int, list
                 slice_args=None,  # list for slice(start, stop, step)
                 target_device_idx=None,
                 precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.indices = indices
        # Use slice arguments to create a slice object
        if slice_args is not None:
            self.slice_obj = slice(*slice_args)
        else:
            self.slice_obj = None
        self.out_value = BaseValue(target_device_idx=target_device_idx)
        self.inputs['in_value'] = InputValue(type=BaseValue)
        self.outputs['out_value'] = self.out_value

    def trigger_code(self):
        value = self.local_inputs['in_value'].value
        if self.indices is not None:
            self.out_value.value = value[self.indices]
        elif self.slice_obj is not None:
            # Use slice object to extract the desired values
            self.out_value.value = value[self.slice_obj]
        else:
            # No slicing, copy the whole value
            self.out_value.value = value.copy()
        self.out_value.generation_time = self.current_time