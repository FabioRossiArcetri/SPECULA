
from collections import OrderedDict, defaultdict

from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue


class DataBuffer(BaseProcessingObj):
    """
    Data buffering processing object.
    Accumulates data and outputs it every N steps.
    """

    def __init__(self, buffer_size: int = 10):
        super().__init__()
        self.buffer_size = buffer_size
        self.storage = defaultdict(OrderedDict)
        self.step_counter = 0
        self.buffered_outputs = {}

    @classmethod
    def input_names(cls):
        return {}

    @classmethod
    def output_names(cls):
        return {}

    def setOutputs(self):
        # Create output objects for each input (like DataStore does)
        for input_name, input_obj in self.inputs.items():
            if input_obj is not None:
                # Create output name and object
                output_name = f"{input_name}_buffered"
                output_obj = BaseValue(target_device_idx=self.target_device_idx)
                self.buffered_outputs[output_name] = output_obj
                self.outputs[output_name] = output_obj

    def trigger_code(self):
        # Accumulate data (same logic as DataStore)
        for k, item in self.local_inputs.items():
            if item is not None and item.generation_time == self.current_time:
                v = item.get_value()
                self.storage[k][self.step_counter] = v.copy()
            elif item is not None:
                # Not fresh this step: skip it rather than buffer a stale
                # value. emit_buffered_data() below only emits step indices
                # common to every input, so a dropped step here just costs
                # one row instead of silently misaligning this input's rows
                # against the others from this point on.
                self.logger.warning(
                    f"DataBuffer: input '{k}' was not fresh at step {self.step_counter} "
                    f"(generation_time={item.generation_time}, current_time={self.current_time}) "
                    f"-- this step will be dropped for '{k}'")

        self.step_counter += 1

        if self.step_counter >= self.buffer_size:
            self.emit_buffered_data()
            self.reset_buffers()

    def emit_buffered_data(self):
        active_keys = [k for k, d in self.storage.items() if len(d) > 0]
        if not active_keys:
            return

        # Only emit step indices present for EVERY buffered input, in a
        # consistent order -- a step dropped for just one input (see
        # trigger_code()'s freshness check) must not shift that input's
        # rows out of alignment with the others.
        common_steps = set(self.storage[active_keys[0]])
        for k in active_keys[1:]:
            common_steps &= set(self.storage[k])
        common_steps = sorted(common_steps)

        if any(len(self.storage[k]) != len(common_steps) for k in active_keys):
            dropped_counts = {k: len(self.storage[k]) - len(common_steps) for k in active_keys}
            dropped_counts = {k: n for k, n in dropped_counts.items() if n > 0}
            self.logger.warning(
                f"DataBuffer: buffered inputs had mismatched step counts "
                f"({dropped_counts} steps dropped per input) -- only the "
                f"{len(common_steps)} steps common to all inputs were emitted, "
                f"to keep rows aligned across inputs")

        for input_name in active_keys:
            output_name = f"{input_name}_buffered"
            values = self.xp.array([self.storage[input_name][s] for s in common_steps])
            if output_name in self.buffered_outputs:
                self.buffered_outputs[output_name].value = values
                self.buffered_outputs[output_name].generation_time = self.current_time
                self.logger.debug(f"DataBuffer: emitted {len(values)} samples for {input_name}")

    def setup(self):
        # We check that all input items
        for k, _input in self.inputs.items():
            item = _input.get(target_device_idx=self.target_device_idx)
            if item is not None and not hasattr(item, 'get_value'):
                raise TypeError(f"Error: don't know how to buffer an object of type {type(item)}")                

    def reset_buffers(self):
        """Clear all buffers and reset counter"""
        self.storage.clear()
        self.step_counter = 0

        self.logger.debug(f"DataBuffer: reset buffers at time {self.current_time}")

    def finalize(self):
        """Emit any remaining data in buffers"""        
        if self.step_counter > 0:
            self.emit_buffered_data()
            self.reset_buffers()
        
        super().finalize()

    def check_input_names(self):
        # DataBuffer inputs are added dynamically via input_list;
        # skip the static input_names validation.
        pass

    def check_output_names(self):
        # DataBuffer outputs are created dynamically in setOutputs();
        # skip the static output_names validation.
        pass
