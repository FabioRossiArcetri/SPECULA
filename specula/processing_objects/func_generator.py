
import numpy as np

from specula.base_value import BaseValue
from specula.base_processing_obj import BaseProcessingObj
from specula.lib.modal_pushpull_signal import modal_pushpull_signal


# TODO
class Vibrations():
    pass


class FuncGenerator(BaseProcessingObj):
    def __init__(self, 
                 func_type='SIN', 
                 nmodes: int=None, 
                 time_hist=None, 
                 psd=None, 
                 fr_psd=None, 
                 continuous_psd=None, 
                 constant: list=None, 
                 amp: list=None, 
                 freq: list=None, 
                 offset: list=None, 
                 vect_amplitude: list=None,
                 nsamples: int=1,
                 seed: int=None, 
                 ncycles: int=1,
                 vsize: int=1,
                 target_device_idx: int=None, 
                 precision: int=None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.type = func_type.upper()
        if self.type == 'PUSHPULLREPEAT':
            repeat_ncycles = True
            self.type = 'PUSHPULL'
        else:
            repeat_ncycles = False

        if nsamples != 1 and self.type != 'PUSHPULL':
            raise ValueError('nsamples can only be used with PUSHPULL or PUSHPULLREPEAT types')

        if str(seed).strip() == 'auto':
            self.seed = self.xp.around(self.xp.random.random() * 1e4)
        elif seed is not None:
            self.seed = self.xp.array(seed, dtype=self.dtype)
        else:
            self.seed = 0

        self.vsize = vsize
        self.constant = self.xp.array(constant, dtype=self.dtype) if constant is not None else 0.0
        self.amp = self.xp.array(amp, dtype=self.dtype) if amp is not None else 0.0
        self.freq = self.xp.array(freq, dtype=self.dtype) if freq is not None else 0.0
        self.offset = self.xp.array(offset, dtype=self.dtype) if offset is not None else 0.0
        self.vect_amplitude = self.xp.array(vect_amplitude, dtype=self.dtype) if vect_amplitude is not None else 0.0
        self.output = BaseValue(target_device_idx=target_device_idx, value=self.xp.zeros(self.vsize, dtype=self.dtype))
        self.vib = None

        if seed is not None:
            self.seed = seed

        # Initialize attributes based on the type
        if self.type == 'SIN':
            pass
        
        elif self.type == 'SQUARE_WAVE':
            pass

        elif self.type == 'LINEAR':
            self.slope = 0.0

        elif self.type == 'RANDOM' or self.type == 'RANDOM_UNIFORM':
            pass

        elif self.type == 'VIB_HIST':
            raise NotImplementedError('VIB_HIST type is not implemented')
        
            if nmodes is None:
                raise ValueError('NMODES keyword is mandatory for type VIB_HIST')
            if time_hist is None:
                raise ValueError('TIME_HIST keyword is mandatory for type VIB_HIST')
            self.vib = Vibrations(nmodes, time_hist=time_hist)

        elif self.type == 'VIB_PSD':
            raise NotImplementedError('VIB_PSD type is not implemented')

            if nmodes is None:
                raise ValueError('NMODES keyword is mandatory for type VIB_PSD')
            if psd is None and continuous_psd is None:
                raise ValueError('PSD or CONTINUOUS_PSD keyword is mandatory for type VIB_PSD')
            if fr_psd is None:
                raise ValueError('FR_PSD keyword is mandatory for type VIB_PSD')
            self.vib = Vibrations(nmodes, psd=psd, freq=fr_psd, continuous_psd=continuous_psd, seed=seed)

        elif self.type == 'PUSH':
            if nmodes is None:
                raise ValueError('NMODES keyword is mandatory for type PUSH')
            if amp is None and vect_amplitude is None:
                raise ValueError('AMP or VECT_AMPLITUDE keyword is mandatory for type PUSH')
            self.time_hist = modal_pushpull_signal(nmodes, amplitude=amp, vect_amplitude=vect_amplitude, only_push=True, ncycles=ncycles)

        elif self.type == 'PUSHPULL':
            if nmodes is None:
                raise ValueError('NMODES keyword is mandatory for type PUSHPULL')
            if amp is None and vect_amplitude is None:
                raise ValueError('AMP or VECT_AMPLITUDE keyword is mandatory for type PUSHPULL')
            self.time_hist = modal_pushpull_signal(nmodes, amplitude=amp, vect_amplitude=vect_amplitude, ncycles=ncycles, repeat_ncycles=repeat_ncycles, nsamples=nsamples)

        elif self.type == 'TIME_HIST':
            if time_hist is None:
                raise ValueError('TIME_HIST keyword is mandatory for type TIME_HIST')
            self.time_hist = self.xp.array(time_hist)

        else:
            raise ValueError(f'Unknown function type: {self.type}')

        self.nmodes = nmodes
        self.outputs['output'] = self.output
        self.output_value = None

    def prepare_trigger(self, t):
        super().prepare_trigger(t)

    def trigger_code(self):
        if self.type == 'SIN':
            phase = self.freq*2 * self.xp.pi*self.current_time_seconds + self.offset
            self.output_value = self.amp * self.xp.sin(phase, dtype=self.dtype) + self.constant
        elif self.type == 'SQUARE_WAVE':
            phase = self.freq*2 * self.xp.pi*self.current_time_seconds + self.offset
            self.output_value = self.amp * self.xp.sign(self.xp.sin(phase, dtype=self.dtype)) + self.constant
        elif self.type == 'LINEAR':
            self.output_value = self.slope * self.current_time_seconds + self.constant

        elif self.type == 'RANDOM':
            self.output_value = self.xp.random.normal(size=len(self.amp)) * self.amp + self.constant

        elif self.type == 'RANDOM_UNIFORM':
            lowv = self.constant - self.amp/2
            highv = self.constant + self.amp/2
            self.output_value = self.xp.random.uniform(low=lowv, high=highv)

        elif self.type in ['VIB_HIST', 'VIB_PSD', 'PUSH', 'PUSHPULL', 'TIME_HIST']:
            self.output_value = self.get_time_hist_at_current_time()

        else:
            raise ValueError(f'Unknown function generator type: {self.type}')

    def post_trigger(self):
        
        if self.vsize > 1:
            self.output.value[:] = self.output_value * self.xp.ones(self.vsize, dtype=self.dtype)
        else:
            self.output.value = self.output_value

        self.output.generation_time = self.current_time

    def get_time_hist_at_current_time(self):
        t = self.current_time
        i = int(np.round(t / self._loop_dt))
        return self.xp.array(self.time_hist[i])

    def setup(self, loop_dt, loop_niters):
        super().setup(loop_dt, loop_niters)
        if self.vib:
            self.vib.set_niters(loop_niters + 1)
            self.vib.set_samp_freq(1.0 / self.t_to_seconds(loop_dt))
            self.vib.compute()
            self.time_hist = self.vib.get_time_hist()

#        if self.type in ['SIN', 'LINEAR', 'RANDOM']:
#            self.build_stream()

