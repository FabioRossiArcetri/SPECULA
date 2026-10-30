'''
Options of the merger (see :mod:`iteras.merger`).

A section of the parameter file whose class is Merger makes Simul run the GPU
work of the objects as CUDA graphs chained on one stream per device, without
the host waiting for the GPU between objects.
'''


class Merger:
    '''
    Merger options.

    Parameters
    ----------
    exclude : list of str, optional
        Names of objects that are not merged (they run as without merger).
    capture_after : int, optional
        Number of triggers of each object before its CUDA graph is captured
        (default 1): the first triggers allocate the buffers and build the
        FFT plans that the graph reuses.
    check_pointers : bool, optional
        Before each launch of the graph of an object, check that the arrays of
        the object and of its inputs have not been reallocated since the
        capture (default True). An object whose arrays change is not merged
        anymore: its graph would read stale memory.
    report_file : str, optional
        File where the merger writes, at the end of the run, which objects were
        merged and why the others were not (default: only logged).
    '''

    def __init__(self,
                 exclude: list = None,
                 capture_after: int = 1,
                 check_pointers: bool = True,
                 report_file: str = None):
        self.exclude = list(exclude or [])
        if int(capture_after) < 1:
            raise ValueError(f'Merger: capture_after must be at least 1, got {capture_after}')
        self.capture_after = int(capture_after)
        self.check_pointers = bool(check_pointers)
        self.report_file = report_file

    def as_dict(self):
        return {'exclude': sorted(self.exclude), 'capture_after': self.capture_after,
                'check_pointers': self.check_pointers}
