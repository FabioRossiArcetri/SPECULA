'''
Merger: run the GPU work of the processing objects as a chain of CUDA graphs.

Without merger, each object runs its trigger_code() as Python code launching
its kernels one by one (or replays its own CUDA graph, waiting for it in
post_trigger()). In small simulations, this host work, not the GPU, sets the
time per step.

With the merger, on each GPU:

* the trigger_code() of each merged object is captured once in a CUDA graph,
  which is then launched at each step instead of running the Python code;
* all the GPU work of the objects (graphs, and the kernels of the phases that
  still run as Python code) goes to the default stream of the device, in
  trigger order, so that it runs in the right order without the host waiting
  for the GPU between objects (BaseProcessingObj.post_trigger() does not wait);
* check_ready(), prepare_trigger() and post_trigger() still run as Python code
  at each step, so that values computed on the host from the time or the
  inputs (like the time of the generators, copied to the GPU in
  prepare_trigger()) are up to date. Only trigger_code() is frozen in the graph.

An object is merged if it runs on a GPU, is not excluded by the Merger
options, and its can_merge() method returns True when its graph is about to
be captured (see BaseProcessingObj.can_merge()). Each object has its own
graph, so objects that do not trigger at every step are simply not launched,
as without merger.

Safety: CuPy refuses, with an exception, the calls that cannot be part of a
graph (cuBLAS, cuRAND, copies between the host and the GPU). If the capture
fails, trigger_code() runs again as Python code for that step, and the object
is not merged anymore. Before each launch, the arrays of the object, of the
data objects it holds and of its inputs are checked to be the same as when
the graph was captured: the graph reads and writes their memory, so an array
that was reallocated (for example by prepare_trigger()) would make it use
stale or freed memory. Such an object is not merged anymore either.
'''
import iteras
from iteras.base_data_obj import BaseDataObj
from iteras.log import get_iteras_logger

_logger = get_iteras_logger(__name__)


def _capture_stream(device_id):
    '''A stream of the device used only to capture the graphs'''
    if device_id not in _capture_stream.streams:
        with iteras.cp.cuda.Device(device_id):
            _capture_stream.streams[device_id] = iteras.cp.cuda.Stream(non_blocking=True)
    return _capture_stream.streams[device_id]


_capture_stream.streams = {}


def _items(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def array_refs(obj):
    '''
    [(holder, attribute, address)] of the GPU arrays that the graph of *obj*
    may use: those of the object, of the data objects it holds (directly or in
    lists) and of its local inputs.
    '''
    holders = [obj]
    seen = {id(obj)}

    def add(v):
        if isinstance(v, BaseDataObj) and id(v) not in seen:
            seen.add(id(v))
            holders.append(v)

    for v in list(obj.__dict__.values()):
        if isinstance(v, (list, tuple)):
            for item in v:
                add(item)
        else:
            add(v)
    for v in obj.local_inputs.values():
        for item in _items(v):
            add(item)
    refs = []
    for h in holders:
        for k, v in h.__dict__.items():
            if isinstance(v, iteras.cp.ndarray):
                refs.append((h, k, v.data.ptr))
    return refs


def changed_ref(refs):
    '''First (holder, attribute) of *refs* whose array is not at the same address anymore'''
    ndarray = iteras.cp.ndarray
    for h, k, ptr in refs:
        v = h.__dict__.get(k)
        if v.__class__ is not ndarray or v.data.ptr != ptr:
            return h, k
    return None


class GraphRunner:
    '''
    Runs the trigger_code() of a merged object: as Python code for the first
    triggers, then captured in a CUDA graph launched at each trigger.
    Set as obj._merge by Merger.apply().
    '''

    def __init__(self, obj, options):
        self.obj = obj
        self.options = options
        self.graph = None
        self.pool = None
        self.refs = []
        self.outputs = []       # data objects whose generation_time trigger_code() sets
        self.triggers = 0
        self.status = 'not captured yet'

    def trigger(self):
        obj = self.obj
        if self.triggers == 0:
            self.release_own_graph()
        self.triggers += 1
        if self.graph is not None:
            if self.options.check_pointers:
                changed = changed_ref(self.refs)
                if changed is not None:
                    h, k = changed
                    where = k if h is obj else f'{type(h).__name__}.{k}'
                    self.detach(f'array {where} reallocated after the capture')
                    obj.trigger()
                    return
            self.graph.launch(stream=iteras.cp.cuda.get_current_stream())
            t = obj.current_time
            for o in self.outputs:
                o.generation_time = t
            return
        if self.triggers <= self.options.capture_after:
            obj.trigger_code()
            return
        reason = obj.can_merge()
        if reason is not True:
            # Retried at the next trigger: it may become possible (e.g. after a start time)
            self.status = f'can_merge(): {reason or False}'
            obj.trigger_code()
            return
        self.capture()

    def capture(self):
        obj = self.obj
        t = obj.current_time
        outputs = [o for out in obj.outputs.values() for o in _items(out)]
        before = [o.generation_time for o in outputs]
        stream = _capture_stream(obj.target_device_idx)
        pool = iteras.cp.cuda.MemoryPool()
        error = None
        graph = None
        # The work queued before the graph (prepare_trigger, the other objects) is on the
        # default stream, where the graph is launched: capture it on a separate stream
        with stream, iteras.cp.cuda.using_allocator(pool.malloc):
            stream.begin_capture(mode=iteras.cp.cuda.runtime.streamCaptureModeRelaxed)
            try:
                obj.trigger_code()
            except Exception as e:
                error = e
            finally:
                try:
                    graph = stream.end_capture()
                except Exception as e:
                    error = error or e
        if error is not None:
            graph = None
            pool.free_all_blocks()
            # The Python code ran without its GPU work: run it again, then as without merger
            first = str(error).splitlines()[0] if str(error) else ''
            self.detach(f'capture failed: {type(error).__name__}: {first[:200]}')
            obj.trigger_code()
            return
        graph.launch(stream=iteras.cp.cuda.get_current_stream())
        self.graph = graph
        self.pool = pool
        self.outputs = [o for o, b in zip(outputs, before) if o.generation_time == t and b != t]
        self.refs = array_refs(obj) if self.options.check_pointers else []
        # The graph keeps the memory of the temporary arrays of trigger_code()
        obj.gpu_bytes_used += pool.total_bytes()
        self.status = 'merged'
        _logger.debug(f'Merger: {obj.name}: graph captured ({pool.total_bytes() / 2**20:.1f} MB)')

    def release_own_graph(self):
        '''The CUDA graph that the object captured by itself (see build_stream()) is not used'''
        obj = self.obj
        obj.cuda_graph = None
        obj._cuda_graph_invalid = False
        if obj._graph_mempool is not None:
            obj._graph_mempool.free_all_blocks()

    def invalidate(self):
        '''Something frozen in the graph changed: capture it again at the next trigger'''
        if self.graph is not None:
            self.graph = None
            self.obj.gpu_bytes_used -= self.pool.total_bytes()
            self.pool.free_all_blocks()
            self.pool = None
            self.triggers = min(self.triggers, self.options.capture_after)

    def detach(self, reason):
        '''The object is not merged anymore: it runs as without merger from now on'''
        obj = self.obj
        _logger.warning(f'Merger: {obj.name} ({type(obj).__name__}) is not merged anymore: {reason}')
        self.invalidate()
        self.status = f'not merged: {reason}'
        obj._merge = None
        if obj.stream is not None:
            # Its own CUDA graph was released by the merger: capture it again
            obj._cuda_graph_invalid = True


class Merger:
    '''
    Applies the merger options (iteras.data_objects.merger.Merger) to the
    processing objects of this process.
    '''

    def __init__(self, options):
        self.options = options
        self.rows = []          # (object, GraphRunner or None, reason it is not merged)

    def apply(self, objs):
        '''Set a GraphRunner on each object of *objs* (processing objects of this process) that can be merged'''
        from iteras.base_processing_obj import BaseProcessingObj
        for obj in objs:
            reason = None
            if obj.target_device_idx < 0:
                reason = 'on the CPU'
            elif obj.name in self.options.exclude:
                reason = 'excluded'
            elif type(obj).trigger is not BaseProcessingObj.trigger:
                reason = 'its class overrides trigger()'
            runner = None
            if reason is None:
                runner = obj._merge = GraphRunner(obj, self.options)
            self.rows.append((obj, runner, reason))

    def runners(self):
        return [runner for _, runner, _ in self.rows if runner is not None]

    def report(self):
        '''Text table: which objects are merged, and why the others are not'''
        lines = ['Merger:']
        n = 0
        for obj, runner, reason in self.rows:
            if runner is not None:
                reason = runner.status
            if reason == 'merged':
                n += 1
            lines.append(f'  {obj.name:30s} {type(obj).__name__:26s} {reason}')
        lines.insert(1, f'  {n} of {len(self.rows)} objects merged')
        return '\n'.join(lines)

    def finish(self):
        text = self.report()
        _logger.info(text)
        if self.options.report_file:
            with open(self.options.report_file, 'w') as f:
                f.write(text + '\n')
