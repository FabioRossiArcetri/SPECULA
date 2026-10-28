'''
Memory and time per step of a placement.

A placement maps each processing object to (process, slot), where a slot is
a device of a node (see machine.Slot). The time per step is estimated by
simulating a few steps of the loop as LoopControl runs it:

* each process walks the trigger levels in order, with only its objects; for
  each level it gathers the inputs of all its objects (waiting for messages
  from other processes, copying data between its devices), then triggers
  them, then runs post_trigger and sends the outputs to other processes;
* the host work of a process is sequential; the GPU work of each device is
  a sequential stream, which runs while the host goes on, except for objects
  that wait for the GPU in trigger or post_trigger (see profiler) and before
  copying data out of a GPU;
* a large MPI message moves only while its sender is in an MPI call.

The estimate is taken in steady state, so that a process running ahead of
the others (allowed by the asynchronous MPI sends) is accounted for.
'''
import bisect
from collections import defaultdict
from dataclasses import dataclass

MPI_BUFFER_BYTES = 40e6     # attached MPI buffer of each process (see iteras.__init__)
EAGER_BYTES = 64 * 1024     # larger MPI messages need the sender to call MPI to progress
MAX_PASSES = 10             # simulation passes to settle the MPI progress of large messages


@dataclass(frozen=True)
class Proc:
    rank: int
    node: int


@dataclass
class Evaluation:
    step_ns: float
    comm_bytes: float
    memory: dict            # slot -> bytes
    feasible: bool
    overflow: dict          # slot -> bytes over capacity

    def key(self):
        return (not self.feasible, self.step_ns, self.comm_bytes)


class Estimator:
    '''
    Parameters
    ----------
    graph : model.SimGraph
    machine : machine.Machine
    costs : dict
        {(name, kind): Cost} for every object and every slot kind it may use.
    refs : dict
        {data object name: bytes}
    shared : dict
        {kind: [(owner names, bytes)]}, arrays shared by objects on one device.
    overhead_ns : float
        Loop overhead per step and per process, not attributed to any object.
    steps : int
        Number of simulated steps; the second half is used.
    '''

    def __init__(self, graph, machine, costs, refs, shared, overhead_ns=0.0, steps=8):
        self.graph = graph
        self.machine = machine
        self.costs = costs
        self.refs = refs
        self.shared = shared
        self.overhead_ns = overhead_ns
        self.steps = steps
        levels = defaultdict(list)
        for name in graph.procs:
            levels[graph.level[name]].append(name)
        self.levels = [levels[k] for k in sorted(levels)]
        self.ref_consumers = {d: graph.ref_consumers[d] for d in refs}

    def cost(self, name, slot):
        return self.costs.get((name, self.machine.slots[slot].kind))

    def out_bytes(self, edge, slot):
        c = self.cost(edge.src, slot)
        return c.outputs.get(edge.out, 0.0) if c is not None else 0.0

    # -- memory --------------------------------------------------------

    def memory(self, placement):
        slots = self.machine.slots
        use = defaultdict(float)
        for name, (_, s) in placement.items():
            use[s] += self.cost(name, s).mem
        on_slot = defaultdict(set)
        for name, (_, s) in placement.items():
            on_slot[s].add(name)
        for s, names in on_slot.items():
            for owners, size in self.shared.get(slots[s].kind, []):
                if names.intersection(owners):
                    use[s] += size
        for d, consumers in self.ref_consumers.items():
            for s in {placement[c][1] for c in consumers if c in placement}:
                use[s] += self.refs[d]
        clones = set()
        procs_on_gpu = set()
        for name, (p, s) in placement.items():
            if slots[s].is_gpu:
                procs_on_gpu.add((p, s))
            for e in self.graph.in_edges[name]:
                if e.src not in placement:
                    continue
                ps, ss = placement[e.src]
                if ss != s and (e.src, e.out, s) not in clones:
                    clones.add((e.src, e.out, s))
                    use[s] += self.out_bytes(e, ss)
        for p, s in procs_on_gpu:
            use[s] += self.machine.context_overhead
        for p in {p for p, _ in placement.values()}:
            use[self.machine.nodes[p.node].cpu_slot] += MPI_BUFFER_BYTES
        return dict(use)

    # -- time ----------------------------------------------------------

    def step_time(self, placement):
        '''
        Steady-state time per step (ns) and bytes exchanged between devices per step.

        A large MPI message (above EAGER_BYTES) moves only while its sender is
        inside an MPI call (no asynchronous progress): it is delivered no earlier
        than the next send or receive of the sender. Since those times depend on
        the deliveries, the simulation is repeated with the MPI calls of the
        previous pass, until the time per step settles.
        '''
        remote = any(placement[e.src][0] != placement[e.dst][0] for e in self.graph.edges
                     if e.src in placement and e.dst in placement)
        events = None
        per_step = None
        stable = 0
        for _ in range(MAX_PASSES if remote else 1):
            new_step, comm_bytes, events = self._simulate(placement, events)
            if per_step is not None and abs(new_step - per_step) <= 0.005 * per_step:
                stable += 1
                if stable == 2:
                    per_step = new_step
                    break
            else:
                stable = 0
            per_step = new_step
        return per_step, comm_bytes

    @staticmethod
    def _next_progress(intervals, t):
        '''First time >= t at which a process is inside an MPI call, or None'''
        i = bisect.bisect_left(intervals, (t, t))
        if i > 0 and intervals[i - 1][1] >= t:
            return t
        return intervals[i][0] if i < len(intervals) else None

    def _simulate(self, placement, prev_events):
        m = self.machine
        slots = m.slots
        host = defaultdict(float)       # process -> time
        stream = defaultdict(float)     # GPU slot -> time when its queued work ends
        done = {}                       # (name, step) -> time when its results are ready
        arrival = {}                    # (name, step, process) -> time when its messages arrive
        events = defaultdict(list)      # process -> [(start, end)] of its MPI calls
        remote_dests = {}
        proc_slot = {p: s for p, s in placement.values()}
        comm_bytes = 0.0
        for name, (p, s) in placement.items():
            dests = defaultdict(float)
            for e in self.graph.out_edges[name]:
                if e.dst in placement and placement[e.dst][0] != p:
                    dests[placement[e.dst][0]] += self.out_bytes(e, s)
            remote_dests[name] = dests
            for e in self.graph.in_edges[name]:
                if e.src in placement and placement[e.src][1] != s:
                    comm_bytes += self.out_bytes(e, placement[e.src][1])

        step_end = []
        for k in range(self.steps):
            for level in self.levels:
                by_proc = defaultdict(list)
                for name in level:
                    if name in placement:
                        by_proc[placement[name][0]].append(name)
                for p, names in by_proc.items():
                    t = host[p]
                    # 1. inputs
                    for name in names:
                        s = placement[name][1]
                        c = self.cost(name, s)
                        for e in self.graph.in_edges[name]:
                            ks = k + e.delay
                            if ks < 0 or e.src not in placement:
                                continue
                            ps, ss = placement[e.src]
                            if ps != p:
                                t0 = t
                                t = max(t, arrival.get((e.src, ks, p), 0.0))
                                events[p].append((t0, t))
                                t += m.host_gpu_ns(s, self.out_bytes(e, ss))
                            elif ss != s:
                                t = max(t, done.get((e.src, ks), 0.0))
                                t += m.copy_ns(ss, s, self.out_bytes(e, ss))
                        t += c.h_in
                    # 2. trigger
                    for name in names:
                        s = placement[name][1]
                        c = self.cost(name, s)
                        t += c.h_prep + c.h_trig
                        if slots[s].is_gpu:
                            start = max(t, stream[s])
                            stream[s] = start + c.g
                            done[(name, k)] = stream[s]
                            if c.syncs == 'trigger':
                                t = max(t, stream[s])
                        else:
                            done[(name, k)] = t
                    # 3. post_trigger and sends
                    for name in names:
                        s = placement[name][1]
                        c = self.cost(name, s)
                        if c.syncs == 'post':
                            t = max(t, done[(name, k)])
                        t += c.h_post
                        dests = remote_dests[name]
                        if dests:
                            t = max(t, done[(name, k)])
                            for q, nbytes in dests.items():
                                bw, lat = m.link(s, proc_slot[q])
                                t += m.host_gpu_ns(s, nbytes) + c.h_send / len(dests)
                                events[p].append((t, t))
                                deliver = t + lat + nbytes / bw
                                if nbytes > EAGER_BYTES and prev_events is not None:
                                    # the next MPI call after this send (1 ns later)
                                    nxt = self._next_progress(prev_events.get(p, []), t + 1)
                                    if nxt is not None:
                                        deliver = max(deliver, nxt + lat + nbytes / bw)
                                arrival[(name, k, q)] = deliver
                    host[p] = t
            for p in list(host):
                host[p] += self.overhead_ns
            step_end.append(max(list(host.values()) + list(stream.values()) + [0.0]))
        half = self.steps // 2
        per_step = (step_end[-1] - step_end[half - 1]) / (self.steps - half)
        for p in events:
            events[p].sort()
        return per_step, comm_bytes, events

    def evaluate(self, placement):
        mem = self.memory(placement)
        overflow = {s: use - self.machine.slots[s].memory for s, use in mem.items()
                    if use > self.machine.slots[s].memory}
        step, comm = self.step_time(placement)
        return Evaluation(step, comm, mem, not overflow, overflow)
