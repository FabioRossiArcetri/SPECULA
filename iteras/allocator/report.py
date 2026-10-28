'''
Text report of an allocation, and evaluation of placements given by hand.
'''
from collections import defaultdict

from .cost import Proc
from .machine import GiB

MiB = 1024 ** 2


def make_report(graph, machine, est, solution, profile, allocator, allocation):
    placement = solution.placement()
    ev = solution.evaluation
    lines = ['Resources:', machine.describe(), '']
    lines.append(f'Layout: {solution.layout.policy}, {allocation["n_ranks"]} MPI rank(s)')
    for p in allocation['processes']:
        lines.append(f'  rank {p["rank"]}: node {p["node"]}, devices {p["devices"]}')
    lines.append('')
    lines.append(f'Estimated time per step: {ev.step_ns / 1e6:.2f} ms '
                 f'(data copied between devices: {ev.comm_bytes / MiB:.1f} MiB per step)')
    calib = getattr(est, 'calibration', None)
    if calib:
        lines.append(f'Profiling run (all objects on one GPU): measured {calib[1] / 1e6:.2f} ms per step, '
                     f'model {calib[0] / 1e6:.2f} ms; loop overhead added per step and process: '
                     f'{est.overhead_ns / 1e6:.2f} ms')
    lines.append('')
    lines.append(memory_table(machine, ev.memory))
    lines.append('')
    lines.append(object_table(graph, machine, est, placement))
    estimated = sorted({n for (n, k), c in est.costs.items()
                        if not c.measured and any(placement[n][1] == s for s, sl in enumerate(machine.slots)
                                                  if sl.kind == k)})
    if estimated:
        lines.append('')
        lines.append('GPU times scaled from another GPU model (relative_speed) for: '
                     + ', '.join(estimated))
    return '\n'.join(lines) + '\n'


def memory_table(machine, memory):
    lines = ['Memory per device (GiB, used / available):']
    for s, slot in enumerate(machine.slots):
        if s in memory:
            lines.append(f'  {slot.label():16s} {memory[s] / GiB:8.2f} / {slot.memory / GiB:8.2f}')
    return '\n'.join(lines)


def object_table(graph, machine, est, placement):
    lines = [f'{"object":24s} {"class":22s} {"rank":>4s} {"device":>6s} '
             f'{"host ms":>8s} {"GPU ms":>8s} {"mem MiB":>9s}']
    for name in graph.procs:
        if name not in placement:
            continue
        p, s = placement[name]
        c = est.cost(name, s)
        dev = machine.slots[s].idx
        lines.append(f'{name[:24]:24s} {graph.cls(name).__name__[:22]:22s} {p.rank:4d} {dev:6d} '
                     f'{c.host / 1e6:8.3f} {c.g / 1e6:8.3f} {c.mem / MiB:9.1f}')
    return '\n'.join(lines)


def placement_from_params(graph, machine, ranks_per_node=1, default_device=0):
    '''
    Placement written in the parameter file (target_rank, target_device_idx),
    as used by Estimator. Ranks are assumed to be numbered node by node, with
    *ranks_per_node* ranks per node (as with slurm's block distribution);
    objects without target_rank are on rank 0, objects without
    target_device_idx on *default_device*.
    '''
    placement = {}
    procs = {}
    for name in graph.procs:
        pars = graph.params[name]
        rank = pars.get('target_rank', 0) or 0
        dev = pars.get('target_device_idx', default_device)
        if dev is None:
            dev = default_device
        node = rank // ranks_per_node
        if node >= len(machine.nodes):
            raise ValueError(f'{name}: rank {rank} is on node {node}, which does not exist')
        n = machine.nodes[node]
        if not graph.device_settable[name]:
            dev = default_device
        if dev < 0:
            slot = n.cpu_slot
        else:
            if dev >= len(n.gpu_slots):
                raise ValueError(f'{name}: GPU {dev} does not exist on node {node}')
            slot = n.gpu_slots[dev]
        p = procs.setdefault(rank, Proc(rank, node))
        placement[name] = (p, slot)
    return placement


def describe_placement(graph, machine, placement):
    '''Short text: objects per (rank, device)'''
    groups = defaultdict(list)
    for name in graph.procs:
        if name in placement:
            p, s = placement[name]
            groups[(p.rank, machine.slots[s].label())].append(name)
    return '\n'.join(f'  rank {r} {lbl}: {", ".join(names)}' for (r, lbl), names in sorted(groups.items()))
