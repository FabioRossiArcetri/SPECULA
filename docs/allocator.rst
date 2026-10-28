.. _allocator:

Automatic allocation
====================

The allocator places the objects of a simulation on the available resources
(nodes, MPI processes, GPUs and CPUs), choosing the placement with the lowest
estimated time per step that fits in the memory of each device. It is part of
the iteras framework, so it works for any plugin.

It is enabled by adding a section whose class is ``Architecture`` to the
parameter files, for example in a separate file given after the others::

    specula params.yml architecture.yml

Describing the resources
------------------------

On a single computer, the GPUs and the CPU can be detected at run time:

.. code-block:: yaml

    architecture:
      class: 'Architecture'
      nodes: 'local'
      memory_source: 'probe'      # memory free right now (the GPUs may be shared)

On a cluster, the nodes are described explicitly. The memory is assumed to be
fully available (``memory_source: 'declared'``, the default), as on nodes
granted by slurm. For example, two Leonardo Booster nodes:

.. code-block:: yaml

    architecture:
      class: 'Architecture'
      nodes:
        - name: 'booster'
          count: 2
          cpu: {cores: 32, memory_gb: 512}
          gpus: {count: 4, memory_gb: 64, model: 'A100'}
          links: {gpu_gpu_gb_s: 100, host_gpu_gb_s: 12}
      inter_node: {bandwidth_gb_s: 10, latency_us: 5}
      processes: 'per_gpu'

With ``memory_source: 'slurm'``, the number of nodes, cores, memory and GPUs
are read from the slurm environment of the job, and the rest from the first
node type.

The main options (see :class:`iteras.data_objects.architecture.Architecture`
for all of them):

``mode``
    ``'apply'`` (default): allocate, then run the simulation. ``'plan'``: only
    write the allocation and its report (also ``--allocate`` on the command line).
``processes``
    ``'auto'`` (default) compares one process per GPU with a single process
    driving all the GPUs of the node; ``'per_gpu'``; ``'single'``.
``override_placement``
    Objects with ``target_rank`` or ``target_device_idx`` in the parameter files
    are not moved, unless this is ``true``.
``allocation_file``
    Where the allocation is written (default ``allocation.yml``), with a report
    (``allocation_report.txt``) and the measured costs (``allocation_profile.json``).
``memory_margin``
    Fraction of each device kept free (default 0.1).

How it works
------------

1. **Profiling.** The costs of the objects are measured by running the
   simulation itself for a few steps on the local GPU with most free memory
   (or on the CPU if there is none): host time of each phase, GPU time, memory,
   arrays shared between objects, size of each output, size of the data objects
   referenced with ``_ref``. The warm-up goes past the largest ``start_time``
   parameter, and the measurement covers two periods of the slowest object
   (largest ``dt``). If the objects do not fit on the GPU, they are measured in
   chunks, the rest of the simulation running on the CPU. The CPU is measured
   only if requested (``profile_cpu: true``), or for a few objects when the GPUs
   are too small for the whole simulation. The costs are kept in the profile
   file and reused.
2. **Estimation.** The time per step of a placement is estimated by simulating
   a few steps of the loop: each process runs its objects in trigger order, the
   GPU work of each device runs in parallel with the host, data is copied between
   devices and sent between processes as the framework does it.
3. **Search.** Objects that must stay together (an object and the processing
   objects it references, the users of a data object that some object modifies)
   are placed as a unit. An initial placement (memory-aware HEFT) is improved by
   moving and swapping objects, for each process layout.

Objects of classes that must run on the CPU or in the main process (see
:doc:`extension_api`) are placed accordingly. Objects of classes whose
constructor does not accept ``target_device_idx`` stay on the default device of
their process. A data object used by objects on several devices is duplicated
on each of them.

Using the allocation
--------------------

The allocation file contains ``<object>_override`` sections with
``target_rank`` and ``target_device_idx``, so it can also be given as an
additional parameter file without the ``Architecture`` section. With the
``Architecture`` section, later runs reuse it; if the simulation, the
placements in the parameter files or the architecture have changed, the run
stops with an error that names the difference. Use ``--reallocate`` (or
``recompute: true``) to compute a new one.

If the allocation needs several processes, the simulation must be launched
with MPI and the number of ranks given in the report::

    mpirun -n 8 specula --mpi params.yml architecture.yml

Limitations
-----------

* The trigger order is not changed (``schedule: 'fixed'``).
* GPUs of a model that was not profiled use the times of the profiled one,
  divided by ``relative_speed``.
* Transfers between processes go through the host, as in the framework.
