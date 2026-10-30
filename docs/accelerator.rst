.. _accelerator:

Accelerating objects (CUDA graphs)
==================================

In a simulation of moderate size, the time per step is often set by the host,
not by the GPU: each processing object runs Python code that launches its
kernels one by one, and some objects wait for the GPU at each step. The accelerator
removes most of this host work. It is part of the iteras framework, so it works
for any plugin.

It is enabled with ``--accelerate`` on the command line::

    specula --accelerate params.yml

or with a section whose class is ``Accelerator`` in the parameter files, which also
takes options:

.. code-block:: yaml

    accelerator:
      class: 'Accelerator'
      exclude: ['psf']            # objects not accelerated
      report_file: 'accelerator.txt'   # which objects were accelerated, and why not the others

It works with the :doc:`allocator <allocator>` and with :ref:`MPI
<running_with_mpi>`: each process accelerates the objects it runs, on their devices.

What it does
------------

On each GPU, with the accelerator:

* the ``trigger_code()`` of each accelerated object is captured in a CUDA graph at
  its second trigger (``capture_after`` option), and the graph is launched at
  the following triggers instead of running the Python code. Each object has
  its own graph, so an object that does not trigger at some step is not
  launched, as without accelerator;
* all the GPU work goes to the default stream of the device, in trigger order:
  the graphs, and the kernels launched by the code that still runs in Python.
  The GPU work therefore runs in the right order without the host waiting for
  it: ``post_trigger()`` no longer waits for the graph of the object. The host
  waits only where it reads results (copies to the host, for MPI, DataStore or
  objects that read GPU values);
* ``check_ready()``, ``prepare_trigger()`` and ``post_trigger()`` still run as
  Python code at each step. So the values computed on the host at each step,
  like the time of the generators or the time step of the atmosphere, are up
  to date: the generators write the time in a GPU array in
  ``prepare_trigger()``, which their graph reads.

The objects keep their parameters, inputs and outputs: an accelerated simulation is
the same simulation, and its results agree with those of the normal run to
rounding (different kernels are used for the matrix products, see below). The
GPU work of the objects is not changed: the accelerator removes host time, not GPU
time. Simulations whose time per step is set by the GPU gain little.

Which objects are accelerated
-----------------------------

An object is accelerated if it runs on a GPU, is not in ``exclude``, its class does
not override ``trigger()``, and its ``can_accelerate()`` method returns ``True``
when the graph is about to be captured (see :doc:`extension_api`). This method
declares that the graph will do the same as ``trigger_code()`` at all the
following steps:

* objects that build their own CUDA graph (``build_stream()``), like
  ``AtmoEvolution``, ``AtmoPropagation``, ``ModulatedPyramid``, ``SH`` and
  ``ShSlopec``, follow these rules already: they are accelerated by default (their
  own graph is released and replaced by the accelerator's one);
* other classes declare it after checking their ``trigger_code()``:
  ``WaveGenerator``, ``IirFilter`` and ``Integrator``, ``BaseSlicer``,
  ``LinearCombination``, ``PyrSlopec``, ``Modalrec``, ``ModalAnalysis``;
  ``BaseOperation`` once its delayed inputs have been produced; ``DM`` without
  force limitation; ``CCD`` when it reads out at every step, after its start
  time;
* the others are not accelerated, for example ``PSF`` (it logs the Strehl ratio,
  read from the GPU, at each step), ``PushPullGenerator`` and
  ``ModalrecImplicitPolc`` (they choose on the host what to compute at each
  step), ``DataStore`` (it copies its inputs to the host), and the objects on
  the CPU.

At the end of the run, the accelerator logs a table of the objects with their
status (``accelerated``, or why not), also written to ``report_file`` if given.

Safety checks
-------------

Two checks protect the results when an object breaks the rules:

* CuPy refuses, with an exception, the calls that cannot be part of a graph:
  cuBLAS, cuRAND and the copies between host and GPU. If the capture of an
  object fails, its ``trigger_code()`` runs again as Python code for that step
  and the object is not accelerated anymore (with a warning). This is the case, for
  example, of a ``CCD`` with noise, whose random numbers come from cuRAND or
  need copies from the host.
* The graph reads and writes the arrays the object had when it was captured.
  Before each launch, the accelerator checks that the arrays of the object, of the
  data objects it holds and of its inputs are still the same (same memory
  address). An array that was reallocated, for example by ``prepare_trigger()``,
  would make the graph use stale or freed memory: such an object is not accelerated
  anymore (with a warning naming the array), and runs as without accelerator. The
  check takes a few microseconds per object; ``check_pointers: false`` disables
  it.

What is not checked: a Python value computed in ``trigger_code()`` from
something that changes at each step (the time, a counter, a value read from
an input on the host) is frozen in the graph at its capture value. This is
what ``can_accelerate()`` declares not to happen.

Matrix products
---------------

CuPy does not allow cuBLAS during a graph capture, so ``a @ b`` cannot be
captured. ``specula.lib.graph_matmul.matmul(a, b, xp)`` is ``a @ b`` (cuBLAS)
outside a capture; inside one it uses kernels that can be captured, chosen by
the memory layout of the matrix: ``matvec()`` (one warp per row of the matrix)
and ``vecmat()`` (partial sums over chunks of rows, added in a fixed order).
Both are deterministic and, on the matrices of MORFEO and SOUL (influence
functions, mode-to-command and reconstruction matrices), as fast as cuBLAS,
without copying the transposed matrix. ``Modalrec``, ``DM``, ``ModalAnalysis``
and the slope computers use it, so their results with and without accelerator
differ by rounding.

With the allocator
------------------

With an ``Accelerator`` section (or ``--accelerate``), the allocator profiles the objects
accelerated, as they will run: their host time is the time to launch their graph,
and the accelerated objects with their own graph no longer wait for the GPU in
``post_trigger()``. These costs are kept in the profile file apart from those
measured without accelerator, and an allocation computed with the accelerator is stale
without it (and the reverse). Accelerating does not constrain the placement: each
object is accelerated on the device where it is placed.

Options
-------

See :class:`iteras.data_objects.accelerator.Accelerator`.

``exclude``
    Names of objects not to accelerate.
``capture_after``
    Triggers of each object before its graph is captured (default 1): the first
    triggers allocate the buffers and the FFT plans the graph reuses.
``check_pointers``
    Check, before each launch, that the arrays of the object have not been
    reallocated (default ``true``).
``report_file``
    File where the table of the accelerated objects is written at the end.

Measured gains and accuracy
---------------------------

Time per step on a computer with two NVIDIA RTX PRO 5000 Blackwell GPUs, for
the configurations in ``config/`` (full runs, median after the first 20% of
the steps), without and with the accelerator, on one GPU and with the placement of
the :doc:`allocator <allocator>` (one process):

.. list-table::
   :header-rows: 1
   :widths: 22 13 13 13 13 26

   * - Simulation
     - normal
     - accelerated
     - allocated
     - allocated, accelerated
     - accelerated objects
   * - SCAO SH (``scao``)
     - 2.29 ms
     - 1.76 ms
     - 2.27 ms
     - 1.51 ms
     - 10 of 13
   * - SCAO pyramid (``scao``)
     - 3.20 ms
     - 2.35 ms
     - 3.01 ms
     - 1.95 ms
     - 10 of 13
   * - space8m
     - 3.19 ms
     - 2.76 ms
     - 3.54 ms
     - 2.49 ms
     - 6 of 12
   * - SOUL
     - 3.56 ms
     - 2.61 ms
     - 3.76 ms
     - 2.19 ms
     - 12 of 14
   * - ANDES
     - 35.0 ms
     - 34.2 ms
     - 33.3 ms
     - 30.4 ms
     - 13 of 19
   * - MORFEO (full)
     - 64.1 ms
     - 64.1 ms
     - 40.9 ms
     - 38.6 ms
     - 46 of 63

The small simulations, whose time is set by the host, run 1.3 to 1.6 times
faster; the large ones, set by the GPUs, gain little from the accelerator alone.

Accuracy: with the noise of the detectors off, the outputs of the accelerated runs
(slopes, commands, residual fields, Strehl ratios over 100 steps) agree with
those of the normal run within 1e-5 of their range (they differ by rounding:
other kernels for the matrix products, and the order of some sums). With noise,
a rounding difference eventually changes a random sample (a Poisson count of a
detector pixel), after which the noise realization, and the run, differ: the
statistics agree (for example the mean Strehl ratio of SOUL, 0.8960 over 18700
steps, in all the modes). Compare accelerated and normal runs with the noise off.

Writing objects that can be accelerated
---------------------------------------

See ``can_accelerate()`` in :doc:`extension_api`. In short, ``trigger_code()`` must
only launch GPU work on arrays that are updated in place, with no Python value
that changes from one step to the next: values that change (time, inputs read
on the host, counters) are written to GPU arrays in ``prepare_trigger()``, which
runs at each step, and ``trigger_code()`` reads them from there. Use
``graph_matmul.matmul()`` for matrix products, and no random numbers from
cuRAND (``cupy.random.default_rng()`` keeps its state on the GPU and can be
captured, except for the distributions that copy data from the host, like
``poisson``).
