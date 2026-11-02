.. _cnn_reconstructor:

CNN reconstructor
=================

A convolutional neural network (a U-Net) can replace the linear modal
reconstructor: it is trained inside a simulation, on the data the
simulation produces, and then closes the loop with ``Conv2dNetRec``. This
page describes the objects and how to use them; the tutorial
:doc:`tutorials/cnn_rms_tutorial` shows the training and test steps on a
small example.

The CNN classes need PyTorch, an optional dependency: ``pip install specula[nn]``.

Objects
-------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Object
     - Inputs and outputs
   * - ``Slopes2D``
     - ``in_slopes`` (``Slopes``) -> ``out_value``: the 2D map of the slopes
       (``Slopes.get2d()``), x and y maps for ordinary slopes, one map with
       ``slopes_from_intensity``. It makes the slopes bufferable.
   * - ``DataBuffer``
     - Collects ``buffer_size`` steps of each input of ``input_list`` and emits
       them together as ``<prefix>_buffered`` (see the tutorial). With ``align``
       (default) the rows of all the outputs refer to the same steps.
   * - ``Conv2dNetTrainer``
     - ``input_2d_batch`` (the buffered maps) and ``labels`` (the buffered
       modes) -> ``loss``, ``val_loss``. Optional ``baseline`` (the network
       learns ``labels - baseline``) and ``gain_mod`` (samples taken at reduced
       loop gain are dropped or weighted). Writes the checkpoint.
   * - ``Conv2dNetTester``
     - The same inputs -> ``prediction``, ``loss``, and per-mode statistics at
       the end of the run. Optional ``reference``: another estimate of the same
       modes (e.g. a linear reconstructor's output, buffered alongside), scored
       on the same frames.
   * - ``Conv2dNetRec``
     - ``in_slopes`` -> ``out_modes``, as ``Modalrec``: the network at each step.
       Optional ``baseline`` input for networks trained on a residual.
   * - ``ModalGainModulation``
     - Optional ``gain_mod`` -> ``out_gain_mod``: random per-mode loop gain
       factors, redrawn every ``update_interval`` seconds, for the ``gain_mod``
       input of an integrator.

The checkpoint is two files: the weights (``<name>.pth``) and
``<name>_stats.json``, with the network architecture, the input and output
normalization and the per-mode prediction gain. ``Conv2dNetTester`` and
``Conv2dNetRec`` read everything from it: only the file name is configured.

Training
--------

The network is trained while a simulation runs. The labels are the modes
the reconstructor should output, typically ``ModalAnalysis.out_modes`` of
the residual phase; the input is the 2D map of the slopes. Both are buffered
together and each batch is one training step:

.. code-block:: yaml

   slopes2d:
     class:             'Slopes2D'
     inputs:
       in_slopes:       'slopec.out_slopes'
     outputs: ['out_value']

   data_buffer:
     class:             'DataBuffer'
     buffer_size:       1000
     inputs:
       input_list: ['input_2d-slopes2d.out_value', 'modes-modal_analysis.out_modes']

   nn_trainer:
     class:             'Conv2dNetTrainer'
     network_filename:  './trained_models/my_network.pth'
     nmodes:            100
     input_channels:    2                   # x and y slope maps
     depth:             3
     loss_delta:        10.0                # [label units] Huber loss transition
     replay_size:       20000
     inputs:
       input_2d_batch:  'data_buffer.input_2d_buffered'
       labels:          'data_buffer.modes_buffered'
     outputs: ['loss', 'val_loss']

The data can come from a loop closed by a linear reconstructor, or from the
CNN itself (see below). Main parameters:

* ``input_channels``: the channels of the buffered maps, 2 for x/y slope maps,
  1 for one map per frame. A mismatch is an error. ``channel_product`` uses
  the product of two channels as a single-channel input.
* ``label_offset``, ``nmodes``: the network predicts ``labels[label_offset:label_offset + nmodes]``.
* ``depth``: each level of the U-Net halves the map, which must be at least
  ``2 ** depth`` pixels wide. ``head_type: 'spatial'`` keeps the position of the
  features (needed for many modes); its ``head_grid`` should be about the map size.
* ``loss_delta``, ``max_label_rms``: in the units of the labels (nm for modes in nm).
* ``replay_size``, ``replay_ratio``: a replay buffer of past samples. Without
  it the network forgets the conditions of the previous batches.
* ``load_from_file``: resume from an existing checkpoint, which must have the
  same architecture.

The validation loss and FVU (error variance / label variance) are computed
on the most recent frames of each batch, which the network has not trained
on. For a fair test, use ``Conv2dNetTester`` on an independent run.

Closing the loop
----------------

``Conv2dNetRec`` replaces the linear reconstructor:

.. code-block:: yaml

   modalrec:
     class:             'Conv2dNetRec'
     network_filename:  './trained_models/my_network.pth'
     calibrate_gain:    true
     inputs:
       in_slopes:       'slopec.out_slopes'
     outputs: ['out_modes']

A network shrinks its predictions towards the mean on the modes it predicts
poorly, which in a loop acts as a lower gain on those modes. The trainer
measures this per-mode gain and stores it in the checkpoint;
``calibrate_gain`` divides it out, so that the integrator gains keep their
meaning across retrainings. Do not use it for a network trained on a
residual (with ``baseline``): it amplifies the correction.

Training in closed loop
-----------------------

The network can keep training on the data of the loop it closes: the
trainer and ``Conv2dNetRec`` use the same checkpoint, and the reconstructor
reloads it every ``reload_interval`` seconds, so the network closing the loop
stays fixed between reloads. ``snapshot_interval`` keeps a copy of the
network at regular times (``<name>_t<seconds>s.pth``), to evaluate the stages
of a long training:

.. code-block:: yaml

   modalrec_override:
     reload_interval:   20.0

   nn_trainer_override:
     load_from_file:    true
     snapshot_interval: 20.0
     gain_mod_threshold: 0.0               # keep the frames at reduced gain...
     settle_frames:     50                  # ...and weight them, with the first
     transient_weight:  0.3                 # frames after the loop is back at full gain
     inputs:
       input_2d_batch:  'data_buffer.input_2d_buffered'
       labels:          'data_buffer.modes_buffered'
       gain_mod:        'data_buffer.gain_mod_buffered'

A loop driven by a poorly trained network can lose the residual, and a
network trained on such data learns to lose it: check the runs (Strehl
ratio, limit cycles, growing residuals) before continuing a training from
their checkpoints. A gradual increase of the loop gain at the start (a
``ScheduleGenerator`` on the integrator's ``gain_mod``) helps the first closed
loops of a new network.

Notes
-----

* With ``--accelerate`` the CNN objects run outside the CUDA graphs and the
  rest of the loop is accelerated. The results are not identical to a normal
  run: the float32 sums of the captured kernels differ in the last bits, and
  cuDNN computes the convolutions in TF32 by default, which turns such a
  difference into about 1e-4 in the network output; a closed loop then
  amplifies it.
* The training is not seeded: two trainings give slightly different networks.
