Plugins
=======

The simulation framework used by SPECULA (base classes, connections, loop
control, MPI support, data storage, displays) lives in a separate package,
``iteras``, which knows nothing about adaptive optics. SPECULA is an iteras
*plugin*: a package that adds data objects, processing objects and displays
for a specific domain. Other packages can be plugins too, of iteras directly
or of another plugin.

This page describes how plugins are registered, how the classes named in a
parameter file are found, and how this relates to the ``add_modules``
parameter.

What a plugin is
----------------

A plugin is an ordinary Python package whose classes derive from the iteras
base classes (``BaseDataObj``, ``BaseProcessingObj``, ``BaseDisplay``, ...),
directly or through another plugin. The only convention to follow is the one
already used in SPECULA: class ``FooBar`` is defined in a module called
``foo_bar.py``, inside one of the packages that the plugin registers. This is
how a ``class: FooBar`` entry in a parameter file is resolved.

Plugins can be stacked, since a plugin is just code that imports another
package. For example::

    iteras        framework: base classes, Simul, LoopControl, DataStore, displays, ...
      specula     AO plugin: ElectricField, Pixels, IFunc, DM, PyrSlopec, SimulParams, ...
        speculab  laboratory plugin: PlicoDM, PysilicoCamera, ... (uses Pixels, IFunc, ...)

The registry
------------

The module ``iteras.registry`` holds the information provided by the plugins.
It is filled when the plugins are imported, and is shared by the whole
process.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Entry
     - Filled by
     - Used by
   * - Search packages
     - the iteras packages first, then the packages of each plugin, in
       registration order
     - ``iteras.lib.utils.import_class()``, which resolves the class names of
       the parameter files, the object types stored by ``DataStore`` (read back
       by ``DataSource``) and the main parameters class
   * - Plugin roots
     - ``register_plugin(root, ...)``
     - logger names: an object of a class defined in a registered top-level
       package logs as ``<root>.<ClassName>`` (e.g. ``specula.DM``). Also the
       GPU memory accounting, which follows objects of these packages
   * - Calibration subdirectories
     - ``register_plugin(calib_subdirs=...)``
     - ``CalibManager`` (e.g. ``IFunc`` objects are stored in ``ifunc/``),
       see :doc:`calibration_manager`
   * - Environment variable aliases
     - ``register_plugin(env_aliases=...)``
     - ``iteras.registry.getenv()``: for example ``SPECULA_DISABLE_GPU`` is an
       alias of ``ITERAS_DISABLE_GPU``, and ``SPECULA_PORT`` of ``ITERAS_PORT``
   * - Application name
     - ``set_application_name()``
     - messages, the prompt of ``TerminalInput`` and the header of the trace
       files (``SPECULA`` when SPECULA is imported, ``iteras`` otherwise)

How a plugin is registered
--------------------------

A plugin registers itself when its top-level package is imported, calling
``iteras.registry.register_plugin()``. This is what SPECULA does in
``specula/__init__.py``:

.. code-block:: python

    from iteras import registry

    registry.register_plugin(
        'specula',
        packages=['specula.processing_objects',
                  'specula.data_objects',
                  'specula.display'],
        calib_subdirs=CALIB_SUBDIRS,
        env_aliases={'ITERAS_DISABLE_GPU': 'SPECULA_DISABLE_GPU',
                     'ITERAS_PORT': 'SPECULA_PORT'},
    )
    registry.set_application_name('SPECULA')

Any program that imports the plugin, or any of its modules, therefore has it
registered before a parameter file is read. This is the case of the ``specula``
command, and of any script that calls ``specula.init()``.

A plugin can also declare an ``iteras.plugins`` entry point in its package
metadata. The ``iteras`` command imports all the plugins declared in this way
before building the simulation, so that it can run their parameter files
without importing them explicitly. SPECULA declares it in ``pyproject.toml``:

.. code-block:: toml

    [project.entry-points."iteras.plugins"]
    specula = "specula"

Entry points are part of the installed package metadata: after adding one,
the package must be installed again (``pip install -e .`` for a development
installation).

How classes are found
---------------------

When ``Simul`` builds a simulation, ``import_class()`` converts each class name
to a module name (``FooBar`` to ``foo_bar``) and tries to import that module
from the following packages, in order. The first module found must define the
class.

1. the iteras packages: ``iteras.processing_objects``, ``iteras.data_objects``,
   ``iteras.display``;
2. the packages of the registered plugins, in registration order;
3. the packages listed in the ``add_modules`` parameter of the main parameters.

The main parameters are the section whose class is ``SimParams`` (defined by
iteras), or a subclass of it, like SPECULA's ``SimulParams``. ``Simul`` finds it
by resolving the class of each section, looking also in the ``add_modules``
packages listed in any section, so that the main parameters class itself can
come from one of them.

The add_modules parameter
-------------------------

``add_modules`` is a parameter of ``SimParams`` (and therefore of
``SimulParams``) listing packages to search, after the registered ones, when
building that simulation:

.. code-block:: yaml

    main:
      class:        'SimulParams'
      root_dir:     './calib'
      add_modules:  ['my_lab_objects']

``add_modules`` and plugin registration can be used together, and differ in
scope:

- ``add_modules`` only extends the class search of the simulation that lists
  it. It does not register anything: the classes found in this way log as
  ``iteras.<ClassName>``, and they cannot add calibration subdirectories or
  environment variable aliases. It suits objects that are local to a project,
  or experimental.
- Registration is permanent for the process and applies to every simulation:
  it suits a package installed and used across many simulations, and it is the
  only way to add calibration subdirectories, environment variable aliases and
  a logger root.

Listing a registered package in ``add_modules`` too is harmless.

Writing a plugin
----------------

As an example, SpecuLab (``speculab``) provides processing objects that drive
laboratory hardware, and loads them with ``add_modules: ['speculab']``. This
keeps working unchanged. Since its classes use SPECULA data objects (``Pixels``,
``ElectricField``, ``IFunc``), it would be a plugin of SPECULA rather than of
iteras, and would need two additions to become one.

In ``speculab/__init__.py``:

.. code-block:: python

    import specula                  # SPECULA is registered first
    from iteras import registry

    registry.register_plugin('speculab', packages=['speculab'])

In ``setup.py`` (or ``pyproject.toml``):

.. code-block:: python

    entry_points={'iteras.plugins': ['speculab = speculab']},

After this, ``add_modules: ['speculab']`` is no longer needed in programs that
import ``speculab`` or with the ``iteras`` command, and the SpecuLab objects log
as ``speculab.<ClassName>``.

For a new plugin, the steps are:

1. create a package, with the classes in modules named after them (one class
   per module, ``FooBar`` in ``foo_bar.py``), possibly grouped in subpackages
   like ``processing_objects`` and ``data_objects``;
2. derive the classes from the iteras base classes, or from the classes of the
   plugin they extend;
3. call ``iteras.registry.register_plugin()`` in the top-level ``__init__.py``,
   listing the packages that contain the classes;
4. optionally, declare an ``iteras.plugins`` entry point;
5. optionally, define a subclass of ``SimParams`` to add domain-specific main
   parameters.

Compatibility with older SPECULA code
-------------------------------------

The framework modules moved from SPECULA to iteras are still available from
their old paths: for example ``specula.base_processing_obj`` is the same module
object as ``iteras.base_processing_obj``, so its classes are the same objects
and ``isinstance()`` checks, ``unittest.mock.patch()`` targets and pickled
objects referring to the old paths keep working. The values set by
``specula.init()`` (``cp``, ``xp``, ``global_precision``, ...) are read from
iteras when accessed through ``specula``. Plugins written for SPECULA can keep
importing from ``specula`` and do not need any change.

Planned changes
---------------

The following changes are planned before iteras 1.0, and will be described
here when implemented:

- an error when a class name matches different classes in more than one
  package, instead of silently using the first one, and qualified class names
  in the parameter files (e.g. ``class: specula.IirFilter``) to choose one
  explicitly;
- automatic loading of the installed plugins (entry points) when a class is not
  found, so that they are found by any program, not only by the ``iteras``
  command;
- classes found through ``add_modules`` logging under the application name
  (e.g. ``specula.<ClassName>``) instead of ``iteras.<ClassName>``;
- removal of the global application name, which depends on which plugin is
  imported last.
