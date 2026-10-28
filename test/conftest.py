import os

# Limit BLAS/OpenMP threads before numpy is imported by any test module.
# Test matrices are small, and with the default (one thread per core) the
# LAPACK calls are dominated by threading overhead, especially on a loaded machine.
for _var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '4')


import pytest

# Values set by init() (iteras.init(), also available as specula.init()).
# The framework reads them when objects are created, so each test must see
# the values set by its own file (see _restore_init_state())
_INIT_STATE = ['xp', 'cp', 'gpuEnabled', 'global_precision', 'float_dtype_list',
               'complex_dtype_list', 'gpu_float_dtype_list', 'gpu_complex_dtype_list',
               'array_types', 'float_dtype', 'complex_dtype', 'default_target_device_idx',
               'default_target_device', 'process_comm', 'process_rank']


# Values set by init() right after each test file was imported, by module node id
_module_init_state = {}


def _iteras_state():
    import iteras
    return {name: getattr(iteras, name) for name in _INIT_STATE}


def pytest_collectreport(report):
    '''
    Called after each test file is collected, that is, right after it has been
    imported: record the values set by the init() call at its top, if any.
    '''
    if report.nodeid.endswith('.py') and report.passed:
        _module_init_state[report.nodeid] = _iteras_state()


@pytest.fixture(autouse=True)
def _restore_init_state(request):
    '''
    Run each test with the values set by the init() call at the top of its
    file, and restore the previous values after the test.

    All the test files are imported before running any test, so without this
    the tests would see the values set by the last file imported (for example
    single precision instead of double), and a test calling init(), directly
    or through main_simul() (which uses single precision by default), would
    change the device and precision of the objects created by the following tests.
    '''
    import iteras
    saved = _iteras_state()
    module_state = _module_init_state.get(request.node.nodeid.split('::')[0])
    if module_state is not None:
        for name, value in module_state.items():
            setattr(iteras, name, value)
    yield
    for name, value in saved.items():
        setattr(iteras, name, value)
