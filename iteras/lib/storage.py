'''
File formats used by DataStore to save the time series of its inputs, and by
DataSource to read them back.

Each stored input is a file named after the input, containing:

- the data, an array whose first axis is time;
- the times of the samples, in the integer units of the simulation clock;
- a header describing the data object, whose OBJ_TYPE entry is the name of
  its class (see BaseDataObj.get_fits_header() and from_header()).

Supported formats, with their file extensions:

- ``fits`` (``.fits``): data in the primary HDU, times in the second HDU,
  header in both;
- ``pickle`` (``.pickle``): a dictionary with "data", "times" and "hdr";
- ``npz`` (``.npz``): numpy arrays "data" and "times", and the header as a
  JSON string in "header" (read without pickle);
- ``hdf5`` (``.h5``): datasets "data" and "times", header entries as
  attributes of the file. Needs the optional h5py package.

The default format is set by the ITERAS_DATA_FORMAT environment variable
(or one of its aliases, see iteras.registry), and is ``fits`` if it is not set.
'''
import json
import os

import numpy as np

from iteras import registry

FORMATS = ('fits', 'pickle', 'npz', 'hdf5')

EXTENSIONS = {'fits': '.fits', 'pickle': '.pickle', 'npz': '.npz', 'hdf5': '.h5'}


def check_format(data_format, source='data_format'):
    '''Raise ValueError if *data_format* is not supported'''
    if data_format not in FORMATS:
        raise ValueError(f'Unsupported {source} {data_format!r}: must be one of {", ".join(FORMATS)}')


def default_format():
    '''Format used when none is given: ITERAS_DATA_FORMAT, or "fits" if it is not set'''
    data_format = registry.getenv('ITERAS_DATA_FORMAT', 'fits')
    check_format(data_format, source='ITERAS_DATA_FORMAT environment variable value')
    return data_format


def filename(directory, name, data_format):
    '''Path of the file of input *name* in *directory*'''
    return os.path.join(directory, name + EXTENSIONS[data_format])


def find_format(directory, name):
    '''
    Format of the file of input *name* in *directory*, looking for the
    extension of each format, or None if no file is found
    '''
    for data_format in FORMATS:
        if os.path.exists(filename(directory, name, data_format)):
            return data_format
    return None


def header_to_dict(header):
    '''
    Header entries as a dictionary of plain values (str, int, float, bool),
    from a FITS header or a dictionary. Comments and history are left out.
    '''
    result = {}
    for key in header.keys():
        if key in ('COMMENT', 'HISTORY', ''):
            continue
        value = header[key]
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, (str, int, float, bool)):
            value = str(value)
        result[key] = value
    return result


def write_npz(path, data, times, header):
    '''Write an npz file with the data, the times and the header (as JSON)'''
    with open(path, 'wb') as f:
        np.savez(f, data=data, times=times, header=np.array(json.dumps(header_to_dict(header))))


def read_npz(path):
    '''Read an npz file written by write_npz(): returns (data, times, header dict)'''
    with np.load(path, allow_pickle=False) as f:
        return f['data'], f['times'], json.loads(str(f['header']))


def _h5py():
    try:
        import h5py
    except ImportError:
        raise ImportError('The hdf5 data format needs the h5py package (pip install h5py)') from None
    return h5py


def write_hdf5(path, data, times, header):
    '''Write an HDF5 file with the data, the times and the header (as file attributes)'''
    h5py = _h5py()
    with h5py.File(path, 'w') as f:
        f.create_dataset('data', data=data)
        f.create_dataset('times', data=times)
        for key, value in header_to_dict(header).items():
            f.attrs[key] = value


def read_hdf5(path):
    '''Read an HDF5 file written by write_hdf5(): returns (data, times, header dict)'''
    h5py = _h5py()
    with h5py.File(path, 'r') as f:
        header = header_to_dict(dict(f.attrs))
        return f['data'][()], f['times'][()], header


def read_header(path, data_format):
    '''Header of a stored file, as a dictionary'''
    if data_format == 'fits':
        from astropy.io import fits
        return header_to_dict(fits.getheader(path))
    if data_format == 'pickle':
        import pickle
        with open(path, 'rb') as handle:
            return header_to_dict(pickle.load(handle).get('hdr', {}))
    if data_format == 'npz':
        return read_npz(path)[2]
    if data_format == 'hdf5':
        return read_hdf5(path)[2]
    check_format(data_format)
