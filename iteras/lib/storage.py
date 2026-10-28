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


# ---------------------------------------------------------------------------
# Data objects (save() / restore(), calibration files)
#
# Data objects write FITS files with their own layout (header entries and one
# or more image HDUs). The same content can be stored in npz or HDF5 files:
# each HDU is stored with its data array and its header cards, so that the
# FITS file can be rebuilt exactly and read by the restore() method of the
# class (see BaseDataObj).
# ---------------------------------------------------------------------------

OBJECT_FORMATS = ('fits', 'npz', 'hdf5')

_OBJECT_EXTENSIONS = {'.fits': 'fits', '.fit': 'fits', '.fts': 'fits',
                      '.npz': 'npz', '.h5': 'hdf5', '.hdf5': 'hdf5'}

# FITS structure keywords, recomputed by astropy from the data
_STRUCTURE_KEYWORDS = {'SIMPLE', 'BITPIX', 'EXTEND', 'XTENSION', 'PCOUNT', 'GCOUNT',
                       'BZERO', 'BSCALE'}


def object_format(path):
    '''Format of a data object file, from its extension, or None if the extension is not known'''
    return _OBJECT_EXTENSIONS.get(os.path.splitext(os.fspath(path))[1].lower())


def default_object_format():
    '''
    Format of the data object files written without an explicit extension: the
    default format (see default_format()), or 'fits' if that is not available
    for data objects (pickle).
    '''
    data_format = default_format()
    return data_format if data_format in OBJECT_FORMATS else 'fits'


def object_filename(path):
    '''*path*, with the extension of the default object format if it has no known extension'''
    path = os.fspath(path)
    if object_format(path) is not None:
        return path
    return path + EXTENSIONS[default_object_format()]


def find_object_file(path):
    '''
    Existing file of a data object, given its path without extension: the
    file with the extension of the default object format, or of the other
    formats, or None if none exists.
    '''
    path = os.fspath(path)
    first = default_object_format()
    for data_format in [first] + [f for f in OBJECT_FORMATS if f != first]:
        candidate = path + EXTENSIONS[data_format]
        if os.path.exists(candidate):
            return candidate
    return None


def _header_cards(header):
    '''Header cards as a JSON-serializable list of [keyword, value, comment]'''
    cards = []
    for card in header.cards:
        keyword = card.keyword
        if keyword in _STRUCTURE_KEYWORDS or keyword.startswith('NAXIS'):
            continue
        value = card.value
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            value = str(value)
        cards.append([keyword, value, card.comment])
    return cards


def _native(data):
    '''Array in native byte order (FITS data is big-endian)'''
    data = np.asarray(data)
    return data.astype(data.dtype.newbyteorder('='), copy=False)


def write_hdulist(path, hdul, data_format):
    '''Store the HDUs of a FITS HDUList (data and header cards) in an npz or HDF5 file'''
    hdus = [(None if hdu.data is None else _native(hdu.data), json.dumps(_header_cards(hdu.header)))
            for hdu in hdul]
    if data_format == 'npz':
        arrays = {'n_hdus': np.array(len(hdus))}
        for i, (data, cards) in enumerate(hdus):
            arrays[f'hdu{i}_header'] = np.array(cards)
            if data is not None:
                arrays[f'hdu{i}_data'] = data
        with open(path, 'wb') as f:
            np.savez(f, **arrays)
    elif data_format == 'hdf5':
        h5py = _h5py()
        with h5py.File(path, 'w') as f:
            f.attrs['n_hdus'] = len(hdus)
            for i, (data, cards) in enumerate(hdus):
                group = f.create_group(f'hdu{i}')
                group.attrs['header'] = cards
                if data is not None:
                    group.create_dataset('data', data=data)
    else:
        raise ValueError(f'Unsupported format {data_format!r} for HDUs: must be npz or hdf5')


def read_hdulist(path, data_format):
    '''Rebuild the FITS HDUList stored by write_hdulist()'''
    from astropy.io import fits
    hdus = []
    if data_format == 'npz':
        with np.load(path, allow_pickle=False) as f:
            for i in range(int(f['n_hdus'])):
                data = f[f'hdu{i}_data'] if f'hdu{i}_data' in f.files else None
                hdus.append((data, str(f[f'hdu{i}_header'])))
    elif data_format == 'hdf5':
        h5py = _h5py()
        with h5py.File(path, 'r') as f:
            for i in range(int(f.attrs['n_hdus'])):
                group = f[f'hdu{i}']
                data = group['data'][()] if 'data' in group else None
                cards = group.attrs['header']
                hdus.append((data, cards.decode() if isinstance(cards, bytes) else str(cards)))
    else:
        raise ValueError(f'Unsupported format {data_format!r} for HDUs: must be npz or hdf5')

    hdul = fits.HDUList()
    for i, (data, cards) in enumerate(hdus):
        header = fits.Header([fits.Card(*card) for card in json.loads(cards)])
        hdu_class = fits.PrimaryHDU if i == 0 else fits.ImageHDU
        hdul.append(hdu_class(data=data, header=header))
    return hdul


def fits_to_object_file(fits_path, path, data_format):
    '''Convert a FITS file written by a data object to an npz or HDF5 file'''
    from astropy.io import fits
    with fits.open(fits_path, memmap=False) as hdul:
        write_hdulist(path, hdul, data_format)


def object_file_to_fits(path, data_format, fits_path):
    '''Convert an npz or HDF5 file written by fits_to_object_file() back to FITS'''
    read_hdulist(path, data_format).writeto(fits_path)
