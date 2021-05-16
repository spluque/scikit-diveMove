"""Tools to read and write DataArray and Dataset attributes

Easily dump a template to set up metadata for a given Dataset or DataArray
that can subsequenty be read-in and append attributes to the objects.


Functions
---------

.. autosummary::

   dump_config_template
   assign_xr_attrs


API
---

"""

import json

__all__ = ["dump_config_template", "assign_xr_attrs"]

_SENSOR_DATA_CONFIG = {
    'sampling': "regular",
    'sampling_rate': "1",
    'sampling_rate_unit': "Hz",
    'history': "",
    'name': "",
    'full_name': "",
    'description': "",
    'units': "",
    'units_name': "",
    'units_label': "",
    'column_name': "",
    'frame': "",
    'axes': "",
    'files': ""
}

_DATASET_CONFIG = {
    'dep_id': "",
    'dep_device_tzone': "",
    'dep_device_regional_settings': "YYYY-mm-dd HH:MM:SS",
    'dep_device_time_beg': "",
    'deploy': {
        'locality': "",
        'lon': "",
        'lat': "",
        'device_time_on': "",
        'method': ""
    },
    'project': {
        'name': "",
        'date_beg': "",
        'date_end': ""
    },
    'provider': {
        'name': "",
        'affiliation': "",
        'email': "",
        'license': "",
        'cite': "",
        'doi': ""
    },
    'data': {
        'source': "",
        'format': "",
        'creation_date': "",
        'nfiles': ""
    },
    'device': {
        'serial': "",
        'make': "",
        'type': "",
        'model': "",
        'url': ""
    },
    'sensors': {
        'firmware': "",
        'software': "",
        'list': ""
    },
    'animal': {
        'id': "",
        'species_common': "",
        'species_science': "",
        'dbase_url': ""
    }
}


def dump_config_template(fname, config_type):
    """Dump configuration file

    Dump a json configuration template file to build metadata for a Dataset
    or DataArray.

    Parameters
    ----------
    fname : str
        A valid string path for output file.
    config_type : {"dataset", "sensor"}
        The type of config to dump.

    Examples
    --------
    >>> import skdiveMove.metadata as metadata
    >>> metadata.dump_config_template("mydataset.json",
    ...                               "dataset")  # doctest: +SKIP
    >>> metadata.dump_config_template("mysensor.json",
    ...                               "sensor")  # doctest: +SKIP

    edit the files to your specifications.

    """
    with open(fname, "w") as ofile:

        if config_type == "dataset":
            json.dump(_DATASET_CONFIG, ofile, indent=2)
        elif config_type == "sensor":
            json.dump(_SENSOR_DATA_CONFIG, ofile, indent=2)


def assign_xr_attrs(obj, config_file):
    """Assign attributes to xarray.Dataset or xarray.DataArray

    The `config_file` should have only one-level of nesting.

    Parameters
    ----------
    obj : {xarray.Dataset, xarray.DataArray}
        Object to assign attributes to.
    config_file : str
        A valid string path for input json file with metadata attributes.

    Returns
    -------
    out : {xarray.Dataset, xarray.DataArray}

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> import xarray as xr
    >>> import skdiveMove.metadata as metadata

    Synthetic dataset with depth and speed

    >>> nsamples = 60 * 60 * 24
    >>> times = pd.date_range("2000-01-01", freq="1s", periods=nsamples,
    ...                       name="time")
    >>> cycles = np.sin(2 * np.pi * np.arange(nsamples) / (60 * 20))
    >>> ds = xr.Dataset({"depth": (("time"), 1 + cycles),
    ...                  "speed": (("time"), 3 + cycles)},
    ...                 {"time": times})

    Dump dataset and sensor templates

    >>> metadata.dump_config_template("mydataset.json",
    ...                               "dataset")  # doctest: +SKIP
    >>> metadata.dump_config_template("P_sensor.json",
    ...                               "sensor")  # doctest: +SKIP
    >>> metadata.dump_config_template("S_sensor.json",
    ...                               "sensor")  # doctest: +SKIP

    Edit the templates as appropriate, load and assign to objects

    >>> assign_xr_attrs(ds, "mydataset.json")       # doctest: +SKIP
    >>> assign_xr_attrs(ds.depth, "P_sensor.json")  # doctest: +SKIP
    >>> assign_xr_attrs(ds.speed, "S_sensor.json")  # doctest: +SKIP

    """
    with open(config_file) as ifile:
        config = json.load(ifile)

    # Parse the dict
    for key, val in config.items():
        top_kname = "{}".format(key)
        if not val:
            continue

        if type(val) is dict:
            for key_n, val_n in val.items():
                if not val_n:
                    continue
                lower_kname = "{0}_{1}".format(top_kname, key_n)
                obj.attrs[lower_kname] = val_n
        else:
            obj.attrs[top_kname] = val
