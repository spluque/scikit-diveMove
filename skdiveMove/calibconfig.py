"""Read and write TDR calibration configuration files

"""
import json

__all__ = ["dump_config_template", "dump_config", "read_config"]

_DEFAULT_CONFIG = {
    'log_level': "INFO",
    'read': {
        'depth_name': "depth",
        'has_speed': False,
        'load_dataset_kwargs': {}
    },
    'zoc': {
        'required': True,
        'method': "offset",
        'parameters': {'offset': 0}
    },
    'wet_dry': {
        'dry_thr': 70,
        'wet_thr': 3610,
        'interp_wet': False
    },
    'dives': {
        'dive_thr': 4,
        'dive_model': "unimodal",
        'smooth_par': 0.1,
        'knot_factor': 3,
        'descent_crit_q': 0,
        'ascent_crit_q': 0
    },
    'speed_calib': {
        'required': False,
        'tau': 0.1,
        'contour_level': 0.1,
        'z': 0,
        'bad': [0, 0]
    }
}

_DUMP_INDENT = 4


def dump_config_template(fname):
    """Dump configuration template file

    Dump a json configuration template file to set up TDR calibration.

    Parameters
    ----------
    fname : str or file-like
        A valid string path, or `file-like` object, for output file.

    Examples
    --------
    >>> dump_config_template("tdr_config.json")  # doctest: +SKIP

    Edit the file to your specifications.

    """
    with open(fname, "w") as ofile:
        json.dump(_DEFAULT_CONFIG, ofile, indent=_DUMP_INDENT)


def read_config(config_file):
    """Read configuration file into dictionary

    Parameters
    ----------
    config_file : str or file-like
        A valid string path, or `file-like` object, for input file.

    Returns
    -------
    out : dict

    """
    with open(config_file, "r") as ifile:
        config = json.load(ifile)

    return(config)


def dump_config(fname, config_dict):
    """Dump configuration dictionary to file

    Dump a dictionary onto a JSON configuration file to set up TDR
    calibration.

    Parameters
    ----------
    fname : str or file-like
        A valid string path, or `file-like` object, for output file.
    config_dict : dict
        Dictionary to dump.

    """
    with open(fname, "w") as ofile:
        json.dump(config_dict, ofile, indent=_DUMP_INDENT)


if __name__ == '__main__':
    dump_config_template("tdr_config.json")
    config = read_config("tdr_config.json")
