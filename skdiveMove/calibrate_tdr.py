"""Convenience function to perform all major TDR calibration operations

"""
import logging
from skdiveMove.tdr import TDR
import skdiveMove.calibconfig as skconfig


def calibrate(tdr_file, config_file=None):
    """Perform all major TDR calibration operations

    Parameters
    ----------
    tdr_file : str, Path or xarray.backends.*DataStore
        As first argument for :func:`xarray.load_dataset`.
    config_file : str
        A valid string path for TDR calibration configuration file.

    Returns
    -------
    out : TDR

    """
    if config_file is None:
        config = skconfig._DEFAULT_CONFIG
    else:
        config = skconfig.read_config(config_file)

    logging.basicConfig(level=config["log_level"])
    logger = logging.getLogger(__name__)

    load_dataset_kwargs = config["read"].pop("load_dataset_kwargs")
    logger.info("Reading config: {}, {}"
                .format(config["read"], load_dataset_kwargs))
    tdr = TDR(tdr_file, **config["read"], **load_dataset_kwargs)

    do_zoc = config["zoc"].pop("required")
    if do_zoc:
        logger.info("ZOC config: {}".format(config["zoc"]))
        tdr.zoc(config["zoc"]["method"], **config["zoc"]["parameters"])

    logger.info("Wet/Dry config: {}".format(config["wet_dry"]))
    tdr.detect_wet(**config["wet_dry"])

    logger.info("Dives config: {}".format(config["dives"]))
    tdr.detect_dives(config["dives"].pop("dive_thr"))
    tdr.detect_dive_phases(**config["dives"])

    do_speed_calib = bool(config["speed_calib"].pop("required"))
    if do_speed_calib:
        logger.info("Speed calibration config: {}"
                    .format(config["speed_calib"]))
        tdr.calibrate_speed(**config["speed_calib"], plot=False)

    return(tdr)
