"""Read and write TDR calibration configuration files

"""
import json

from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Literal


# __all__ = ["dump_config_template", "dump_config", "read_config"]

_DEFAULT_CONFIG = {
    'read': {
        'depth_name': "depth",
        'time_name': "timestamp",
        'subsample': "5s",
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
        'knot_factor': 3,
        'descent_crit_q': 0,
        'ascent_crit_q': 0
    },
    'speed_calib': {
        'required': False,
        'tau': 0.1,
        'contour_level': 0.1
    },
    'log_level': "INFO"
}

_DUMP_INDENT = 4


@dataclass
class ConfigTDRRead:
    """Configuration of variables for reading source TDR dataset.

    Attributes
    ----------
    depth_name : str
        Name of the depth variable in the dataset.
    time_name : str
        Name of the time variables in the dataset.
    subsample : str
        Interval frequency to subsample the input dataset at, expressed as
        in pandas offset aliases (:func:`pandas.Timedelta`).
    has_speed : bool
        Whether the dataset includes a speed variable.
    load_dataset_kwargs : dict
        Dictionary with optional key-value pairs for
        :meth:`~xarray.load_dataset`.

    """
    depth_name: str = "depth"
    time_name: str = "timestamp"
    subsample: str = "5s"
    has_speed: bool = False
    load_dataset_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigTDRZOC:
    """Configuration for zero offset correction procedure.

    Attributes
    ----------
    required : bool
        Whether ZOC is required for the input data.
    method : {"offset", "filter"}
        The ZOC method to use.
    parameters : dict
        A dictionary with parameters for the selected method.

    """
    required: bool = True
    method: Literal["offset", "filter"] = "offset"
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate method."""
        if self.method not in ["offset", "filter"]:
            raise ValueError("`method` must be one of 'offset' or 'filter'")


@dataclass
class ConfigTDRWetDry:
    """Configuration of variables for wet/dry phase identification.

    Attributes
    ----------
    dry_thr : float
        The maximum dry time [s] allowed between wet phases. Dry phases
        longer than this threshold generate a dry phase, otherwise they are
        considered part of the encompassing wet phase.
    wet_thr : float
        The minimum wet time [s] allowed between dry phases without
        interrupting them. Wet phases longer than this threshold generate a
        wet phase, otherwise they are considered trivial wet phases.
    interp_wet : bool
        Whether to interpolate missing depth records in wet periods (after
        ZOC).

    """
    dry_thr: float = 70.0
    wet_thr: float = 3610.0
    interp_wet: bool = False

    def __post_init__(self):
        """Validate thresholds."""
        if self.dry_thr <= 0:
            raise ValueError("`dry_thr` must be greater than zero.")
        if self.wet_thr <= 0:
            raise ValueError("`wet_thr` must be greater than zero.")


@dataclass
class ConfigTDRDives:
    """Configuration of variables for dive identification.

    Attributes
    ----------
    dive_thr : float
        Threshold depth below which an underwater phase defines a dive.
    dive_model : {"unimodal", "smooth.spline"}
        Model to use for dive phase identification.
    knot_factor : int
        Multiplier to generate the knots in the dive spline.
    smooth_par : float
        Amount of smoothing (when ``dive_model="smooth.spline"``.
    descent_crit_q, ascent_crit_q : float
        Critical quantiles for rate of descent or ascent defining the end
        of descent, or the beginning of ascent phases.

    """
    smooth_par: float = 0.1
    dive_thr: float = 4.0
    dive_model: Literal["unimodal", "smooth.spline"] = "unimodal"
    knot_factor: int = 3
    descent_crit_q: float = 0
    ascent_crit_q: float = 0

    def __post_init__(self):
        # Validate dive threshold
        if self.dive_thr <= 0:
            raise ValueError("`dive_thr` must be greater than zero.")
        # Validate dive model
        if self.dive_model not in ["unimodal", "smooth.spline"]:
            raise ValueError(
                "`dive_model` must be one of 'unimodal' or 'smooth.spline'")
        # Validate knot factor
        if self.knot_factor <= 0:
            raise ValueError("`knot_factor` must be greater than zero.")
        # Validate smooth paramater
        if self.smooth_par is not None and not (0 < self.smooth_par <= 1):
            raise ValueError(
                "`smooth_par` must lie in the (0, 1] interval.")
        # Validate critical quantiles
        if not (0 <= self.descent_crit_q <= 1):
            raise ValueError(
                "`descent_crit_q` must lie in the [0, 1] interval.")
        if not (0 <= self.ascent_crit_q <= 1):
            raise ValueError(
                "`ascent_crit_q` must lie in the [0, 1] interval.")


@dataclass
class ConfigTDRSpeed:
    """Configuration of variables for speed calibration.

    Attributes
    ----------
    required : bool
        Whether speed calibration is required for the input data.
    tau : float
        Quantile on which to regress speed on rate of depth change.
    contour_level : float
        Contour level to use for quantile regression.

    """
    required: bool = False
    tau: float = 0.1
    contour_level: float = 0.1

    def __post_init__(self):
        """Validate tau and contour level."""
        if not (0 <= self.tau <= 1):
            raise ValueError("`tau` must lie in the [0, 1] interval.")
        if not (0 <= self.contour_level <= 1):
            raise ValueError(
                "`contour_level` must lie in the [0, 1] interval.")


@dataclass
class ConfigTDR:
    """Configuration for TDR processing

    Encapsulates variables required by methods and functions for processing
    TDR records.

    Attributes
    ----------
    log_level :  str
        Logging level.
    read : ConfigTDRRead
    zoc : ConfigTDRZOC
    wet_dry : ConfigTDRWetDry
    dives : ConfigTDRDives
    speed_calib : ConfigTDRSpeed

    """
    read: ConfigTDRRead = field(
        default_factory=lambda: ConfigTDRRead(**_DEFAULT_CONFIG["read"]))
    zoc: ConfigTDRZOC = field(
        default_factory=lambda: ConfigTDRZOC(**_DEFAULT_CONFIG["zoc"]))
    wet_dry: ConfigTDRWetDry = field(
        default_factory=lambda: ConfigTDRWetDry(**_DEFAULT_CONFIG["wet_dry"]))
    dives: ConfigTDRDives = field(
        default_factory=lambda: ConfigTDRDives(**_DEFAULT_CONFIG["dives"]))
    speed_calib: ConfigTDRSpeed = field(
        default_factory=lambda: ConfigTDRSpeed(
            **_DEFAULT_CONFIG["speed_calib"]))
    log_level: str = _DEFAULT_CONFIG["log_level"]

    @classmethod
    def from_dict(
            cls: "ConfigTDR",
            obj: dict) -> "ConfigTDR":
        read_d = obj.get("read", _DEFAULT_CONFIG["read"])
        zoc_d = obj.get("zoc", _DEFAULT_CONFIG["zoc"])
        wet_dry_d = obj.get("wet_dry", _DEFAULT_CONFIG["wet_dry"])
        dives_d = obj.get("dives", _DEFAULT_CONFIG["dives"])
        speed_calib_d = obj.get(
            "speed_calib", _DEFAULT_CONFIG["speed_calib"])
        log_level = obj.get(
            "log_level", _DEFAULT_CONFIG["log_level"])

        return cls(
            read=ConfigTDRRead(**read_d),
            zoc=ConfigTDRZOC(**zoc_d),
            wet_dry=ConfigTDRWetDry(**wet_dry_d),
            dives=ConfigTDRDives(**dives_d),
            speed_calib=ConfigTDRSpeed(**speed_calib_d),
            log_level=log_level)

    @classmethod
    def from_config_file(
            cls: "ConfigTDR",
            config_json: Path) -> "ConfigTDR":
        """Instantiate from json configuration file.

        Parameters
        ----------
        config_json : PosixPath
            Path to json configuration file.

        Returns
        -------
        ConfigTDR

        """

        try:
            with open(config_json, "r") as cfg:
                config = json.load(cfg)
        except json.JSONDecodeError as err:
            raise ValueError(
                f"Invalid JSON syntax in {config_json}: {err}") from err

        return cls.from_dict(config)

    def dump_config(self, fname):
        """Dump configuration dictionary to file

        Dump instance dictionary onto a JSON configuration file to set up
        TDR calibration.

        Parameters
        ----------
        fname : str or file-like
            A valid string path, or `file-like` object, for output file.

        """
        with open(fname, "w") as ofile:
            json.dump(asdict(self), ofile, indent=_DUMP_INDENT)


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


if __name__ == '__main__':
    dump_config_template("tdr_config.json")
