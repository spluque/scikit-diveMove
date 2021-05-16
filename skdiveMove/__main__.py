import sys
import argparse
from skdiveMove import calibrate


def main():
    _DESCRIPTION = "Perform TDR calibration, given a configuration file"
    _FORMATERCLASS = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=_DESCRIPTION,
                                     formatter_class=_FORMATERCLASS)
    parser.add_argument("--config-file",
                        help="Path to JSON configuration file.")
    parser.add_argument("tdr_file",
                        help="Path to NetCDF TDR data file.")
    args = parser.parse_args()
    tdr = calibrate(args.tdr_file, args.config_file)
    return(tdr)


sys.exit(main())
