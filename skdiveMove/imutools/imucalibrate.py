"""Calibrate IMU measurements for temperature effects and errors

"""

import logging
import re
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.signal as signal
import xarray as xr
from skdiveMove.tdrsource import _load_dataset
from .imu import (IMUBase,
                  _ACCEL_NAME, _OMEGA_NAME, _MAGNT_NAME, _DEPTH_NAME)

_TRIAXIAL_VARS = [_ACCEL_NAME, _OMEGA_NAME, _MAGNT_NAME]
_MONOAXIAL_VARS = [_DEPTH_NAME, "light_levels"]
_AXIS_NAMES = list("xyz")

logger = logging.getLogger(__name__)
# Add the null handler if importing as library; whatever using this library
# should set up logging.basicConfig() as needed
logger.addHandler(logging.NullHandler())


class IMUcalibrate(IMUBase):
    r"""Calibration framework for IMU measurements

    Measurements from most IMU sensors are influenced by temperature, among
    other artifacts.  The IMUcalibrate class implements the following
    procedure to remove the effects of temperature from IMU signals:

    - For each axis, fit a piecewise or simple linear regression of
      measured (lowpass-filtered) data against temperature.
    - Compute predicted signal from the model.
    - Select a reference temperature :math:`T_{\alpha}` to standardize all
      measurements at.
    - The standardized measurement (:math:`x_\sigma`) at :math:`T_{\alpha}`
      is calculated as:

      .. math::
         :label: 5

         x_\sigma = x - (\hat{x} - \hat{x}_{\alpha})

      where :math:`\hat{x}` is the value predicted from the model at the
      measured temperature, and :math:`\hat{x}_{\alpha}` is the predicted
      value at :math:`T_{\alpha}`.

    The models fit to signals from a *motionless* (i.e. experimental) IMU
    device in the first step can subsequently be used to remove or minimize
    temperature effects from an IMU device measuring motions of interest,
    provided the temperature is within the range observed in experiments.

    In addition to attributes in :class:`IMUBase`, ``IMUcalibrate`` adds
    the attributes listed below.

    Attributes
    ----------
    periods : list
        List of slices with the beginning and ending timestamps defining
        periods in ``x_calib`` where valid calibration data are available.
        Periods are assumed to be ordered chronologically.
    models_l : list
        List of dictionaries as long as there are periods, with each
        element corresponding to a sensor, in turn containing another
        dictionary with each element corresponding to each sensor axis.
    axis_order : list
        List of characters specifying which axis ``x``, ``y``, or ``z`` was
        pointing in the same direction as gravity in each period in
        ``periods``.

    Examples
    --------
    Construct IMUcalibrate from NetCDF file with samples of IMU signals and
    a list with begining and ending timestamps for experimental periods:

    >>> import importlib.resources as rsrc
    >>> import skdiveMove.imutools as imutools
    >>> icdf = (rsrc.files("skdiveMove") / "tests" / "data" /
    ...         "cats_temperature_calib.nc")
    >>> pers = [slice("2021-09-20T09:00:00", "2021-09-21T10:33:00"),
    ...         slice("2021-09-21T10:40:00", "2021-09-22T11:55:00"),
    ...         slice("2021-09-22T12:14:00", "2021-09-23T11:19:00")]
    >>> imucal = (imutools.IMUcalibrate
    ...           .read_netcdf(icdf, periods=pers,
    ...                        axis_order=list("zxy"),
    ...                        time_name="timestamp_utc"))
    >>> print(imucal)  # doctest: +ELLIPSIS
    IMU -- Class IMUcalibrate object
    Source File          None
    IMU: <xarray.Dataset>
    Dimensions:           (timestamp_utc: 268081, axis: 3)
    Coordinates:
      * axis              (axis) object 'x' 'y' 'z'
      * timestamp_utc     (timestamp_utc) datetime64[ns] ...
    Data variables:
        acceleration      (timestamp_utc, axis) float64 ...
        angular_velocity  (timestamp_utc, axis) float64 ...
        magnetic_density  (timestamp_utc, axis) float64 ...
        depth             (timestamp_utc) float64 ...
        temperature       (timestamp_utc) float64 ...
    Attributes:...
        history:  Resampled from 20 Hz to 1 Hz
    Periods:
    0:['2021-09-20T09:00:00', '2021-09-21T10:33:00']
    1:['2021-09-21T10:40:00', '2021-09-22T11:55:00']
    2:['2021-09-22T12:14:00', '2021-09-23T11:19:00']

    Plot signals from a given period:

    >>> fig, axs, axs_temp = imucal.plot_experiment(0, var="acceleration")

    Build temperature models for a given variable and chosen
    :math:`T_{\alpha}`, without low-pass filtering the input signals:

    >>> fs = 1.0
    >>> acc_cal = imucal.build_tmodels("acceleration", T_alpha=8,
    ...                                use_axis_order=True,
    ...                                 win_len=int(2 * 60 * fs) - 1)

    Plot model of IMU variable against temperature:

    >>> fig, axs = imucal.plot_var_model("acceleration",
    ...                                  use_axis_order=True)


    Notes
    -----
    This class redefines :meth:`IMUBase.read_netcdf`.

    """

    def __init__(self, x_calib, periods, axis_order=list("xyz"),
                 **kwargs):
        """Set up attributes required for calibration

        Parameters
        ----------
        x_calib : xarray.Dataset
            Dataset with temperature and tri-axial data from *motionless*
            IMU experiments.  Data are assumed to be in FLU coordinate
            frame.
        periods : list
            List of slices with the beginning and ending timestamps
            defining periods in `x_calib` where valid calibration data are
            available.  Periods are assumed to be ordered chronologically.
        axis_order : list
            List of characters specifying which axis ``x``, ``y``, or ``z``
            was pointing in the same direction as gravity in each period in
            ``periods``.
        **kwargs
            Optional keyword arguments passed to the `IMUBase.__init__` for
            instantiation.

        """
        super(IMUcalibrate, self).__init__(x_calib, **kwargs)
        self.periods = periods
        models_l = []
        for period in periods:
            models_1d = {i: dict() for i in _MONOAXIAL_VARS}
            models_2d = dict.fromkeys(_TRIAXIAL_VARS)
            for k in models_2d:
                models_2d[k] = dict.fromkeys(axis_order)
            models_l.append(dict(**models_1d, **models_2d))

        self.models_l = models_l
        self.axis_order = axis_order
        # Private attribute collecting DataArrays with standardized data
        self._stdda_l = []

    @classmethod
    def read_netcdf(cls, imu_nc, load_dataset_kwargs=dict(), **kwargs):
        """Create IMUcalibrate from NetCDF file and list of slices

        This method redefines :meth:`IMUBase.read_netcdf`.

        Parameters
        ----------
        imu_nc : str
            Path to NetCDF file.
        load_dataset_kwargs : dict, optional
            Dictionary of optional keyword arguments passed to
            :func:`xarray.load_dataset`.
        **kwargs
            Additional keyword arguments passed to
            :meth:`IMUcalibrate.__init__` method, except ``has_depth`` or
            ``imu_filename``.  The input ``Dataset`` is assumed to have a
            depth ``DataArray``.

        Returns
        -------
        out :

        """
        imu = _load_dataset(imu_nc, **load_dataset_kwargs)
        ocls = cls(imu, **kwargs)

        return ocls

    def __str__(self):
        super_str = super(IMUcalibrate, self).__str__()
        pers_ends = []
        for per in self.periods:
            pers_ends.append([per.start, per.stop])
        msg = ("\n".join("{}:{}".format(i, per)
                         for i, per in enumerate(pers_ends)))
        return super_str + "\nPeriods:\n{}".format(msg)

    def savgol_filter(self, var, period_idx, win_len, polyorder=1):
        """Apply Savitzky-Golay filter on tri-axial IMU signals

        Parameters
        ----------
        var : str
            Name of the variable in ``x`` with tri-axial signals.
        period_idx : int
            Index of period to plot (zero-based).
        win_len : int
            Window length for the low pass filter.
        polyorder : int, optional
            Polynomial order to use.

        Returns
        -------
        xarray.DataArray
            Array with filtered signals, with the same coordinates,
            dimensions, and updated attributes.

        """
        darray = self.subset_imu(period_idx)[var]
        var_df = darray.to_dataframe().unstack()
        var_sg = signal.savgol_filter(var_df, window_length=win_len,
                                      polyorder=polyorder, axis=0)
        new_history = (("{}: Savitzky-Golay filter: win_len={}, "
                        "polyorder={}\n")
                       .format(pd.to_datetime("today")
                               .strftime("%Y-%m-%d"), win_len, polyorder))
        darray_new = xr.DataArray(var_sg, coords=darray.coords,
                                  dims=darray.dims, name=darray.name,
                                  attrs=darray.attrs)
        darray_new.attrs["history"] = (darray_new.attrs["history"] +
                                       new_history)
        return darray_new

    def build_tmodels(self, var, T_alpha=None, T_brk=None,
                      use_axis_order=False, filter_sig=True, **kwargs):
        r"""Build temperature models for experimental tri-axial IMU sensor
        signals

        Perform thermal compensation on *motionless* tri-axial IMU sensor
        data.  A simple approach is used for the compensation:

          - For each axis, build a piecewise or simple linear regression of
            measured data against temperature.  If a breakpoint is known,
            as per manufacturer specifications or experimentation, use
            piecewise regression.
          - Compute predicted signal from the model.
          - Select a reference temperature :math:`T_{\alpha}` to
            standardize all measurements at.
          - The standardized measurement at :math:`T_{\alpha}` is
            calculated as :math:`x - (\hat{x} - x_{T_{\alpha}})`, where
            :math:`\hat{x}` is the value predicted from the model at the
            measured temperature, and :math:`x_{T_{\alpha}}` is the
            predicted value at :math:`T_{\alpha}`.

        Parameters
        ----------
        var : str
            Name of the variable in `x` with tri-axial data.
        T_alpha : float, optional
            Reference temperature at which all measurements will be
            adjusted to.  Defaults to the mean temperature for each period,
            rounded to the nearest integer.
        T_brk : float, optional
            Temperature change point separating data to be fit differently.
            A piecewise regression model is fit.  Default is a simple
            linear model is fit.
        use_axis_order : bool, optional
            Whether to use axis order from the instance.  If True, only one
            sensor axis per period is considered to have valid calibration
            data for the correction.  Otherwise, all three axes for each
            period are used in the correction.
        filter_sig : bool, optional
            Whether to filter in the measured signal for thermal
            correction.  Default is to apply a Savitzky-Golay filter to the
            signal for characterizing the temperature relationship, and to
            calculate the standardized signal.
        **kwargs
            Optional keyword arguments passed to `savgol_filter`
            (e.g. ``win_len`` and ``polyorder``).

        Returns
        -------
        list
            List of tuples as long as there are periods, with tuple elements:

              - Dictionary with regression model objects for each sensor
                axis.
              - DataFrame with hierarchical column index with sensor axis
                label at the first level.  The following columns are in the
                second level:

                  - temperature
                  - var_name
                  - var_name_pred
                  - var_name_temp_refC
                  - var_name_adj

        Notes
        -----
        A new DataArray with signal standardized at :math:`T_{\alpha}` is
        added to the instance Dataset.  These signals correspond to the
        lowpass-filtered form of the input used to build the models.

        See Also
        --------
        apply_model

        """
        # Iterate through periods
        per_l = []              # output list as long as periods
        for idx in range(len(self.periods)):
            per = self.subset_imu(idx)
            # Subset the requested variable, smoothing if necessary
            if filter_sig:
                per_var = self.savgol_filter(var, idx, **kwargs)
            else:
                per_var = per[var]
            per_temp = per["temperature"]
            var_df = xr.merge([per_var, per_temp]).to_dataframe()
            if T_alpha is None:
                t_alpha = np.rint(per_temp.mean().to_numpy().item())
                logger.info("Period {} T_alpha set to {:.2f}"
                            .format(idx, t_alpha))
            else:
                t_alpha = T_alpha

            odata_l = []
            models_d = self.models_l[idx]
            if use_axis_order:
                axis_names = [self.axis_order[idx]]
            elif len(per_var.dims) > 1:
                axis_names = per_var[per_var.dims[-1]].to_numpy()
            else:
                axis_names = [per_var.dims[0]]

            std_colname = "{}_std".format(var)
            pred_colname = "{}_pred".format(var)
            for i, axis in enumerate(axis_names):  # do all axes
                if isinstance(var_df.index, pd.MultiIndex):
                    data_axis = var_df.xs(axis, level="axis").copy()
                else:
                    data_axis = var_df.copy()

                if T_brk is not None:
                    temp0 = (data_axis["temperature"]
                             .where(data_axis["temperature"] < T_brk, 0))
                    data_axis.loc[:, "temp0"] = temp0
                    temp1 = (data_axis["temperature"]
                             .where(data_axis["temperature"] > T_brk, 0))
                    data_axis.loc[:, "temp1"] = temp1

                    fmla = "{} ~ temperature + temp0 + temp1".format(var)
                else:
                    fmla = "{} ~ temperature".format(var)

                model_fit = smf.ols(formula=fmla, data=data_axis).fit()
                models_d[var][axis] = model_fit
                data_axis.loc[:, pred_colname] = model_fit.fittedvalues
                # Data at reference temperature
                ref_colname = "{}_{}C".format(var, t_alpha)

                if T_brk is not None:
                    if t_alpha < T_brk:
                        pred = model_fit.predict(exog=dict(
                            temperature=t_alpha,
                            temp0=t_alpha, temp1=0)).to_numpy().item()
                        data_axis[ref_colname] = pred
                    else:
                        pred = model_fit.predict(exog=dict(
                            temperature=t_alpha,
                            temp0=0, temp1=t_alpha)).to_numpy().item()
                        data_axis[ref_colname] = pred
                    data_axis.drop(["temp0", "temp1"], axis=1, inplace=True)
                else:
                    pred = model_fit.predict(exog=dict(
                        temperature=t_alpha)).to_numpy().item()
                    data_axis.loc[:, ref_colname] = pred

                logger.info("Predicted {} ({}, rounded) at {:.2f}: {:.3f}"
                            .format(var, axis, t_alpha, pred))
                data_axis[std_colname] = (data_axis[var] -
                                          (data_axis[pred_colname] -
                                           data_axis[ref_colname]))
                odata_l.append(data_axis)
                # Update instance models_l attribute
                self.models_l[idx][var][axis] = model_fit

            if var in _MONOAXIAL_VARS:
                odata = pd.concat(odata_l)
                std_data = xr.DataArray(odata.loc[:, std_colname],
                                        name=std_colname)
            else:
                odata = pd.concat(odata_l, axis=1,
                                  keys=axis_names[:i + 1],
                                  names=["axis", "variable"])
                std_data = xr.DataArray(odata.xs(std_colname,
                                                 axis=1, level=1),
                                        name=std_colname)
            per_l.append((models_d, odata))
            std_data.attrs = per_var.attrs
            new_description = ("{} standardized at {}C"
                               .format(std_data.attrs["description"],
                                       t_alpha))
            std_data.attrs["description"] = new_description
            new_history = ("{}: temperature_model: temperature models\n"
                           .format(pd.to_datetime("today")
                                   .strftime("%Y-%m-%d")))
            std_data.attrs["history"] = (std_data.attrs["history"] +
                                         new_history)
            # Update instance _std_da_l attribute with DataArray having an
            # additional dimension for the period index
            std_data = std_data.expand_dims(period=[idx])
            self._stdda_l.append(std_data)

        return per_l

    def plot_experiment(self, period_idx, var, units_label=None, **kwargs):
        """Plot experimental IMU

        Parameters
        ----------
        period_idx : int
            Index of period to plot (zero-based).
        var : str
            Name of the variable in with tri-axial data.
        units_label : str, optional
            Label for the units of the chosen variable.  Defaults to the
            "units_label" attribute available in the DataArray.
        **kwargs
            Optional keyword arguments passed to
            :func:`~matplotlib.pyplot.subplots` (e.g. ``figsize``).

        Returns
        -------
        fig : matplotlib.Figure
        axs : array_like
            Array of :class:`~matplotlib.axes.Axes` instances in ``fig``
            with IMU signal plots.
        axs_temp : array_like
            Array of :class:`~matplotlib.axes.Axes` instances in ``fig``
            with temperature plots.

        See Also
        --------
        plot_var_model
        plot_standardized

        """
        per_da = self.subset_imu(period_idx)
        per_var = per_da[var]
        per_temp = per_da["temperature"]

        def _plot(var, temp, ax):
            """Plot variable and temperature"""
            ax_temp = ax.twinx()
            var.plot.line(ax=ax, label="measured", color="k",
                          linewidth=0.5)
            temp.plot.line(ax=ax_temp, label="temperature", color="r",
                           linewidth=0.5, alpha=0.5)
            ax.set_title("")
            ax.set_xlabel("")
            # Adjust ylim to exclude outliers
            ax.set_ylim(var.quantile(1e-5).to_numpy().item(),
                        var.quantile(1 - 1e-5).to_numpy().item())
            # Axis locators and formatters
            dlocator = mdates.AutoDateLocator(minticks=3, maxticks=7)
            dformatter = mdates.ConciseDateFormatter(dlocator)
            ax.xaxis.set_major_locator(dlocator)
            ax.xaxis.set_major_formatter(dformatter)
            ax.xaxis.set_tick_params(rotation=0)

            return ax_temp

        if units_label is None:
            units_label = per_var.attrs["units_label"]
        ylabel_pre = "{} [{}]".format(per_var.attrs["full_name"],
                                      units_label)
        temp_label = "{} [{}]".format(per_temp.attrs["full_name"],
                                      per_temp.attrs["units_label"])

        ndims = len(per_var.dims)
        axs_temp = []
        if ndims == 1:
            fig, axs = plt.subplots(**kwargs)
            ax_temp = _plot(per_var, per_temp, axs)
            axs.set_xlabel("")
            axs.set_title("")
            axs.set_ylabel(ylabel_pre)
            ax_temp.set_ylabel(temp_label)
            axs_temp.append(ax_temp)
        else:
            fig, axs = plt.subplots(3, 1, sharex=True, **kwargs)
            ax_x, ax_y, ax_z = axs
            axis_names = per_var[per_var.dims[-1]].to_numpy()
            for i, axis in enumerate(axis_names):
                ymeasured = per_var.sel(axis=axis)
                ax_temp = _plot(ymeasured, per_temp, axs[i])
                axs[i].set_title("")
                axs[i].set_xlabel("")
                axs[i].set_ylabel("{} {}".format(ylabel_pre, axis))
                if i == 1:
                    ax_temp.set_ylabel(temp_label)
                else:
                    ax_temp.set_ylabel("")
                axs_temp.append(ax_temp)
            ax_z.set_xlabel("")

        return fig, axs, axs_temp

    def plot_var_model(self, var, use_axis_order=True, units_label=None,
                       axs=None, **kwargs):
        """Plot IMU variable against temperature and fitted model

        A multi-panel plot of the selected variable against temperature
        from all periods.

        Parameters
        ----------
        var : str
            IMU variable to plot.
        use_axis_order : bool
            Whether to use axis order from the instance.  If True, only one
            sensor axis per period is considered to have valid calibration
            data for the correction.  Otherwise, all three axes for each
            period are used in the correction.  Ignored for uniaxial
            variables.
        units_label : str
            Label for the units of the chosen variable.  Defaults to the
            "units_label" attribute available in the DataArray.
        axs : array_like, optional
            Array of Axes instances to plot in.
        **kwargs
            Optional keyword arguments passed to
            :func:`~matplotlib.pyplot.subplots` (e.g. ``figsize``).

        Returns
        -------
        fig : matplotlib.Figure
        axs : array_like
            Array of :class:`~matplotlib.axes.Axes` instances in ``fig``
            with IMU signal plots.

        See Also
        --------
        plot_experiment
        plot_standardized

        """
        def _plot_signal(x, y, idx, model_fit, ax):
            ax.plot(x, y, ".", markersize=2, alpha=0.03,
                    label="Period {}".format(idx))
            # Adjust ylim to exclude outliers
            ax.set_ylim(np.quantile(y, 1e-3), np.quantile(y, 1 - 1e-3))
            # Linear model
            xpred = np.linspace(x.min(), x.max())
            ypreds = (model_fit
                      .get_prediction(exog=dict(Intercept=1,
                                                temperature=xpred))
                      .summary_frame())
            ypred_0 = ypreds["mean"]
            ypred_l = ypreds["obs_ci_lower"]
            ypred_u = ypreds["obs_ci_upper"]
            ax.plot(xpred, ypred_0, color="k", alpha=0.5)
            ax.plot(xpred, ypred_l, color="k", linestyle="dashed",
                    linewidth=1, alpha=0.5)
            ax.plot(xpred, ypred_u, color="k", linestyle="dashed",
                    linewidth=1, alpha=0.5)

        per0 = self.subset_imu(0)
        if units_label is None:
            units_label = per0[var].attrs["units_label"]
        xlabel = "{} [{}]".format(per0["temperature"].attrs["full_name"],
                                  per0["temperature"].attrs["units_label"])
        ylabel_pre = "{} [{}]".format(per0[var].attrs["full_name"],
                                      units_label)
        nperiods = len(self.periods)
        if axs is not None:
            fig = plt.gcf()

        if var in _MONOAXIAL_VARS:
            if axs is None:
                fig, axs = plt.subplots(1, nperiods, **kwargs)

            for per_i in range(nperiods):
                peri = self.subset_imu(per_i)
                per_var = peri[var]
                per_temp = peri["temperature"]
                xdata = per_temp.to_numpy()
                ydata = per_var.to_numpy()
                # Linear model
                model_fit = self.get_model(var, period=per_i,
                                           axis=per_var.dims[0])
                ax_i = axs[per_i]
                _plot_signal(x=xdata, y=ydata, idx=per_i,
                             model_fit=model_fit, ax=ax_i)
                ax_i.set_xlabel(xlabel)
            axs[0].set_ylabel(ylabel_pre)
        elif use_axis_order:
            if axs is None:
                fig, axs = plt.subplots(3, 1, **kwargs)
            axs[-1].set_xlabel(xlabel)
            for i, axis in enumerate(_AXIS_NAMES):
                idx = self.axis_order.index(axis)
                peri = self.subset_imu(idx)
                xdata = peri["temperature"].to_numpy()
                ydata = peri[var].sel(axis=axis).to_numpy()
                # Linear model
                model_fit = self.get_model(var, period=idx, axis=axis)
                ax_i = axs[i]
                _plot_signal(xdata, y=ydata, idx=idx,
                             model_fit=model_fit, ax=ax_i)
                ax_i.set_ylabel("{} {}".format(ylabel_pre, axis))
                ax_i.legend(loc=9, bbox_to_anchor=(0.5, 1),
                            frameon=False, borderaxespad=0)
        else:
            if axs is None:
                fig, axs = plt.subplots(3, nperiods, **kwargs)
            for vert_i in range(nperiods):
                peri = self.subset_imu(vert_i)
                xdata = peri["temperature"].to_numpy()
                axs_xyz = axs[:, vert_i]
                for i, axis in enumerate(_AXIS_NAMES):
                    ydata = (peri[var].sel(axis=axis).to_numpy())
                    # Linear model
                    model_fit = self.get_model(var, period=vert_i,
                                               axis=axis)
                    ax_i = axs_xyz[i]
                    _plot_signal(xdata, y=ydata, idx=vert_i,
                                 model_fit=model_fit, ax=ax_i)
                    ax_i.set_ylabel("{} {}".format(ylabel_pre, axis))
                axs_xyz[0].set_title("Period {}".format(vert_i))
                axs_xyz[-1].set_xlabel(xlabel)

        return fig, axs

    def plot_standardized(self, var, use_axis_order=True, units_label=None,
                          ref_val=None, axs=None, **kwargs):
        r"""Plot IMU measured and standardized variable along with temperature

        A multi-panel time series plot of the selected variable, measured
        and standardized, for all periods.

        Parameters
        ----------
        var : str
            IMU variable to plot.
        use_axis_order : bool, optional
            Whether to use axis order from the instance.  If True, only one
            sensor axis per period is considered to have valid calibration
            data for the correction.  Otherwise, all three axes for each
            period are used in the correction.
        units_label : str, optional
            Label for the units of the chosen variable.  Defaults to the
            "units_label" attribute available in the DataArray.
        ref_val : float
            Reference value for the chosen variable (e.g. gravity, for
            acceleration).  If provided, a horizontal line is included in
            the plot for reference.
        axs : array_like, optional
            Array of Axes instances to plot in.
        **kwargs
            Optional keyword arguments passed to
            :func:`~matplotlib.pyplot.subplots` (e.g. ``figsize``).

        Returns
        -------
        fig : matplotlib.Figure
        axs : array_like
            Array of :class:`~matplotlib.axes.Axes` instances in ``fig``
            with IMU signal plots.
        axs_temp : array_like
            Array of :class:`~matplotlib.axes.Axes` instances in ``fig``
            with temperature plots.

        See Also
        --------
        plot_experiment
        plot_var_model

        """
        def _plot_signal(ymeasured, ystd, temp, ax, neg_ref=False):
            ax_temp = ax.twinx()
            (ymeasured.plot.line(ax=ax, label="measured", color="k",
                                 linewidth=0.5))
            (ystd.plot.line(ax=ax, label="standardized", color="b",
                            linewidth=0.5, alpha=0.5))
            temp.plot.line(ax=ax_temp, label="temperature", color="r",
                           linewidth=0.5, alpha=0.5)
            txt_desc = ystd.attrs["description"]
            t_alpha_match = re.search(r'[-+]?\d+\.\d+', txt_desc)
            ax_temp.axhline(float(txt_desc[t_alpha_match.start():
                                           t_alpha_match.end()]),
                            linestyle="dashed", linewidth=1,
                            color="r", label=r"$T_\alpha$")
            q0 = ymeasured.quantile(1e-5).to_numpy().item()
            q1 = ymeasured.quantile(1 - 11e-5).to_numpy().item()
            if ref_val is not None:
                # Assumption of FLU with axes pointing against field
                if neg_ref:
                    ref_i = -ref_val
                else:
                    ref_i = ref_val
                ax.axhline(ref_i, linestyle="dashdot", color="m",
                           linewidth=1, label="reference")
                ylim0 = np.minimum(q0, ref_i)
                ylim1 = np.maximum(q1, ref_i)
            else:
                ylim0 = q0
                ylim1 = q1
            ax.set_title("")
            ax.set_xlabel("")
            # Adjust ylim to exclude outliers
            ax.set_ylim(ylim0, ylim1)

            # Axis locators and formatters
            dlocator = mdates.AutoDateLocator(minticks=3, maxticks=7)
            dformatter = mdates.ConciseDateFormatter(dlocator)
            ax.xaxis.set_major_locator(dlocator)
            ax.xaxis.set_major_formatter(dformatter)
            ax.xaxis.set_tick_params(rotation=0)

            return ax_temp

        per0 = self.subset_imu(0)
        if units_label is None:
            units_label = per0[var].attrs["units_label"]
        ylabel_pre = "{} [{}]".format(per0[var].attrs["full_name"],
                                      units_label)
        var_std = var + "_std"
        nperiods = len(self.periods)
        if axs is not None:
            fig = plt.gcf()

        std_ds = xr.merge(self._stdda_l)

        if var in _MONOAXIAL_VARS:
            if axs is None:
                fig, axs = plt.subplots(1, nperiods, **kwargs)
            axs_temp = np.empty_like(axs)
            for per_i in range(nperiods):
                peri = self.subset_imu(per_i)
                per_var = peri[var]
                per_std = std_ds.loc[dict(period=per_i)][var_std]
                per_temp = peri["temperature"]
                ax_i = axs[per_i]
                ax_temp = _plot_signal(per_var, ystd=per_std,
                                       temp=per_temp, ax=ax_i)
                ax_i.set_ylabel(ylabel_pre)
                axs_temp[per_i] = ax_temp
            # legend at center top panel
            axs[1].legend(loc=9, bbox_to_anchor=(0.5, 1.15), ncol=3,
                          frameon=False, borderaxespad=0)
            # Temperature legend at the bottom
            axs_temp[1].legend(loc=9, bbox_to_anchor=(0.5, -0.23), ncol=2,
                               frameon=False, borderaxespad=0)
        elif use_axis_order:
            if axs is None:
                fig, axs = plt.subplots(3, 1, sharex=False, **kwargs)
            axs_temp = np.empty_like(axs)
            for i, axis in enumerate(_AXIS_NAMES):
                idx = self.axis_order.index(axis)
                peri = self.subset_imu(idx)
                per_var = peri[var].sel(axis=axis, drop=True)
                per_std = (std_ds.loc[dict(period=idx)][var_std]
                           .sel(axis=axis, drop=True))
                per_temp = peri["temperature"]
                ax_i = axs[i]
                if axis == "x":
                    neg_ref = True
                else:
                    neg_ref = False
                ax_temp = _plot_signal(per_var, ystd=per_std,
                                       temp=per_temp, ax=ax_i,
                                       neg_ref=neg_ref)
                ax_i.set_xlabel("Period {}".format(idx))
                ax_i.set_ylabel("{} {}".format(ylabel_pre, axis))
                axs_temp[i] = ax_temp
            # legend at top panel
            axs[0].legend(loc=9, bbox_to_anchor=(0.5, 1.15), ncol=3,
                          frameon=False, borderaxespad=0)
            # Temperature legend at the bottom
            axs_temp[i].legend(loc=9, bbox_to_anchor=(0.5, -0.23), ncol=2,
                               frameon=False, borderaxespad=0)
        else:
            if axs is None:
                fig, axs = plt.subplots(3, nperiods, **kwargs)
            axs_temp = np.empty_like(axs)
            for vert_i in range(nperiods):
                axs_xyz = axs[:, vert_i]
                for i, axis in enumerate(_AXIS_NAMES):
                    peri = self.subset_imu(vert_i)
                    per_var = peri[var].sel(axis=axis, drop=True)
                    per_std = (std_ds.loc[dict(period=vert_i)][var_std]
                               .sel(axis=axis, drop=True))
                    per_temp = peri["temperature"]
                    ax_i = axs_xyz[i]
                    ax_temp = _plot_signal(per_var, ystd=per_std,
                                           temp=per_temp, ax=ax_i)
                    axs_temp[i, vert_i] = ax_temp
                    if vert_i == 0:
                        ax_i.set_ylabel("{} {}".format(ylabel_pre, axis))
                    else:
                        ax_i.set_ylabel("")
                axs_xyz[0].set_title("Period {}".format(vert_i))
            # legend at bottom panel
            leg0 = axs[-1, 1].legend(loc=9, bbox_to_anchor=(0.5, -0.23),
                                     ncol=3, frameon=False, borderaxespad=0)
            # Temperature legend at bottom panel
            leg1 = axs_temp[-1, 1].legend(loc=9, bbox_to_anchor=(0.5, -0.37),
                                          ncol=2, frameon=False,
                                          borderaxespad=0)
            axs[-1, 1].add_artist(leg0)
            axs_temp[-1, 1].add_artist(leg1)

        return fig, axs, axs_temp

    def get_model(self, var, period, axis=None):
        """Retrieve linear model for a given IMU sensor axis signal

        Parameters
        ----------
        var : str
            Name of the variable to calculate offset for.
        period: int
            Period containing calibration model to use.
        axis : str, optional
            Name of the sensor axis the signal comes from, if `var` is
            tri-axial; ignored otherwise.

        Returns
        -------
        RegressionResultsWrapper

        """
        if var in _MONOAXIAL_VARS:
            model_d = self.models_l[period][var]
            model_fit = [*model_d.values()][0]
        else:
            model_fit = self.models_l[period][var][axis]

        return model_fit

    def get_offset(self, var, period, T_alpha, ref_val, axis=None):
        """Calculate signal ofset at given temperature from calibration model

        Parameters
        ----------
        var : str
            Name of the variable to calculate offset for.
        period: int
            Period (zero-based) containing calibration model to use.
        T_alpha : float
            Temperature at which to compute offset.
        ref_val : float
           Reference value for the chosen variable (e.g. gravity, for
           acceleration).
        axis : str, optional
            Name of the sensor axis the signal comes from, if ``var`` is
            tri-axial; ignored otherwise.

        Returns
        -------
        float

        Notes
        -----
        For obtaining offset and gain of magnetometer signals, the
        ellipsoid method from the the ``ellipsoid`` module yields far more
        accurate results, as it allows for the simultaneous
        three-dimensional estimation of the offset.

        """
        if var in _MONOAXIAL_VARS:
            model_fit = self.get_model(var, period=period)
        else:
            model_fit = self.get_model(var, period=period, axis=axis)
        ypred = (model_fit.predict(exog=dict(temperature=T_alpha))
                 .to_numpy().item())
        logger.info("Predicted {} ({}, rounded) at T_alpha: {:.3f}"
                    .format(var, axis, ypred))
        offset = ypred - ref_val
        return offset

    def apply_model(self, var, dataset, T_alpha=None, ref_vals=None,
                    use_axis_order=True, model_idx=None):
        """Apply fitted temperature compensation model to Dataset

        The selected models for tri-axial sensor data are applied to input
        Dataset, standardizing signals at :math:`T_{\alpha}`, optionally
        subtracting the offset at :math:`T_{\alpha}`.

        Parameters
        ----------
        var : str
            Name of the variable with tri-axial data.
        dataset : xarray.Dataset
            Dataset with temperature and tri-axial data from motionless IMU.
        T_alpha : float, optional
            Reference temperature at which all measurements will be
            adjusted to.  Default is the mean temperature in the input
            dataset.
        ref_vals : list, optional
            Sequence of three floats with target values to compare against
            the signal from each sensor axis.  If provided, the offset of
            each signal at :math:`T_{\alpha}` is computed and subtracted from
            the temperature-standardized signal.  The order should be the
            same as in the `axis_order` attribute if `use_axis_order` is
            True, or ``x``, ``y``, ``z`` otherwise.
        use_axis_order : bool, optional
            Whether to use axis order from the instance.  If True, retrieve
            model to apply using instance's ``axis_order`` attribute.
            Otherwise, use the models defined by ``model_idx`` argument.
            Ignored if `var` is monoaxial.
        model_idx : list or int, optional
            Sequence of three integers identifying the period (zero-based)
            from which to retrieve the models for ``x``, ``y``, and ``z``
            sensor axes, in that order.  If ``var`` is monoaxial, an integer
            specifying the period for the model to use.  Ignored if
            ``use_axis_order`` is True.

        Returns
        -------
        xarray.Dataset

        """
        temp_obs = dataset["temperature"]
        darray = dataset[var]

        if T_alpha is None:
            T_alpha = temp_obs.mean().item()
            logger.info("T_alpha set to {:.2f}".format(T_alpha))

        def _standardize_array(darray, model_fit, period_idx, axis=None):
            x_hat = (model_fit
                     .get_prediction(exog=dict(Intercept=1,
                                               temperature=temp_obs))
                     .predicted_mean)
            x_alpha = (model_fit
                       .get_prediction(exog=dict(Intercept=1,
                                                 temperature=T_alpha))
                       .predicted_mean)
            x_sigma = darray - (x_hat - x_alpha)

            if ref_vals is not None:
                off = self.get_offset(var, axis=axis, period=period_idx,
                                      T_alpha=T_alpha,
                                      ref_val=ref_vals[period_idx])
                x_sigma -= off

            return x_sigma

        darray_l = []
        if var in _MONOAXIAL_VARS:
            model_fit = self.get_model(var, period=model_idx)
            x_sigma = _standardize_array(darray, model_fit=model_fit,
                                         period_idx=model_idx)
            darray_l.append(x_sigma)
        elif use_axis_order:
            for i, axis in enumerate(_AXIS_NAMES):
                idx = self.axis_order.index(axis)
                model_fit = self.get_model(var, period=idx, axis=axis)
                x_i = darray.sel(axis=axis)
                x_sigma = _standardize_array(x_i, model_fit=model_fit,
                                             period_idx=idx, axis=axis)
                darray_l.append(x_sigma)
        else:
            for i, axis in enumerate(_AXIS_NAMES):
                model_fit = self.get_model(var, period=model_idx[i],
                                           axis=axis)
                x_i = darray.sel(axis=axis)
                x_sigma = _standardize_array(x_i, model_fit=model_fit,
                                             period_idx=model_idx[i],
                                             axis=axis)
                darray_l.append(x_sigma)

        if len(darray_l) > 1:
            darray_new = xr.concat(darray_l, dim="axis").transpose()
        else:
            darray_new = darray_l[0]
        darray_new.attrs = darray.attrs
        new_history = ("{}: Applied temperature model at: T={}\n"
                       .format(pd.to_datetime("today")
                               .strftime("%Y-%m-%d"), T_alpha))
        darray_new.attrs["history"] = (darray_new.attrs["history"] +
                                       new_history)

        return darray_new

    def subset_imu(self, period_idx):
        """Subset IMU dataset given a period index

        The dataset is subset using the slice corresponding to the period
        index.

        Parameters
        ----------
        period_idx : int
            Index of the experiment period to subset.

        Returns
        -------
        xarray.Dataset

        """
        time_name = self.time_name
        return self.imu.loc[{time_name: self.periods[period_idx]}]
