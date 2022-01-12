=========================================
 IMU-frame to body-frame transformations
=========================================

Consider loading the `logging` module and setting up a logger to monitor
progress:

.. jupyter-execute::

   # Set up
   import pkg_resources as pkg_rsrc
   import os.path as osp
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import skdiveMove.imutools as imutools
   from scipy.spatial.transform import Rotation as R
   from skdiveMove.imutools.imu import (_ACCEL_NAME,
   				        _OMEGA_NAME,
   				        _MAGNT_NAME)
   from skdiveMove.imutools.imu2body import (_LEG3X1_ANCHOR,
   				             _euler_ctr2body)

   _FIG1X1 = (11, 5)
   _FIG3D1X1 = (11, 8)
   _FIG3X1 = (11, 11)
   _FIG4X1 = (10, 10)

   pd.set_option("display.precision", 3)
   np.set_printoptions(precision=3, sign="+")
   %matplotlib inline

Load saved pre-processed (properly calibrated and in a right-handed,
front-left-up frame) data:

.. jupyter-execute::
   :linenos:

   icdf = (pkg_rsrc
           .resource_filename("skdiveMove",
	                      osp.join("tests", "data", "gertrude",
			               "gert_imu_frame.nc")))
   icsv = (pkg_rsrc
           .resource_filename("skdiveMove",
	                      osp.join("tests", "data", "gertrude",
			               "gert_long_srfc.csv")))

Set up an instance to use throughout:

.. jupyter-execute::
   :linenos:

   # If data are in NetCDF and CSV files, one can take advantage of the
   # class method `IMU2Body.from_csv_nc` to instantiate directly from
   # the file names:
   imu2whale = imutools.IMU2Body.from_csv_nc(icsv, icdf,
   	       	  		             savgol_parms=(99, 2))
   # Print summary info
   print(imu2whale)
   # Choose an index from surface details table
   idx = imu2whale.surface_details.index[33]
   idx_title = idx.strftime("%Y%m%dT%H%M%S")

Visualize low-pass filter (Savitzky-Golay filter) job:

.. jupyter-execute::
   :hide-code:

   fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
   for i, ax in enumerate(axs):
       (imu2whale.get_surface_vectors(idx, _ACCEL_NAME)[:, i]
        .plot.line(ax=axs[i], label="measured"))
       (imu2whale.get_surface_vectors(idx, _ACCEL_NAME,
                                      smoothed_accel=True)[:, i]
        .plot.line(ax=axs[i], linestyle="--", label="SG filtered"))
       axs[i].set_xlabel("")
       axs[i].set_title("")
   axs[0].set_title(idx_title)
   axs[0].margins(x=0)
   axs[-1].tick_params(labelrotation=0, which="both")
   leg = axs[2].legend(loc=9, bbox_to_anchor=_LEG3X1_ANCHOR,
                       frameon=False, borderaxespad=0, ncol=2)
   leg.get_texts()[0].set_text("measured")
   leg.get_texts()[1].set_text("SG filtered");

Quick plots of smoothed acceleration and magnetic density from the
segment:

.. jupyter-execute::
   :linenos:

   acc_imu = imu2whale.get_surface_vectors(idx, _ACCEL_NAME,
                                           smoothed_accel=True)
   depth = imu2whale.get_surface_vectors(idx, "depth")
   # Alternatively, use the function of the same name as method below
   ax = imu2whale.scatterIMU3D(idx, _MAGNT_NAME, normalize=True,
                               animate=False, figsize=_FIG3D1X1)
   ax.view_init(azim=-30);

Below shows that the IMU was deployed facing forward and on the left side
of the whale, so in the above plot negative `x` is forward and negative `y`
is left as per our right-handed coordinate system.  As above, we can use
the method of the same name to produce the plot:

.. jupyter-execute::

   imu2whale.tsplotIMU_depth(_ACCEL_NAME, idx, smoothed_accel=True,
                             figsize=_FIG4X1);

Calculate orientation for the segment above, and produce an animated plot
of the orientation.  This can be done in a single step with
`IMU2Body.get_orientation`.

.. jupyter-execute::
   :linenos:
   :hide-output:

   Rctr2i, svd = imu2whale.get_orientation(idx, plot=False, animate=False)
   anim_file = "source/.static/video/gert_imu_{}.mp4".format(idx_title)
   imutools.scatterIMU_svd(acc_imu, svd, Rctr2i, normalize=True, center=True,
                  	   animate=True, animate_file=anim_file,
		  	   title="IMU-Frame Centered Acceleration [g]",
		  	   figsize=_FIG3D1X1)

.. raw:: html

   <video controls width="800" height="420">
   <source src="_static/video/gert_imu_20170810T120654.mp4" type="video/mp4">
   </video>

Apply the inverse transformation to get to the animal frame:

.. jupyter-execute::
   :linenos:

   # Orient the surface segment using the estimated rotation
   imu_bodyi = imu2whale.orient_surfacing(idx, Rctr2i)
   # Have a look at corrected acceleration
   acci = imu_bodyi[_ACCEL_NAME]

An animation may be useful to visualize the normalized animal-frame data:

.. jupyter-execute::
   :linenos:
   :hide-output:

   anim_file = "source/.static/video/gert_body_{}.mp4".format(idx_title)
   imutools.scatterIMU3D(acci, imu_bodyi["depth"].to_numpy().flatten(), normalize=True,
                	 animate=True, animate_file=anim_file,
                	 title=r"Normalized Animal-Frame Acceleration [$g$]",
                	 cbar_label="Depth [m]", figsize=_FIG3D1X1)

.. raw:: html

   <video controls width="800" height="420">
   <source src="_static/video/gert_body_20170810T120654.mp4" type="video/mp4">
   </video>

Obtain orientations for all surface periods, and retrieve the quality
indices for each estimate:

.. jupyter-execute::

   orientations = imu2whale.get_orientations()
   # Unpack quality of the estimates
   qual = orientations["quality"].apply(pd.Series)
   qual.columns = ["q_index", "phi_std"]

Look at the quality of the orientation estimates:

.. jupyter-execute::
   :hide-code:

   # Plot the quality index
   fig = plt.figure(figsize=_FIG1X1)
   ax = fig.add_subplot(111)
   ax.set_ylabel("Quality index")
   qual["q_index"].plot(ax=ax, style="-o", rot=0)
   qual["phi_std"].plot(ax=ax, style="-o", rot=0)
   ax.axhline(0.05, linestyle="dashed", color="k")
   ax.set_xlabel("")
   leg = ax.legend(loc=9, bbox_to_anchor=(0.5, -0.08),
                   frameon=False, borderaxespad=0, ncol=2)
   leg.get_texts()[0].set_text("quality index")
   leg.get_texts()[1].set_text("rolling std")

Remove bad quality estimates:

.. jupyter-execute::
   :linenos:

   imu2whale.filter_surfacings((0.04, 0.06))

Plot "ok" Euler angles:

.. jupyter-execute::
   :hide-code:

   fig, axs = plt.subplots(3, 1, sharex=True, figsize=_FIG3X1)
   (imu2whale.orientations[["phi", "theta", "psi"]]
    .plot(ax=axs, style="o-", subplots=True, legend=False,
          rot=0))
   axs[0].set_ylabel(r"$\phi$ [deg]")
   axs[1].set_ylabel(r"$\theta$ [deg]")
   axs[2].set_ylabel(r"$\psi$ [deg]")
   for ax in axs:
       ax.axhline(0, linestyle="dashed", color="k")
   # Summary
   imu2whale.orientations[["phi", "theta", "psi"]].describe()

Can we use an average (median?) rotation matrix?  This requires retrieving
the direction cosine matrices of the centered data, which can be expressed
as Euler angles with respect to the centered-data frame:

.. jupyter-execute::
   :linenos:

   euler_xyz = (imu2whale.orientations["R"]
                .apply(lambda x: x.as_euler("XYZ", degrees=True))
                .apply(pd.Series))
   euler_xyz.rename(columns={0: "phi_ctr", 1: "theta_ctr", 2: "psi_ctr"},
                    inplace=True)
   euler_avg = euler_xyz.mean()
   Rctr2i_avg = R.from_euler("XYZ", euler_avg.values, degrees=True)
   Rctr2i_avg.as_euler("XYZ", degrees=True)
   _euler_ctr2body(Rctr2i_avg)

Check the effect of using this common transformation with the first
remaining period:

.. jupyter-execute::
   :linenos:

   idx0 = imu2whale.surface_details.index[0]
   imu_bodyi = imu2whale.orient_surfacing(idx0, Rctr2i_avg)
   # Plot the time series; not bad
   imutools.tsplotIMU_depth(imu_bodyi[_ACCEL_NAME], imu_bodyi["depth"],
                            figsize=_FIG4X1);

Orient all surface periods with average rotation -- note we get a
hierarchical dataframe output:

.. jupyter-execute::
   :linenos:

   imu_bodys = imu2whale.orient_surfacings(R_all=Rctr2i_avg)
   # Check out plot of a random sample
   idxs = imu_bodys.coords.get("endasc")  # values in topmost level
   idx_rnd = idxs[np.random.choice(idxs.size)]
   idx_rnd_title = idx_rnd.dt.strftime("%Y%m%dT%H%M%S").item()
   # Compare with period-specific orientation
   Rctr2i = imu2whale.orientations.loc[idx_rnd.to_pandas()]["R"]
   imu_bodyi = imu2whale.orient_surfacing(idx_rnd.to_pandas(), Rctr2i)

.. jupyter-execute::
   :hide-code:

   fig, axs = plt.subplots(3, 1, sharex=True, figsize=_FIG3X1)
   axs[0].set_title(idx_rnd_title)
   for i, ax in enumerate(axs):
       (imu_bodyi[_ACCEL_NAME][:, i]
        .plot.line(x="timestamp", ax=ax, label="segment-specific"))
       (imu_bodys.sel(endasc=idx_rnd)[_ACCEL_NAME][:, i]
        .plot.line(x="timestamp", ax=ax, linestyle="dashed", label="common"))
       axs[i].set_xlabel("")
       if i < 2:
           hlev = 0
       else:
           hlev = 1
       ax.axhline(hlev, linestyle="dashed", color="k")
   axs[0].margins(x=0)
   axs[-1].tick_params(labelrotation=0, which="both")
   leg = axs[2].legend(loc=9, bbox_to_anchor=_LEG3X1_ANCHOR,
                       frameon=False, borderaxespad=0, ncol=2);

Orient the entire `IMU` object with common rotation:

.. jupyter-execute::

   gert_frame = imu2whale.orient_IMU(Rctr2i_avg)

Or with segment-specific rotations:

.. jupyter-execute::
   :linenos:

   gert_frames = imu2whale.orient_IMU()
   # Check out IMUs
   idx = imu2whale.surface_details.index[1]
   imu_i = gert_frames.sel(endasc=idx)
   imutools.tsplotIMU_depth(imu_i[_ACCEL_NAME], imu_i["depth"],
                            figsize=_FIG4X1);

Feel free to download a copy of this demo
(:jupyter-download:script:`demo_imu2body`).
