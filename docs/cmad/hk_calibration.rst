Housekeeping Calibration
------------------------

This section describes the calibration of the non-science data values such as housekeeping data.

hvps_temp
^^^^^^^^^

.. plot::

   import matplotlib.pyplot as plt
   from astropy.io import ascii

   from padre_meddea.calibration import calib_hk
   calib_directory = calib_hk._calibdata_directory
   calib_data = ascii.read(calib_directory / "hvps_temp.csv")
   f = calib_hk.get_hk_cal_func('hvps_temp')
   adc = calib_data['adc']
   plt.plot(adc, calib_data['deg_C'], 'x')
   plt.plot(adc, f(adc), '-')
   plt.title('hvps_temp')


hvps_setpoint
^^^^^^^^^^^^^^

.. plot::

   import matplotlib.pyplot as plt
   from astropy.io import ascii

   from padre_meddea.calibration import calib_hk
   calib_directory = calib_hk._calibdata_directory

   calib_data = ascii.read(calib_directory / "hvps_setpoint.csv")
   f = calib_hk.get_hk_cal_func('hvps_setpoint')
   adc = calib_data['adc']
   plt.plot(adc, calib_data['volt'], 'x')
   plt.plot(adc, f(adc), '-')
   plt.title('hvps_setpoint')


hvps_vsense
^^^^^^^^^^^

.. plot::

   import matplotlib.pyplot as plt
   from astropy.io import ascii

   from padre_meddea.calibration import calib_hk
   calib_directory = calib_hk._calibdata_directory

   calib_data = ascii.read(calib_directory / "hvps_vsense.csv")
   f = calib_hk.get_hk_cal_func('hvps_vsense')
   adc = calib_data['adc']
   plt.plot(adc, calib_data['volt'], 'x')
   plt.plot(adc, f(adc), '-')
   plt.title('hvps_vsense')


hvps_csense
^^^^^^^^^^^

.. plot::

   import matplotlib.pyplot as plt
   from astropy.io import ascii

   from padre_meddea.calibration import calib_hk

   calib_directory = calib_hk._calibdata_directory
   calib_data = ascii.read(calib_directory / "hvps_csense.csv")
   f = calib_hk.get_hk_cal_func('hvps_csense')
   adc = calib_data['adc']
   plt.plot(adc, calib_data['nanoamp'], 'x')
   plt.plot(adc, f(adc), '-')
   plt.title('hvps_csense')


pulser_setpoint
^^^^^^^^^^^^^^^^

.. plot::

   import matplotlib.pyplot as plt
   from astropy.io import ascii

   from padre_meddea.calibration import calib_hk
   calib_directory = calib_hk._calibdata_directory
 
   calib_data = ascii.read(calib_directory / "pulser_setpoint.csv")
   f = calib_hk.get_hk_cal_func('pulser_setpoint')
   adc = calib_data['adc']
   plt.plot(adc, calib_data['volt'], 'x')
   plt.plot(adc, f(adc), '-')
   plt.title('pulser_setpoint')


fp_temp
^^^^^^^

.. plot::

   import matplotlib.pyplot as plt
   from astropy.io import ascii

   from padre_meddea.calibration import calib_hk
   calib_directory = calib_hk._calibdata_directory
  
   calib_data = ascii.read(calib_directory / "fp_temp.csv")
   
   f = calib_hk.get_hk_cal_func('fp_temp')
   adc = calib_data['adc']
   plt.plot(adc, calib_data['deg_C'], 'x')
   plt.plot(adc, f(adc), '-')
   plt.title('fp_temp')


csense_15v, csense_33vd, csense_33va
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. plot::

   import matplotlib.pyplot as plt
   from astropy.io import ascii

   from padre_meddea.calibration import calib_hk
   
   adc = [20000, 30000, 40000]
   mamps = [230.65, 122.89, 15.13]
   f = calib_hk.get_hk_cal_func('csense_15v')
   plt.plot(adc, mamps, 'x')
   plt.plot(adc, f(adc), '-')
   plt.title('csense_15v, csense_33vd, csense_33va')

