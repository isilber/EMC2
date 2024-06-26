try:
    from act.io.arm import read_arm_netcdf as read_netcdf
except:
    print('Using act-atmos v1.5.3 or earlier. Please update to v2.0.0 or newer')
    from act.io.armfiles import read_netcdf


def load_arm_file(filename, **kwargs):
    """
    Loads an ARM-compliant netCDF file.

    Parameters
    ----------
    filename: str
       The name of the file to load.

    Additional keyword arguments are passed into :py:func:`act.io.arm.read_arm_netcdf`

    Returns
    -------
    ds: ACT dataset
        The xarray dataset containing the file data.
    """

    return read_netcdf(filename, **kwargs)
