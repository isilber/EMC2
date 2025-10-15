import numpy as np
from .subcolumn import set_convective_sub_col_frac, set_precip_sub_col_frac
from .subcolumn import set_stratiform_sub_col_frac, set_q_n
from .lidar_moments import calc_lidar_moments, calc_LDR_and_ext, calc_total_alpha_beta
from .radar_moments import calc_radar_moments, calc_total_reflectivity
from .attenuation import calc_radar_Ze_min
from .classification import lidar_classify_phase, lidar_emulate_cosp_phase, radar_classify_phase


def make_simulated_data(model, instrument, N_columns, do_classify=False, unstack_dims=False,
                        skip_subcol_gen=False, finalize_fields=False, calc_spectral_width=True,
                        subcol_gen_only=False,
                        **kwargs):
    """
    This procedure will make all of the subcolumns and simulated data for each model column.

    NOTE:
    When starting a parallel task (in microphysics approach), it is recommended
    to wrap the top-level python script calling the EMC^2 processing ('lines_of_code')
    with the following command (just below the 'import' statements):
    
    .. code-block:: python
    
        if __name__ == “__main__”:
            lines_of_code

    Parameters
    ----------
    model: :func:`emc2.core.Model`
        The model to make the simulated parameters for.
    instrument: :func:`emc2.core.Instrument`
        The instrument to make the simulated parameters for.
    N_columns: int or None
        The number of subcolumns to generate. Set to None to automatically
        detect from LES 4D data.
    do_classify: bool
        run hydrometeor classification routines when True.
    unstack_dims: bool
        True - unstack the time, lat, and lon dimensions after processing in cases
        of regional model output.
    skip_subcol_gen: bool
        True - skip the subcolumn generator (e.g., in case subcolumn were already generated).
    finalize_fields: bool
        True - set absolute 0 values in"sub_col"-containing fields to np.nan enabling analysis
        and visualization.
    calc_spectral_width: bool
        If False, skips spectral width calculations since these are not always needed for an application
        and are the most computationally expensive. Default is True.
    subcol_gen_only: bool
        If True, only returns mass and number distributed among subcolumns and skips moment calculations
    Additional keyword arguments are passed into :func:`emc2.simulator.calc_lidar_moments` or
    :func:`emc2.simulator.calc_radar_moments`

    Returns
    -------
    model: :func:`emc2.core.Model`
        The model with all of the simulated parameters generated.
    """
    hydrometeor_classes = model.conv_frac_names.keys()

    # Cache frequently used attributes to avoid repeated attribute lookups
    process_conv = getattr(model, 'process_conv', False)
    model_name = getattr(model, 'model_name', None)
    mcphys_scheme = getattr(model, 'mcphys_scheme', '').lower()
    inst_class = getattr(instrument, 'instrument_class', '').lower()

    use_rad_logic = kwargs.pop('use_rad_logic', True)

    OD_from_sfc = kwargs.pop('OD_from_sfc', instrument.OD_from_sfc)

    parallel = kwargs.pop('parallel', True)

    chunk = kwargs.pop('chunk', None)

    convert_zeros_to_nan = kwargs.pop('convert_zeros_to_nan', False)

    mask_height_rng = kwargs.pop('mask_height_rng', None)

    hyd_types = kwargs.pop('hyd_types', None)

    mie_val = kwargs.pop('mie_for_ice', None)
    if mie_val is not None:
        # allow a single value to be provided and apply to both conv/strat
        mie_for_ice = {"conv": mie_val, "strat": mie_val}
    else:
        if use_rad_logic:
            mie_for_ice = {"conv": False, "strat": False}
        elif mcphys_scheme == "p3":
            mie_for_ice = {"conv": False, "strat": False}  # ice shape integrated into microphysics
        else:
            mie_for_ice = {"conv": False, "strat": True}  # use True for strat (micro), False for conv (rad)
    use_empiric_calc = kwargs.pop('use_empiric_calc', False)

    if skip_subcol_gen:
        print("Skipping subcolumn generator (make sure subcolumns were already generated).")
    else:
        print("## Creating subcolumns...")
        if process_conv:
            for hyd_type in hydrometeor_classes:
                model = set_convective_sub_col_frac(
                    model, hyd_type, N_columns=N_columns,
                    use_rad_logic=use_rad_logic)
        else:
            print(f"No convective processing for {model_name}")

        # Subcolumn Generator
        model = set_stratiform_sub_col_frac(
            model, use_rad_logic=use_rad_logic, N_columns=N_columns, parallel=parallel, chunk=chunk)

        # Build precip_types excluding cloud-only classes
        precip_types = [h for h in hydrometeor_classes if h not in {"cl", "ci"}]

        # Call set_precip_sub_col_frac for stratiform (is_conv=False) and optionally convective
        for is_conv in (False, True) if process_conv else (False,):
            if not is_conv:
                model = set_precip_sub_col_frac(
                    model, is_conv=is_conv, use_rad_logic=use_rad_logic, parallel=parallel, chunk=chunk,
                    precip_types=precip_types)
            else:
                model = set_precip_sub_col_frac(
                    model, is_conv=is_conv, use_rad_logic=use_rad_logic, parallel=parallel, chunk=chunk)

        # Distribute q and N among subcolumns for each hydrometeor type and convective state
        is_conv_list = (False, True) if process_conv else (False,)
        for hyd_type in hydrometeor_classes:
            for is_conv in is_conv_list:
                # For cloud liquid 'cl', use qc_flag=True only for stratiform (is_conv=False)
                if hyd_type == 'cl' and not is_conv:
                    qc_flag = True
                else:
                    qc_flag = False
                model = set_q_n(
                    model, hyd_type, is_conv=is_conv,
                    qc_flag=qc_flag, use_rad_logic=use_rad_logic, parallel=parallel, chunk=chunk)

    # Skip moment calculations and return only subcolumn-distributed q and N, else continue to simulator
    if subcol_gen_only:
        print("User chose to return only subcolumn generation and skip moment calculations")
    else:

        # Radar Simulator
        if inst_class == "radar":
            print("Generating radar moments...")
            # allow callers to override the reference range via kwargs; default to 1000
            ref_rng = kwargs.pop('ref_rng', 1000)

            model = calc_radar_moments(
                instrument, model, False, OD_from_sfc=OD_from_sfc, hyd_types=hyd_types,
                parallel=parallel, chunk=chunk, mie_for_ice=mie_for_ice["strat"],
                use_rad_logic=use_rad_logic,
                use_empiric_calc=use_empiric_calc, calc_spectral_width=calc_spectral_width,**kwargs)
            if process_conv:
                model = calc_radar_moments(
                    instrument, model, True, OD_from_sfc=OD_from_sfc, hyd_types=hyd_types,
                    parallel=parallel, chunk=chunk, mie_for_ice=mie_for_ice["conv"],
                    use_rad_logic=use_rad_logic,
                    use_empiric_calc=use_empiric_calc, calc_spectral_width=calc_spectral_width,**kwargs)

            model = calc_radar_Ze_min(instrument, model, ref_rng)
            model = calc_total_reflectivity(model, detect_mask=True)

            if do_classify is True:
                model = radar_classify_phase(
                    instrument, model, mask_height_rng=mask_height_rng,
                    convert_zeros_to_nan=convert_zeros_to_nan)

        # Lidar Simulator
        elif inst_class == "lidar":
            print("Generating lidar moments...")
            ext_OD = kwargs.pop('ext_OD', instrument.ext_OD)
            eta = kwargs.pop('eta', instrument.eta)
            model = calc_lidar_moments(
                instrument, model, False, OD_from_sfc=OD_from_sfc, hyd_types=hyd_types,
                parallel=parallel, eta=eta, chunk=chunk,
                mie_for_ice=mie_for_ice["strat"], use_rad_logic=use_rad_logic,
                use_empiric_calc=use_empiric_calc, **kwargs)
            if process_conv:
                model = calc_lidar_moments(
                    instrument, model, True, OD_from_sfc=OD_from_sfc, hyd_types=hyd_types,
                    parallel=parallel, eta=eta, chunk=chunk,
                    mie_for_ice=mie_for_ice["conv"], use_rad_logic=use_rad_logic,
                    use_empiric_calc=use_empiric_calc, **kwargs)
            model = calc_total_alpha_beta(model, OD_from_sfc=OD_from_sfc, eta=eta)
            model = calc_LDR_and_ext(model, ext_OD=ext_OD, OD_from_sfc=OD_from_sfc, hyd_types=hyd_types)

            if do_classify is True:
                model = lidar_classify_phase(
                    instrument, model, convert_zeros_to_nan=convert_zeros_to_nan)
                model = lidar_emulate_cosp_phase(
                    instrument, model, eta=eta, OD_from_sfc=OD_from_sfc,
                    convert_zeros_to_nan=convert_zeros_to_nan, hyd_types=hyd_types)
        else:
            raise ValueError("Currently, only lidars and radars are supported as instruments.")

    
    
    
    if finalize_fields:
        model.finalize_subcol_fields()
        
    
    # Unstack dims in case of regional model output (typically done at the end of all EMC^2 processing)
    sd = getattr(model, 'stacked_time_dim', None)
    if (sd is not None) and unstack_dims:
        print(f"Unstacking the {sd} dimension (time, lat, and lon dimensions)")
        model.unstack_time_lat_lon()
        
    return model
