#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FERMI LDM online data reducer

@license: GPL v3 (c)2022
@author: Michele Devetta <michele.devetta@cnr.it>

"""
import numpy as np
from DataReducer import DataReducer
import ProcessingFunctions
from Logger import Logger


# ================
# REMOTE PATH
# Path to the remote data on the online storage. It should be the root path of the current beamtime
remote_path = "/home/ldm/ExperimentalData/Online4LDM/<beamtime number>"


# ================
# LOCAL PATH
# Path to a local folder where the script should copy the acquired data
# Configure only if you want a local copy of the data. Otherwise, leave 'None'
local_path = None


# ================
# SAVE PATH
# Path were to save reduced data. By default reduced data is saved in the 'work' directory of each run.
save_path = None


# ================
# CONFIGURATION
# Configure log file
logfile = '/tmp/reducer.log'

# Set the time in seconds after a run is marked as 'stale', i.e. is old enough to not expect other files to come
# This is needed when a run is aborted, otherwise the script automatically teminate when it reaches the expected number of files
stale_run = 5 * 60

# Number of workers: number of parallel processes that process acquisition runs
number_of_workers = 4

# Runs to skip: set the first run to consider when processing (useful when the reducer is restarted)
runs_to_skip = 0

# FEL source in use (1 or 2)
fel = 2

# Enable DEBUG
DEBUG = False


options = {}
## ============================================================================
# PROCESSING CONFIGURATION
# IMPORTANT: never use "spectrum", "spec_wl", "spec_fwhm", "spec_int", "vmi", "tof" or "advanced" as tags.
# All tags MUST be unique.
# NOTE: the 'background' shot-to-shot tag has a special meaning. It should contain a logical array. All shots marked
# as true will be assigned to background, other to signal.


# => SHOT-TO-SHOT
# Datasets to be stored in output shot-by-shot (I0, BN, etc)
options['s2s'] = [
    # I0 monitor
    {'tag': 'i0_uh',    'dataset': f'photon_diagnostics/FEL{fel:02d}/I0_monitor/iom_uh_a'},
    {'tag': 'i0_sh',    'dataset': f'photon_diagnostics/FEL{fel:02d}/I0_monitor/iom_sh_a'},
    {'tag': 'i0_uh_pc', 'dataset': f'photon_diagnostics/FEL{fel:02d}/I0_monitor/iom_uh_a_pc'},
    {'tag': 'i0_sh_pc', 'dataset': f'photon_diagnostics/FEL{fel:02d}/I0_monitor/iom_sh_a_pc'},

    # Mirror photocurrent
    {'tag': 'vdm',         'dataset': 'photon_diagnostics/VDM-LDM/photocurrent'},
    {'tag': f'pm{fel:d}a', 'dataset': f'photon_diagnostics/PM{fel:d}A/photocurrent'},

    # LDM photodiode
    #{'tag': 'i0_ldm',    'dataset': 'photon_diagnostics/photodiode_ldm/Id'},
    #{'tag': 'i0_ldm_pc', 'dataset': 'photon_diagnostics/photodiode_ldm/Id_pC'},

    # SLU
    {'tag': 'delay',       'dataset': 'user_laser/delay_line/position'},
    {'tag': 'slu_i0',      'dataset': 'user_laser/energy_meter/Energy1'},
    {'tag': 'slu_i0_25hz', 'dataset': 'user_laser/energy_meter/Energy2'},

    # PADRES spectrometer
    { # All spectra for each shot
        'tag': 'spectrum',
        'processing': ProcessingFunctions.padres_spectrum,
        'dataset': {
            'spectrum': 'photon_diagnostics/Spectrometer/hor_spectrum',
            'roi':      'photon_diagnostics/Spectrometer/ROI',
        },
    },
    { # Wavelength and FWHM from a gaussian fit of the peak (return nan if fit fails)
        'tag': 'spectrum_fit',
        'processing': ProcessingFunctions.fit_padres_spectrum,
        'dataset': {
            'spectrum':   'photon_diagnostics/Spectrometer/hor_spectrum',
            'wavelength': 'photon_diagnostics/Spectrometer/Wavelength',
            'p2m':        'photon_diagnostics/Spectrometer/Pixel2micron',
            'span':       'photon_diagnostics/Spectrometer/WavelengthSpan',
            'roi':        'photon_diagnostics/Spectrometer/ROI',
        },
    },
    { # Integral of the spectrum
        'tag': 'spectrum_int',
        'processing': ProcessingFunctions.spectrum_integral,
        'dataset': {
            'spectrum': 'photon_diagnostics/Spectrometer/hor_spectrum',
            'roi':      'photon_diagnostics/Spectrometer/ROI',
        },
    },

    # Background sequence
    # { # Background sequence with the background period on the molecular source
    #     'tag': 'background',
    #     'processing': ProcessingFunctions.background_sequence,
    #     'extra_args': {
    #         'phase': 0,
    #     },
    #     'dataset': {
    #         'bunches': 'bunches',
    #         'period': 'Background_Period',
    #     }
    # },
    { # Background sequence on the SLU at 25 Hz
        'tag': 'background',
        'processing': ProcessingFunctions.background_sequence,
        'extra_args': {
            'period': 2,
            'phase': 1,
        },
        'dataset': {
            'bunches': 'bunches',
        },
    },

    # Integrate TOF peaks
    # {
    #     'tag': 'tof_peaks',
    #     'processing': ProcessingFunctions.integrate_digitizer,
    #     'extra_args': {
    #         'threshold': 12,
    #         'baseline': [0, 2000],
    #         'peaks': [(5360, 5390), (5395, 5415)],
    #     },
    #     'dataset': {
    #         'trace': 'digitizer/channel3',
    #     },
    # },
]


# => METADATA
# Values to be stored only once
options['metadata'] = [
    {'tag': 'wavelength',   'dataset': f'photon_source/FEL{fel:02d}/wavelength'},
    {'tag': 'harmonic',     'dataset': f'photon_source/FEL{fel:02d}/harmonic_number'},
    #{'tag': 'polarization', 'dataset': 'photon_source/FEL{fel:02d}/polarization_status'},
    #{'tag': 'back_p',       'dataset': 'endstation/analog_in/I1'},
    {'tag': 'gas_cell',     'dataset': f'photon_diagnostics/FEL{fel:02d}/Gas_Attenuator/Pressure'},
    #{'tag': 'phase',        'dataset': f'photon_source/FEL{fel:02d}/PhaseShifter5/DeltaPhase'},
    {
        'tag': 'spectrum_wl',
        'processing': ProcessingFunctions.padres_spectrum_wavelength,
        'dataset': {
            'wavelength': 'photon_diagnostics/Spectrometer/Wavelength',
            'p2m':        'photon_diagnostics/Spectrometer/Pixel2micron',
            'span':       'photon_diagnostics/Spectrometer/WavelengthSpan',
            'roi':        'photon_diagnostics/Spectrometer/ROI',
        },
        'extra_args': {},
    },
    {
        'tag': 'mbes_retardation',
        'processing': ProcessingFunctions.mbes_retardation,
        'dataset': {
            'ret_v1': 'endstation/MagneticBottle/voltage_ch1',
            'ret_v1_en': 'endstation/MagneticBottle/ch1_is_enabled',
            'ret_v2': 'endstation/MagneticBottle/voltage_ch2',
            'ret_v2_en': 'endstation/MagneticBottle/ch2_is_enabled',
        },
    },
]


# => MAIN DATA REDUCTION
options['main'] = [

    # VMI: processing of the VMI image sum
    # {
    #     'tag': 'vmi',
    #     'dataset': 'vmi/andor',
    #     'preprocess': ProcessingFunctions.camera_baseline, # Subtract camera baseline
    #     'binning': [
    #         #{'tag': 'i0', 'dataset': 'spectrum_int', 'bin_edges': [0, ] + list(range(60,80,2)) + [80, 100]}
    #     ],
    # },

    # DIGITIZER: processing of the digitizer trace for TOF or MBES
    {
        'tag': 'mbes',
        #'dataset': 'digitizer/channel1',
        'dataset': 'digitizer/channel3',
        #'preprocess': lambda x: ProcessingFunctions.tof_baseline(x, [1, 2000]), # Subtract digitizer baseline
        'preprocess': lambda x: ProcessingFunctions.tof_with_threshold(x, 12, [1,2000]), # Subtract digitizer baseline and apply threshold
        'filters': [
            {'dataset': 'spectrum_fit', 'processing': lambda x: np.logical_not(np.isnan(x[:,0]))},
            {'dataset': 'spectrum_int', 'processing': lambda x: np.logical_and(~np.isnan(x), x > 5e5)},
        ],
        'binning': [
            #{'tag': 'i0', 'dataset': 'spectrum_int', 'bin_edges': [0.4e7, 0.6e7, 0.8e7, 1e7, 1.2e7]},
            #{'tag': 'wl', 'dataset': 'spectrum_fit', 'preprocessing': lambda x: x[:, 0], 'bin_edges': [6.242, 6.2433, 6.2445]},
        ],
    },
]


# => ADVANCED PROCESSING
# Advanced processing. You must provide a function that takes as first parameter a state, i.e. a dictionary of output variables,
# that is maintained for the full run and should be returned by the function for the next call. Then a second parameter is the
# background mask for the current file. Then the function can take any number of dataset as kwargs. The args name must correspond
# to the keys in the datasets option
# options['advanced'] = [
#     {
#         'tag': 'adv_test',
#         'processing': sample_processing_function,
#         'datasets': {
#             'test1': 'bunches',
#             'test2': 'Background_Period',
#         },
#     },
# ]


#################################################################################################
# RUN DATA REDUCER - DO NOT MODIFY
dr = DataReducer(options, remote_path, local_path, save_path=save_path, nworkers=number_of_workers, skip_runs=runs_to_skip, log_level=Logger.DEBUG if DEBUG else Logger.INFO)
dr.setStaleTimeRun(stale_run)
dr.addLogFile(logfile)
dr.run()
