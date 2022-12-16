# -*- coding: utf-8 -*-
"""
Pre-processing functions for online data reducer

@license: GPL v3 (c)2022
@author: Michele Devetta <michele.devetta@cnr.it>

"""

import numpy as np
from scipy.optimize import curve_fit


# =========================
# Processing PADRES spectrometer

def padres_spectrum(spectrum, roi):
    """ Return the PADRES spectrum filterd by ROI
    """
    return spectrum[:, roi[0]:roi[2]]

def padres_spectrum_wavelength(wavelength, p2m, span, roi):
    """ Compute wavelength axis for PADRES spectrometer
    """
    wl = wavelength + p2m * span * np.arange(-500, 500, dtype=np.float64) / 1000.0
    return wl[roi[0]:roi[2]]

def spectrum_integral(spectrum, roi):
    """ Compute integral of PADRES spectrometer
    """
    spectrum = padres_spectrum(spectrum, roi)
    baseline_value = np.mean(np.hstack((spectrum[:, 0:5], spectrum[:, -5:])), axis=1)
    baseline = np.tile(np.expand_dims(baseline_value, axis=1), (1, spectrum.shape[1]))
    spectrum = spectrum - baseline
    return np.sum(spectrum, axis=1)

def fit_spectrum(x, y):
    """ Fit with a gaussian line the spectrum and returns the central wavelenght and FWHM in nm
    """
    def gauss(x, base, a, m, s):
        # Split parameters
        return base + a * np.exp( -(x - m)**2 / s**2)

    # Baseline
    im = np.argmax(y)
    if len(y)-im < 5 or im < 5:
        # Too near the edge. probably no spectrum
        return (0, 0)
    if im > len(y)/2:
        base0 = np.mean(y[0:20])
    else:
        base0 = np.mean(y[-20:])

    # Amplitude
    a0 = np.max(y) - base0

    # Mean
    m0 = x[im]

    # Sigma
    yhalf = np.abs(y - base0 - a0 / 2)
    imin1 = np.argmin(yhalf[0:im])
    imin2 = np.argmin(yhalf[im:]) + im
#   print("IM =", im, "IMIN1 =", imin1, "IMIN2 =", imin2, "SHAPE(x) =", x.shape, "SHAPE(y) =", y.shape)
    s0 = np.abs(x[imin2] - x[imin1]) / 2 / np.sqrt(2 * np.log(2))

    p0 = [base0, a0, m0, s0]
    try:
        p, pconv = curve_fit(gauss, x, y, p0)
        #p, pconv = curve_fit(gauss, x, y, p0, bounds=([0, 0, x[-1], 0], [base0*2, a0*2, x[0], 5*s0]))
        return (p[2], 2 * np.sqrt(2 * np.log(2)) * np.abs(p[3]))
    except Exception:
        # Fit failed. Probably no spectrum
        return (np.nan, np.nan)

def fit_padres_spectrum(spectrum, wavelength, p2m, span, roi):
    wl = padres_spectrum_wavelength(wavelength, p2m, span, roi)
    spec = padres_spectrum(spectrum, roi)
    out = np.zeros(shape=(spec.shape[0], 2), dtype=np.float64)
    for i in range(spec.shape[0]):
        try:
            out[i, 0], out[i, 1] = fit_spectrum(wl, spec[i,:])
        except Exception as e:
            out[i, :] = np.nan
    return out


# =========================
# Processing functions for background sequence
def background_sequence(bunches, period, phase):
    return np.mod(bunches, period) == phase

def generic_background_sequence(bunches, period, phase, slu_dec, slu_bn_start, slu_sequence):
    if period != -1 and slu_dec == 'OFF':
        # Source background sequence is on
        return np.mod(bunches, period) == phase

    if period == -1 and slu_dec != 'OFF':
        # Slu decimation is on
        slu_period = len(slu_sequence)
        slu_phase = np.mod(slu_bn_start, slu_period)
        return np.logical_not(np.mod(bunches + slu_phase, slu_period))

    return np.full(bunches.shape, False, dtype=bool)


# =========================
# Processing function for online data analysis


def camera_baseline(x):
    """ Camera baseline: convert to float and subtract the Andor camera baseline of 100 counts
    """
    x = np.float64(x) - 100
    return x


def camera_baseline_threshold(x, threshold):
    """ Camera baseline with threshold: convert to float and subtract the Andor camera baseline of 100 counts
    Then apply a threshold to remove noise.
    """
    x = np.float64(x) - 100
    x[x < threshold] = 0
    return x


def tof_baseline(x, baseline_roi=[0, 2000]):
    """ Tof baseline: preprocess digitizer trace removing baseline and inverting signal
    """
    return np.mean(x[:, baseline_roi[0]:baseline_roi[1]], axis=1)[:, np.newaxis] - x


def tof_with_threshold(x, threshold, baseline_roi=[0, 2000]):
    """ Tof with threshold: preprocess digitizer trace removing baseline and inverting signal.
    Then apply a threshold to remove noise and keep only peaks from actual MCP signals.
    """
    x = np.mean(x[:, baseline_roi[0]:baseline_roi[1]], axis=1)[:, np.newaxis] - x
    return x * (x > threshold)


def sample_processing_function(state, bkg_mask, test1, test2):
    if state is None:
        state = {}
        state['bunches'] = np.array([], dtype=np.float64)

    state['bunches'] = np.append(state['bunches'], test1 / float(test2))
    return state


def mbes_retardation(ret_v1, ret_v1_en, ret_v2, ret_v2_en):
    return ret_v1 * ret_v1_en + ret_v2 * ret_v2_en


def integrate_digitizer(trace, peaks, threshold, baseline):
    trace = tof_with_threshold(trace, threshold, baseline)
    out = np.zeros(shape=(trace.shape[0], len(peaks)), dtype=np.float64)
    for i, p in enumerate(peaks):
        out[:, i] = np.sum(trace[:, p[0]:p[1]], axis=1)
    return out
