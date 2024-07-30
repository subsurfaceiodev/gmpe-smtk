#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2017 GEM Foundation and G. Weatherill
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.
"""
Strong motion utilities.
"""
# WARNING: this module is intended to collect functions used in various places
# throughout the code. Consequently, try to limit the amount of stuff here and in
# particular the amount of imports, which might slow down the code unnecessarily
import os
import sys
import re
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.constants import g

if sys.version_info[0] >= 3:
    import pickle
else:
    import cPickle as pickle  # pylint: disable=import-error


def get_time_vector(time_step, number_steps):
    """
    General SMTK utils
    """
    return np.cumsum(time_step * np.ones(number_steps, dtype=float)) - time_step


def nextpow2(nval):
    m_f = np.log2(nval)
    m_i = np.ceil(m_f)
    return int(2.0 ** m_i)


def convert_accel_units(acceleration, from_, to_='cm/s/s'):  # noqa
    """
    Converts acceleration from/to different units

    :param acceleration: the acceleration (numeric or numpy array)
    :param from_: unit of `acceleration`: string in "g", "m/s/s", "m/s**2",
        "m/s^2", "cm/s/s", "cm/s**2" or "cm/s^2"
    :param to_: new unit of `acceleration`: string in "g", "m/s/s", "m/s**2",
        "m/s^2", "cm/s/s", "cm/s**2" or "cm/s^2". When missing, it defaults
        to "cm/s/s"

    :return: acceleration converted to the given units (by default, 'cm/s/s')
    """
    m_sec_square = ("m/s/s", "m/s**2", "m/s^2")
    cm_sec_square = ("cm/s/s", "cm/s**2", "cm/s^2")
    acceleration = np.asarray(acceleration)
    if from_ == 'g':
        if to_ == 'g':
            return acceleration
        if to_ in m_sec_square:
            return acceleration * g
        if to_ in cm_sec_square:
            return acceleration * (100 * g)
    elif from_ in m_sec_square:
        if to_ == 'g':
            return acceleration / g
        if to_ in m_sec_square:
            return acceleration
        if to_ in cm_sec_square:
            return acceleration * 100
    elif from_ in cm_sec_square:
        if to_ == 'g':
            return acceleration / (100 * g)
        if to_ in m_sec_square:
            return acceleration / 100
        if to_ in cm_sec_square:
            return acceleration

    raise ValueError("Unrecognised time history units. "
                     "Should take either ''g'', ''m/s/s'' or ''cm/s/s''")


def get_velocity_displacement(time_step, acceleration, units="cm/s/s",
                              velocity=None, displacement=None):
    """
    Returns the velocity and displacment time series using simple integration
    :param float time_step:
        Time-series time-step (s)
    :param numpy.ndarray acceleration:
        Acceleration time-history
    :returns:
        velocity - Velocity Time series (cm/s)
        displacement - Displacement Time series (cm)
    """
    acceleration = convert_accel_units(acceleration, units)
    if velocity is None:
        velocity = time_step * cumulative_trapezoid(acceleration, initial=0.)
    if displacement is None:
        displacement = time_step * cumulative_trapezoid(velocity, initial=0.)
    return velocity, displacement


def _save_image(filename, fig, format='png', dpi=300, **kwargs):  # noqa
    """
    Saves the matplotlib figure `fig` to `filename`. Wrapper around `fig.savefig`
    with `dpi=300` by default and `format` inferred from `filename` extension
    or, if no extension is found, set as "png".
    If filename is empty this function does nothing and return

    :param str filename: str, the file path
    :param figure: a :class:`matplotlib.figure.Figure` (e.g. via
        `matplotlib.pyplot.figure()`)
    :param format: string, the image format. Default: 'png'. This argument is
        ignored if `filename` has a file extension, as `format` will be set
        equal to the extension without leading dot.
    :param str kwargs: additional keyword arguments to pass to `fig.savefig`.
        For details, see:
        https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
    """
    if not filename:
        return

    name, ext = os.path.splitext(filename)
    if ext:
        format = ext[1:]  # noqa
    else:
        filename = name + '.' + format

    fig.savefig(filename, dpi=dpi, format=format, **kwargs)


def load_pickle(pickle_file):
    """
    Python 2 & 3 compatible way of loading a Python Pickle file
    """
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


# mean utilities (geometric, arithmetic, ...):
SCALAR_XY = {
    "Geometric": lambda x, y: np.sqrt(x * y),
    "Arithmetic": lambda x, y: (x + y) / 2.,
    "Larger": lambda x, y: np.max(np.array([x, y]), axis=0),
    "Vectorial": lambda x, y: np.sqrt(x ** 2. + y ** 2.)
}


def get_interpolated_period(target_period, periods, values):
    """
    Returns the spectra interpolated in loglog space

    :param float target_period: Period required for interpolation
    :param np.ndarray periods: Spectral Periods
    :param np.ndarray values: Ground motion values
    """
    if (target_period < np.min(periods)) or (target_period > np.max(periods)):
        raise ValueError("Period not within calculated range: %s" %
                         str(target_period))
    lval = np.where(periods <= target_period)[0][-1]
    uval = np.where(periods >= target_period)[0][0]

    if (uval - lval) == 0:
        return values[lval]

    d_y = np.log10(values[uval]) - np.log10(values[lval])
    d_x = np.log10(periods[uval]) - np.log10(periods[lval])
    return 10.0 ** (
            np.log10(values[lval]) +
            (np.log10(target_period) - np.log10(periods[lval])) * d_y / d_x
    )
