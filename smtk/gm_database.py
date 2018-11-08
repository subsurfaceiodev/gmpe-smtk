#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2018 GEM Foundation and G. Weatherill
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
Basic classes for the GMDatabase (HDF5 database) and parsers
"""

import os
import sys
import re
import csv
import hashlib
import shlex
import tokenize
from tokenize import generate_tokens, TokenError, untokenize
from io import StringIO
from datetime import datetime
from contextlib import contextmanager
from collections import defaultdict
import tables
from tables.file import File
from tables.table import Table
from tables.group import Group
from tables.exceptions import NoSuchNodeError
import numpy as np
from scipy.constants import g

from tables.description import IsDescription, Int64Col, StringCol, \
    Int16Col, UInt16Col, Float32Col, Float16Col, TimeCol, BoolCol, \
    UInt8Col, Float64Col, Int8Col, UInt64Col, UInt32Col, EnumCol
from smtk import sm_utils

# custom defaults. Defaults will be considered equal to "missing":
COLUMN_DEFAULTS = {
    "StringCol": b'',
    # 'BoolCol': False
    "EnumCol": b'',
    "IntCol": np.iinfo(np.int).min,
    "Int8Col": np.iinfo(np.int8).min,
    "Int16Col": np.iinfo(np.int16).min,
    "Int32Col": np.iinfo(np.int32).min,
    "Int64Col": np.iinfo(np.int64).min,
    "UIntCol": 0,
    "UInt8Col": 0,
    "UInt16Col": 0,
    "UInt32Col": 0,
    "UInt64Col": 0,
    "FloatCol": float('nan'),
    "Float16Col": float('nan'),
    "Float32Col": float('nan'),
    "Float64Col": float('nan'),
    "Float96Col": float('nan'),
    "Float128Col": float('nan'),
    "ComplexCol": complex(float('nan'), float('nan')),
    "Complex32Col": complex(float('nan'), float('nan')),
    "Complex64Col": complex(float('nan'), float('nan')),
    "Complex128Col": complex(float('nan'), float('nan')),
    "Complex192Col": complex(float('nan'), float('nan')),
    "Complex256Col": complex(float('nan'), float('nan')),
    # "TimeCol": 0,
    # "Time32Col": 0,
    # "Time64Col": 0
}


# rewrite pytables description column types to account for default
# values meaning MISSING, and bounds (min and max)
def _col(col_class, **kwargs):
    '''utility function returning a pytables Column. The rationale behind
    this simple wrapper (`_col(StringCol, ...)` equals `StringCol(...)`)
    is twofold:

    1. Pytables columns do not allow bounds (min, max), which can be
    specified here as 'min' and 'max' arguments. None or missing values will
    mean: no check on the relative bound (any value allowed)

    2. When inserting/updating records, each missing value will be set as the
    relative column's default (attribute "dflt") defined for the column.
    This value is not always distinguishable from a value input from the user.
    Therefore, when not explicitly provided as 'dflt' argument in `kwargs`,
    this function sets a value which can interpreted as "missing", whenever
    possible, and will be (column types not listed below - e.g. TimeColumns -
    will not change their default):

    -----------------------+----------------------------------------
    column's type          | dflt
    -----------------------+----------------------------------------
    string                 | "" (same as pytables)
    -----------------------+----------------------------------------
    unsigned integer       |
    (uint8, uint16, ...)   | 0 (same as pytables)
    -----------------------+----------------------------------------
    float                  | nan
    (float8, float16, ...) |
    -----------------------+----------------------------------------
    int (int8, int16, ...) | the type minimum value
    -----------------------+----------------------------------------
    Enum                   | "" (if not in the enum list of values,
                           |     it will be added)
    -----------------------+----------------------------------------
    bool                   | False (same as pytables). Note that having
                           | booleans only two possible values, they
                           | can not have a default interpretable as
                           | missing
    -----------------------+----------------------------------------

    :param: col_class: the pytables column class, e.g. StringCol. You can
        also supply the String "DateTime" which will set default to StringCol
        adding the default 'itemsize' to `kwargs` and a custom attribute
        'is_datetime_str' to the returned object. The attribute will be used
        in the `expr` class to properly cast passed values into the correct
        date-time ISO-formatted string
    :param kwargs: keyword argument to be passed to `col_class` during
        initialization. Note thtat the `dflt` parameter, if provided
        will be overridden. See the `atom` module of pytables for a list
        of arguments for each Column class
    '''
    is_iso_dtime = col_class == 'DateTime'
    if is_iso_dtime:
        col_class = StringCol
        if 'itemsize' not in kwargs:
            kwargs['itemsize'] = 19  # '1999-31-12T01:02:59'

    if 'dflt' not in kwargs:
        if col_class.__name__ in COLUMN_DEFAULTS:
            dflt = COLUMN_DEFAULTS[col_class.__name__]
            if col_class == EnumCol:
                dflt = ''
                if dflt not in kwargs['enum']:
                    kwargs['enum'].insert(0, dflt)
            kwargs['dflt'] = dflt

    min_, max_ = kwargs.pop('min', None), kwargs.pop('max', None)
    ret = col_class(**kwargs)
    ret.min_value, ret.max_value = min_, max_
    if is_iso_dtime:
        ret.is_datetime_str = True  # will be used in selection syntax to
        # convert string values in the correct format %Y-%m-%dT%H:%M:%s
    return ret


class GMDatabaseTable(IsDescription):  # pylint: disable=too-few-public-methods
    """
    Implements a GMDatabase as `pytable.IsDescription` class.
    This class is the skeleton of the data structure of HDF5 tables, which
    map flatfiles data (in CSV) in an HDF5 file.

    **Remember that, with the exception of `BoolCol`s, default values
    are interpreted as 'missing'. Usually, no dflt argument has to be passed
    here as it will be set by default (see `_col` function)
    """
    # id = UInt32Col()  # no default. Starts from 1 incrementally
    # max id: 4,294,967,295
    record_id = _col(StringCol, itemsize=20)
    event_id = _col(StringCol, itemsize=20)
    event_name = _col(StringCol, itemsize=40)
    event_country = _col(StringCol, itemsize=30)
    event_time = _col("DateTime")  # In ISO Format YYYY-MM-DDTHH:mm:ss
    # Note: if we want to support YYYY-MM-DD only be aware that:
    # YYYY-MM-DD == YYYY-MM-DDT00:00:00
    # Note2: no support for microseconds for the moment
    event_latitude = _col(Float64Col, min=-90, max=90)
    event_longitude = _col(Float64Col, min=-180, max=180)
    hypocenter_depth = _col(Float32Col)
    magnitude = _col(Float16Col)
    magnitude_type = _col(StringCol, itemsize=5)
    magnitude_uncertainty = _col(Float32Col)
    tectonic_environment = _col(StringCol, itemsize=30)
    strike_1 = _col(Float32Col)
    strike_2 = _col(Float32Col)
    dip_1 = _col(Float32Col)
    dip_2 = _col(Float32Col)
    rake_1 = _col(Float32Col)
    rake_2 = _col(Float32Col)
    style_of_faulting = _col(Float32Col)
    depth_top_of_rupture = _col(Float32Col)
    rupture_length = _col(Float32Col)
    rupture_width = _col(Float32Col)
    station_id = _col(StringCol, itemsize=20)
    station_name = _col(StringCol, itemsize=40)
    station_latitude = _col(Float64Col, min=-90, max=90)
    station_longitude = _col(Float64Col, min=-180, max=180)
    station_elevation = _col(Float32Col)
    vs30 = _col(Float32Col)
    vs30_measured = _col(BoolCol, dflt=True)
    vs30_sigma = _col(Float32Col)
    depth_to_basement = _col(Float32Col)
    z1 = _col(Float32Col)
    z2pt5 = _col(Float32Col)
    repi = _col(Float32Col)  # epicentral_distance
    rhypo = _col(Float32Col)  # Float32Col
    rjb = _col(Float32Col)  # joyner_boore_distance
    rrup = _col(Float32Col)  # rupture_distance
    rx = _col(Float32Col)
    ry0 = _col(Float32Col)
    azimuth = _col(Float32Col)
    digital_recording = _col(BoolCol, dflt=True)
#     acceleration_unit = _col(EnumCol, enum=['cm/s/s', 'm/s/s', 'g'],
#                              base='uint8')
    type_of_filter = _col(StringCol, itemsize=25)
    npass = _col(Int8Col)
    nroll = _col(Float32Col)
    hp_h1 = _col(Float32Col)
    hp_h2 = _col(Float32Col)
    lp_h1 = _col(Float32Col)
    lp_h2 = _col(Float32Col)
    factor = _col(Float32Col)
    lowest_usable_frequency_h1 = _col(Float32Col)
    lowest_usable_frequency_h2 = _col(Float32Col)
    lowest_usable_frequency_avg = _col(Float32Col)
    highest_usable_frequency_h1 = _col(Float32Col)
    highest_usable_frequency_h2 = _col(Float32Col)
    highest_usable_frequency_avg = _col(Float32Col)
    pga = _col(Float64Col)
    pgv = _col(Float64Col)
    pgd = _col(Float64Col)
    duration_5_75 = _col(Float64Col)
    duration_5_95 = _col(Float64Col)
    arias_intensity = _col(Float64Col)
    cav = _col(Float64Col)
    sa = _col(Float64Col, shape=(111,))


class GMDatabaseParser(object):
    '''
    Implements a base class for parsing flatfiles in csv format into
    GmDatabase files in HDF5 format. The latter are Table-like heterogeneous
    datasets (each representing a flatfile) organized in subfolders-like
    structures called groups.
    See the :class:`GmDatabaseTable` for a description of the Table columns
    and types.

    The parsing is done in the `parse` method. The typical workflow
    is to implement a new subclass for each new flatfile released.
    Subclasses should override the `mapping` dict where flatfile
    specific column names are mapped to :class:`GmDatabaseTable` column names
    and optionally the `parse_row` method where additional operation
    is performed in-place on each flatfile row. For more details, see
    the :method:`parse_row` method docstring
    '''
    _accel_units = ["g", "m/s/s", "m/s**2", "m/s^2",
                    "cm/s/s", "cm/s**2", "cm/s^2"]

    _ref_periods = [0.010, 0.020, 0.022, 0.025, 0.029, 0.030, 0.032,
                    0.035, 0.036, 0.040, 0.042, 0.044, 0.045, 0.046,
                    0.048, 0.050, 0.055, 0.060, 0.065, 0.067, 0.070,
                    0.075, 0.080, 0.085, 0.090, 0.095, 0.100, 0.110,
                    0.120, 0.130, 0.133, 0.140, 0.150, 0.160, 0.170,
                    0.180, 0.190, 0.200, 0.220, 0.240, 0.250, 0.260,
                    0.280, 0.290, 0.300, 0.320, 0.340, 0.350, 0.360,
                    0.380, 0.400, 0.420, 0.440, 0.450, 0.460, 0.480,
                    0.500, 0.550, 0.600, 0.650, 0.667, 0.700, 0.750,
                    0.800, 0.850, 0.900, 0.950, 1.000, 1.100, 1.200,
                    1.300, 1.400, 1.500, 1.600, 1.700, 1.800, 1.900,
                    2.000, 2.200, 2.400, 2.500, 2.600, 2.800, 3.000,
                    3.200, 3.400, 3.500, 3.600, 3.800, 4.000, 4.200,
                    4.400, 4.600, 4.800, 5.000, 5.500, 6.000, 6.500,
                    7.000, 7.500, 8.000, 8.500, 9.000, 9.500, 10.000,
                    11.000, 12.000, 13.000, 14.000, 15.000, 20.000]

    # the regular expression used to parse SAs periods. Note capturing
    # group for the SA period:
    _sa_periods_re = re.compile(r'^\s*sa\s*\((.*)\)\s*$',
                                re.IGNORECASE)  # @UndefinedVariable

    # the regular expression used to parse PGA periods. Note capturing
    # group for the PGA unit
    _pga_unit_re = re.compile(r'^\s*pga\s*\((.*)\)\s*$',
                              re.IGNORECASE)  # @UndefinedVariable

    # this field is a list of strings telling which are the column names
    # of the event time. If:
    # 1. A list of a single item => trivial case, it denotes the event time
    # column, which must be supplied as ISO format
    # 2. A list of length 3: => then it denotes the column names of the year,
    # month and day, respectively, all three int-parsable strings
    # 3. A list of length 6: then it denotes the column names of the
    # year, month, day hour minutes seconds, repsectively, all six int-parsable
    # strings
    _event_time_colnames = ['year', 'month', 'day', 'hour', 'minute', 'second']

    # The csv column names will be then converted according to the
    # `mappings` dict below, where a csv flatfile column is mapped to its
    # corresponding Gm database column name. The mapping is the first
    # operation performed on any row. After that:
    # 1. Columns matching `_sa_periods_re` will be parsed and log-log
    # interpolated with '_ref_periods'. The resulting data will be put in the
    # Gm database 'sa' column.
    # 2. If a column 'event_time' is missing, then the program searches
    # for '_event_time_colnames' (ignoring case) and parses the date. The
    # resulting date (in ISO formatted string) will be put in the Gm databse
    # column 'event_time'.
    # 3. If a column matching `_pga_unit_re` is found, then the unit is
    # stored and the Gm databse column 'pga' is filled with the PGA value,
    # converted to cm/s/s.
    # 4. The `parse_row` method is called. Therein, the user should
    # implement any more complex operation
    # 5 a row is written as record of the output HDF5 file. The columns
    # 'event_id', 'station_id' and 'record_id' are automatically filled to
    # uniquely identify their respective entitites
    mappings = {}

    @classmethod
    def parse(cls, flatfile_path, output_path, mode='a'):
        '''Parses a flat file and writes its content in the GM database file
        `output_path`, which is a HDF5 organized hierarchically in groups
        (sort of sub-directories) each of which identifies a parsed
        input flatfile. Each group's `table` attribute is where
        the actual GM database data is stored and can be accessed later
        with the module's :function:`get_table`.
        The group will have the same name as `flatfile_path` (more precisely,
        the file basename without extension).

        :param flatfile_path: string denoting the path to the input CSV
            flatfile
        :param output_path: string: path to the output GM database file.
        :param mode: either 'w' or 'a'. It is NOT the `mode` option of the
            `open_file` function (which is always 'a'): 'a' means append to
            the existing **table**, if it exists (otherwise create a new one),
            'w' means write (i.e. overwrite the existing table, if any).
            In case of 'a' and the table exists, it is up to the user not to
            add duplicated entries
        :return: a dictionary holding information with keys:
            'total': the total number of csv rows
            'written': the number of parsed rows written on the db table
            'error': a list of integers denoting the position
                (0 = first row) of the parsed rows not written on the db table
                because of errors
            'missing_values': a dict with table column names as keys, mapped
                to the number of rows which have missing values for that
                column (e.g., invalid/empty values in the csv, or most
                likely, a column not found, if the number of missing values
                equals 'total').
            'outofbound_values': a dict with table column names as keys,
                mapped to the number of rows which had out-of-bound values for
                that column.

            Missing and out-of-bound values are stored in the GM database with
            the column default, which is usually NaN for floats, the minimum
            possible value for integers, the empty string for strings
        '''
        dbname = os.path.splitext(os.path.basename(flatfile_path))[0]
        with cls.get_table(output_path, dbname, mode) as table:

            i, error, missing, outofbound = \
                -1, [], defaultdict(int), defaultdict(int)

            for i, rowdict in enumerate(cls._rows(flatfile_path)):

                if rowdict:
                    tablerow = table.row
                    missingcols, outofboundcols = \
                        cls._writerow(rowdict, tablerow, dbname)
                    tablerow.append()  # pylint: disable=no-member
                    table.flush()
                else:
                    missingcols, outofboundcols = [], []
                    error.append(i)

                # write statistics:
                for col in missingcols:
                    missing[col] += 1
                for col in outofboundcols:
                    outofbound[col] += 1

            return {'total': i+1, 'written': i+1-len(error), 'error': error,
                    'missing_values': missing, 'outofbound_values': outofbound}

    @staticmethod
    @contextmanager
    def get_table(filepath, name, mode='r'):
        '''Yields a pytable Table object representing a Gm database
        in the given hdf5 file `filepath`. Creates such a table if mode != 'r'
        and the table does not exists.

        Example:
        ```
            with GmDatabaseParser.get_table(filepath, name, 'r') as table:
                # ... do your operation here
        ```

        :param filepath: the string denoting the path to the hdf file
            previously created with this method. If `mode`
            is 'r', the file must exist
        :param name: the name of the database table
        :param mode: the mode ('a', 'r', 'w') whereby the **table** is opened.
            I.e., 'w' does not overwrites the whole file, but the table data.
            More specifically:
            'r': opens file in 'r' mode, raises if the file or the table in
                the file content where not found
            'w': opens file in 'a' mode, creates the table if it does not
                exists, clears all table data if it exists. Eventually it
                returns the table
            'a': open file in 'a' mode, creates the table if it does not
                exists, does nothing otherwise. Eventually it returns the table

        :raises: :class:`tables.exceptions.NoSuchNodeError` if mode is 'r'
            and the table was not found in `filepath`, IOError if the
            file does not exist
        '''
        with tables.open_file(filepath, mode if mode == 'r' else 'a') \
                as h5file:
            table = None
            tablename = 'table'
            tablepath = '/%s/%s' % (name, tablename)
            try:
                table = h5file.get_node(tablepath, classname=Table.__name__)
                if mode == 'w':
                    h5file.remove_node(tablepath, recursive=True)
                    table = None
            except NoSuchNodeError as _:
                if mode == 'r':
                    raise
                table = None
                # create parent group node
                try:
                    h5file.get_node("/%s" % name, classname=Group.__name__)
                except NoSuchNodeError as _:
                    h5file.create_group(h5file.root, name)

            if table is None:
                table = h5file.create_table("/%s" % name, tablename,
                                            description=GMDatabaseTable)
            yield table

    @classmethod
    def _rows(cls, flatfile_path):  # pylint: disable=too-many-locals
        '''Yields each row from the CSV file `flatfile_path` as
        dictionary, after performing SA conversion and running custom code
        implemented in `cls.parse_row` (if overridden by
        subclasses). Yields empty dict in case of exceptions'''
        ref_log_periods = np.log10(cls._ref_periods)
        mappings = getattr(cls, 'mappings', {})
        with cls._get_csv_reader(flatfile_path) as reader:

            newfieldnames = [mappings[f] if f in mappings else f for f in
                             reader.fieldnames]
            # get spectra fieldnames and priods:
            try:
                spectra_fieldnames, spectra_periods =\
                    cls._get_sa_columns(newfieldnames)
            except Exception as exc:
                raise ValueError('Unable to parse SA columns: %s' % str(exc))

            # get event time fieldname(s):
            try:
                evtime_fieldnames = \
                    cls._get_event_time_columns(newfieldnames, 'event_time')
            except Exception as exc:
                raise ValueError('Unable to parse event '
                                 'time column(s): %s' % str(exc))

            # get pga fieldname and units:
            try:
                pga_col, pga_unit = cls._get_pga_column(newfieldnames)
            except Exception as exc:
                raise ValueError('Unable to parse PGA column: %s' % str(exc))

            for rowdict in reader:
                # re-map keys:
                for k in mappings:
                    rowdict[mappings[k]] = rowdict.pop(k)

                # assign values (sa, event time, pga):
                try:
                    rowdict['sa'] = cls._get_sa(rowdict, spectra_fieldnames,
                                                ref_log_periods,
                                                spectra_periods)
                except Exception as _:  # pylint: disable=broad-except
                    pass

                try:
                    rowdict['event_time'] = \
                        cls._get_event_time(rowdict, evtime_fieldnames)
                except Exception as _:  # pylint: disable=broad-except
                    pass

                try:
                    acc_unit = rowdict[pga_unit] \
                        if pga_unit == 'acceleration_unit' else pga_unit
                    rowdict['pga'] = cls._get_pga(rowdict, pga_col, acc_unit)
                except Exception as _:  # pylint: disable=broad-except
                    pass

                try:
                    # custom post processing, if needed in subclasses:
                    cls.parse_row(rowdict)
                except Exception as _:  # pylint: disable=broad-except
                    pass

                if not cls._sanity_check(rowdict):
                    rowdict = {}

                # yield row as dict:
                yield rowdict

    @classmethod
    def _sanity_check(cls, rowdict):
        '''performs sanity checks on the csv row `rowdict` before
        writing it. Note that  pytables does not support roll backs,
        and when closing the file pending data is automatically flushed.
        Therefore, the data has to be checked before, on the csv row'''
        # for the moment, just do a pga/sa[0] check for unit consistency
        # other methods might be added in the future
        return cls._pga_sa_unit_ok(rowdict)

    @classmethod
    def _pga_sa_unit_ok(cls, rowdict):
        '''Checks that pga unit and sa unit are in accordance
        '''
        # if the PGA and the acceleration in the shortest period of the SA
        # columns differ by more than an order of magnitude then certainly
        # there is something wrong and the units of the PGA and SA are not
        # in agreement and an error should be raised.
        try:
            pga, sa0 = float(rowdict['pga']) / (100*g),\
                float(rowdict['sa'][0])
            retol = abs(max(pga, sa0) / min(pga, sa0))
            if not np.isnan(retol) and round(retol) >= 10:
                return False
        except Exception as _:  # disable=broad-except
            # it might seem weird to return true on exceptions, but this method
            # should only check wheather there is certainly a unit
            # mismatch between sa and pga, and int that case only return True
            pass
        return True

    @staticmethod
    @contextmanager
    def _get_csv_reader(filepath, dict_reader=True):
        '''opends a csv file and yields the relative reader. To be used
        in a with statement to properly close the csv file'''
        # according to the docs, py3 needs the newline argument
        kwargs = {'newline': ''} if sys.version_info[0] >= 3 else {}
        with open(filepath, **kwargs) as csvfile:
            reader = csv.DictReader(csvfile) if dict_reader else \
                csv.reader(csvfile)
            yield reader

    @classmethod
    def _get_sa_columns(cls, csv_fieldnames):
        """Returns the field names, the spectra fieldnames and the periods
        (numoy array) of e.g., a parsed csv reader's fieldnames
        """
        spectra_fieldnames = []
        periods = []
        reg = cls._sa_periods_re
        for fname in csv_fieldnames:
            match = reg.match(fname)
            if match:
                periods.append(float(match.group(1)))
                spectra_fieldnames.append(fname)

        return spectra_fieldnames, np.array(periods)

    @staticmethod
    def _get_sa(rowdict, spectra_fieldnames, ref_log_periods, spectra_periods):
        '''gets sa values with log log interpolation if needed'''
        sa_values = np.array([rowdict.get(key) for key in spectra_fieldnames],
                             dtype=float)
        logx = np.log10(spectra_periods)
        logy = np.log10(sa_values)
        return np.power(10.0, np.interp(ref_log_periods, logx, logy))

    @classmethod
    def _get_event_time_columns(cls, csv_fieldnames, default_colname):
        '''returns the event time column names'''
        if default_colname in csv_fieldnames:
            return [default_colname]
        evtime_defnames = {_.lower(): i for i, _ in
                           enumerate(cls._event_time_colnames)}
        evtime_names = [None] * 6
        for fname in csv_fieldnames:
            index = evtime_defnames.get(fname.lower(), None)
            if index is not None:
                evtime_names[index] = fname

        for _, caption in zip(evtime_names, ['year', 'month', 'day']):
            if _ is None:
                raise Exception('column "%s" not found' % caption)

        return evtime_names

    @classmethod
    def _get_event_time(cls, rowdict, evtime_fieldnames):
        '''returns the event time column names'''
        dtime = rowdict[evtime_fieldnames[0]]
        if len(evtime_fieldnames) > 1:
            args = [int(rowdict[fieldname] if i < 3 else
                        rowdict.get(fieldname, 0))
                    for i, fieldname in enumerate(evtime_fieldnames)]
            dtime = datetime(*args)

        return cls.normalize_dtime(dtime)

    @staticmethod
    def normalize_dtime(dtime):
        '''Returns a datetime *string* in ISO format ('%Y-%m-%dT%H:%M:%S')
        representing `dtime`

        :param dtime: string or datetime. In the former case, it must be
            in any of these formats:
            '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y'
        :return: ISO formatted string representing `dtime`
        :raises: ValueError (string not parsable) or TypeError (`dtime`
            neither datetime not string)
        '''
        base_format = '%Y-%m-%dT%H:%M:%S'
        if not isinstance(dtime, datetime):
            formats_ = [base_format, '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y']
            for frmt in formats_:
                try:
                    dtime = datetime.strptime(dtime, frmt)
                    break
                except ValueError:
                    pass
            else:
                raise ValueError('Unparsable as date-time: "%s"' % str(dtime))
        return dtime.strftime(base_format)

    @classmethod
    def _get_pga_column(cls, csv_fieldnames):
        '''returns the column name denoting the PGA and the PGA unit.
        The latter is usually retrieved in the PGA column name. Otherwise,
        if a column 'PGA' *and* 'acceleration_unit' are found, returns
        the names of those columns'''
        reg = cls._pga_unit_re
        # store fields 'pga' and 'acceleration_unit', if present:
        pgacol, pgaunitcol = None, None
        for fname in csv_fieldnames:
            if not pgacol and fname.lower().strip() == 'pga':
                pgacol = fname
                continue
            if not pgaunitcol and fname.lower().strip() == 'acceleration_unit':
                pgaunitcol = fname
                continue
            match = reg.match(fname)
            if match:
                unit = match.group(1)
                if unit not in cls._accel_units:
                    raise Exception('unit not in %s' % str(cls._accel_units))
                return fname, unit
        # no pga(<unit>) column found. Check if we had 'pga' and
        # 'acceleration_unit'
        pgacol_ok, pgaunitcol_ok = pgacol is not None, pgaunitcol is not None
        if pgacol_ok and not pgaunitcol_ok:
            raise ValueError("provide field 'acceleration_unit' or "
                             "specify unit in '%s'" % pgacol)
        elif not pgacol_ok and pgaunitcol_ok:
            raise ValueError("missing field 'pga'")
        elif pgacol_ok and pgaunitcol_ok:
            return pgacol, pgaunitcol
        raise Exception('no matching column found')

    @classmethod
    def _get_pga(cls, rowdict, pga_column, pga_unit):
        '''Returns the pga value from the given `rowdict[pga_column]`
        converted to cm/^2
        '''
        return sm_utils.convert_accel_units(float(rowdict[pga_column]),
                                            pga_unit)

    @classmethod
    def parse_row(cls, rowdict):
        '''This method is intended to be overridden by subclasses (by default
        is no-op) to perform any further operation on the given csv row
        `rowdict` before writing it to the GM databse file.

        Please **keep in mind that**:

        1. This method should process `rowdict` in place, the returned value is
           ignored. Any exception raised here is hanlded in the caller method.
        2. `rowdict` keys might not be the same as the csv
           field names (first csv row). See `mappings` class attribute
        3. The values of `rowdict` are all strings, i.e. they have still to be
           parsed to the correct column type, except those mapped to the keys
           'sa', 'pga' and 'event_time', if present.
        4. the `rowdict` keys 'event_id', 'station_id' and 'record_id' are
           reserved and their values will be anyway overridden, as they
           must represent hash string whereby comparing same
           events, stations and records, respectively

        :param rowdict: a row of the csv flatfile, as Python dict
        '''
        pass

    @classmethod
    def _writerow(cls, csvrow, tablerow, dbname):
        '''writes the content of csvrow into tablerow. Returns two lists:
        The missing column names (a missing column is also a column for which
        the csv value is invalid, i.e. it raised during assignement), and
        the out-of-bounds column names (in case bounds were provided in the
        column class. In this case, the default of that column will be set
        in `tablerow`)'''
        missing_colnames, outofbounds_colnames = [], []
        for col, colobj in tablerow.table.coldescrs.items():
            if col not in csvrow:
                missing_colnames.append(col)
                continue
            try:
                # remember: if val is a castable string -> ok
                #   (e.g. table column float, val is '5.5')
                # if val is out of bounds for the specific type, -> ok
                #   (casted to the closest value)
                # if val is scalar and the table column is a N length array,
                # val it is broadcasted
                #   (val= 5, then tablerow will have a np.array of N 5s)
                # TypeError is raised when there is a non castable element
                #   (e.g. 'abc' for a Float column): in this case pass
                tablerow[col] = csvrow[col]

                bound = getattr(colobj, 'min_value', None)
                if bound is not None and \
                        (np.asarray(tablerow[col]) < bound).any():
                    tablerow[col] = colobj.dflt
                    outofbounds_colnames.append(col)
                    continue

                bound = getattr(colobj, 'max_value', None)
                if bound is not None and \
                        (np.asarray(tablerow[col]) > bound).any():
                    tablerow[col] = colobj.dflt
                    outofbounds_colnames.append(col)
                    continue  # actually useless, but if we add code below ...

            except (ValueError, TypeError):
                missing_colnames.append(col)

        # build a record hashes as ids:
        evid, staid, recid = cls.get_ids(tablerow, dbname)
        tablerow['event_id'] = evid
        tablerow['station_id'] = staid
        tablerow['record_id'] = recid

        return missing_colnames, outofbounds_colnames

    @classmethod
    def get_ids(cls, tablerow, dbname):
        '''Returns the tuple record_id, event_id and station_id from
        the given HDF5 row `tablerow`'''
        toint = cls._toint
        ids = (dbname,
               toint(tablerow['pga'], 0),  # (first two decimals of pga in g)
               toint(tablerow['event_longitude'], 5),
               toint(tablerow['event_latitude'], 5),
               toint(tablerow['hypocenter_depth'], 3),
               tablerow['event_time'],
               toint(tablerow['station_longitude'], 5),
               toint(tablerow['station_latitude'], 5))
        # return event_id, station_id, record_id:
        return cls._hash(*ids[2:6]), cls._hash(*ids[6:]), cls._hash(*ids)

    @classmethod
    def _toint(cls, value, decimals):
        '''returns an integer by multiplying value * 10^decimals
        and rounding the result to int. Returns nan if value is nan'''
        return value if np.isnan(value) else \
            int(round((10**decimals)*value))

    @classmethod
    def _hash(cls, *values):
        '''generates a 160bit (20bytes) hash bytestring which uniquely
        identifies the given tuple of `values`.
        The returned string is assured to be the same for equal `values`
        tuples (note that the order of values matters).
        Conversely, the probability of colliding hashes, i.e., returning
        the same bytestring for two different tuples of values, is 1 in
        100 billion for roughly 19000 hashes (roughly 10 flatfiles with
        all different records), and apporaches 50% for for 1.42e24 hashes
        generated (for info, see
        https://preshing.com/20110504/hash-collision-probabilities/#small-collision-probabilities)

        :param values: a list of values, either bytes, str or numeric
            (no support for other values sofar)
        '''
        hashalg = hashlib.sha1()
        # use the slash as separator as it is unlikely to be in value(s):
        hashalg.update(b'/'.join(cls._tobytestr(v) for v in values))
        return hashalg.digest()

    @classmethod
    def _tobytestr(cls, value):
        '''converts a value to bytes. value can be bytes, str or numeric'''
        if not isinstance(value, bytes):
            value = str(value).encode('utf8')
        return value

#########################################
# Database selection / maniuplation
#########################################


def get_dbnames(filepath):
    '''Returns he database names of the given Gm database (HDF5 file)
    The file should have been created with the `GMDatabaseParser.parse`
    method.

    :param filepath: the path to the HDF5 file
    :return: a list of strings identyfying the database names in the file
    '''
    with tables.open_file(filepath, 'r') as h5file:
        root = h5file.get_node('/')
        return [group._v_name for group in  # pylint: disable=protected-access
                h5file.list_nodes(root, classname='Group')]
        # note: h5file.walk_groups() might raise a ClosedNodeError.
        # This error is badly documented (as much pytables styff),
        # the only mention is (pytables pdf doc): "CloseNodeError: The
        # operation can not be completed because the node is closed. For
        # instance, listing the children of a closed group is not allowed".
        # I suspect it deals with groups deleted / overwritten and the way
        # hdf5 files mark portions of files to be "empty". However,
        # the list_nodes above seems not to raise anymore


def get_table(filepath, dbname):
    '''Returns a Gm database table from the given database name `dbname`
    located in the specific HDF5 file with path `filepath`. To be used within
    a "with" statement:
    ```
    with get_table(filepath, dbname):
        # do your stuff here
    '''
    return GMDatabaseParser.get_table(filepath, dbname, 'r')


def records_where(table, condition, limit=None):
    '''Returns an iterator yielding records (Python dicts) of the
    database table "dbname" stored inside the HDF5 file with path `filepath`.
    The records returned will be filtered according to `condition`.
    IMPORTANT: This function is designed to be used inside a `for ...` loop
    to avoid loading all data into memory. Do **not** do this as it fails:
    `list(records_where(...))`.
    If you want all records in a list (be aware of potential meory leaks
    for huge amount of data) use :function:`read_where`

    Example:
    ```
        # given a variable dtime representing a datetime object:

        condition = between('pga', 0.14, 1.1) & ne('pgv', 'nan') & \
                    lt('event_time', dtime)

        with get_table(...) as table:
            for rec in records_where(table, condition):
                # do your stuff with `rec`, e.g. access the fields:
                sa = rec["sa"]
                pga = rec['pga']  # and so on...
    ```
    The same can be obtained by specifying `condition` with the default
    pytables string expression syntax (note however that
    this approach has some caveats, see [1]):
    ```
        condition = "(pga < 0.14) | (pga > 1.1) & (pgv == pgv) & \
            (event_time < %s)" % \
            dtime.strftime(''%Y-%m-%d %H:%M:%S').encode('utf8')

        # the remainder of the code is the same as the example above
    ```

    :param table: The pytables Table object. See module function `get_table`
    :param condition: a string expression denoting a selection condition.
        See https://www.pytables.org/usersguide/tutorials.html#reading-and-selecting-data-in-a-table

        `condition` can be also given with the expression objects imoplemented
        in this module, which handle some caveats (see note [1]) and also
        ease the casting and construction of string expressions from python
        variables. All expression objeects are actually enhanced Python strings
        supporting also logical operators: negation ~, logical and & and or |.
        They are:
        ```
        eq(column, *values)  # column value equal to any given value
        ne(column, *values)  # column value differs from all given value(s)
        lt(column, value)  # column value lower than the given value
        gt(column, value)  # column value greater than the given value
        le(column, value)  # column value lower or equal to the given value
        ge(column, value)  # column value greater or equal to the given value
        between(column, min, max)  # column between (or equal to) min and max
        isaval(column)  # column value is available (i.e. not the default)
            # (for boolean columns, isaval always returns all records)
        ```
        All values can be given as Python objects or strings (including
        'nan' or float('nan')): the casting is automatically done according to
        the column type

    :param limit: integer (defaults: None) implements a SQL 'limit'
        when provided, yields only the first `limit` matching rows

    --------------------------------------------------------------------------

    [1] The use of the module level expression objects in the `condition`
    handles some caveats that users implementing strings should be
    aware of:
    1. expressions concatenated with & or | should be put into brakets:
        "(pga <= 0.5) & (pgv > 9.5)"
    2. NaNs should be compared like this:
        "pga != pga"  (pga is nan)
        "pga == pga"  (pga is not nan)
    3. String column types (e.g., 'event_country') should be compared with
    quoted strings:
        "event_country == 'Germany'" (or "Germany")
    (in pytables documentation, they claim that in Python3 the above do not
    work either, as they should be encoded into bytes:
    "event_country == %s" % "Germany".encode('utf8')
    **but** when tested in Python3.6.2 these work, so the claim is false or
    incomplete. Maybe it works as long as `value` has ascii characters only?).
    '''
    for count, row in enumerate(table.iterrows() if condition in ('', None)
                                else table.where(_parse_condition(condition))):
        if limit is None or count < limit:
            yield row


def read_where(table, condition, limit=None):
    '''Returns a list of records (Python dicts) of the
    database table "dbname" stored inside the HDF5 file with path `filepath`.
    The records returned will be filtered according to `condition`.
    IMPORTANT: This function loads all data into memory
    To avoid potential memory leaks (especially if for some reason
    `condition` is 'True' or 'true' or None), use :function:`records_where`.

    All parameters are the same as :function:`records_where`
    '''
    return table.read_where(_parse_condition(condition))[:limit]


def _parse_condition(condition):
    '''processes the given `condition` string (numexpr syntax) to be used
    in record selection in order to handle some caveats when using numexpr
    syntax in pytables selection:
    1. expressions concatenated with & or | should be put into brakets:
        "(pga <= 0.5) & (pgv > 9.5)". This function raises if the logical
        operators are not preceeded by a ")" or not followed by a "("
    2. NaNs should be compared like this:
        "pga != pga"  (pga is nan)
        "pga == pga"  (pga is not nan)
        This method converts expression of the type "column != nan" to
        "column == column"
    3. String column types (e.g., 'event_country') should be compared with
    bytes strings in Python 3:
        "event_country == b'Germany'"
    This method checks for quoted strings, unquotes them and converts to
    bytes, if necessary (py3).
    Note: The last conversion (reported in pytables documentation) is made for
    safety **but** when tested in Python3.6.2 these work, so the claim is
    false or incomplete. Maybe it works as long as `value` has ascii
    characters only?).
    '''
    py3 = sys.version_info[0] >= 3
    nan_operators = {'==': '!=', '!=': '=='}
    nan_indices = []
    strings_indices = []
    result = []

    def last_tokenstr():
        return '' if not result else result[-1][1]

    def raise_invalid_logical_op_if(bool_value):
        if bool_value:
            raise ValueError('Logical operators (&|~) allowed only with '
                             'parenthezised expressions')

    STRING, OP, NAME = tokenize.STRING, tokenize.OP, tokenize.NAME
    try:
        for token in generate_tokens(StringIO(condition).readline):
            tokentype, tokenstr = token[0], token[1]

            raise_invalid_logical_op_if(tokenstr in ('&', '|')
                                        and last_tokenstr() != ')')
            raise_invalid_logical_op_if(last_tokenstr() in ('~', '|', '&')
                                        and tokenstr not in ('~', '('))

            if tokentype == STRING and py3 and tokenstr[0] != 'b':
                strings_indices.append(len(result))
            elif tokentype == NAME and tokenstr in ('nan', 'NAN', 'NaN') \
                    and len(result) > 1:
                if result[-2][0] == NAME and result[-1][0] == OP:
                    operator = result[-1][1]
                    if operator not in nan_operators:
                        raise ValueError('only != and == can be compared '
                                         'with nan')
                    nan_indices.append(len(result))

            result.append(list(token))

    except TokenError as terr:
        # tokenizer seems to do some weird stuff at the end of the parsed
        # stringas, raising TokenErrors for "unclosed string or brakets".
        # We do not want to raise this kind of stuff, as the idea here is
        # to check only for logical operatorsm, nans, and bytes conversion
        if untokenize(result).strip() != condition.strip():
            raise ValueError(str(terr))

    raise_invalid_logical_op_if(last_tokenstr() in ('&', '|', '~'))

    # replace nans and strings at the real end:
    for i in strings_indices:
        tokenstr = result[i][1]
        string = shlex.split(tokenstr)[0]
        result[i][1] = str(string.encode('utf8'))

    for i in nan_indices:
        varname = result[i-2][1]
        operator = result[i-1][1]
        result[i-1][1] = nan_operators[operator]
        result[i][1] = varname

    return untokenize(result)
