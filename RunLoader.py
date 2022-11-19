# -*- coding: utf-8 -*-
""" RunLoader module

This module implements Run and RunCollection, two classes to conveniently access reduced data from the DataReducer.

@license: GPL v3 (c)2022
@author: Michele Devetta <michele.devetta@cnr.it>

"""

import os
import numpy as np
import h5py as h5
from matplotlib import pyplot as plt


class RunGroup:
    """ This class represent a HDF5 Group inside the reduced data
    Should not be used alone.
    """
    def __init__(self, fh, h5path):
        self.fh = fh
        self.h5path = h5path

    def keys(self):
        return list(self.fh[self.h5path].keys())

    def __getattr__(self, key):
        # Compute full path
        path = "/".join((self.h5path, key))
        # Get element
        data = self.fh[path]
        # Check type
        if type(data) is h5.Dataset:
            data = data[()]
        elif type(data) is h5.Group:
            data = RunGroup(self.fh, path)
        # Store for future (caching)
        setattr(self, key, data)
        # Return data
        return data

class Run:
    """ This class represent a Run. It should be constructed with the basepath where to search for reduced files
    and the run number. The path can be a local directory with all the reduced files inside or the main directory
    structure with the raw data.
    """
    def __init__(self, basepath, run, suffix="reduced"):
        self.fh = None
        self.fh_s2s = None
        self.run = run
        self.suffix = suffix

        # Format names
        name = f"Run_{self.run:03d}_{self.suffix}.h5"
        name_s2s = f"Run_{self.run:03d}_s2s.h5"

        if os.path.exists(os.path.join(basepath, name)):
            # Reduced file is in basepath
            self.filename = os.path.join(basepath, name)
        else:
            # Scan experiments in basepath
            elements = sorted(os.listdir(basepath))
            elements = ['', ] + elements  # Add empty experiment to check if basepath is already an experiment
            for e in elements:
                if e == 'results':
                    continue
                if os.path.isdir(os.path.join(basepath, e)):
                    # Check if run is here
                    if os.path.isdir(os.path.join(basepath, e, f"Run_{run:03d}")):
                        self.filename = os.path.join(basepath, e, f"Run_{run:03d}", 'work', name)
                        break
            else:
                # Run was not found!
                raise OSError(f"Run {self.run} not found")

        # Check if we have the s2s file
        self.filename_s2s = os.path.join(os.path.dirname(self.filename), name_s2s)
        if not os.path.exists(self.filename_s2s):
            self.filename_s2s = None

        try:
            self.fh = h5.File(self.filename, 'r')
            if self.filename_s2s is not None:
                self.fh_s2s = h5.File(self.filename_s2s, 'r')
        except Exception as e:
            print(f"Failed to open file '{self.filename}' for run {self.run:d} (Error: {e})")
            raise e

    def __del__(self):
        try:
            if self.fh is not None:
                self.fh.close()
            if self.fh_s2s is not None:
                self.fh_s2s.close()
        except:
            pass

    def keys(self):
        k = list(self.fh.keys())
        if self.fh_s2s:
            k += list(self.fh_s2s.keys())
        return sorted(k)

    def __getattr__(self, key):
        try:
            data = self.fh[key]
            f = self.fh
        except Exception as e:
            # Not found
            if self.fh_s2s is not None:
                data = self.fh_s2s[key]
                f = self.fh_s2s
            else:
                raise e

        if type(data) is h5.Dataset:
            data = data[()]
        elif type(data) is h5.Group:
            data = RunGroup(f, key)
        setattr(self, key, data)
        return data

    def __lt__(self, obj):
        return self.run < obj.run
    def __le__(self, obj):
        return self.run <= obj.run
    def __gt__(self, obj):
        return self.run > obj.run
    def __ge__(self, obj):
        return self.run >= obj.run
    def __eq__(self, obj):
        return self.run == obj.run
    def __ne__(self, obj):
        return self.run != obj.run


class RunCollection:
    """ This represent a collection of Runs. Useful to cycle of many runs.
    """
    def __init__(self, basepath, runs=[], suffix="reduced"):
        self.__runs = []
        self.basepath = basepath
        self.suffix = suffix
        for r in runs:
            self.__runs.append(Run(basepath, r, suffix))
        self.__runs.sort()

    def runs(self):
        return [r.run for r in self.__runs]

    def __len__(self):
        return len(self.__runs)

    def __getitem__(self, key):
        for r in self.__runs:
            if r.run == key:
                return r
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        if key != value.run:
            raise ValueError("Key must correspond to run number")
        self.__runs.append(value)
        self.__runs.sort()

    def addRuns(self, run):
        if type(run) not in (tuple, list):
            run = [run, ]
        for r in run:
            val = Run(self.basepath, r, self.suffix)
            self.__runs.append(val)
        self.__runs.sort()

    def addRunRange(self, first, last):
        self.addRuns(list(range(first, last+1)))

    class _RunCollectionIterator:
        def __init__(self, collection):
            self._collection = collection
            self._runs = collection.runs()
            self._index = 0

        def __next__(self):
            if self._index < len(self._runs):
                val = self._collection[self._runs[self._index]]
                self._index += 1
                return val
            raise StopIteration

    def __iter__(self):
        return self._RunCollectionIterator(self)
