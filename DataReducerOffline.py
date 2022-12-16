# -*- coding: utf-8 -*-
"""

Multiprocessing based data reducer for FERMI

@license: GPL v3 (c)2022
@author: Michele Devetta <michele.devetta@cnr.it>

"""

import os
import re
import time
import signal
import multiprocessing
import numpy as np

# Logger server
from Logger import Logger, LoggerListener
from DataWorker import DataWorker


class DataReducerOffline(object):

    def __init__(
            self,
            options: dict,
            remote_path: str,
            save_path: str = None,
            save_suffix: str = "reduced",
            nworkers: int = 2,
            log_level: int = Logger.INFO
        ):
        """ Create DataReducerOffline object
            - remote path for raw data
            - save path to save reduced data
            - save suffix to mark the files differently from s2s data
        """
        # Logging
        self._logging_queue = multiprocessing.Queue()
        self._logger_listener = LoggerListener(self._logging_queue, level=log_level)
        self._logger_listener.start()
        self.logger = Logger(self._logging_queue, 'DR OFF Master', log_level)
        self.logger.info("Starting DataReducer")

        # Store data paths
        self.rpath = remote_path
        self.spath = save_path

        # Manager to share variables between processes
        self.terminate_flag = multiprocessing.Value('i', 0)

        # Setup signal handler for termination
        signal.signal(signal.SIGINT, self.intHandler)
        signal.siginterrupt(signal.SIGINT, False)

        # Create job queue
        self.job_queue = multiprocessing.Queue()

        # Start worker processes
        self.workers = []
        for i in range(nworkers):
            self.workers.append(DataWorker(self.job_queue, self.terminate_flag, options, self._logging_queue, log_level))
            self.workers[-1].name = "Worker-{0:d}".format(i+1)
            self.workers[-1].save_suffix = save_suffix

    def intHandler(self, signum, frame):
        # Set terminate flag
        self.logger.info("Got INT signal")
        with self.terminate_flag.get_lock():
            self.terminate_flag.value = 1

    def addLogFile(self, filename: str):
        self._logger_listener.addLogFile(filename)

    def list_experiments(self, basepath):
        exp = []
        ls = os.listdir(basepath)
        for el in ls:
            if el != "results":
                exp.append(el)
        return exp

    def list_runs(self, basepath, experiment):
        ls = os.listdir(os.path.join(basepath, experiment))
        p = re.compile("Run_(\d+)")
        runs = []
        for el in ls:
            fp = os.path.join(basepath, experiment, el)
            if os.path.isdir(fp):
                m = p.match(el)
                if m is not None:
                    runs.append(int(m.groups()[0]))
        return runs

    def run(self, runs2reduce: list):

        # Inspect runs2reduce
        __runs2reduce = []
        for i, r in enumerate(runs2reduce):
            if r is Ellipsis:
                if i == 0 or i == len(runs2reduce) - 1:
                    # First and last element cannot be Ellipsis
                    continue
                s = runs2reduce[i - 1]
                e = runs2reduce[i + 1]
                if s is Ellipsis or e is Ellipsis:
                    continue
                __runs2reduce += list(range(s + 1, e))
            elif type(r) is int:
                __runs2reduce.append(r)

        # Start workers
        for w in self.workers:
            w.start()

        # List experiments
        experiments = self.list_experiments(self.rpath)

        for exp in experiments:

            # List runs for an experiment
            runs = self.list_runs(self.rpath, exp)

            for r in runs:
                if r not in __runs2reduce:
                    continue

                # New run. Create paths
                run_rpath = os.path.join(self.rpath, exp, "Run_{0:03d}".format(r))

                # Submit job to be processed
                self.logger.info("Found new run {0:d}".format(r))
                self.job_queue.put( (r, run_rpath, None, self.spath) )

        # Wait for the queue to empty
        while not self.job_queue.empty():
            if self.terminate_flag.value:
                self.logger.info("Data reduction was interrupted. Terminating...")
                break
            time.sleep(2.0)

        # Check that all workers are now idle
        worker_idle = np.full((len(self.workers), ), False, dtype=bool)
        while True:
            for i, w in enumerate(self.workers):
                if w.idle.value == 1:
                    worker_idle[i] = True
            if np.all(worker_idle):
                break
            time.sleep(2.0)

        # Set the terminate flag to shutdown
        if not self.terminate_flag.value:
            with self.terminate_flag.get_lock():
                self.terminate_flag.value = 1
            self.logger.info("Data reduction complete. Terminating...")

        # Clean up
        for w in self.workers:
            w.join()
            self.logger.info(f"Successfully joined worker {w.name} (Exitcode: {w.exitcode})")

        time.sleep(0.5)

        # Clear job queue
        while not self.job_queue.empty():
            self.job_queue.get()

        self._logger_listener.stop()
