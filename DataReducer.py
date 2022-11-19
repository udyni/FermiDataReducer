# -*- coding: utf-8 -*-
"""

Multiprocessing based data reducer for FERMI

@license: GPL v3 (c)2022
@author: Michele Devetta <michele.devetta@cnr.it>

"""
import sys
import os
import re
import signal
import time
import multiprocessing
from DataWorker import DataWorker

# Logger server
from Logger import Logger, LoggerListener


class DataReducer(object):

    def __init__(self, options, remote_path, local_path=None, save_path=None, nworkers=2, skip_runs=0, log_level=Logger.INFO):
        """ Create DataReducer object
            - remote path for raw data
            - local path to copy raw data (if 'None' a local copy of the data is not used)
        """
        # Logging
        self._logging_queue = multiprocessing.Queue()
        self._logger_listener = LoggerListener(self._logging_queue, level=log_level)
        self._logger_listener.start()
        self.logger = Logger(self._logging_queue, 'DR Master', log_level)
        self.logger.info("Starting DataReducer")

        # Store data paths
        self.rpath = remote_path
        self.lpath = local_path
        self.spath = save_path

        # Set last run. Useful to restart the application and avoid to rescan all the runs...
        self.last_run = skip_runs

        # Manager to share variables between processes
        self.terminate = multiprocessing.Value('i', 0)

        # Setup signal handler for termination
        signal.signal(signal.SIGINT, self.intHandler)
        signal.siginterrupt(signal.SIGINT, False)

        # Create job queue
        self.job_queue = multiprocessing.Queue()

        # Start worker processes
        self.workers = []
        for i in range(nworkers):
            self.workers.append(DataWorker(self.job_queue, self.terminate, options, self._logging_queue, log_level))
            self.workers[-1].name = f"Worker-{i+1}"
            self.workers[-1].start()

    def intHandler(self, signum, frame):
        # Set terminate flag
        with self.terminate.get_lock():
            self.terminate.value = 1

    def addLogFile(self, filename):
        self._logger_listener.addLogFile(filename)

    def list_experiments(self, basepath):
        exp = []
        ls = sorted(os.listdir(basepath))
        for el in ls:
            if el != "results":
                exp.append(el)
        return exp

    def list_runs(self, basepath, experiment):
        ls = sorted(os.listdir(os.path.join(basepath, experiment)))
        p = re.compile("Run_(\d+)")
        runs = []
        for el in ls:
            fp = os.path.join(basepath, experiment, el)
            if os.path.isdir(fp):
                m = p.match(el)
                if m is not None:
                    runs.append(int(m.groups()[0]))
        return runs

    def run(self):
        # List of known experiments
        known_experiments = []

        # Start main process
        self.logger.info("Starting run broker")

        while not self.terminate.value:

            try:
                # List experiments
                experiments = self.list_experiments(self.rpath)

                for exp in experiments:

                    if exp not in known_experiments:
                        known_experiments.append(exp)
                        self.logger.info("Found new experiment '{0}'".format(exp))

                    # List runs for an experiment
                    runs = self.list_runs(self.rpath, exp)

                    for r in runs:
                        if r <= self.last_run:
                            continue

                        # New run. Create paths
                        run_rpath = os.path.join(self.rpath, exp, "Run_{0:03d}".format(r))
                        run_lpath = run_rpath.replace(self.rpath, self.lpath) if self.lpath is not None else None

                        # Submit job to be processed
                        self.logger.info("Found new run {0:d}".format(r))
                        self.job_queue.put( (r, run_rpath, run_lpath, self.spath) )
                        self.last_run = r
            except FileNotFoundError:
                with self.terminate.get_lock():
                    self.terminate.value = 1

            time.sleep(0.5)

        # Start main process
        self.logger.info("Terminating run broker")

        # Clean up
        for w in self.workers:
            w.join()
            self.logger.info(f"Successfully joined worker {w.name} (Exitcode: {w.exitcode})")

        time.sleep(0.5)

        # Clear job queue
        while not self.job_queue.empty():
            self.job_queue.get()

        self._logger_listener.stop()

    def setStaleTimeRun(self, t):
        for w in self.workers:
            with w.stale_run.get_lock():
                w.stale_run.value = t
