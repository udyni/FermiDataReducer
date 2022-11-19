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

# Logger server
from Logger import Logger, LoggerListener
from DataWorker import DataWorker


class DataReducerOffline(object):

    def __init__(self, options, remote_path, save_path=None, save_suffix="reduced", nworkers=2, log_level=Logger.INFO):
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

    def addLogFile(self, filename):
        if self.log_server is not None:
            self.log_server.addLogFile(filename)

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

    def run(self, runs2reduce):

        # Start workers
        for w in self.workers:
            w.start()

        # List experiments
        experiments = self.list_experiments(self.rpath)

        for exp in experiments:

            # List runs for an experiment
            runs = self.list_runs(self.rpath, exp)

            for r in runs:
                if r not in runs2reduce:
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
            time.sleep(5.0)

        # Set the terminate flag to shutdown
        if not self.terminate_flag.value:
            with self.terminate_flag.get_lock():
                self.terminate_flag.value = 1
            self.logger.info("Date reduction complete. Terminating...")

        # Clean up
        for w in self.workers:
            self.logger.info("Joining %s...", w.name)
            w.join()

        time.sleep(0.5)

        # Clear job queue
        while not self.job_queue.empty():
            self.job_queue.get()

        self._logger_listener.stop()