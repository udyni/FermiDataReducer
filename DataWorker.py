# -*- coding: utf-8 -*-
"""

Data reduction worker

@license: GPL v3 (c)2022
@author: Michele Devetta <michele.devetta@cnr.it>

"""

import sys
import traceback
import os
import re
import time
import glob
import h5py as h5
import numpy as np
import multiprocessing
import queue
import shutil
import copy
import gc

# Logger server
from Logger import Logger


class H5File:
    """ Cached H5 file
    """
    def __init__(self, filename, mode='r'):
        self.__fh = None
        self.__fh = h5.File(filename, mode)
        self.data = {}

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def __getitem__(self, key):
        if key not in self.data:
            self.data[key] = self.__fh[key][()]
        return self.data[key]

    def __getattr__(self, key):
        return getattr(self.__fh, key)

    def close(self):
        try:
            self.__fh.close()
        except:
            pass


class BrokerS2S:
    """ Broker to access S2S datasets (use previously reduced data if available)
    """
    def __init__(self, datasets, filename, logger):
        self.s2s_index = 0
        self.datasets = datasets
        self.logger = logger
        self.__data = {}
        try:
            self.__fh = h5.File(filename, 'r+')
            self.__bn = self.__fh['bunches'][()]
            self.__nshots = len(self.__bn)
            self.__data['bunches'] = self.__bn
        except Exception:
            self.__fh = None
            self.__bn = None
            self.__nshots = None

    def __del__(self):
        try:
            if self.__fh is not None:
                self.__fh.close()
        except:
            pass

    def addFile(self, newfile):
        # First define the total number of shots if not already defined
        if self.__nshots is None:
            self.__nshots = newfile['Files_per_acquisition'] * newfile['ShotsPerFile']

        # Shot in the files
        bn = newfile['bunches']
        fs = len(bn)

        # Check that bunch numbers are loaded
        if self.__bn is None:
            if 'bunches' not in self.__data:
                self.__data['bunches'] = np.zeros(shape=(self.__nshots, ), dtype=bn.dtype)
            self.__data['bunches'][self.s2s_index:self.s2s_index+fs] = bn

        for d in self.datasets:
            tag = d['tag']
            dset = d['dataset']
            last_dset = None
            if self.__fh is not None and tag in self.__fh:
                # Dataset is already present from a previous reduction
                continue

            try:
                # Load data
                f = d.get('processing', None)
                if f is not None:
                    # Check if we need to load multiple datasets
                    if type(dset) is dict:
                        input_dset = {}
                        for k, v in dset.items():
                            self.logger.debug("Loading dataset %s", v)
                            last_dset = v
                            input_dset[k] = newfile[v]
                        input_dset.update(d.get('extra_args', {}))
                        data = f(**input_dset)
                    else:
                        self.logger.debug("Loading dataset %s", dset)
                        last_dset = dset
                        data = f(newfile[dset])
                else:
                    self.logger.debug("Loading dataset %s", dset)
                    last_dset = dset
                    data = newfile[dset]

                if tag not in self.__data:
                    if len(getattr(data, 'shape', ())) > 1:
                        self.__data[tag] = np.zeros(shape=(self.__nshots, ) + data.shape[1:], dtype=data.dtype)
                    else:
                        self.__data[tag] = np.zeros(shape=(self.__nshots, ), dtype=getattr(data, 'dtype', np.float64))

                self.__data[tag][self.s2s_index:self.s2s_index+fs, ...] = data
            except Exception as e:
                self.logger.error("Failed to load dataset '%s' from file '%s' (Error: %s)", last_dset, os.path.basename(newfile.filename), e)
                # If the dataset is float, set as nan the missing values
                if tag in self.__data and self.__data[tag].dtype == np.float64:
                    self.__data[tag][self.s2s_index:self.s2s_index+fs, ...] = np.nan

        # Increment s2s index
        self.s2s_index += fs

        # Return bunches to handle further processing
        return bn

    def getDataset(self, dataset, bunches):
        # Load dataset if is the first access
        if dataset not in self.__data:
            self.__data[dataset] = self.__fh[dataset][()]
        # Create selector
        if self.__bn is not None:
            selector = np.isin(self.__bn, bunches)
        else:
            selector = np.isin(self.__data['bunches'], bunches)
        # Return data
        return self.__data[dataset][selector]

    def __contains__(self, key):
        for d in self.datasets:
            if d['tag'] == key:
                return True
        return False

    def update(self, filename):
        if self.__fh is None:
            self.__fh = h5.File(filename, 'w')
        # Update or create datasets
        for k, v in self.__data.items():
            if k not in self.__fh:
                self.__fh.create_dataset(k, data=v, compression='gzip')

    def save_metadata(self, metadata):
        for k, v in metadata.items():
            if k in self.__fh:
                del self.__fh[k]
            self.__fh.create_dataset(k, data=v)

    def close(self):
        if self.__fh:
            self.__fh.close()
            self.__fh = None


class DataWorker(multiprocessing.Process):

    def __init__(self, job_queue, terminate_flag, options, log_level=Logger.INFO, log_port=9999):
        """ Worker process constructor
        Worker(job_queue)
        """

        # base class initialization
        multiprocessing.Process.__init__(self)

        # Log info
        self.log_level = log_level
        self.log_port = log_port

        # job management stuff
        self.job_queue = job_queue

        # Offline worker (if true exits when the queue is empty)
        self.offline = False

        # Store terminate flag
        self.terminate = terminate_flag

        # Options
        self.options = options

        # Output file suffix
        self.save_suffix = "reduced"

        # Stale times for files and runs
        self.stale_run = multiprocessing.Value('i', 5 * 60)

    def run(self):
        # Connect to logger server
        self.logger = Logger(multiprocessing.current_process().name, self.log_level, f"localhost:{self.log_port}")

        # Start worker
        self.logger.info("Starting worker '%s'", self.name)

        # General pattern to match raw data filenames
        p = re.compile("Run_(\d+)_(\d+).h5")

        while not self.terminate.value:

            # Get a new run job from the queue
            try:
                (run_number, remote_path, local_path, save_path) = self.job_queue.get(block=False)

                # We have a run to reduce
                self.logger.info("Processing run %d", run_number)

                # Extract save path
                if save_path is None:
                    save_path = os.path.join(remote_path, "work")

                # Reset variables to start processing another run
                self.total_files = None     # Total number of files to be processed
                self.file_shots = None     # Total number of shots for S2S datasets
                self.s2s_index = 0
                processed_files = []        # List of files successfully processed
                bad_files = {}              # Dictionary of bad files (key is filename, value is number of failures)
                last_file = time.time()     # Timestamp of when the last file has been processed
                last_file_ts = time.time()  # Timestamp of last modification time
                self.clean()

                # Check if the run has already been reduced, so that we can load s2s datasets
                self.data_s2s = BrokerS2S(self.options['s2s'], os.path.join(save_path, f"Run_{run_number:3d}_s2s.h5"), self.logger)

                # Create local_path if needed
                if local_path is not None:
                    os.makedirs(os.path.join(local_path, "rawdata"), exist_ok=True)
                    os.makedirs(os.path.join(local_path, "work"), exist_ok=True)

                search_path = os.path.join(remote_path, "rawdata", "*.h5")

                while not self.terminate.value:
                    # Scan for new files
                    files = sorted(glob.glob(search_path))
                    new_files = []
                    for f in files:
                        if f not in processed_files:
                            new_files.append(f)

                    if len(new_files):
                        self.logger.debug("[Run %d] Found %d new files", run_number, len(new_files))

                        # We have new files
                        for f in new_files:
                            # We process one file at a time. To keep the ordering, if we found a problem with one file
                            # we exit and start again after waiting a few seconds.

                            # Check filename
                            m = p.match(os.path.basename(f))
                            if m is not None:
                                if int(m.groups()[1]) == 0:
                                    # Bad filename generated by an abort.
                                    # We skip it but we add it to the processed files so that
                                    # we do not have to check it again
                                    processed_files.append(f)
                                    last_file = time.time()
                                    continue

                            # Try to open the file
                            try:
                                # Get file stats
                                s = os.stat(f)

                                # If the file is newer that 120 seconds, check that the file size is stable before trying to open it
                                if (time.time() - s.st_mtime) < 120:
                                    self.logger.debug("[Run %d] File %s is newer that 120s. Checking size", run_number, f)
                                    last_size = s.st_size
                                    count = 0
                                    while True:
                                        time.sleep(2)
                                        s = os.stat(f)
                                        if s.st_size != last_size:
                                            count = 0
                                            last_size = s.st_size
                                        else:
                                            count += 1

                                        # If the size was the same for the last 3 checks than go on
                                        if count >= 3:
                                            break

                                # Try to open file
                                try:
                                    h5.File(f, 'r')
                                    self.logger.debug("[Run %d] Successfully open file %s", run_number, f)
                                except Exception:
                                    self.logger.debug("[Run %d] Failed to open file %s", run_number, f)
                                    # Failed to open the file. It may be still in processing...
                                    if f not in bad_files:
                                        bad_files[f] = 0
                                    bad_files[f] += 1
                                    if bad_files[f] >= 5:
                                        # We try 5 times, then give up and skip the file
                                        processed_files.append(f)
                                        continue
                                    else:
                                        break

                                # Copy new file if local path is defined
                                if local_path is not None:
                                    # Copy file if not already copied
                                    need_copy = False
                                    lf = f.replace(remote_path, local_path)
                                    if os.path.exists(lf):
                                        sl = os.stat(lf)
                                        if sl.st_size != s.st_size:
                                            self.logger.debug("[Run %d] Size of file %s is different (%d != %d)", run_number, f, sl.st_size, s.st_size)
                                            need_copy = True
                                    else:
                                        need_copy = True

                                    if need_copy:
                                        self.logger.info("[Run %d] Copying %s to %s", run_number, f, lf)
                                        shutil.copy2(f, lf)
                                    process_file = lf
                                else:
                                    process_file = f
                            except Exception as e:
                                self.logger.error("[Run %d] Handling of file '%s' failed (Error: %s)", run_number, os.path.basename(f), str(e))
                                break

                            # Process file
                            try:
                                self.process(process_file)
                                processed_files.append(f)
                                last_file = time.time()
                                last_file_ts = os.stat(f).st_mtime
                                # Call garbage collector to remove work variables and prevent memory exhaution
                                gc.collect()

                                self.logger.info("[Run %d] Done processing file %d/%d (%s)", run_number, len(processed_files), self.total_files, os.path.basename(f))

                            except Exception as e:
                                self.logger.error("[Run %d] Failed to process file %s (Error: %s)", run_number, os.path.basename(f), e)
                                # Check if file has been marked as processed
                                if f not in processed_files:
                                    processed_files.append(f)

                            if f in bad_files:
                                del bad_files[f]

                            # Check terminate flag so that we stop even if the new files list is very long...
                            if self.terminate.value:
                                break

                    # Check if the number of processed files is the same as the total number of files
                    if self.total_files is not None and len(processed_files) >= self.total_files:
                        # Finished processing the run
                        break

                    elif len(bad_files) == 0 and time.time() - last_file_ts > 10 * 60:
                        # No bad files and the last file is older that 10 minutes
                        break

                    elif (time.time() - last_file) > self.stale_run.value:
                        # 5 minutes elapsed without any new file. We can assume that the run was aborted.
                        break

                    else:
                        # Wait for new files
                        time.sleep(2.0)

                # Done processing. Save results
                self.save(run_number, save_path)

            except queue.Empty:
                # Nothing to be processed
                if self.offline:
                    break
                time.sleep(2.0)

            except Exception as e:
                # Unhandled exception
                self.logger.error("Unhandled exception (Error: %s)", e)

        # Terminating worker
        self.logger.info("Terminating worker '%s'", self.name)

    def process(self, f):
        # Add one file to the processed data
        try:
            # Open file
            with H5File(f, 'r') as fh:

                # Check if we need to get number of files
                if self.total_files is None:
                    # This is the first file. We should calculate the total number of shots for S2S datasets
                    self.total_files = fh['Files_per_acquisition']
                    self.file_shots = fh['ShotsPerFile']

                # Update shot-to-shot data and get current file bunches
                bn = self.data_s2s.addFile(fh)

                # Load background sequence
                bkg_mask = np.zeros((self.file_shots, ), dtype=np.bool)
                try:
                    if 'background' in self.data_s2s:
                        bkg_mask = self.data_s2s.getDataset('background', bn)

                except Exception as e:
                    self.logger.error("Failed to process background subtraction for file '%s' (Error: %s)", os.path.basename(f), str(e))

                # TODO: Read metadata
                for dset in self.options['metadata']:
                    try:
                        if dset['tag'] not in self.output['metadata']:
                            self.output['metadata'][dset['tag']] = fh[dset['dataset']]
                        else:
                            # TODO: We may check that the metadata is consistent through all the files...
                            pass

                    except Exception as e:
                        self.logger.error("Failed to load metadata '%s' for file '%s' (Error: %s)", dset['tag'], os.path.basename(f), str(e))

                # ============================================================================================================
                # Recursive dataset binning function
                def dataset_binning(dout, binning, mask=np.ones(shape=(self.file_shots,), dtype=np.bool), indexes=()):
                    # Get binning information
                    loc_bin = binning.pop(0)

                    # Dataset must be in s2s data
                    bin_data = self.data_s2s.getDataset(loc_bin['dataset'], bn)
                    if 'preprocessing' in loc_bin:
                        bin_data = loc_bin['preprocessing'](bin_data)

                    # Cycle over bins
                    for i in range(len(loc_bin['bin_edges'])-1):
                        bin_mask = np.logical_and(mask, bin_data >= loc_bin['bin_edges'][i])
                        bin_mask = np.logical_and(bin_mask, bin_data < loc_bin['bin_edges'][i+1])

                        if len(binning):
                            dataset_binning(dout, copy.deepcopy(binning), bin_mask, indexes + (i, ))

                        else:
                            # We have reached the end of the binning definitions
                            self.output[dout]['b_sig']['data'][indexes + (i, )] += np.squeeze(np.sum(data[np.logical_and(bin_mask, s_i), ...], axis=0))
                            self.output[dout]['b_sig']['indexes'][indexes + (i, slice(self.s2s_index, self.s2s_index+self.file_shots), )] = np.logical_and(bin_mask, s_i)
                            if np.sum(b_i):
                                self.output[dout]['b_bkg']['data'][indexes + (i, )] += np.squeeze(np.sum(data[np.logical_and(bin_mask, b_i), ...], axis=0))
                                self.output[dout]['b_bkg']['indexes'][indexes + (i, slice(self.s2s_index, self.s2s_index+self.file_shots), )] = np.logical_and(bin_mask, b_i)

                # ============================================================================================================
                # Check main data reduction
                for m in self.options['main']:
                    tag = m['tag']

                    try:
                        # Load data
                        if 'preprocess' in m and callable(m['preprocess']):
                            data = m['preprocess'](fh[m['dataset']])
                        else:
                            data = fh[m['dataset']]

                        # Data filtering
                        good = np.ones(shape=(data.shape[0], ), dtype=np.bool)
                        if 'filters' in m and len(m['filters']) > 0:
                            for f in m['filters']:
                                try:
                                    d = self.data_s2s.getDataset(f['dataset'], bn)
                                    good = np.logical_and(good, f['processing'](d))
                                except Exception as e:
                                    self.logger.error(f"Failed to filter '{tag}' reduction on '{f['dataset']}' (Error: {e})")

                        # Initialize if needed
                        if tag not in self.output:
                            self.output[tag] = {}

                            # Signal without binning
                            self.output[tag]['sig'] = {}
                            self.output[tag]['sig']['data'] = np.zeros(shape=data.shape[1:], dtype=np.float64)
                            self.output[tag]['sig']['indexes'] = np.zeros(shape=(self.total_files * self.file_shots, ), dtype=np.bool)

                            # Background without binning
                            self.output[tag]['bkg'] = {}
                            self.output[tag]['bkg']['data'] = np.zeros(shape=data.shape[1:], dtype=np.float64)
                            self.output[tag]['bkg']['indexes'] = np.zeros(shape=(self.total_files * self.file_shots, ), dtype=np.bool)

                            # Initialize binned data if needed
                            if 'binning' in m and len(m['binning']) > 0:
                                # Save binning information for later save
                                self.output[tag]['__binning'] = copy.deepcopy(m['binning'])

                                # Check binning dimensions
                                sizes = ()
                                for b in m['binning']:
                                    sizes += (len(b['bin_edges']) - 1, )

                                # Binned signal tensor
                                self.output[tag]['b_sig'] = {}
                                self.output[tag]['b_sig']['data'] = np.zeros(shape=sizes + data.shape[1:], dtype=np.float64)
                                self.output[tag]['b_sig']['indexes'] = np.zeros(shape=sizes + (self.total_files * self.file_shots, ), dtype=np.bool)

                                # Binned background tensor
                                self.output[tag]['b_bkg'] = {}
                                self.output[tag]['b_bkg']['data'] = np.zeros(shape=sizes + data.shape[1:], dtype=np.float64)
                                self.output[tag]['b_bkg']['indexes'] = np.zeros(shape=sizes + (self.total_files * self.file_shots, ), dtype=np.bool)

                        # Signal and background masks
                        s_i = np.logical_and(good, np.logical_not(bkg_mask))
                        b_i = np.logical_and(good, bkg_mask)

                        # Run recursive binning
                        if 'binning' in m and len(m['binning']) > 0:
                            dataset_binning(tag, copy.deepcopy(m['binning']))

                        # Sum without binning
                        self.output[tag]['sig']['data'] += np.squeeze(np.sum(data[s_i, ...], axis=0))
                        self.output[tag]['sig']['indexes'][self.s2s_index:self.s2s_index+self.file_shots] = np.logical_not(bkg_mask)
                        if np.sum(b_i):
                            self.output[tag]['bkg']['data'] += np.squeeze(np.sum(data[b_i, ...], axis=0))
                            self.output[tag]['bkg']['indexes'][self.s2s_index:self.s2s_index+self.file_shots] = bkg_mask

                    except KeyError as e:
                        self.logger.error(f"Failed to reduce '{tag}' data for file '{os.path.basename(f)}' (Missing dataset: '{e}')")

                    except Exception as e:
                        self.logger.error(f"Failed to reduce '{tag}' data for file '{os.path.basename(f)}' (Error: '{e}')")

                # ============================================================================================================
                # Check advanced processing
                if 'advanced' in self.options and len(self.options['advanced']):
                    for adv in self.options['advanced']:
                        try:
                            datasets = {}
                            for k, dset in adv['datasets'].items():
                                datasets[k] = fh[dset]
                        except KeyError as e:
                            self.logger.error("Advanced processing '%s': missing dataset '%s'", adv.get('tag', 'Unknown'), str(e))
                            continue

                        try:
                            if adv['tag'] not in self.output['advanced']:
                                self.output['advanced'][adv['tag']] = None
                            self.output['advanced'][adv['tag']] = adv['processing'](self.output['advanced'][adv['tag']], bkg_mask, **datasets)
                        except Exception as e:
                            self.logger.error("Failed to process advanced output '%s' (Error: %s)", adv.get('tag', 'Unknown'), str(e))

                # Increment S2S index
                self.s2s_index += self.file_shots

        except Exception as e:
            sys_info = sys.exc_info()
            exc_txt = traceback.format_exception(*sys_info)
            self.logger.error("Processing failed with error {0!s}".format(e))
            self.logger.error("".join(exc_txt))

    def __save_dataset(self, obj, name, data, compression=True):
        try:
            if data is None:
                return
            if compression and len(data) == 1 or type(data) in (str, int, float):
                # Disable compression for scalars
                compression = False
            obj.create_dataset(name, data=data, compression='gzip' if compression else None)
        except Exception as e:
            self.logger.error("Failed to save dataset '%s' (Error: %s)", name, e)

    def save(self, run_number, path):
        # Save the results when the run is all processed
        s2s_file = os.path.join(path, f"Run_{run_number:03d}_s2s.h5")
        save_file = os.path.join(path, f"Run_{run_number:03d}_{self.save_suffix}.h5")
        self.logger.info("Saving file %s", save_file)

        try:
            # Create or update s2s
            self.data_s2s.update(s2s_file)

            # Open output file
            with h5.File(save_file, 'w') as fh:

                for k, data in self.output.items():
                    # Save metadata
                    if k == 'metadata':
                        self.data_s2s.save_metadata(data)

                    # Save advanced processing
                    elif k == 'advanced':
                        for k, v in self.output['advanced'].items():
                            try:
                                ga = fh.create_group(k)
                                for tag, dset in v.items():
                                    self.__save_dataset(ga, tag, data=dset)
                            except Exception as e:
                                self.logger.error("Failed to save output of advanced processing '%s' (Error: %s)", k, e)

                    # Save main processing
                    else:
                        gv = fh.create_group(k)
                        self.__save_dataset(gv, 'sig', data=data['sig']['data'])
                        self.__save_dataset(gv, 'sig_indexes', data=data['sig']['indexes'])
                        if 'bkg' in data:
                            self.__save_dataset(gv, 'bkg', data=data['bkg']['data'])
                            self.__save_dataset(gv, 'bkg_indexes', data=data['bkg']['indexes'])
                        if '__binning' in data:
                            # Save bin information
                            n = len(data['__binning'])
                            self.__save_dataset(gv, 'binning_dim', data=n, compression=False)
                            for i in range(n):
                                self.__save_dataset(gv, "binning_dset_{0:d}".format(i), data=data['__binning'][i]['dataset'], compression=False)
                                self.__save_dataset(gv, "binning_edge_{0:d}".format(i), data=data['__binning'][i]['bin_edges'])
                            # Save binned data
                            self.__save_dataset(gv, 'b_sig', data=data['b_sig']['data'])
                            self.__save_dataset(gv, 'b_sig_indexes', data=data['b_sig']['indexes'])
                            if 'bkg' in data:
                                self.__save_dataset(gv, 'b_bkg', data=data['b_bkg']['data'])
                                self.__save_dataset(gv, 'b_bkg_indexes', data=data['b_bkg']['indexes'])

            # Actively close s2s file
            self.data_s2s.close()

        except Exception as e:
            sys_info = sys.exc_info()
            exc_txt = traceback.format_exception(*sys_info)
            self.logger.error("Save failed with error {0!s}".format(e))
            self.logger.error("".join(exc_txt))

    def clean(self):
        # Reset variables to start processing another run
        self.output = {}

        # Metadata
        self.output['metadata'] = {}

        # Advanced processing
        if 'advanced' in self.options:
            self.output['advanced'] = {}
