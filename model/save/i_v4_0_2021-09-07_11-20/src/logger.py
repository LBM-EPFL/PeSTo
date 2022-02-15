"""This file contains the logger object to print and save the debug informations and results."""
import os
import pandas as pd
from time import time
from datetime import timedelta


class Logger:
    """Object to store and write debug informations and results."""

    def __init__(self, log_dir, log_name, verbose=True):
        """Create logger with a log directory and an identifing name."""
        # define log filepath
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_str_filepath = os.path.join(log_dir, log_name+'.log')
        self.log_lst_filepath = os.path.join(log_dir, log_name+'.dat')

        # define logs
        self.log_s = ''
        self.log_l = []

        # debug flag
        self.verbose = verbose

        # start timer
        self.t0 = time()
        self.ts = self.t0

    def print(self, line_raw):
        """Write string to log file and print to console if verbose active."""
        # convert line to string
        line = str(line_raw)

        # update log and append to log file
        self.log_s += line + '\n'

        # update log file
        with open(self.log_str_filepath, 'a') as fs:
            fs.write(line + '\n')

        # debug print
        if self.verbose:
            print(line)

    def store(self, **kwargs):
        """Write to file any optional arguments in json format."""
        # update list log
        self.log_l.append(kwargs)

        # create series
        s = pd.Series(kwargs)

        # update log file
        with open(self.log_lst_filepath, 'a') as fs:
            fs.write(s.to_json()+'\n')

    def print_profiling_info(self, n_curr, n_step, n_total):
        """Log print profiling information about elapsed time and estimated time of arrival."""
        dt = time() - self.t0
        dts = time() - self.ts
        eta = (n_total-n_curr)*dt/n_step
        self.print("> Elapsed time: {}".format(timedelta(seconds=dt)))
        self.print("> Since last call: {}".format(timedelta(seconds=dts)))
        self.print("> ETA: {}".format(timedelta(seconds=eta)))
        self.ts = time()

    def restart_timer(self):
        """Restart timer for the profiler."""
        self.ts = time()
