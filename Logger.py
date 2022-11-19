# -*- coding: utf-8 -*-

""" Logger module

This module implements a general purpose logger class based on the logging
module.

@license: GPL v3 (c)2022
@author: Michele Devetta <michele.devetta@cnr.it>

"""

import logging
import logging.handlers
import sys
import traceback


class Logger(object):

    """ Logger class
    This class defines a simple general purpose logger on top of the standard
    logging module.

    """

    def __init__(self, queue, name="Default", level=logging.INFO):
        """ Constructor.
        Accept as optional input parameters the logger name and the logging
        level.
        """
        self._name = name
        self._hname = name + "_handler"
        self._queue = queue
        self._level = level
        self._init_logger()

    def _init_logger(self):
        """ Initialize the logger object. """
        # Setup logger
        self._logger = logging.getLogger(self._name)
        self._logger.setLevel(self._level)

        if len(self._logger.handlers) == 1 and self._logger.handlers[0].name == self._hname:
            return

        if len(self._logger.handlers) > 0:
            self._logger.handlers = []

        # Add handler
        _handler = logging.handlers.QueueHandler(self._queue)
        _handler.setLevel(self._level)
        _handler.name = self._hname
        self._logger.addHandler(_handler)

    def __getstate__(self):
        """ Enable the logger object to be pickled. """
        odict = self.__dict__.copy()  # copy the dict since we change it
        del odict['_logger']          # remove logger entry
        return odict

    def __setstate__(self, idict):
        """ Enable the logger object to be unpickled. """
        self.__dict__.update(idict)   # restore dict
        self._init_logger()

    def level(self):
        """ Return current logger level. """
        return self._level()

    def critical(self, msg, *args, **kwargs):
        """ Equivalent to logging.critical """
        if 'exc_info' in kwargs and bool(kwargs['exc_info']) == True and self._logger.level != logging.DEBUG:
            kwargs['exc_info'] = False
        self._logger.log(logging.CRITICAL, msg, *args, **kwargs)
        if self._logger.isEnabledFor(logging.DEBUG):
            sys_info = sys.exc_info()
            exc_txt = traceback.format_exception(*sys_info)
            self._logger.log(logging.DEBUG, "".join(exc_txt))

    def error(self, msg, *args, **kwargs):
        """ Equivalent to logging.error """
        if 'exc_info' in kwargs and bool(kwargs['exc_info']) == True and self._logger.level != logging.DEBUG:
            kwargs['exc_info'] = False
        self._logger.log(logging.ERROR, msg, *args, **kwargs)
        if self._logger.isEnabledFor(logging.DEBUG):
            sys_info = sys.exc_info()
            exc_txt = traceback.format_exception(*sys_info)
            self._logger.log(logging.DEBUG, "".join(exc_txt))

    def warning(self, msg, *args, **kwargs):
        """ Equivalent to logging.warning """
        if 'exc_info' in kwargs and bool(kwargs['exc_info']) == True and self._logger.level != logging.DEBUG:
            kwargs['exc_info'] = False
        self._logger.log(logging.WARN, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """ Equivalent to logging.info """
        if 'exc_info' in kwargs and bool(kwargs['exc_info']) == True and self._logger.level != logging.DEBUG:
            kwargs['exc_info'] = False
        self._logger.log(logging.INFO, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        """ Equivalent to logging.debug """
        self._logger.log(logging.DEBUG, msg, *args, **kwargs)

    # Log levels
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARN = logging.WARN
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LoggerListener(object):

    def __init__(self, queue, level):
        self._level = level

        # Default StreamHandler
        h = logging.StreamHandler()
        h.setLevel(level)
        h.setFormatter(logging.Formatter('[%(asctime)s] %(name)s:%(levelname)s:%(message)s', '%b %d, %H:%M:%S'))

        # Create listener
        self._listener = logging.handlers.QueueListener(queue, h)

    def addLogFile(self, filename):
        h = logging.FileHandler(filename)
        h.setLevel(self._level)
        h.setFormatter(logging.Formatter('[%(asctime)s] %(name)s:%(levelname)s:%(message)s', '%b %d, %H:%M:%S'))
        self._listener.handlers = self._listener.handlers + (h, )

    def start(self):
        self._listener.start()

    def stop(self):
        self._listener.stop()
