# -*- coding: utf-8 -*-
"""
LoggerServer

A UDP server that can collect datagrams from logging extension
The server will listen on localhost, port 9999

@license: GPL v3 (c)2021
@author: Michele Devetta <michele.devetta@cnr.it>

"""

import sys
import errno
import threading
import socketserver
import pickle
import logging
import select


class LoggerServerHandler(socketserver.BaseRequestHandler):

    """ Handler log messages sent via UDP. """

    def handle(self):
        """ Handle datagram. """
        try:
            # Discard first 4 bytes (length of message)
            rec = logging.makeLogRecord(pickle.loads(self.request[0][4:]))
            self.server.logger.handle(rec)

        except Exception as e:
            self.server.logger.error("Error handling a log message (Error: %s)", e)


class LoggerServer(threading.Thread):

    """ Logger server main class """

    def __init__(self, name=None, level=logging.INFO, port=9999):
        """ Constructor. """
        # Parent constructor
        threading.Thread.__init__(self)

        # Create logger UDP server
        self.server = socketserver.UDPServer(('localhost', port), LoggerServerHandler)

        # Init server logger
        self.server.logger = logging.getLogger(name)
        self.server.logger.setLevel(level)
        self.server.logger.handlers = []
        h = logging.StreamHandler()
        h.setLevel(level)
        h.setFormatter(logging.Formatter('[%(asctime)s] %(name)s:%(levelname)s:%(message)s', '%b %d, %H:%M:%S'))
        self.server.logger.addHandler(h)

    def addLogFile(self, filename):
        try:
            h = logging.FileHandler(filename)
            h.setLevel(self.server.logger.level)
            h.setFormatter(logging.Formatter('[%(asctime)s] %(name)s:%(levelname)s:%(message)s', '%b %d, %H:%M:%S'))
            self.server.logger.addHandler(h)
        except Exception as e:
            self.server.logger.error("Failed to add log file (Error: %s)", e)

    def run(self):
        """ Main entrypoint of logger thread. """
        self.server.logger.info("Starting logging server.")
        while True:
            try:
                self.server.serve_forever()
                break
            except select.error as e:
                if e.args[0] == errno.EINTR:
                    self.server.logger.debug("Logging server listener was interrupted by a syscall. Restarting.")
                else:
                    raise e
        self.server.logger.info("Logging server terminated")

    def setLevel(self, level):
        """ Set logger level. """
        try:
            level = getattr(logging, level)
        except AttributeError:
            self.server.logger.error("Error setting log level to '%s'.", level)
        else:
            self.server.logger.setLevel(level)
            for h in self.server.logger.handlers:
                h.setLevel(level)
