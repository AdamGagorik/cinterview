"""
Python logging helpers.
"""
import contextlib
import logging
import sys


def setup_logging(level=logging.DEBUG, filename='tree.log'):
    """
    Configure logging module.

    Parameters:
        level (int): logging level
        filename (str): log file name

    Examples:
        >>> setup_logging(level=logging.DEBUG)
        >>> logging.debug('Hello!')
    """
    lfmt = '[%(levelname)s][%(asctime)s][%(process)d][%(name)s] %(message)s'
    dfmt = '%Y-%m-%d][%H:%M:%S'

    logging.basicConfig(level=level, format=lfmt, datefmt=dfmt)

    if filename:
        formatter = logging.Formatter(fmt=lfmt, datefmt=dfmt)
        handler = logging.FileHandler(filename=filename)
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logging.getLogger().addHandler(handler)

    logging.addLevelName(logging.CRITICAL, 'F')
    logging.addLevelName(logging.WARNING, 'W')
    logging.addLevelName(logging.ERROR, 'E')
    logging.addLevelName(logging.DEBUG, 'D')
    logging.addLevelName(logging.INFO, 'I')


@contextlib.contextmanager
def capture(fatal=True):
    """
    Capture exceptions and log them.

    Parameters:
        fatal (bool): exit program if an exception is raised?

    Examples:
        >>> with capture(fatal=True):
        >>>     raise RuntimeError('This error is logged!')
    """
    try:
        logging.captureWarnings(True)
        try:
            yield
        except Exception:
            logging.exception('caught unhandled exception!')
            if fatal:
                logging.error('exiting')
                sys.exit(-1)
    finally:
        logging.captureWarnings(False)
