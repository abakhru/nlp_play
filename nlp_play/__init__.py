"""Top-level package for NLP Play."""

__author__ = """Amit Bakhru"""
__email__ = 'bakhru@me.com'
__version__ = '0.1.0'

import logging

from colorlog import ColoredFormatter

LOGGER = logging.getLogger(__name__)

LOG_FORMAT = ('%(asctime)s '
              '%(log_color)s'
              '%(process)d %(name)s %(levelname)s | %(pathname)s:%(lineno)s | '
              '%(reset)s'
              '%(log_color)s%(message)s%(reset)s')
V_LEVELS = {0: logging.ERROR,
            1: logging.WARNING,
            2: logging.INFO,
            3: logging.DEBUG,
            }
stream = logging.StreamHandler()
stream.setFormatter(ColoredFormatter(LOG_FORMAT))
level = V_LEVELS.get(logging.INFO, logging.DEBUG)
logging.basicConfig(handlers=[stream], level=level)
LOGGER.setLevel('DEBUG')
