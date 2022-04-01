# coding=utf-8
import logging
import configparser
from pathlib import Path
import sys
from typing import Optional, List
import jax.numpy as jnp
import sys

# configure console logger for the module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False
# create console handler
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
handler.setLevel(logging.INFO)
logger.addHandler(handler)
# create file handler
file_handler = logging.FileHandler("to_plot.log", 'w')
file_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

def load_config(*argkeys, **IGNORE):
    r'''
     Read a key from the config file
     Parameters
     ---------
     argkeys: Iterable
             List of (section, keyname) tuple to read from the config file
     Returns
     -------
     None or key vale

     Test
     -----
     >>> load_config(("DEFAULT", "BATCH_SIZE"), ("DEFAULT", "LR"))
    '''
    config_parser = configparser.ConfigParser()
    config_file_path = Path(__file__).parent / "config.ini"
    # This function does not raise an exception
    # Even if the file does not exist
    # Instead it return an empty list
    is_file_read = config_parser.read(config_file_path)
    if not is_file_read:
        logger.warning('Config file not found !')
        # we should return nothing
        return
    key_values = []
    for (section, key) in argkeys:
        key_value_read = None
        try:
            key_value_read = config_parser[section][key]
        except KeyError:
            logger.error("Key %s for section %s was not found in the config file", key, section)
        finally:
            key_values += [key_value_read]
    return key_values

def mse_loss(x, y):
    return jnp.mean(jnp.square(x - y))
