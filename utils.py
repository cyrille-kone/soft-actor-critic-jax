# coding=utf-8
r"""
@git  Team
PyCharm Editor
"""
import logging
import configparser
from pathlib import Path
from typing import Optional, List
import jax.numpy as jnp

# configure logger for the module
logger = logging.getLogger(__name__)
# create console handler
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)


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
