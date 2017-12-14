import time 
import os
import sys
import numpy as np
import importlib
from  .get_args import *
from .io_util import *
from .geotiff_util import *

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

#!!!it does not work yet - do not use it!!!!
def import_file(path):
    cpath, cfile = os.path.split(path)
    sys.path.append(cpath)    
    #import config file
    config = importlib.import_module(cfile.split('.')[0])
    return config
    
