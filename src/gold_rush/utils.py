import numpy as np
import matplotlib.pyplot as plt
import corner

import zeus21
import gold_rush.fitting

from gold_rush.model import Signal21cm
from gold_rush.fitting import *

def get_class_vars(cls):
    return {
        k: v for k, v in cls.__dict__.items()
        if not callable(v) and not k.startswith('__')
    }