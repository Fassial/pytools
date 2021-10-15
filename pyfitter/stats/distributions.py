"""
Created on 00:50, Oct. 15th, 2021
Author: fassial
Filename: distributions.py
"""
# import distns
from . import _discrete_distns
from . import _continuous_distns
from ._discrete_distns import *
from ._continuous_distns import *

# init __all__
__all__ = []
# add only the distribution names, not the *_gen names
__all__ += _continuous_distns._distn_names
__all__ += _discrete_distns._distn_names
